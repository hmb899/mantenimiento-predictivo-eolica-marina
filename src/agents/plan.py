"""
agente 2, planificador

mismo diseno que el agente 1 (diagnose.py), las reglas deciden los numeros y el llm solo
redacta. el plan lo produce NSGA-II (solver_tools.solve_plan), nunca el llm.

build_request
    construye un PlanRequest validado con los AlarmReport escalados, el catalogo y el escenario
    el subsistema sale de las reglas, o se pasa a mano en los casos de estudio
    la duracion, el coste y el riesgo salen del catalogo src/scheduler/kb/tareas.yaml
    la ventana viene de la urgencia del diagnostico
    en modo llm el modelo elige el subsistema, con fallback a las reglas

explain
    el knee del frente es el plan elegido por regla, el llm redacta el trade-off
    cita solo numeros del frente, con fallback a plantilla
    las alternativas son los extremos del frente en coste, riesgo e impacto, deterministas

comparar_escenarios
    narra la diferencia entre dos escenarios ya resueltos
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from src.agents.contracts import (AlarmReport, CandidatePlan, MaintenancePlan, ParetoFront,
                                   PlanRequest, TipoPrecursor, TriageAction)
from src.agents.llm import LLMClient
from src.agents.tools.catalog_tools import catalog_task, recursos_catalogo, subsistemas_catalogo
from src.agents.tools.env_tools import load_kb, veto_jackup_periodo_sensible

SYSTEM_PROMPT = """Eres un planificador de mantenimiento offshore que explica un plan a quien debe
DECIDIR (tecnico o ingeniero de O&M). El plan lo produce un optimizador multi-objetivo (coste,
riesgo de fallo, impacto ambiental sobre la fauna del Mar del Norte); tu tarea NO es decidir, sino
explicar de forma clara y accionable QUE hacer y QUE implica. Cita SOLO numeros que te den; no
inventes valores."""

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

# traduccion representativa de tipo de precursor a subsistema del catalogo
# los canales estan anonimizados y no dejan identificar el subsistema fisico
# los casos de estudio pueden fijarlo a mano, y el coste es un valor relativo
_SUBSISTEMA_POR_TIPO = {
    # ruptura de correlaciones, degradacion mecanica lenta
    TipoPrecursor.RELACIONAL: "multiplicadora",
    # grupo coherente de sensores del mismo subsistema
    TipoPrecursor.GRUPO_COHERENTE: "rodamiento_principal",
    # un unico sensor se dispara, fallo local o electrico
    TipoPrecursor.CANAL_UNICO: "convertidor",
    TipoPrecursor.MIXTO: "generador",
    TipoPrecursor.DIFUSO: "generico",
    TipoPrecursor.SIN_SENAL: "generico",
}


def _obj_tuple(cp: CandidatePlan):
    # el orden importa, _alternativas indexa por posicion
    o = cp.objetivos
    return o.coste_esperado, o.riesgo_esperado, o.impacto_ambiental


def _fmt_sched(cp: CandidatePlan, req: PlanRequest) -> str:
    partes = []
    for s in cp.schedule:
        f = req.t0 + timedelta(days=s.dia_inicio)  # dia relativo pasado a fecha
        partes.append(f"{s.task_id}@d{s.dia_inicio} ({MESES[f.month - 1]})")
    return ", ".join(partes)


class PlannerAgent:
    def __init__(self, llm: LLMClient | None = None, trace=None) -> None:
        self.llm = llm
        self.trace = trace


    # build_request
    def build_request(self, reports: list[AlarmReport], t0: date, horizonte_dias: int, *,
                      subsistemas: dict[int, str] | None = None, con_restricciones: bool = True,
                      weather_provider: str = "synthetic", n_escenarios: int = 50, seed: int = 42,
                      incluir: tuple[TriageAction, ...] = (TriageAction.ESCALAR,),
                      mode: str = "rules", politica: str = "derate") -> PlanRequest:
        """convierte en tareas los AlarmReport con triaje en `incluir` y devuelve un
        PlanRequest. por defecto solo ESCALAR, que es la salida fiable del agente 1.
        con_restricciones activa el calendario de fauna como objetivo blando y el veto de
        jack-up como objetivo duro."""

        subsistemas = subsistemas or {}
        activos = [r for r in reports if r.triage.action in incluir]
        if not activos:
            raise ValueError("ningun report con triaje en 'incluir': no hay tareas que planificar")

        tasks = []
        for r in activos:
            ev = r.evidencia
            sub = subsistemas.get(ev.event_id) or self._subsistema(r, mode)
            urg = r.urgencia_dias if r.urgencia_dias is not None else horizonte_dias

            # ventana generosa, no un corte duro en la urgencia, que chocaria con el veto
            # y degeneraria el frente; urg es el tiempo a fallo sin actuar y al deratear el
            # riesgo sube monotono a 1 en factor_ventana_derating por urg y diferir penaliza f2
            t = catalog_task(sub, task_id=f"{ev.farm}-ev{ev.event_id}", farm=ev.farm,
                             asset_id=ev.asset_id, origen_event_id=ev.event_id,
                             ventana_fin_dia=horizonte_dias - 1)
            crec = max(t.riesgo_crecimiento_dia,
                       round((t.p_fallo_max - t.riesgo_fallo_base)
                             / max(t.factor_ventana_derating * int(urg), 1.0), 6))
            t = t.model_copy(update={"riesgo_crecimiento_dia": crec})
            tasks.append(t)
            if self.trace:
                self.trace.log("task_from_report", event_id=ev.event_id, subsistema=sub,
                               urgencia_dias=urg, riesgo_crec=crec, jackup=t.requiere_jackup)

        periodos = load_kb()[0] if con_restricciones else []
        vetos = veto_jackup_periodo_sensible(periodos) if con_restricciones else []
        return PlanRequest(t0=t0, horizonte_dias=horizonte_dias, tasks=tasks,
                           recursos=recursos_catalogo(), n_escenarios_meteo=n_escenarios,
                           weather_provider=weather_provider, seed=seed,
                           periodos_ambientales=periodos, vetos_duros=vetos, politica_espera=politica)


    def _subsistema(self, report: AlarmReport, mode: str) -> str:
        """devuelve la clave del catalogo. en modo llm decide el modelo, con fallback a la
        regla."""

        regla = _SUBSISTEMA_POR_TIPO.get(report.tipo_precursor, "generico")
        if mode != "llm" or self.llm is None:
            return regla
        try:
            sub = self._subsistema_llm(report)
            if self.trace:
                self.trace.log("subsistema_llm", event_id=report.evidencia.event_id,
                               llm=sub, regla=regla, concordante=(sub == regla))
            return sub
        except Exception as exc:  # noqa: BLE001
            if self.trace:
                self.trace.log("subsistema_fallback", event_id=report.evidencia.event_id,
                               motivo=str(exc))
            return regla


    def _subsistema_llm(self, report: AlarmReport) -> str:
        catalogo = subsistemas_catalogo()
        sys = ("Eres un ingeniero de mantenimiento. Dado el diagnostico de una alarma, ELIGE el "
               "subsistema mas probable de esta lista (responde SOLO con una palabra de la lista, "
               "sin texto adicional): " + ", ".join(catalogo) + ".")
        ev = report.evidencia

        user = (f"tipo_precursor={report.tipo_precursor.value} canal={ev.canal_disparador} "
                f"margen={ev.margen_sh} hipotesis={report.hipotesis}\nSubsistema:")

        txt = self.llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}])
        # el modelo puede devolver su razonamiento delante, se descarta
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip().lower()
        for k in catalogo:  # primera clave del catalogo mencionada
            if k in txt:
                return k
        raise ValueError(f"el LLM no eligio un subsistema del catalogo: {txt[:60]!r}")


    # explain
    def explain(self, front: ParetoFront, req: PlanRequest, reports: list[AlarmReport], *,
                mode: str = "narrate", escenario_desc: str = "") -> MaintenancePlan:
        """elige el knee del frente por regla, pide la narrativa al llm con fallback a
        plantilla, y anade las alternativas."""

        idx = front.knee_idx if front.knee_idx is not None else 0
        elegido = front.soluciones[idx]
        alternativas = self._alternativas(front, idx, req)
        narrativa = self._narrativa_fallback(front, elegido, req, escenario_desc)
        if mode == "narrate" and self.llm is not None:
            try:
                narrativa = self._narrar(front, elegido, req, escenario_desc)
                if self.trace:
                    self.trace.log("explain_narrate", knee_idx=idx,
                                   coste=elegido.objetivos.coste_esperado)
            except Exception as exc:  # noqa: BLE001
                if self.trace:
                    self.trace.log("narrate_fallback", motivo=str(exc))
        return MaintenancePlan(elegido=elegido, narrativa=narrativa, alternativas=alternativas,
                               reports_origen=[r.evidencia.event_id for r in reports],
                               trace_id=(self.trace.trace_id if self.trace else None))


    def _alternativas(self, front: ParetoFront, knee_idx: int, req: PlanRequest) -> list[str]:
        """un one-liner por extremo del frente, distinto del knee, para ver el rango
        disponible."""

        sols = front.soluciones
        C = [_obj_tuple(s) for s in sols]
        etiquetas = {0: "min coste", 1: "min riesgo", 2: "min impacto ambiental"}
        vistos, out = {knee_idx}, []
        for eje, nombre in etiquetas.items():
            i = min(range(len(sols)), key=lambda j: C[j][eje])  # mejor solucion en ese eje
            if i in vistos:
                continue  # ya salio como knee o como otro extremo
            vistos.add(i)
            c, r, a = C[i]
            out.append(f"{nombre}: [{_fmt_sched(sols[i], req)}] coste {c:,.0f}, riesgo {r:.2f}, impacto {a:.1f}")
        return out


    def _contexto_numerico(self, front: ParetoFront, elegido: CandidatePlan, req: PlanRequest,
                           escenario_desc: str) -> str:
        c, r, a = _obj_tuple(elegido)
        C = [_obj_tuple(s) for s in front.soluciones]
        cmin, cmax = min(x[0] for x in C), max(x[0] for x in C)
        amin, amax = min(x[2] for x in C), max(x[2] for x in C)
        veto = "; ".join(v.motivo for v in req.vetos_duros) or "ninguna"
        lineas = [
            f"escenario: {escenario_desc or req.weather_provider}",
            f"tareas: " + "; ".join(
                f"{t.task_id} {t.subsistema}{' [jack-up]' if t.requiere_jackup else ''} "
                f"ventana d{t.ventana_inicio_dia}-d{t.ventana_fin_dia}" for t in req.tasks),
            f"restricciones ambientales (veto duro): {veto}",
            f"PLAN ELEGIDO (knee): [{_fmt_sched(elegido, req)}] coste {c:,.0f}, riesgo {r:.2f}, impacto {a:.1f}",
            f"rango del frente ({len(front.soluciones)} soluciones): coste [{cmin:,.0f}, {cmax:,.0f}], "
            f"impacto ambiental [{amin:.1f}, {amax:.1f}]",
        ]
        if front.baseline_greedy is not None:
            gc, gr, ga = _obj_tuple(front.baseline_greedy)
            lineas.append(f"politica greedy (miope, ignora ambiente): coste {gc:,.0f}, riesgo {gr:.2f}, impacto {ga:.1f}")
        return "\n".join(lineas)


    def _narrar(self, front: ParetoFront, elegido: CandidatePlan, req: PlanRequest,
                escenario_desc: str) -> str:
        user = (self._contexto_numerico(front, elegido, req, escenario_desc)
                + "\n\nRedacta la explicacion en TRES lineas con estas etiquetas exactas:\n"
                "'Recomendacion:' cuando intervenir, en lenguaje llano (con el mes).\n"
                "'Por que:' que equilibra y que restriccion ambiental respeta.\n"
                "'A que atenerse:' cuanto cuesta frente a la alternativa mas barata y que se "
                "sacrificaria.\nCita SOLO los numeros de arriba. Sin markdown, sin preambulo.")
        txt = self.llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": user}])
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
        return txt[:700] or self._narrativa_fallback(front, elegido, req, escenario_desc)


    def _narrativa_fallback(self, front: ParetoFront, elegido: CandidatePlan, req: PlanRequest,
                            escenario_desc: str) -> str:
        c, r, a = _obj_tuple(elegido)
        cmin = min(s.objetivos.coste_esperado for s in front.soluciones)
        por = "equilibra coste, riesgo e impacto ambiental"
        if req.vetos_duros:
            por += "; respeta el veto de " + "; ".join(v.motivo for v in req.vetos_duros)
        aten = f"coste esperado {c:,.0f} (riesgo {r:.2f}, impacto {a:.1f})"
        if c - cmin > 1:  # el sobrecoste solo se menciona si es apreciable
            aten += (f"; ~{c - cmin:,.0f} mas que la opcion mas barata del frente, "
                     "a cambio de menor impacto y/o riesgo")
        return (f"Recomendacion: intervenir en [{_fmt_sched(elegido, req)}].\n"
                f"Por que: {por}.\n"
                f"A que atenerse: {aten}.")


    # ¿y si...?
    def comparar_escenarios(self, caso_a, caso_b, *, mode: str = "narrate") -> str:
        """narra la diferencia entre dos escenarios ya resueltos.
        caso_x es una tupla de etiqueta, PlanRequest, ParetoFront y MaintenancePlan."""

        (la, ra, _fa, pa), (lb, rb, _fb, pb) = caso_a, caso_b
        ctx = "\n".join([
            f"[{la}] plan [{_fmt_sched(pa.elegido, ra)}] coste {pa.elegido.objetivos.coste_esperado:,.0f} "
            f"riesgo {pa.elegido.objetivos.riesgo_esperado:.2f} impacto {pa.elegido.objetivos.impacto_ambiental:.1f}",
            f"[{lb}] plan [{_fmt_sched(pb.elegido, rb)}] coste {pb.elegido.objetivos.coste_esperado:,.0f} "
            f"riesgo {pb.elegido.objetivos.riesgo_esperado:.2f} impacto {pb.elegido.objetivos.impacto_ambiental:.1f}",
        ])
        fallback = (f"Comparativa {la} vs {lb}: el impacto ambiental pasa de "
                    f"{pa.elegido.objetivos.impacto_ambiental:.1f} a {pb.elegido.objetivos.impacto_ambiental:.1f} "
                    f"y el coste de {pa.elegido.objetivos.coste_esperado:,.0f} a "
                    f"{pb.elegido.objetivos.coste_esperado:,.0f}.")
        if mode != "narrate" or self.llm is None:
            return fallback
        try:
            user = (ctx + f"\n\nEn 2-3 frases, di a quien decide que cambia entre '{la}' y '{lb}', "
                    "QUE conviene hacer y por que (ventana meteorologica, periodo sensible de fauna). "
                    "Cita SOLO los numeros de arriba. Sin markdown.")
            txt = self.llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user}])
            return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()[:600] or fallback
        except Exception:  # noqa: BLE001
            return fallback
