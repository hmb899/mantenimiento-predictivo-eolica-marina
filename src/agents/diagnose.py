"""
agente 1, diagnostico de alarmas

DiagnoseAgent
    el LLM va en el borde, recibe el expediente objetivo (AlarmEvidence, computado por
    xai_tools) y produce un AlarmReport (tipo de precursor, hipotesis y triaje) validado
    contra el contrato
    si el LLM falla o no valida, el fallback determinista (clasificar_precursor mas reglas de
    triaje) produce un AlarmReport correcto, o sea que el pipeline nunca se cae por el LLM

F4 implementa el prompt con few-shot de los casos de los notebooks (ev11/ev66/ev53/ev86), los
umbrales calibrados y la evaluacion de triaje (CARE post-filtro sobre los 15 eventos de B)
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.agents.contracts import (AlarmEvidence, AlarmReport, TipoPrecursor, TriageAction,
                                   TriageDecision)
from src.agents.llm import LLMClient
from src.agents.tools.xai_tools import clasificar_precursor

SYSTEM_PROMPT = """Eres un ingeniero de diagnostico de aerogeneradores offshore. Recibes el
expediente objetivo de una alarma (evidencia computada, con numeros verificados) y produces un
diagnostico estructurado. NO recibes etiquetas de fallo: juzga SOLO por la evidencia. REGLAS:
(1) cita SOLO numeros presentes en la evidencia, jamas inventes valores; (2) clasifica el tipo
de precursor segun los umbrales indicados; (3) decide el triaje: escalar (precursor creible y
persistente), vigilar (senal debil o ambigua) o descartar (patron de falsa alarma: excursion
transitoria que revierte, margen bajo, sin coherencia entre canales); (4) se breve y tecnico."""

UMBRALES = """Taxonomia del precursor (umbrales calibrados):
- canal_unico: el counterfactual del canal top ya explica >=80% de la caida del score.
- grupo_coherente: la caida es escalonada (varios canales del mismo subsistema).
- difuso: el counterfactual acumulado top-5 explica <50% (senal repartida, sin foco).
- relacional: IG propio <10% (el sensor no se desvia; el resto de la turbina deja de predecirlo).
- mixto: IG propio 20-50%.
- sin_senal: margen S/h < 1 o sin alarma."""

# few-shot de los casos caracterizados en los notebooks, evidencia y diagnostico correcto
FEWSHOT = [
    ("farm=C event=11 margen_sh=600.3 lead_h=664.7 canal=val__sensor_15 "
     "top=[sensor_15 z264, sensor_186 z95, sensor_99 z36] cf_acum=[64,86,88,89,92,93] "
     "ig_contexto=84% ig_propio=2%",
     '{"tipo_precursor":"relacional","confianza":0.85,"hipotesis":"Precursor relacional fuerte '
     '(margen 600x, IG propio 2%): sensor_15 deja de ser predicho por la turbina semanas antes; '
     'el counterfactual acumulado (64->93%) revela un grupo coherente 15/186.","triage":'
     '{"action":"escalar","razones":["margen 600x","IG propio 2% = deriva relacional","cf 64->93% grupo"]},'
     '"urgencia_dias":21}'),
    ("farm=C event=15 margen_sh=0.2 lead_h=null canal=null top=[] cf_acum=[] "
     "ig_contexto=null ig_propio=null",
     '{"tipo_precursor":"sin_senal","confianza":0.9,"hipotesis":"Sin alarma al punto de operacion '
     '(margen 0.2 < 1): no hay precursor en los valores.","triage":{"action":"descartar",'
     '"razones":["margen 0.2 < 1","sin canal disparador"]},"urgencia_dias":null}'),
]

# traduccion a lenguaje llano del precursor, la decision no cambia, solo se explica mejor
# con canales anonimizados no se puede nombrar el componente fisico, asi que el subsistema y su
# coste son una asignacion representativa por tipo de patron, y la narrativa del plan lo dice
_QUE_PASA = {
    TipoPrecursor.RELACIONAL: "el sensor {canal} (indicador) ya no encaja con lo que predice el resto "
                              "de la turbina: el componente que vigila se esta desviando (deriva "
                              "relacional), patron compatible con una degradacion incipiente semanas antes",
    TipoPrecursor.CANAL_UNICO: "el sensor {canal} marca una anomalia localizada que no arrastra al resto, "
                               "tipico de un fallo local o electrico en el componente que monitoriza",
    TipoPrecursor.GRUPO_COHERENTE: "varios sensores del mismo subsistema marcan una desviacion coordinada, "
                                   "compatible con la degradacion del componente que vigilan",
    TipoPrecursor.MIXTO: "el sensor {canal} se desvia y ademas arrastra a su contexto (patron mixto)",
    TipoPrecursor.DIFUSO: "la senal esta repartida entre muchos sensores, sin un foco claro",
    TipoPrecursor.SIN_SENAL: "ningun sensor marca una desviacion relevante",
}


class _Diagnosis(BaseModel):
    """salida estructurada del LLM, solo el diagnostico, la evidencia no se re-emite."""

    tipo_precursor: TipoPrecursor
    confianza: float = Field(ge=0.0, le=1.0)
    hipotesis: str
    triage: TriageDecision
    urgencia_dias: int | None = Field(default=None, ge=0)


def _evidence_prompt(ev: AlarmEvidence) -> str:
    top = ", ".join(f"{c.canal} z{c.z:.0f}" for c in ev.top_canales[:5]) or "-"
    return (f"farm={ev.farm} event={ev.event_id} asset={ev.asset_id} detector={ev.detector}\n"
            f"margen_sh={ev.margen_sh} lead_h={ev.lead_h}\n"
            f"canal_disparador={ev.canal_disparador}\ntop_canales=[{top}]\n"
            f"counterfactual_acumulado_top_k(% caida del score)={ev.cf_acumulado_pct}\n"
            f"ig_contexto_pct={ev.ig_contexto_pct} ig_propio_pct={ev.ig_propio_pct}")


class DiagnoseAgent:
    def __init__(self, llm: LLMClient | None = None, trace=None, use_slope: bool = False) -> None:
        self.llm = llm
        self.trace = trace
        self.use_slope = use_slope  # rescate por crecimiento (z_slope) en banda debil


    def run(self, evidencia: AlarmEvidence, mode: str = "narrate") -> AlarmReport:
        """produce el AlarmReport en dos modos, 'narrate' donde deciden las reglas y el LLM
        solo redacta, y 'decide' donde el LLM decide todo. sin LLM los dos caen al fallback."""

        if mode == "decide":
            # peor en la prueba, sobre-escala los normales debiles, se conserva para comparar
            return self._decide_with_llm(evidencia)
        # 'narrate' es el recomendado en produccion, las reglas calibradas deciden tipo, triaje
        # y urgencia de forma reproducible y el LLM no puede corromper el numero
        rep = self.fallback(evidencia)  # las reglas deciden, es lo autoritativo
        if self.llm is not None:
            try:
                rep = rep.model_copy(update={"hipotesis": self._narrate(evidencia, rep)})
                if self.trace:
                    self.trace.log("diagnose_narrate", event_id=evidencia.event_id,
                                   tipo=rep.tipo_precursor.value, action=rep.triage.action.value)
            except Exception as exc:  # noqa: BLE001
                if self.trace:
                    self.trace.log("narrate_fallback", event_id=evidencia.event_id, motivo=str(exc))
        return rep


    def _narrate(self, evidencia: AlarmEvidence, rep: AlarmReport) -> str:
        """el LLM reescribe en prosa natural el diagnostico ya redactado por reglas,
        manteniendo sentido, framing y numeros."""

        # anclarlo al fallback evita que derive a cosas como 'revisar el sensor' o 'planificar
        # en X dias', y si algo falla se devuelve la propia plantilla
        sys = ("Eres un ingeniero de fiabilidad que explica una alarma a quien decide el mantenimiento "
               "(tecnico/ingeniero de O&M). Te doy un diagnostico YA redactado por reglas; reescribelo en "
               "prosa natural y clara MANTENIENDO el mismo sentido, las mismas tres lineas "
               "('Recomendacion:', 'Que pasa:', 'A que atenerse:') y los mismos numeros. Reglas que NO "
               "puedes romper: (1) el sensor es un INDICADOR — nunca digas 'revisar/cambiar el sensor'; lo "
               "que se degrada es el COMPONENTE que vigila; (2) la urgencia es el tiempo estimado hasta el "
               "FALLO si NO se actua, NO un plazo para actuar. No inventes numeros. Sin markdown, sin preambulo.")
        user = "DIAGNOSTICO YA REDACTADO (reescribelo natural, mismo sentido y numeros):\n" + rep.hipotesis
        txt = self.llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}])
        # se quita el bloque <think> de los modelos que razonan en voz alta y se corta a 700
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
        return txt[:700] or rep.hipotesis


    def _decide_with_llm(self, evidencia: AlarmEvidence) -> AlarmReport:
        """el LLM estructurado decide todo, se valida, y si falla se cae al fallback. es el
        modo inferior, ver run."""

        if self.llm is None:
            return self.fallback(evidencia)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + UMBRALES}]
        # los ejemplos entran como turnos alternos de usuario y asistente, no dentro del system
        for ev_txt, diag in FEWSHOT:
            msgs.append({"role": "user", "content": "EVIDENCIA:\n" + ev_txt})
            msgs.append({"role": "assistant", "content": diag})
        msgs.append({"role": "user", "content": "EVIDENCIA:\n" + _evidence_prompt(evidencia)})
        try:
            d = self.llm.structured(msgs, _Diagnosis)
            return AlarmReport(evidencia=evidencia, tipo_precursor=d.tipo_precursor,
                               confianza=d.confianza, hipotesis=d.hipotesis, triage=d.triage,
                               urgencia_dias=d.urgencia_dias)
        except Exception as exc:  # noqa: BLE001
            if self.trace:
                self.trace.log("diagnose_fallback", event_id=evidencia.event_id, motivo=str(exc))
            return self.fallback(evidencia)


    def fallback(self, evidencia: AlarmEvidence) -> AlarmReport:
        """diagnostico por reglas, sin LLM, clasificar_precursor mas triaje por umbrales."""

        tipo, conf = clasificar_precursor(evidencia)
        action, razones, urgencia = self._triage(evidencia, tipo, self.use_slope)
        return AlarmReport(
            evidencia=evidencia, tipo_precursor=tipo, confianza=conf,
            hipotesis=self._explicacion(evidencia, tipo, action, urgencia),
            triage=TriageDecision(action=action, razones=razones), urgencia_dias=urgencia)

    MARGEN_FUERTE = 5.0  # margen S/h por encima del cual el precursor es claro
    # por debajo = alarma transitoria, activa <10% de las ventanas tras disparar
    PERSIST_MIN = 0.10
    SLOPE_MIN = 2.0  # pre-registrado, z_slope >= 2 train-std = trayectoria divergiendo


    @classmethod
    def _triage(cls, ev: AlarmEvidence, tipo: TipoPrecursor, use_slope: bool = False):
        """reglas de triaje label-free, con la evidencia y el tipo. los umbrales estan
        pre-registrados y no se ajustaron al resultado."""

        # en la banda debil decide la persistencia, un precursor real es sostenido y se vigila,
        # una falsa alarma es transitoria y se descarta
        if tipo is TipoPrecursor.SIN_SENAL:
            return TriageAction.DESCARTAR, [f"margen {ev.margen_sh} < 1 o sin canal"], None
        fuerte = {TipoPrecursor.CANAL_UNICO, TipoPrecursor.GRUPO_COHERENTE,
                  TipoPrecursor.RELACIONAL, TipoPrecursor.MIXTO}
        if tipo in fuerte and ev.margen_sh >= cls.MARGEN_FUERTE:
            urg = 21 if ev.lead_h is None else int(min(30, max(3, round(ev.lead_h / 24.0))))
            return (TriageAction.ESCALAR,
                    [f"precursor {tipo.value}", f"margen {ev.margen_sh:.0f}x"], urg)
        # banda debil, margen por debajo de MARGEN_FUERTE
        p = ev.persistencia_frac
        if p is not None and p < cls.PERSIST_MIN:
            g = ev.crecimiento_z
            # use_slope rescata a vigilar un transitorio que acelera, precursor debil real y no
            # falsa alarma plana, que es el caso ev27
            if use_slope and g is not None and g >= cls.SLOPE_MIN:
                return (TriageAction.VIGILAR,
                        [f"margen {ev.margen_sh:.1f} transitorio (persist {p:.2f})",
                         f"pero ACELERA (z_slope {g:.1f} >= {cls.SLOPE_MIN}) -> precursor debil real"], None)
            return (TriageAction.DESCARTAR,
                    [f"margen {ev.margen_sh:.1f} debil", f"transitorio (persist {p:.2f} < {cls.PERSIST_MIN})"], None)
        det = f"persist {p:.2f}" if p is not None else "sin persistencia"
        return (TriageAction.VIGILAR,
                [f"precursor {tipo.value}", f"margen {ev.margen_sh:.1f} debil pero sostenido ({det})"], None)


    @staticmethod
    def _explicacion(ev: AlarmEvidence, tipo: TipoPrecursor, action: TriageAction,
                     urgencia: int | None) -> str:
        """explicacion operativa en dos capas, el fallback determinista con el mismo registro
        que el LLM, Recomendacion, Que pasa y A que atenerse."""

        # cita solo numeros de la evidencia, el que decide sabe que hacer y a que atenerse sin
        # leer la jerga del modelo
        canal = ev.canal_disparador or "el canal disparador"
        que = _QUE_PASA[tipo].format(canal=canal)
        if tipo is TipoPrecursor.SIN_SENAL:
            return (f"Recomendacion: descartar; no requiere accion.\n"
                    f"Que pasa: {que} (margen {ev.margen_sh:.1f} < 1).\n"
                    f"A que atenerse: ausencia de precursor accionable en los valores.")
        nivel = "senal fuerte" if ev.margen_sh >= DiagnoseAgent.MARGEN_FUERTE else "senal debil"
        if ev.persistencia_frac is not None:
            sost = "sostenida" if ev.persistencia_frac >= 0.5 else "poco sostenida"
            fuerza = (f"{nivel} y {sost} (margen {ev.margen_sh:.0f}x, activa el "
                      f"{ev.persistencia_frac * 100:.0f}% del tiempo tras dispararse)")
        else:
            fuerza = f"{nivel} (margen {ev.margen_sh:.0f}x)"
        if action is TriageAction.ESCALAR:
            rec = "escalar a planificacion: reparar DENTRO de la ventana, antes de la rotura prevista"
            aten = (f"si no se actua, lo mas probable es que el componente quede inservible en ~{urgencia} "
                    "dias (estimacion del detector); operar derateada frena el daño y da margen para "
                    "planificar la reparacion en una buena ventana")
        elif action is TriageAction.VIGILAR:
            rec = "mantener en vigilancia y revisar en los proximos ciclos; no intervenir aun"
            aten = ("no requiere accion inmediata; si la senal se refuerza o se vuelve sostenida, "
                    "pasara a escalar")
        else:  # descartar en banda debil, transitorio
            rec = "descartar; no requiere accion"
            aten = "probable excursion transitoria que revierte; no es un precursor accionable"
        return (f"Recomendacion: {rec}.\n"
                f"Que pasa: {que}; {fuerza}.\n"
                f"A que atenerse: {aten}.")
