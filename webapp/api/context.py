"""
nucleo de acceso a datos de la consola, en solo lectura sobre src/. todas las funciones pesadas
se cachean en proceso con lru_cache, que la app corre con un unico worker uvicorn, asi que el
contexto TrajDev, la evidencia XAI y los frentes de Pareto se calculan una vez y se sirven
calientes.

no re-implementa nada del pipeline, reutiliza xai_tools, diagnose, plan, solver_tools y
scheduler con la misma receta y los mismos numeros, de modo que lo que sale por la API es
reproducible.

regla del sistema, visible en la UI, las reglas deciden los numeros y el LLM (qwen3) solo
narra. la verdad-terreno (etiqueta CARE) es label-free, solo se sirve en /truth, tras decidir.
"""
from __future__ import annotations

import sys
import threading
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

# serializa el computo pesado (torch de la evidencia, NSGA-II del solver) porque conviven el
# worker de peticiones y el hilo de precalentamiento. evita contencion y trabajo duplicado en
# un miss concurrente, el segundo espera y ya encuentra cache. no reentrante, se suelta
_HEAVY_LOCK = threading.Lock()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.contracts import TriageAction
from src.agents.diagnose import DiagnoseAgent
from src.agents.plan import PlannerAgent, _fmt_sched
from src.agents.tools.env_tools import load_kb, sensibilidad_dia
from src.agents.tools.xai_tools import _context, get_alarm_evidence
from src.scheduler.nsga2 import knee_index
from src.scheduler.savings import comparar_estrategias, coste_reactivo_tarea
from src.scheduler.scenarios import Era5Provider

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
CASOS_ESTUDIO = {("C", 11), ("C", 15), ("B", 34), ("B", 53)}  # van primero en el listado
N_TURBINAS = {"C": 22, "B": 9}
MAX_PLAYBACK_PTS = 600  # diezmado de la serie CUSUM para la animacion, payload pequeno


# escalas cualitativas
# los cortes van en la misma unidad que el impacto_ambiental del frente
def impacto_cualitativo(x: float) -> str:
    if x <= 0.05:
        return "nulo"
    if x < 1.5:
        return "muy bajo"
    if x < 3.0:
        return "bajo"
    if x < 5.0:
        return "moderado"
    if x < 7.0:
        return "alto"
    return "muy alto"


def riesgo_cualitativo(x: float) -> str:
    # el techo p_fallo_max (~0.40) de un componente gestionado es "alto", no "inviable", porque
    # lo que decide si un plan es aceptable es la regla de seguridad, si la reparacion cae
    # antes o despues de la rotura con derating, y no este numero
    if x <= 0.02:
        return "nulo"
    if x < 0.10:
        return "muy bajo"
    if x < 0.20:
        return "bajo"
    if x < 0.30:
        return "moderado"
    return "alto"


# LLM (qwen3, opcional)
def ollama_client():
    from src.agents.llm import OllamaClient
    return OllamaClient()


@lru_cache(maxsize=1)
def _ollama_ok_cached() -> bool:
    try:
        ollama_client().ping()
        return True
    except Exception:
        return False


def ollama_available(refresh: bool = False) -> bool:
    if refresh:
        _ollama_ok_cached.cache_clear()
    return _ollama_ok_cached()


# contexto pesado y evidencia
def warm(farm: str, seed: int = 42) -> None:
    """calienta el contexto TrajDev de una granja, bundle, scores y punto de operacion."""
    _context(farm, None, seed)


@lru_cache(maxsize=8)
def _evidence_cached(farm: str, event: int, seed: int):
    _context(farm, None, seed)  # asegura el contexto caliente
    return get_alarm_evidence(farm, event, seed=seed)


def evidence(farm: str, event: int, seed: int = 42):
    # el lock envuelve a la cacheada, no al reves, asi el segundo en llegar ya encuentra cache
    with _HEAVY_LOCK:
        return _evidence_cached(farm, event, seed)


@lru_cache(maxsize=16)
def diagnose(farm: str, event: int, seed: int = 42, use_llm: bool = False):
    """AlarmReport, las reglas deciden tipo, triaje y urgencia y el LLM solo narra la
    hipotesis."""
    ev = evidence(farm, event, seed)
    llm = ollama_client() if use_llm else None
    return DiagnoseAgent(llm).run(ev)


# listado de casos, barato y label-free
def event_ids(farm: str, seed: int = 42) -> list[int]:
    return sorted(int(e) for e in _context(farm, None, seed)["meta"].keys())


def list_cases(farm: str, seed: int = 42) -> list[dict]:
    """resumen barato por evento y sin etiqueta, margen, mes de la alarma y senal. no corre
    torch, usa la serie CUSUM y el meta ya calientes, y tipo y triaje se calculan al elegir."""
    c = _context(farm, None, seed)
    first_alarm, h = c["first_alarm"], c["h"]
    out = []
    for e in event_ids(farm, seed):
        S = c["ev_S"][e]
        m = c["meta"][e]
        margen = float(S.max() / h)
        idx = first_alarm(S, h)
        ts = m["window_timestamps"]
        t_alarma = ts[idx] if idx is not None else None
        mes = (t_alarma.month if t_alarma is not None else None)
        senal = "sin senal" if margen < 1.0 else ("fuerte" if margen >= 5.0 else "debil")
        out.append({
            "event_id": e,
            "asset_id": int(m["asset_id"]),
            "margen_sh": round(margen, 2),
            "senal": senal,
            "mes": (MESES[mes - 1] if mes else None),
            "mes_num": mes,
            "caso_estudio": (farm, e) in CASOS_ESTUDIO,
        })
    # casos de estudio y senal fuerte primero (mas interesantes para la demo)
    out.sort(key=lambda r: (not r["caso_estudio"], r["senal"] != "fuerte", -r["margen_sh"]))
    return out


CONFIRM_WINDOWS = 6  # ventanas seguidas sobre h para confirmar la alarma, ~1 h sostenida


def _confirm_idx(S: np.ndarray, h: float, k: int = CONFIRM_WINDOWS) -> int | None:
    """primer indice con k ventanas seguidas del CUSUM sobre el umbral, la alarma sostenida y
    el momento de decidir (rojo). None si nunca se mantiene, transitorio que se queda ambar."""
    run = 0
    for i, v in enumerate(S):
        if v > h:
            run += 1
            if run >= k:
                return i
        else:
            run = 0
    return None


# reproduccion (animacion CUSUM)
def playback(farm: str, event: int, seed: int = 42) -> dict:
    """serie CUSUM para animar la cuenta atras, barata porque el contexto ya esta caliente.
    se corta en el instante de la decision, que el climax es decidir, no ver lo que vino."""
    c = _context(farm, None, seed)
    S = np.asarray(c["ev_S"][event], dtype=float)
    h = float(c["h"])
    m = c["meta"][event]
    ts = m["window_timestamps"]
    n = len(S)
    cross = c["first_alarm"](S, h)  # primer cruce del umbral, empieza el ambar
    confirm = _confirm_idx(S, h)  # alarma sostenida, rojo y parada, None si es transitorio
    margen = float(S.max() / h)

    # ventana visible, del inicio del evento a la confirmacion, o entera si no confirma
    fin = confirm if confirm is not None else n - 1
    S_vis = S[:fin + 1]
    m_vis = len(S_vis)
    step = max(1, m_vis // MAX_PLAYBACK_PTS)  # paso de diezmado, deja MAX_PLAYBACK_PTS puntos
    serie = [round(float(v), 4) for v in S_vis[::step]]
    # 8% de aire sobre el pico visible o el umbral, el eje-y se reescala al tramo, no al evento
    y_top = max(float(S_vis.max()), h) * 1.08

    # el canal disparador no sale de aqui, lo pone el diagnostico
    return {
        "farm": farm,
        "event_id": event,
        "asset_id": int(m["asset_id"]),
        "n_turbinas": N_TURBINAS.get(farm, 22),
        "cusum": serie,
        "h": round(h, 4),
        "y_max": round(y_top, 4),
        "margen_sh": round(margen, 3),
        "confirmada": confirm is not None,
        # fraccion del cruce del umbral dentro del tramo visible (marca de ambar)
        "alarma_frac": (round(cross / fin, 4) if cross is not None and fin > 0 else None),
        "t_inicio": ts[0].isoformat(),
        "t_alarma": (ts[cross].isoformat() if cross is not None else None),
        "t_decision": ts[fin].isoformat(),  # reloj en el instante de parada (rojo)
        "n_ventanas": n,
        "meteo": weather_month(ts[cross].month if cross is not None else ts[0].month),
    }


def alarm_month(farm: str, event: int, seed: int = 42) -> int | None:
    """mes (1-12) de la alarma real del evento, barato y sin torch. None si no hay alarma."""
    c = _context(farm, None, seed)
    S = np.asarray(c["ev_S"][event], dtype=float)
    idx = c["first_alarm"](S, float(c["h"]))
    ts = c["meta"][event]["window_timestamps"]
    return (ts[idx].month if idx is not None else None)


# verdad del dataset, solo tras decidir
def truth(farm: str, event: int, seed: int = 42) -> dict:
    """etiqueta real CARE del evento, label-free por diseno, solo se sirve en el paso de
    revelacion. el agente nunca la vio, no entra ni en la evidencia ni en los prompts."""
    m = _context(farm, None, seed)["meta"][event]
    label = m["event_label"]
    ev = evidence(farm, event, seed)
    disparo = ev.margen_sh >= 1.0
    # 2x2 entre la etiqueta real y si hubo disparo, la unica lectura con etiqueta de la app
    es_normal = "normal" in str(label).lower()
    if not es_normal and disparo:
        veredicto, glosa = "acierto", "anomalia real detectada"
    elif es_normal and disparo:
        veredicto, glosa = "falsa_alarma", "evento normal que disparo la alarma"
    elif es_normal and not disparo:
        veredicto, glosa = "correcto_negativo", "evento normal, sin alarma"
    else:
        veredicto, glosa = "no_detectado", "anomalia sin alarma al punto de operacion"
    return {"event_label": str(label), "disparo": disparo, "veredicto": veredicto, "glosa": glosa}


# planificacion (Agente 2 y solver)
# n_gen del NSGA-II en la app. la config validada de los notebooks usa 200, y se verifico que
# desde 30 el frente y el knee salen identicos bit a bit en C ev11 y B ev53, derate y stop,
# porque el espacio es de 180 dias en una sola variable y converge mucho antes. 40 repite ese
# mismo resultado con margen y acelera ~4-5x sin mover ningun numero
APP_NGEN = 40


# no toca src/, replica la composicion de solver_tools.solve_plan, nsga2 mas el baseline
# greedy mas la validacion exacta
def _solve_plan_app(req):
    from src.scheduler.exact_check import exact_check
    from src.scheduler.greedy import solve_greedy
    from src.scheduler.nsga2 import solve_nsga2
    from src.scheduler.problem import SchedulingProblem
    prob = SchedulingProblem.from_request(req)
    front = solve_nsga2(req, n_gen=APP_NGEN, prob=prob)
    return front.model_copy(update={"baseline_greedy": solve_greedy(req, prob=prob),
                                    "validacion_exacta": exact_check(req)})


@lru_cache(maxsize=32)
def _solve_cached(farm: str, event: int, t0_iso: str, con: bool, weather: str, nsc: int,
                  seed: int, horizonte: int, politica: str):
    ev = _evidence_cached(farm, event, seed)
    rep = DiagnoseAgent(None).run(ev)
    # sin escalado no hay plan, el triaje corta antes de llegar al solver
    if rep.triage.action is not TriageAction.ESCALAR:
        return None
    planner = PlannerAgent(None)
    req = planner.build_request([rep], date.fromisoformat(t0_iso), horizonte, con_restricciones=con,
                                weather_provider=weather, n_escenarios=nsc, seed=seed, politica=politica)
    return {"req": req, "front": _solve_plan_app(req)}


def solve(farm: str, event: int, t0_iso: str, con: bool, weather: str, nsc: int,
          seed: int, horizonte: int, politica: str = "derate"):
    """diagnostico determinista, sin LLM, de ahi al PlanRequest y al frente de Pareto, y por
    eso reproducible. None si el evento no se escala. politica es 'derate' o 'stop'."""
    with _HEAVY_LOCK:
        return _solve_cached(farm, event, t0_iso, con, weather, nsc, seed, horizonte, politica)


def _objetivos_np(front) -> np.ndarray:
    # (n, 3) con coste, riesgo e impacto, el orden que esperan knee_index y los filtros de _eco
    return np.array([[s.objetivos.coste_esperado, s.objetivos.riesgo_esperado,
                      s.objetivos.impacto_ambiental] for s in front.soluciones])


def _o(obj) -> dict:
    # una fila de la tabla de decision, cada numero redondeado con su etiqueta cualitativa
    return {
        "coste": round(obj.coste_esperado, 0),
        "riesgo": round(obj.riesgo_esperado, 3),
        "riesgo_cual": riesgo_cualitativo(obj.riesgo_esperado),
        "impacto": round(obj.impacto_ambiental, 2),
        "impacto_cual": impacto_cualitativo(obj.impacto_ambiental),
    }


def _fecha(t0: date, dia: int) -> str:
    """fecha real de un dia-offset desde t0 (p.ej. 3 mar 2026). se muestra en vez del offset
    'd30' para no confundir el desplazamiento con un dia del mes, tipo '30 feb'."""
    f = t0 + timedelta(days=int(dia))
    return f"{f.day} {MESES[f.month - 1]} {f.year}"


def _causa_tardia(opts, veto: str) -> str:
    """por que no puede adelantarse la reparacion antes del fallo, el veto de fauna o el lead
    de movilizacion del jack-up, segun si el fallo cae o no dentro de un periodo vetado."""
    if opts.get("fallo_en_veto") and veto:
        return f" porque el veto de jack-up de la temporada sensible ({veto}) prohibe reparar antes"
    return " porque no puede adelantarse la movilizacion del jack-up (no se contrata de un dia para otro)"


def _plan_narrativa(req, opts) -> str:
    """narrativa determinista del plan, en dos capas y con fechas reales, sobre la opcion
    recomendada, que puede ser parar. sin LLM, qwen3 inventaba fechas con los offsets 'd30'."""
    ops = {o["id"]: o for o in opts["opciones"]}
    reco = ops[opts["reco_id"]]
    veto = "; ".join(v.motivo for v in req.vetos_duros)
    # parar, la reparacion no llega antes del fallo previsto con derating
    if opts["reco_id"] == "parar":
        opt, fallo = ops["optimo"], _fecha(req.t0, opts["dia_rotura_derating"])
        l1 = (f"Recomendacion: PARAR la turbina y repararla el {reco['cuando']}. La reparacion no "
              f"puede hacerse antes del fallo previsto con derating (~{fallo}){_causa_tardia(opts, veto)}"
              + "; por SEGURIDAD se detiene la turbina en vez de operarla arriesgando el fallo.")
        l2 = (f"Por que: la parada elimina el riesgo de fallo y deja el impacto ambiental en "
              f"{reco['impacto']:.1f} ({reco['impacto_cual']}, turbina silenciosa en la temporada sensible).")
        l3 = (f"A que atenerse: coste {reco['coste']:,.0f} (se pierde produccion esperando); operar "
              f"derateada saldria por {opt['coste']:,.0f} pero apostando a ~{int(opt['riesgo'] * 100)}% "
              "de fallo antes de reparar.")
    # hibrida, se opera derateada mientras el riesgo aguanta y se para antes del fallo
    elif opts["reco_id"] == "hibrida":
        opt, par = ops["optimo"], ops["parar"]
        fallo = _fecha(req.t0, opts["dia_rotura_derating"])
        l1 = ("Recomendacion: estrategia HIBRIDA - operar DERATEADA mientras el riesgo es aceptable y "
              f"PARAR la turbina antes del fallo (~{fallo}){_causa_tardia(opts, veto)}, reparando el "
              f"{reco['cuando']}.")
        l2 = (f"Por que: el riesgo queda ACOTADO en {reco['riesgo']:.2f} ({reco['riesgo_cual']}) -no se "
              "llega al techo de fallo- y se aprovecha algo de produccion derateada antes de parar "
              f"(impacto {reco['impacto']:.1f}, {reco['impacto_cual']}).")
        l3 = (f"A que atenerse: coste {reco['coste']:,.0f}; "
              + (f"mas barata que Parar del todo ({par['coste']:,.0f}) " if par.get("coste") is not None else "")
              + f"y sin el ~{int(opt['riesgo'] * 100)}% de riesgo de operar derateada hasta el final.")
    # derate, el knee del frente, que se compara con la opcion mas barata que hay en el
    else:
        mc = ops["min_coste"]
        por = "equilibra coste, riesgo e impacto ambiental" + (f"; respeta el veto de {veto}" if veto else "")
        aten = (f"coste esperado {reco['coste']:,.0f} (riesgo {reco['riesgo']:.2f} {reco['riesgo_cual']}, "
                f"impacto {reco['impacto']:.1f} {reco['impacto_cual']})")
        if reco["coste"] - mc["coste"] > 1:
            aten += f"; ~{reco['coste'] - mc['coste']:,.0f} mas que la opcion mas barata del frente"
        l1 = f"Recomendacion: intervenir el {reco['cuando']}."
        l2 = f"Por que: {por}."
        l3 = f"A que atenerse: {aten}."
    return l1 + "\n" + l2 + "\n" + l3


def plan_options(farm: str, event: int, t0_iso: str, con: bool, weather: str, nsc: int,
                 seed: int, horizonte: int, mes: int) -> dict | None:
    """las 7 opciones de la tabla de decision, dejar romper, minimo coste, optimo,
    imp<=moderado, imp<=bajo, parar e hibrida."""
    res = solve(farm, event, t0_iso, con, weather, nsc, seed, horizonte)
    if res is None:
        return None
    req, front = res["req"], res["front"]
    F = _objetivos_np(front)
    task0 = req.tasks[0]
    knee_i = front.knee_idx or 0
    knee = front.soluciones[knee_i]
    ko = knee.objetivos

    rep = diagnose(farm, event, seed, False)
    urgencia = int(rep.urgencia_dias or task0.duracion_dias)
    u_der = int(urgencia * task0.factor_ventana_derating)

    def _dia_mes(sol) -> str:
        return _fecha(req.t0, sol.schedule[0].dia_inicio)

    c_react = coste_reactivo_tarea(task0)  # dejar romper, la via reactiva
    res_stop = solve(farm, event, t0_iso, con, weather, nsc, seed, horizonte, "stop")
    stop_sol = (res_stop["front"].soluciones[res_stop["front"].knee_idx or 0]
                if res_stop is not None else None)
    # la hibrida saca su propia politica del solver, derate y despues la parada optima
    res_hib = solve(farm, event, t0_iso, con, weather, nsc, seed, horizonte, "hibrida_opt")
    hib_sol = (res_hib["front"].soluciones[res_hib["front"].knee_idx or 0]
               if res_hib is not None else None)

    def _eco(ceil: float):
        """mejor plan (el knee del subconjunto) con impacto bajo el techo, y alcanzable=False
        si ningun plan del frente lo cumple, que entonces se devuelve el de minimo impacto."""
        m = F[:, 2] < ceil
        if m.any():
            # knee_index indexa el subconjunto, flatnonzero lo devuelve al indice del frente
            return front.soluciones[int(np.flatnonzero(m)[knee_index(F[m])])], True
        return front.soluciones[int(F[:, 2].argmin())], False

    min_c_sol = front.soluciones[int(F[:, 0].argmin())]
    eo_mod, ok_mod = _eco(5.0)
    eo_baj, ok_baj = _eco(3.0)

    # recomendacion unificada, la misma en la app y en el notebook (savings.recomendar_union).
    # toma el knee de cada politica del menu (derate, hibrida_opt, stop), que son las filas
    # optimo, hibrida y parar de esta tabla, y elige por precio, fuera la turbina perdida
    # (riesgo ~1) y el impacto elevado. ev36 y ev12 salen parar, ev11 optimo
    from src.scheduler.savings import POLITICAS_MENU, recomendar_union
    _sols_menu = {"derate": front.soluciones,
                  "hibrida_opt": (res_hib["front"].soluciones if res_hib is not None else []),
                  "stop": (res_stop["front"].soluciones if res_stop is not None else [])}
    # de la 5-tupla solo interesa la politica ganadora, que se mapea a su fila de la tabla
    _, _, _, _pol_reco, _ = recomendar_union([_sols_menu[p] for p in POLITICAS_MENU], [task0], POLITICAS_MENU)
    reco_id = {"stop": "parar", "hibrida_opt": "hibrida", "derate": "optimo"}.get(_pol_reco, "optimo")
    # solo informativo para la narrativa, el derate repararia tras el fallo con derating?
    dia_rot_der = min(u_der, horizonte - 1)
    unsafe = knee.schedule[0].dia_inicio > dia_rot_der and stop_sol is not None
    # dia de intervencion del recomendado, la linea de tiempo pinta ese plan y no el derate
    _reco_sol = {"optimo": knee, "hibrida": hib_sol, "parar": stop_sol}.get(reco_id) or knee
    dia_reco = _reco_sol.schedule[0].dia_inicio
    # el fallo cae dentro del veto? entonces la causa es el veto, si no el lead de movilizacion
    fallo_en_veto = any(_month_in((req.t0 + timedelta(days=dia_rot_der)).month, v.mes_inicio, v.mes_fin)
                        for v in req.vetos_duros if v.alcance != "jackup" or task0.requiere_jackup)

    opciones = [
        {"id": "romper", "titulo": "Dejar romper", "coste": round(c_react, 0),
         "riesgo": 1.0, "riesgo_cual": "seguro", "impacto": None, "impacto_cual": "a ciegas",
         "cuando": "tras fallo", "recomendado": False, "alcanzable": True},
        {"id": "min_coste", "titulo": "Minimo coste", "recomendado": False, "alcanzable": True,
         "cuando": _dia_mes(min_c_sol), **_o(min_c_sol.objetivos)},
        {"id": "optimo", "titulo": "Optimo", "recomendado": True, "alcanzable": True,
         "cuando": _dia_mes(knee), **_o(ko)},
        {"id": "imp_mod", "titulo": "Imp <= moderado", "recomendado": False, "alcanzable": ok_mod,
         "cuando": (_dia_mes(eo_mod) if ok_mod else None), **_o(eo_mod.objetivos)},
        {"id": "imp_bajo", "titulo": "Imp <= bajo", "recomendado": False, "alcanzable": ok_baj,
         "cuando": (_dia_mes(eo_baj) if ok_baj else None), **_o(eo_baj.objetivos)},
        {"id": "parar", "titulo": "Parar", "recomendado": False, "alcanzable": stop_sol is not None,
         "cuando": (_dia_mes(stop_sol) if stop_sol is not None else None),
         **(_o(stop_sol.objetivos) if stop_sol is not None else {"coste": None, "riesgo": 0.0,
             "riesgo_cual": "nulo", "impacto": None, "impacto_cual": "-"})},
        {"id": "hibrida", "titulo": "Hibrida (derate+parar)", "recomendado": False,
         "alcanzable": hib_sol is not None,
         "cuando": (_dia_mes(hib_sol) if hib_sol is not None else None),
         **(_o(hib_sol.objetivos) if hib_sol is not None else {"coste": None, "riesgo": 0.0,
             "riesgo_cual": "nulo", "impacto": None, "impacto_cual": "-"})},
    ]
    for o in opciones:  # la recomendacion la fija la regla de seguridad
        o["recomendado"] = (o["id"] == reco_id)
    return {
        "opciones": opciones,
        "reco_id": reco_id,
        "unsafe": unsafe,
        "fallo_en_veto": fallo_en_veto,
        "dia_rotura_derating": dia_rot_der,
        "dia_intervencion_reco": dia_reco,
        "reco_sched": _fmt_sched(_reco_sol, req),
        "knee_sched": _fmt_sched(knee, req),
        "urgencia_dias": urgencia,
        "urgencia_derating_dias": u_der,
        "n_soluciones": len(front.soluciones),
        "subsistema": task0.subsistema,
        "requiere_jackup": task0.requiere_jackup,
    }


def diagnose_bundle(farm: str, event: int, seed: int, use_llm: bool) -> dict:
    """todo lo que la pantalla de diagnostico necesita (Agente 1), en un solo objeto."""
    rep = diagnose(farm, event, seed, use_llm)
    ev = rep.evidencia
    # el prefijo val__ del canal se quita aqui, el frontend muestra el nombre pelado
    return {
        "farm": farm, "event_id": event, "asset_id": ev.asset_id, "detector": ev.detector,
        "margen_sh": ev.margen_sh, "tipo_precursor": rep.tipo_precursor.value,
        "confianza": round(rep.confianza, 2),
        "triage": rep.triage.action.value, "razones": rep.triage.razones,
        "urgencia_dias": rep.urgencia_dias, "hipotesis": rep.hipotesis,
        "canal_disparador": (ev.canal_disparador.replace("val__", "") if ev.canal_disparador else None),
        "top_canales": [{"canal": c.canal.replace("val__", ""), "z": c.z} for c in ev.top_canales],
        "cf_acumulado_pct": ev.cf_acumulado_pct,
        "ig_propio_pct": ev.ig_propio_pct, "ig_contexto_pct": ev.ig_contexto_pct,
        "persistencia_frac": ev.persistencia_frac, "lead_h": ev.lead_h,
        "t_alarma": (ev.t_alarma.isoformat() if ev.t_alarma is not None else None),
        "usa_llm": bool(use_llm and ollama_available()),
    }


def plan_bundle(farm: str, event: int, t0_iso: str, con: bool, weather: str, nsc: int,
                seed: int, horizonte: int, use_llm: bool, mes: int) -> dict | None:
    """todo lo que la pantalla de plan necesita (Agente 2 y solver), opciones, narrativa,
    frente, validacion, linea de tiempo, meteo y ahorro. None si el evento no se escala."""
    res = solve(farm, event, t0_iso, con, weather, nsc, seed, horizonte)
    if res is None:
        return None
    req, front = res["req"], res["front"]
    F = _objetivos_np(front)
    knee_i = front.knee_idx or 0
    knee = front.soluciones[knee_i]

    opts = plan_options(farm, event, t0_iso, con, weather, nsc, seed, horizonte, mes)
    narrativa = _plan_narrativa(req, opts)  # determinista, fechas reales, opcion recomendada

    pareto = [{"coste": round(c, 0), "riesgo": round(r, 3), "impacto": round(a, 2)}
              for c, r, a in F]
    greedy = None
    if front.baseline_greedy is not None:
        g = front.baseline_greedy.objetivos
        greedy = {"coste": round(g.coste_esperado, 0), "riesgo": round(g.riesgo_esperado, 3),
                  "impacto": round(g.impacto_ambiental, 2)}
    validacion = None
    if front.validacion_exacta is not None:
        ve = front.validacion_exacta
        validacion = {"gap": round(ve.gap, 4), "optimo": ve.optimo_alcanzado, "detalle": ve.detalle}

    # linea de tiempo, alarma, derating, rotura prevista e intervencion
    urg = opts["urgencia_dias"]
    u_der = opts["urgencia_derating_dias"]
    # el dia que se pinta es el del plan recomendado, no el del knee del derate
    dia_int = opts.get("dia_intervencion_reco", knee.schedule[0].dia_inicio)
    dia_rot_der = min(u_der, horizonte - 1)
    fauna = fauna_curve(t0_iso, horizonte, con)
    t0d = date.fromisoformat(t0_iso)
    veto_dias = _veto_ranges(req, t0d, horizonte)  # tramos [ini,fin] con el jack-up vetado
    timeline = {
        "t0": t0_iso,
        "dia_intervencion": dia_int,
        "dur_intervencion": req.tasks[0].duracion_dias,
        "dia_rotura": urg,
        "dia_rotura_derating": dia_rot_der,
        "sensibilidad": fauna["sensibilidad"],
        "veto": [v.motivo for v in req.vetos_duros],
        "veto_dias": veto_dias,
    }

    # aviso de seguridad cuando la reparacion no llega antes del fallo con derating, por el
    # veto de jack-up, que entonces la recomendacion es parar y no prolongar la operacion
    aviso = _aviso(req, opts, t0d)

    return {
        "escalado": True,
        "opciones": opts["opciones"],
        "knee": _o(knee.objetivos),
        "knee_sched": opts["knee_sched"],
        "n_soluciones": opts["n_soluciones"],
        "subsistema": opts["subsistema"],
        "requiere_jackup": opts["requiere_jackup"],
        "narrativa": narrativa,
        "usa_llm": False,  # la narrativa del plan es determinista, con fechas reales, no qwen3
        "pareto": pareto, "knee_idx": knee_i, "greedy": greedy, "validacion": validacion,
        "timeline": timeline,
        "aviso": aviso,
        "meteo": weather_month(mes),
        "ahorro": _savings_from(req, front),
    }


def _month_in(m: int, ini: int, fin: int) -> bool:
    # con fin < ini el rango cruza el ano (p.ej. nov a feb) y la pertenencia se parte en dos
    return ini <= m <= fin if ini <= fin else (m >= ini or m <= fin)


def _veto_ranges(req, t0d: date, horizonte: int) -> list[list[int]]:
    """tramos [dia_ini, dia_fin] del horizonte con el jack-up vetado, para pintar la zona
    prohibida en la linea de tiempo. solo aplica si la tarea requiere jack-up."""
    task_jackup = req.tasks[0].requiere_jackup
    dias = set()
    for v in req.vetos_duros:
        if v.alcance == "jackup" and not task_jackup:
            continue
        for d in range(horizonte):
            if _month_in((t0d + timedelta(days=d)).month, v.mes_inicio, v.mes_fin):
                dias.add(d)
    # agrupa los dias vetados sueltos en tramos contiguos, cerrando cada uno al primer hueco
    rangos, ini = [], None
    for d in range(horizonte):
        if d in dias and ini is None:
            ini = d
        elif d not in dias and ini is not None:
            rangos.append([ini, d - 1]); ini = None
    if ini is not None:
        rangos.append([ini, horizonte - 1])
    return rangos


def _aviso(req, opts, t0d: date) -> str | None:
    """aviso de seguridad cuando la reparacion no llega antes del fallo con derating, explica
    el veto de jack-up y el riesgo de operar derateada. la recomendacion va en la narrativa."""
    if not opts.get("unsafe"):
        return None
    opt = next((o for o in opts["opciones"] if o["id"] == "optimo"), None)
    if opt is None:
        return None
    veto = "; ".join(v.motivo for v in req.vetos_duros)
    return (f"SEGURIDAD: la reparacion no llega antes del fallo previsto con derating "
            f"(~{_fecha(t0d, opts['dia_rotura_derating'])}){_causa_tardia(opts, veto)}. Operar derateada "
            f"hasta la reparacion implicaria ~{int(opt['riesgo'] * 100)}% de riesgo de fallo, asi que se "
            "recomienda PARAR: no se prolonga la operacion mas alla de la ventana de fallo.")


def _savings_from(req, front) -> dict:
    # por_componente se cae, el desglose no cabe en la respuesta de la API
    s = comparar_estrategias(req, front)
    return {k: (round(v, 0) if isinstance(v, float) else v)
            for k, v in s.items() if k != "por_componente"}


def savings(farm: str, event: int, t0_iso: str, con: bool, weather: str, nsc: int,
            seed: int, horizonte: int) -> dict | None:
    res = solve(farm, event, t0_iso, con, weather, nsc, seed, horizonte)
    if res is None:
        return None
    return _savings_from(res["req"], res["front"])


# meteo y acceso (ERA5 real)
@lru_cache(maxsize=1)
def _era5_monthly() -> dict | None:
    """resumen mensual real del German Bight (2014-2023), viento y Hs medianos y % de dias
    accesibles en CTV. la serie horaria es pesada, se carga una vez y se cachea el resumen."""
    try:
        from src.scheduler.era5_io import HS_MAX_M, WIND_MAX_MS, load_hourly
        from src.scheduler.scenarios import daily_accessibility
        df = load_hourly()
        acc = daily_accessibility(df)
        # medianas por mes de calendario, agregando los diez anos de la serie
        by_month = {}
        for mth in range(1, 13):
            sub = df[df.index.month == mth]
            acc_m = acc[acc.index.month == mth]
            by_month[mth] = {
                "viento": round(float(sub["wind"].median()), 1),
                "hs": round(float(sub["hs"].median()), 2),
                "pct_accesible": round(100.0 * float(acc_m.mean()), 0),
            }
        return {"by_month": by_month, "hs_max": HS_MAX_M, "wind_max": WIND_MAX_MS}
    except Exception:
        return None


def era5_available() -> bool:
    # basta con que haya algun zip descargado, no se abre para no pagar la carga
    from src.scheduler.era5_io import ERA5_DIR
    return bool(list(ERA5_DIR.glob("*.zip")))


def weather_month(mes: int) -> dict | None:
    data = _era5_monthly()
    if data is None or mes is None:
        return None
    d = data["by_month"].get(int(mes))
    if d is None:
        return None
    # hay ventana CTV solo si viento y Hs quedan los dos por debajo de su limite
    ctv = d["viento"] < data["wind_max"] and d["hs"] < data["hs_max"]
    return {"mes": MESES[int(mes) - 1], "viento": d["viento"], "hs": d["hs"],
            "pct_accesible": d["pct_accesible"], "ventana_ctv": ctv,
            "hs_max": data["hs_max"], "wind_max": data["wind_max"]}


# curva de fauna (linea de tiempo)
def fauna_curve(t0_iso: str, horizonte: int, con: bool) -> dict:
    """sensibilidad ambiental diaria, trafico mas ruido, para la linea de tiempo del plan."""
    t0 = date.fromisoformat(t0_iso)
    periodos = load_kb()[0] if con else []
    dias = [t0 + timedelta(days=i) for i in range(horizonte)]
    traf = [sensibilidad_dia(d, "trafico_embarcaciones", periodos) for d in dias]
    ruido = [sensibilidad_dia(d, "ruido_operacional", periodos) for d in dias]
    # la sensibilidad del dia es la suma de las dos presiones, no el maximo
    sens = [round(a + b, 3) for a, b in zip(traf, ruido)]
    return {"t0": t0_iso, "horizonte": horizonte, "sensibilidad": sens}
