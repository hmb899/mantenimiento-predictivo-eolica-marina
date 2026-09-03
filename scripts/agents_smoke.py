"""smoke de agentes (F1), modulo ambiental (F2) y planificador (F5). verifica sin GPU lo que
sigue, y se invoca con python scripts/agents_smoke.py [--llm].

  [1] contratos: construccion + round-trip JSON de un caso realista completo
      + validadores cruzados (rechazan lo invalido) + expediente label-free
  [2] KB ambiental: carga/validacion del yaml + consultas de sensibilidad con la logica de
      meses + contenido contra el EdA (sensibilidades por mes, umbrales UBA, parametros de
      emision)
  [2b] acoustic: rangos 80-150 dB del EdA, monotonia, salto +20 dB en nominal, horas-ruido
  [5] planificador (Agente 2): catalogo citable, build_request (catalogo + veto), solve_plan +
      explain (knee + narrativa + alternativas), y el lucro cesante crece al diferir
  [3] trazas: TraceLogger escribe jsonl
  [4] Ollama: ping (modelos disponibles) y, con --llm, una llamada estructurada validada

el scheduler (F3) tiene su propio gate en scripts/scheduler_smoke.py. el Agente 1 (F4) y los
casos de estudio del planificador se evaluan fuera de este gate.
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import date, datetime
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pydantic import BaseModel, ValidationError

from src.agents.contracts import (AlarmEvidence, AlarmReport, CandidatePlan, ChannelZ,
                                  EvalContext, MaintenancePlan, MaintenanceTask, Objetivos,
                                  ParetoFront, PlanRequest, ScheduledTask, TipoPrecursor,
                                  TriageAction, TriageDecision, ValidacionExacta, VetoDuro)
from src.agents.runner import TraceLogger
from src.agents.tools.env_tools import load_kb, periodos_activos, sensibilidad_dia

FAILS = []


def check(name, ok, detail=""):
    print(f"[{name}] {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILS.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="incluye llamada estructurada real a Ollama")
    args = ap.parse_args()

    # [1] contratos round-trip con un caso realista (ev11 de los notebooks)
    ev = AlarmEvidence(farm="C", event_id=11, asset_id=13,
                       detector="fusion_ae_crgn_phys95maxrate",
                       t_alarma=datetime(2023, 10, 17, 13, 30),
                       margen_sh=600.3, lead_h=524.8, canal_disparador="val__sensor_15",
                       top_canales=[ChannelZ(canal="val__sensor_15", z=264.2),
                                    ChannelZ(canal="val__sensor_186", z=41.0)],
                       cf_acumulado_pct=[64, 86, 88, 88, 89, 92],
                       ig_contexto_pct=92.1, ig_propio_pct=0.9)
    rep = AlarmReport(evidencia=ev, tipo_precursor=TipoPrecursor.RELACIONAL, confianza=0.9,
                      hipotesis="Grupo coherente 15/186 con deriva relacional (IG contexto 92%).",
                      triage=TriageDecision(action=TriageAction.ESCALAR,
                                            razones=["margen 600x", "persistencia semanas"]),
                      urgencia_dias=21)
    gt = EvalContext(farm="C", event_id=11, es_evento_normal=False, lead_real_h=524.8)
    task = MaintenanceTask(task_id="C-ev11", farm="C", asset_id=13, origen_event_id=11,
                           duracion_dias=2, ventana_inicio_dia=0, ventana_fin_dia=30,
                           coste_intervencion=80_000, riesgo_fallo_base=0.35,
                           riesgo_crecimiento_dia=0.01, coste_generacion_dia=9_600,
                           coste_correctivo=600_000)
    req = PlanRequest(t0=date(2026, 6, 1), horizonte_dias=120, tasks=[task],
                      periodos_ambientales=load_kb()[0],
                      vetos_duros=[VetoDuro(alcance="jackup", mes_inicio=6, mes_fin=8,
                                            motivo="reproduccion marsopa jun-ago (KB)")])
    front = ParetoFront(request_resumen="1 tarea, 120d",
                        soluciones=[CandidatePlan(schedule=[ScheduledTask(task_id="C-ev11", dia_inicio=5)],
                                                  objetivos=Objetivos(coste_esperado=95_000,
                                                                      riesgo_esperado=0.05,
                                                                      impacto_ambiental=1.8))],
                        knee_idx=0,
                        validacion_exacta=ValidacionExacta(optimo_alcanzado=True, gap=0.0))
    plan = MaintenancePlan(elegido=front.soluciones[0], narrativa="(F5)", reports_origen=[11])
    for name, obj in (("contratos.AlarmReport", rep), ("contratos.EvalContext", gt),
                      ("contratos.PlanRequest", req), ("contratos.ParetoFront", front),
                      ("contratos.MaintenancePlan", plan)):
        obj2 = type(obj).model_validate_json(obj.model_dump_json())
        check(name, obj2 == obj, "round-trip JSON exacto")

    # [1b] el expediente es label-free y los validadores cruzados rechazan lo invalido
    check("contratos.label_free", "es_evento_normal" not in AlarmEvidence.model_fields,
          "la verdad-terreno vive solo en EvalContext")
    rechazos = []
    for desc, modelo, datos in (
        ("escalar sin urgencia", AlarmReport, {**rep.model_dump(), "urgencia_dias": None}),
        ("ventana fin<inicio", MaintenanceTask, {**task.model_dump(), "ventana_fin_dia": -1}),
        ("knee fuera de rango", ParetoFront, {**front.model_dump(), "knee_idx": 5}),
    ):
        try:
            modelo.model_validate(datos)
            rechazos.append(f"{desc}: aceptado")
        except ValidationError:
            pass
    check("contratos.validadores", not rechazos, rechazos or "3/3 invalidos rechazados")

    # [2] KB ambiental
    periodos, reglas, emision = load_kb()
    check("kb.carga", len(periodos) >= 3 and len(reglas) >= 2,
          f"{len(periodos)} periodos, {len(reglas)} reglas, banda {emision['banda_dominante_hz']}")
    jun, oct_ = date(2026, 6, 15), date(2026, 10, 15)
    s_jun = sensibilidad_dia(jun, "trafico_embarcaciones", periodos)
    s_oct = sensibilidad_dia(oct_, "trafico_embarcaciones", periodos)
    check("kb.estacionalidad", s_jun > s_oct,
          f"trafico jun={s_jun:.1f} > oct={s_oct:.1f}")
    inviernos = periodos_activos(date(2026, 1, 10), periodos)
    check("kb.rango_cruza_anyo", any(p.actividad == "presencia_invernal" for p in inviernos),
          "nov-feb activo en enero")
    check("kb.meses_eda",
          sensibilidad_dia(date(2026, 6, 15), "ruido_operacional", periodos) == 0.9
          and sensibilidad_dia(date(2026, 10, 15), "ruido_operacional", periodos) == 0.8
          and sensibilidad_dia(date(2026, 4, 15), "trafico_embarcaciones", periodos) == 0.6
          and sensibilidad_dia(date(2026, 1, 10), "trafico_embarcaciones", periodos) == 0.5,
          "ruido jun 0.9, oct 0.8 | trafico abr 0.6, ene 0.5")
    check("kb.focas",
          any(p.especie == "Foca comun" for p in periodos_activos(date(2026, 6, 15), periodos))
          and any(p.especie == "Foca gris" for p in periodos_activos(date(2026, 1, 10), periodos)),
          "foca comun en junio, foca gris en invierno")
    uba = {r.nombre: r for r in reglas}
    check("kb.uba", uba["UBA/BSH pico (pilotaje, vinculante)"].umbral_db == 190.0
          and uba["UBA/BSH SEL (pilotaje, vinculante)"].umbral_db == 160.0
          and uba["UBA SEL perturbacion (precautorio)"].umbral_db == 140.0
          and uba["UBA/BSH pico (pilotaje, vinculante)"].distancia_m == 750,
          "190 dBpk, 160 dB SEL y 140 SEL a 750 m")
    check("kb.emision", emision["banda_dominante_hz"] == [0, 1000]
          and emision["nivel_min_db"] == 80.0 and emision["nivel_max_db"] == 150.0
          and emision["correlacion_viento_rotor"] == 0.95,
          "banda <1 kHz, 80-150 dB re 1 uPa, corr >=0.95 (he_editorial_2023)")

    # [2b] acoustic (F2): rangos del EdA, monotonia, salto en nominal, horas-ruido
    from src.envmod.acoustic import emision_params, emission_level_db, noise_hours_in_band
    p = emision_params()
    lo, hi, inc = p["nivel_min_db"], p["nivel_max_db"], p["incremento_nominal_db"]
    v_nom, ancho = p["viento_nominal_norm"], p["anchura_transicion_norm"]
    check("acoustic.rango", abs(emission_level_db(0.0) - lo) < 0.5
          and abs(emission_level_db(1.0) - hi) < 0.5,
          f"{emission_level_db(0.0):.1f} a {emission_level_db(1.0):.1f} dB (EdA 80-150)")
    niveles = emission_level_db(np.linspace(0, 1, 201))
    check("acoustic.monotona", bool(np.all(np.diff(niveles) >= -1e-9)), "no decreciente en [0,1]")
    antes, despues = emission_level_db(v_nom - 3 * ancho), emission_level_db(v_nom + 3 * ancho)
    check("acoustic.salto_nominal", despues > hi - 1.5 and antes < hi - inc + 1.5,
          f"{antes:.1f} a {despues:.1f} dB en v_nom={v_nom} (+{inc:.0f} dB, tonal 198 Hz)")
    nh_max = noise_hours_in_band(np.ones(24), 1.0)
    nh_half = noise_hours_in_band(np.ones(24), 0.5)
    nh_calma = noise_hours_in_band(np.zeros(24), 1.0)
    check("acoustic.horas_ruido", abs(nh_max - 24) < 0.5 and abs(nh_half - nh_max / 2) < 1e-9
          and nh_calma < 0.5,
          f"24h@max sens1={nh_max:.2f} | sens0.5={nh_half:.2f} | calma={nh_calma:.2f}")

    # [5] Agente 2 / planificador (F5): catalogo citable, build_request, solve, explain y coste
    from src.agents.plan import PlannerAgent
    from src.agents.tools.catalog_tools import catalog_task, load_catalog, subsistemas_catalogo
    from src.agents.tools.solver_tools import solve_plan
    from src.scheduler.problem import SchedulingProblem

    cat = load_catalog()
    check("catalogo.carga", "multiplicadora" in cat["subsistemas"]
          and cat["subsistemas"]["multiplicadora"]["requiere_jackup"] is True
          and catalog_task("no_existe", "X", "C", 1, ventana_fin_dia=10).subsistema == "generico",
          f"{len(subsistemas_catalogo())} subsistemas; multiplicadora=jackup; desconocido da generico")

    def _report(ev_id, asset, tipo, urg):
        e = AlarmEvidence(farm="C", event_id=ev_id, asset_id=asset, detector="trajdev_own3_val",
                          margen_sh=50.0, canal_disparador="val__sensor_15",
                          cf_acumulado_pct=[64, 86, 92], ig_propio_pct=2.0)
        return AlarmReport(evidencia=e, tipo_precursor=tipo, confianza=0.85, hipotesis="x",
                           triage=TriageDecision(action=TriageAction.ESCALAR, razones=["x"]),
                           urgencia_dias=urg)

    reports = [_report(11, 13, TipoPrecursor.RELACIONAL, 28),
               _report(20, 7, TipoPrecursor.CANAL_UNICO, 14)]
    pl = PlannerAgent(llm=None)  # sin LLM, todo determinista (fallback/reglas)
    req5 = pl.build_request(reports, date(2026, 6, 1), 150, con_restricciones=True, n_escenarios=20)
    check("planner.build_request",
          len(req5.tasks) == 2 and req5.tasks[0].subsistema == "multiplicadora"
          and req5.tasks[0].requiere_jackup and len(req5.vetos_duros) == 1
          and len(req5.periodos_ambientales) == 6,
          f"2 tareas (mult jackup + convertidor), veto {req5.vetos_duros[0].alcance}, "
          f"{len(req5.periodos_ambientales)} periodos")
    req5_sin = pl.build_request(reports, date(2026, 6, 1), 150, con_restricciones=False, n_escenarios=20)
    check("planner.sin_restricciones", not req5_sin.vetos_duros and not req5_sin.periodos_ambientales,
          "con_restricciones=False deja sin veto ni periodos ambientales")

    front5 = solve_plan(req5)
    plan5 = pl.explain(front5, req5, reports, escenario_desc="junio")
    knee_ok = front5.knee_idx is not None and 0 <= front5.knee_idx < len(front5.soluciones)
    check("planner.solve_explain",
          len(front5.soluciones) >= 1 and knee_ok and len(plan5.narrativa) > 20
          and plan5.reports_origen == [11, 20] and front5.validacion_exacta is not None,
          f"{len(front5.soluciones)} sols, knee {front5.knee_idx}, narrativa {len(plan5.narrativa)} chars")

    # el coste hace una U, el punto dulce: la prima de urgencia encarece el dia 0 (expedita)
    # frente a esperar al lead de contratacion, y diferir mucho (derating + riesgo residual)
    # tambien sube: a igualdad de meteo el minimo esta en el lead, no en el dia 0 ni al final.
    t_gen = catalog_task("generador", "T", "C", 1, ventana_fin_dia=99)
    lead = t_gen.lead_planificacion_dias
    r_gen = PlanRequest(t0=date(2026, 3, 1), horizonte_dias=100, tasks=[t_gen], n_escenarios_meteo=1)
    prob = SchedulingProblem(r_gen, np.ones((1, 100), dtype=bool))
    c0 = prob.evaluate_plan([0]).coste_esperado
    c_lead = prob.evaluate_plan([lead]).coste_esperado
    c_late = prob.evaluate_plan([50]).coste_esperado
    check("planner.punto_dulce", c_lead < c0 and c_lead < c_late,
          f"coste d{lead}(lead)={c_lead:,.0f} < d0={c0:,.0f} (prima urgencia) y < d50={c_late:,.0f} (derating+riesgo)")

    # [3] trazas
    tr = TraceLogger("smoke")
    tr.log("test", foo=1)
    check("trazas.jsonl", tr.path.exists() and tr.path.stat().st_size > 0, str(tr.path.name))

    # [4] Ollama
    try:
        from src.agents.llm import OllamaClient
        cli = OllamaClient()
        modelos = cli.ping()
        check("ollama.ping", len(modelos) > 0, f"{len(modelos)} modelos; dev={cli.model} "
              f"{'disponible' if cli.model in modelos else 'no descargado'}")
        if args.llm:
            class _Echo(BaseModel):
                resumen: str
                margen: float

            out = cli.structured(
                [{"role": "system", "content": "Responde en espanol, tecnico y breve."},
                 {"role": "user", "content": "Alarma con margen S/h de 600.3 sostenida 3 semanas. "
                                             "Resume en una frase y devuelve el margen."}], _Echo)
            check("ollama.structured", abs(out.margen - 600.3) < 1e-6 and len(out.resumen) > 10,
                  f"resumen={out.resumen[:60]!r} margen={out.margen}")
    # el smoke reporta, no explota
    except Exception as exc:  # noqa: BLE001
        check("ollama", False, f"{type(exc).__name__}: {exc}")

    if FAILS:
        print(f"SMOKE F1-F2-F5: FALLOS {FAILS}")
        sys.exit(1)
    print("SMOKE F1-F2-F5: TODO PASS")


if __name__ == "__main__":
    main()
