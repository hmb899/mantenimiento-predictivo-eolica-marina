"""
orquestacion end-to-end y trazas

TraceLogger
    traza jsonl, una linea por paso con entradas, tools, salidas, validaciones y fallbacks
    es artefacto de primera clase, da reproducibilidad y deja constancia de cada paso
run_pipeline
    el recorrido completo, de la evidencia de cada alarma al plan explicado
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "results" / "agent_traces"


class TraceLogger:
    """traza jsonl por ejecucion de agente. funcional desde F1."""

    def __init__(self, agent_name: str) -> None:
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        # nombre de agente, marca de tiempo y sufijo aleatorio, asi dos ejecuciones no se pisan
        self.trace_id = f"{agent_name}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        self.path = TRACES_DIR / f"{self.trace_id}.jsonl"
        self._t0 = time.time()


    def log(self, step: str, **data) -> None:
        # t es el desplazamiento en segundos desde que se creo el logger, no un reloj absoluto
        rec = {"t": round(time.time() - self._t0, 3), "step": step, **data}
        # se abre en modo append por linea, asi una ejecucion interrumpida deja la traza util
        with open(self.path, "a", encoding="utf-8") as f:
            # default=str deja serializar fechas y enums sin convertirlos a mano
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def run_pipeline(farm: str, event_ids: list[int], t0: date, horizonte_dias: int = 120, *,
                 con_restricciones: bool = True, weather_provider: str = "synthetic",
                 n_escenarios: int = 50, seed: int = 42, model: str | None = None,
                 subsistemas: dict[int, str] | None = None, escenario_desc: str = "") -> dict:
    """pipeline completo (F5), de la evidencia label-free de cada alarma al plan explicado.
    devuelve dict(reports, req, front, plan, escalados, trace_id)."""

    # imports diferidos, importar este modulo no arrastra el solver ni el cliente LLM
    from src.agents.diagnose import DiagnoseAgent
    from src.agents.contracts import TriageAction
    from src.agents.llm import OllamaClient
    from src.agents.plan import PlannerAgent
    from src.agents.tools.solver_tools import solve_plan
    from src.agents.tools.xai_tools import get_alarm_evidence

    trace = TraceLogger("pipeline")
    trace.log("start", farm=farm, event_ids=event_ids, t0=t0, con_restricciones=con_restricciones,
              weather=weather_provider, model=model)
    # sin model no se instancia cliente y todo cae al fallback determinista
    llm = OllamaClient(model) if model else None

    # agente 1, por cada evento el expediente label-free de xai_tools, diagnostico y triaje
    diag = DiagnoseAgent(llm, trace)
    reports = []
    for ev_id in event_ids:
        rep = diag.run(get_alarm_evidence(farm, ev_id, seed=seed))
        reports.append(rep)
        trace.log("diagnosed", event_id=ev_id, tipo=rep.tipo_precursor.value,
                  triage=rep.triage.action.value, urgencia=rep.urgencia_dias)

    # solo lo escalado llega al agente 2, que arma un PlanRequest del catalogo; lo vigilado
    # y lo descartado no generan tarea
    escalados = [r for r in reports if r.triage.action is TriageAction.ESCALAR]
    req = front = plan = None
    if escalados:
        planner = PlannerAgent(llm, trace)
        req = planner.build_request(escalados, t0, horizonte_dias, subsistemas=subsistemas,
                                    con_restricciones=con_restricciones,
                                    weather_provider=weather_provider, n_escenarios=n_escenarios,
                                    seed=seed)
        # el frente lo calcula NSGA-II, el agente 2 solo elige y explica
        front = solve_plan(req)
        plan = planner.explain(front, req, escalados, escenario_desc=escenario_desc)
        trace.log("planned", n_tareas=len(req.tasks), knee_idx=front.knee_idx)
    else:
        trace.log("sin_escalados", motivo="ningun evento escalado -> no hay plan")

    return {"reports": reports, "req": req, "front": front, "plan": plan,
            "escalados": escalados, "trace_id": trace.trace_id}
