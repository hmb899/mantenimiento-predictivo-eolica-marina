"""
baseline greedy del scheduling (F3)

solve_greedy
    politica miope de mantenimiento, atiende primero las tareas mas urgentes (deadline mas
    temprano, desempate por riesgo base) y asigna a cada una el primer dia factible, dentro de
    ventana, no vetado y sin exceder capacidad
    ignora el objetivo ambiental, es el punto de comparacion que el frente NSGA-II debe dominar
    sobre todo en el eje ambiental
    la reserva de capacidad es duration-aware, al colocar una tarea reserva sus duracion_dias
    (span nominal, sin contar retrasos meteo) para no solapar mas de max_tareas_por_dia tareas
    el baseline queda factible, evaluate_plan penaliza la sobre-capacidad en los tres objetivos
"""
from __future__ import annotations

from src.agents.contracts import CandidatePlan, PlanRequest
from src.scheduler.problem import SchedulingProblem


def solve_greedy(req: PlanRequest, prob: SchedulingProblem | None = None) -> CandidatePlan:
    prob = prob or SchedulingProblem.from_request(req)
    # orden miope, primero el deadline mas temprano y a igualdad el de mayor riesgo base
    orden = sorted(range(len(req.tasks)),
                   key=lambda i: (req.tasks[i].ventana_fin_dia, -req.tasks[i].riesgo_fallo_base))
    dias = [0] * len(req.tasks)
    carga: dict[int, int] = {}  # dia de trabajo, nº de tareas ocupandolo (span nominal)
    cap = req.recursos.max_tareas_por_dia
    for i in orden:
        t = req.tasks[i]
        hi = min(t.ventana_fin_dia, prob.horizonte - 1)
        # si ningun dia de la ventana sirve se queda el de inicio, que evaluate_plan penalizara
        elegido = t.ventana_inicio_dia
        for d in range(t.ventana_inicio_dia, hi + 1):
            # el span nominal son duracion_dias seguidos desde d, recortado al horizonte
            span = range(d, min(d + t.duracion_dias, prob.horizonte))
            if prob._dia_vetado(t, d) or any(carga.get(x, 0) >= cap for x in span):
                continue
            elegido = d
            break
        dias[i] = elegido
        # se reserva el span entero, no solo el dia de inicio, o dos tareas largas se
        # solaparian
        for x in range(elegido, min(elegido + t.duracion_dias, prob.horizonte)):
            carga[x] = carga.get(x, 0) + 1
    return prob.to_candidate(dias)
