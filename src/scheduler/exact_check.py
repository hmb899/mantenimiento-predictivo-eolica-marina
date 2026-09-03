"""
validacion exacta del solver (gate F3)

exact_check
    construye un sub-caso pequeno y determinista, hasta max_tasks tareas, ventanas cortas y un
    solo escenario meteo, donde se puede enumerar por fuerza bruta todo el espacio de decision
    y calcular el frente de Pareto exacto
    luego corre NSGA-II en el mismo sub-caso y verifica que su frente alcanza el exacto
    el gap es la maxima distancia normalizada de un punto exacto al NSGA mas cercano, 0 lo toca
    devuelve ValidacionExacta, maquina-verificable, que viaja dentro de ParetoFront
"""
from __future__ import annotations

import itertools

import numpy as np

from src.agents.contracts import PlanRequest, ValidacionExacta
from src.scheduler.nsga2 import nondominated_mask, solve_nsga2
from src.scheduler.problem import SchedulingProblem


def _sub_request(req: PlanRequest, max_tasks: int, span_cap: int) -> PlanRequest:
    # recorta el caso, menos tareas, ventanas de span_cap dias y un unico escenario meteo
    tasks = []
    for t in req.tasks[:max_tasks]:
        hi = min(t.ventana_fin_dia, t.ventana_inicio_dia + span_cap, req.horizonte_dias - 1)
        tasks.append(t.model_copy(update={"ventana_fin_dia": hi}))
    return req.model_copy(update={"tasks": tasks, "n_escenarios_meteo": 1})


def _objetivos_matrix(prob: SchedulingProblem, combos) -> np.ndarray:
    # una fila por combo y las columnas en el orden coste, riesgo, impacto, el mismo que
    # produce nsga2._evaluate y el que espera nondominated_mask
    F = []
    for c in combos:
        o = prob.evaluate_plan(c)
        F.append([o.coste_esperado, o.riesgo_esperado, o.impacto_ambiental])
    return np.array(F)


def exact_check(req: PlanRequest, max_tasks: int = 3, span_cap: int = 12,
                tol: float = 1e-3) -> ValidacionExacta:
    # 'hibrida_opt' duplica variables (inicio y parada) y nondominated_mask es O(n^2)
    # con 1 tarea son 2 variables y cientos de combos, gate barato y parada validada
    if req.politica_espera == "hibrida_opt":
        max_tasks = min(max_tasks, 1)
    sub = _sub_request(req, max_tasks, span_cap)
    prob = SchedulingProblem.from_request(sub)

    # 1 rango/tarea (inicio), +1/tarea (parada) en hibrida_opt
    rangos = prob.decision_ranges(span_cap)
    # producto cartesiano de los rangos, o sea el espacio de decision entero del sub-caso
    combos = list(itertools.product(*rangos))
    F = _objetivos_matrix(prob, combos)
    F_exact = F[nondominated_mask(F)]

    # el NSGA recibe el mismo prob, con la misma meteo y los mismos rangos, si no el gap
    # mentiria
    front = solve_nsga2(sub, pop_size=40, n_gen=60, n_seeds=2, prob=prob)
    F_nsga = np.array([[s.objetivos.coste_esperado, s.objetivos.riesgo_esperado,
                        s.objetivos.impacto_ambiental] for s in front.soluciones])

    # normalizacion por el rango del frente exacto, asi el gap es adimensional y comparable
    lo = F_exact.min(axis=0)
    span = np.ptp(F_exact, axis=0)
    span[span == 0] = 1.0  # objetivo constante en el sub-caso, no aporta distancia
    nrm = lambda A: (A - lo) / span  # noqa: E731
    # para cada punto exacto, distancia al NSGA mas cercano, y el gap es la peor de esas
    # distancias
    gap = float(max(min(np.linalg.norm(nrm(fe) - nrm(fn)) for fn in F_nsga) for fe in F_exact))

    return ValidacionExacta(
        optimo_alcanzado=gap <= tol, gap=gap,
        detalle=(f"{len(combos)} combos enumerados, {len(F_exact)} Pareto exactos, "
                 f"{len(F_nsga)} en el frente NSGA-II; gap max normalizado {gap:.4f}"))
