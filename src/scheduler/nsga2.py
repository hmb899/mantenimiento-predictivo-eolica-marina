"""
NSGA-II (pymoo) sobre SchedulingProblem (F3), barre el frente de Pareto de los 3 objetivos
coste, riesgo e impacto ambiental, variando el dia de inicio de cada tarea

nondominated_mask
    filas no dominadas de una matriz de objetivos
knee_index
    el knee, punto mas cercano al ideal en el espacio normalizado
recomendar_idx
    el knee acotado por coste
solve_nsga2
    variables enteras, un dia por tarea dentro de su ventana, via IntegerRandomSampling y
    RoundingRepair
    multi-seed interno con semillas derivadas de req.seed, reproducible, y union de los no
    dominados para estabilizar el frente
"""
from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.agents.contracts import CandidatePlan, Objetivos, ParetoFront, PlanRequest, ScheduledTask
from src.scheduler.problem import SchedulingProblem


class _PymooProblem(ElementwiseProblem):
    def __init__(self, prob: SchedulingProblem) -> None:
        self.prob = prob
        # 1 var/tarea (dia de inicio), +1 var/tarea (dia de parada) en hibrida_opt
        xl, xu = prob.bounds()
        super().__init__(n_var=prob.n_var, n_obj=3, xl=xl, xu=xu)


    def _evaluate(self, x, out, *args, **kwargs) -> None:
        # las 3 columnas van en el orden coste, riesgo, impacto, el mismo que enumera
        # exact_check
        o = self.prob.evaluate_plan(x)
        out["F"] = [o.coste_esperado, o.riesgo_esperado, o.impacto_ambiental]


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    """mascara booleana de filas no-dominadas (minimizacion de las 3 columnas)."""

    # comparacion de todos contra todos, O(n^2), por eso exact_check acota el numero de combos
    n = len(F)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        # una fila ya descartada no puede volver al frente, se salta
        if not mask[i]:
            continue
        for j in range(n):
            # j domina a i si no es peor en ningun objetivo y es mejor en al menos uno
            if i != j and mask[j] and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                mask[i] = False
                break
    return mask


def knee_index(F: np.ndarray) -> int:
    """indice del knee, el punto de menor distancia euclidea al ideal en el espacio
    normalizado."""

    if len(F) == 1:
        return 0
    # cada objetivo se lleva a [0,1] con su propio rango, si no el de mayor escala mandaria
    # solo
    lo = F.min(axis=0)
    span = np.ptp(F, axis=0)
    span[span == 0] = 1.0  # objetivo constante en el frente, no aporta distancia
    # normalizado, el ideal es el origen, asi que la distancia al ideal es la norma de la fila
    return int(np.argmin(np.linalg.norm((F - lo) / span, axis=1)))


def recomendar_idx(F: np.ndarray, coste_max: float | None = None) -> int:
    """indice de la solucion recomendada, el knee restringido a las soluciones cuyo coste
    (columna 0) no supere coste_max."""

    # con coste_max = coste reactivo la recomendacion siempre ahorra frente a no hacer nada y,
    # entre las que ahorran, elige el mejor equilibrio coste-riesgo-impacto
    idx = np.arange(len(F))
    if coste_max is not None:
        # en 'hibrida_opt' el dia de parada ensancha el frente en coste, parar meses baja
        # riesgo e impacto a precio desorbitado y el knee se iria ahi si no se acota
        m = F[:, 0] <= coste_max
        if m.any():
            idx = idx[m]
    # si ninguna cumple la cota, o sea si todas cuestan mas que no hacer nada, no se filtra
    return int(idx[knee_index(F[idx])])


def resumen_request(req: PlanRequest) -> str:
    # one-liner que viaja en el campo request_resumen del ParetoFront
    return (f"{len(req.tasks)} tareas, {req.horizonte_dias}d, {req.n_escenarios_meteo} escenarios "
            f"({req.weather_provider}), t0={req.t0}")


def solve_nsga2(req: PlanRequest, pop_size: int = 100, n_gen: int = 200, n_seeds: int = 3,
                prob: SchedulingProblem | None = None) -> ParetoFront:
    prob = prob or SchedulingProblem.from_request(req)
    pymoo_prob = _PymooProblem(prob)
    # los dias son enteros, RoundingRepair en cruce y mutacion los mantiene asi
    algo = NSGA2(pop_size=pop_size, sampling=IntegerRandomSampling(),
                 crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
                 mutation=PM(eta=20, repair=RoundingRepair()),
                 eliminate_duplicates=True)
    Xs, Fs = [], []
    # semillas consecutivas desde req.seed, repetir la ejecucion da exactamente el mismo frente
    for k in range(n_seeds):
        res = minimize(pymoo_prob, algo, get_termination("n_gen", n_gen),
                       seed=req.seed + k, verbose=False)
        Xs.append(np.atleast_2d(res.X))
        Fs.append(np.atleast_2d(res.F))
    X = np.vstack(Xs).astype(int)
    F = np.vstack(Fs)
    # se apilan los frentes de todas las semillas y se recorta a los no dominados globales
    mask = nondominated_mask(F)
    X, F = X[mask], F[mask]
    # orden estable por coste, riesgo e impacto, lexsort usa la ultima clave como primaria
    order = np.lexsort((F[:, 2], F[:, 1], F[:, 0]))
    X, F = X[order], F[order]

    n = len(req.tasks)
    # hibrida_opt lleva la parada como 2a mitad del vector
    opt = req.politica_espera == "hibrida_opt"

    def _schedule(x):
        # la primera mitad del vector son los inicios y la segunda, si la hay, las paradas
        paradas = x[n:2 * n] if opt else [None] * n
        return [ScheduledTask(task_id=t.task_id, dia_inicio=int(d),
                              dia_parada=(int(p) if p is not None else None))
                for t, d, p in zip(req.tasks, x[:n], paradas)]

    sols = [CandidatePlan(
        schedule=_schedule(x),
        objetivos=Objetivos(coste_esperado=float(f[0]), riesgo_esperado=float(f[1]),
                            impacto_ambiental=float(f[2])))
        for x, f in zip(X, F)]
    return ParetoFront(request_resumen=resumen_request(req), soluciones=sols,
                       knee_idx=knee_index(F))
