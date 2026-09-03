"""tool del solver del Agente 2, puente tipado hacia src/scheduler."""
from __future__ import annotations

from src.agents.contracts import ParetoFront, PlanRequest


def solve_plan(req: PlanRequest) -> ParetoFront:
    """construye la instancia (src/scheduler), corre NSGA-II, greedy y validacion exacta, y
    devuelve el frente con knee, baseline y gate rellenos. determinista dado req.seed."""

    # imports perezosos, el scheduler arrastra pymoo y no hace falta para importar el contrato
    from src.scheduler.exact_check import exact_check
    from src.scheduler.greedy import solve_greedy
    from src.scheduler.nsga2 import solve_nsga2
    from src.scheduler.problem import SchedulingProblem

    # la instancia se construye una vez y la comparten los tres, si no cada uno haria otra
    prob = SchedulingProblem.from_request(req)
    front = solve_nsga2(req, prob=prob)
    return front.model_copy(update={
        "baseline_greedy": solve_greedy(req, prob=prob),
        "validacion_exacta": exact_check(req),
    })


def solve_plan_recomendado(req: PlanRequest) -> ParetoFront:
    """frente recomendable, la union de las politicas del menu (POLITICAS_MENU,
    deratear-y-parar optimo o parar del todo), con knee = recomendacion economica
    (recomendar_union). al considerar la opcion de parar, la recomendacion nunca queda dominada
    por 'Parar'. es la misma definicion que usa el notebook del ahorro."""

    from src.scheduler.savings import POLITICAS_MENU, recomendar_union

    # un frente por politica, con la misma peticion cambiando solo politica_espera
    fronts = [solve_plan(req.model_copy(update={"politica_espera": pol})) for pol in POLITICAS_MENU]
    sols, idx, _, _, _ = recomendar_union([f.soluciones for f in fronts], req.tasks)
    # greedy y validacion se conservan del primer frente (hibrida_opt) como referencia
    return fronts[0].model_copy(update={"soluciones": sols, "knee_idx": idx})
