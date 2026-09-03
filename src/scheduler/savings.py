"""
ahorro del mantenimiento predictivo (F6), compara tres politicas sobre la misma instancia con
el modelo de coste del scheduler. costes relativos (CARE anonimizado), lo interpretable es el
orden y el porcentaje, no el valor absoluto

reactivo o correctivo, sin deteccion
    corre hasta el fallo catastrofico, coste_correctivo (catalogo, reparacion catastrofica mas
    barco de emergencia) mas lucro cesante de una parada larga que no espera a la mejor ventana
predictivo con greedy
    se detecta pronto pero se planifica de forma miope, el baseline del scheduler
predictivo con NSGA-II (knee)
    deteccion (F4) mas planificacion optima, el coste_esperado del plan elegido, que incluye
    intervencion planificada, lucro cesante controlado y riesgo residual

ahorro directo
    coste_correctivo - coste_intervencion por componente, coste cierto que no depende del
    modelo de riesgo, el numero robusto y facil de defender
ahorro esperado
    reactivo menos el predictivo min-coste del frente, con riesgo residual y lucro cesante,
    foto completa y conservadora. el minimo coste y no el knee, que ademas paga por la fauna y
    compararlo en coste seria injusto, el knee va aparte como co-beneficio ambiental, algo mas
    caro y con mucho menos impacto que el reactivo, que interviene a ciegas

supuesto explicito, la parada de emergencia dura duracion_dias +
espera_emergencia_dias (movilizacion urgente y ventana meteo, jack-up ~45 d, CTV ~14 d, del
catalogo citable), mucho mas que la planificada, que son solo los dias de reparacion, y la
produccion perdida domina el coste offshore (~65% del O&M), de ahi el grueso del ahorro
"""
from __future__ import annotations

import numpy as np

from src.agents.contracts import MaintenanceTask, ParetoFront, PlanRequest

# menu de espera del operador y base de la recomendacion, derate puro (80% hasta
# reparar), hibrida_opt (derate y parada optima) y stop (parar ya), compiten las tres
# el derate solo gana sin veto, si difiere pierde la turbina y stop lo domina
POLITICAS_MENU = ("derate", "hibrida_opt", "stop")


def recomendar_union(sols_por_politica, tasks, politicas=POLITICAS_MENU, riesgo_max_gestionado=0.95,
                     impacto_max=7.0):
    """definicion unica de 'plan recomendado', la misma en la app y en el notebook. toma el
    knee de cada politica del menu y entre esos knees elige por precio."""

    from src.scheduler.nsga2 import knee_index

    # coste de no hacer nada, la referencia contra la que se mide cualquier plan
    react = float(sum(coste_reactivo_tarea(t) for t in tasks))
    n = max(len(tasks), 1)
    # el knee de cada politica es su plan representativo, el que equilibra coste, riesgo e
    # impacto, y el de stop, con riesgo 0, es ademas el de minimo impacto de los tres
    knees, pols = [], []
    for pol, sols in zip(politicas, sols_por_politica):
        if not sols:
            continue
        Fp = np.array([[s.objetivos.coste_esperado, s.objetivos.riesgo_esperado, s.objetivos.impacto_ambiental]
                       for s in sols])
        knees.append(sols[knee_index(Fp)])
        pols.append(pol)
    if not knees:
        return [], 0, np.empty((0, 3)), None, react
    F = np.array([[s.objetivos.coste_esperado, s.objetivos.riesgo_esperado, s.objetivos.impacto_ambiental]
                  for s in knees])
    # criterios de la recomendacion, precio sobre todo, impacto no elevado, riesgo ~1 fuera
    # fuera la turbina perdida, un riesgo asi es el artefacto de un plan que la rompe
    ok = np.flatnonzero(F[:, 1] < riesgo_max_gestionado * n)
    if not len(ok):
        ok = np.arange(len(F))  # por seguridad, stop siempre da riesgo 0
    no_elev = ok[F[ok, 2] < impacto_max]  # impacto no elevado
    pool = no_elev if len(no_elev) else ok
    # precio sobre todo, parar gana solo si es lo mas barato (ev36, jack-up vetado) y no por
    # bajar impacto a coste desorbitado, en ev11 el derate gana a la parada de bajo impacto
    idx = int(pool[np.argmin(F[pool, 0])])
    # knees de cada politica, indice del recomendado, F de los knees, politica ganadora y
    # reactivo
    return knees, idx, F, pols[idx], react


def coste_reactivo_tarea(task: MaintenanceTask) -> float:
    """coste de dejar fallar el componente (run-to-failure), el correctivo catastrofico cierto
    mas una parada de emergencia larga."""

    # el correctivo ya incluye prima de emergencia y dano secundario
    # la parada de emergencia son los dias de reparacion mas la espera de barco y ventana meteo
    downtime = task.duracion_dias + task.espera_emergencia_dias
    return task.coste_correctivo + task.coste_generacion_dia * downtime


def _knee(front: ParetoFront):
    # si el frente no trae knee marcado se usa la primera solucion, que es la de menor coste
    return front.soluciones[front.knee_idx if front.knee_idx is not None else 0]


def comparar_estrategias(req: PlanRequest, front: ParetoFront) -> dict:
    """compara reactivo / predictivo-greedy / predictivo-NSGA(knee) sobre la instancia
    `req`."""

    knee = _knee(front)
    # ahorro por componente, la diferencia entre dejarlo romper y repararlo planificado
    por_componente = [
        {"task_id": t.task_id, "subsistema": t.subsistema, "jackup": t.requiere_jackup,
         "correctivo": t.coste_correctivo, "intervencion": t.coste_intervencion,
         "ahorro_directo": t.coste_correctivo - t.coste_intervencion}
        for t in req.tasks]
    corr_tot = sum(t.coste_correctivo for t in req.tasks)
    ahorro_directo = sum(c["ahorro_directo"] for c in por_componente)

    reactivo = sum(coste_reactivo_tarea(t) for t in req.tasks)
    # el predictivo se representa por el minimo coste del frente, no por el knee, para que la
    # comparacion en euros sea justa
    sol_min = min(front.soluciones, key=lambda s: s.objetivos.coste_esperado)
    predictivo_min = sol_min.objetivos.coste_esperado
    greedy = front.baseline_greedy.objetivos.coste_esperado if front.baseline_greedy else None
    return {
        # lectura directa, coste cierto por componente
        "ahorro_directo": ahorro_directo,
        "ahorro_directo_pct": 100.0 * ahorro_directo / corr_tot if corr_tot else 0.0,
        "por_componente": por_componente,
        # lectura esperada, foto completa, el predictivo es el minimo coste del frente
        "reactivo_esperado": reactivo,
        "greedy_esperado": greedy,
        "predictivo_min_coste": predictivo_min,
        "ahorro_esperado": reactivo - predictivo_min,
        "ahorro_esperado_pct": 100.0 * (reactivo - predictivo_min) / reactivo if reactivo else 0.0,
        # co-beneficio ambiental del knee, paga algo mas que el min-coste con un impacto mucho
        # menor
        "predictivo_knee": knee.objetivos.coste_esperado,
        "impacto_knee": knee.objetivos.impacto_ambiental,
        "impacto_min_coste": sol_min.objetivos.impacto_ambiental,
    }


def comparar_fauna(req_con: PlanRequest, front_con: ParetoFront,
                   req_sin: PlanRequest, front_sin: ParetoFront) -> dict:
    """cuantifica en euros lo que cuesta proteger la fauna, comparando sobre la misma campana
    el plan recomendado con el plan de minimo coste que ignora la fauna."""

    from src.scheduler.nsga2 import recomendar_idx

    # las dos campanas llevan la misma politica de espera, hibrida, deratear hasta el umbral
    # de riesgo y entonces parar, para no jugarse la turbina
    assert [t.task_id for t in req_con.tasks] == [t.task_id for t in req_sin.tasks], \
        "comparar_fauna exige la MISMA campanya (mismas tareas) con y sin restricciones"
    # los dos ahorros se miden contra no hacer nada, o sea contra el run-to-failure
    reactivo = sum(coste_reactivo_tarea(t) for t in req_con.tasks)
    # recomendado = selector economico, knee acotado a coste <= reactivo (para 'hibrida_opt' el
    # knee crudo se iria a un rincon carisimo, ver recomendar_idx), en 'hibrida' es el knee
    F_con = np.array([[s.objetivos.coste_esperado, s.objetivos.riesgo_esperado, s.objetivos.impacto_ambiental]
                      for s in front_con.soluciones])
    recomendado = front_con.soluciones[recomendar_idx(F_con, coste_max=reactivo)]
    # req_sin es la misma campana construida con con_restricciones=False, sin calendario de
    # fauna, y de su frente se toma el plan mas barato a secas
    sin_fauna = min(front_sin.soluciones, key=lambda s: s.objetivos.coste_esperado)

    # huella real del plan sin-fauna, fauna activa y veto fuera, se mide la molestia f3
    # y no la penalizacion, misma escala que el knee, un plan ciego cae en ventana sensible
    from src.scheduler.problem import SchedulingProblem
    prob_imp = SchedulingProblem.from_request(req_con.model_copy(update={"vetos_duros": []}))
    # se reevaluan los mismos dias de inicio del plan sin-fauna, tarea a tarea y en el mismo
    # orden
    dias_sin = [next(sc.dia_inicio for sc in sin_fauna.schedule if sc.task_id == t.task_id)
                for t in req_con.tasks]
    impacto_sin_real = prob_imp.evaluate_plan(dias_sin).impacto_ambiental

    c_rec = recomendado.objetivos.coste_esperado
    c_sin = sin_fauna.objetivos.coste_esperado
    ah_rec, ah_sin = reactivo - c_rec, reactivo - c_sin
    return {
        "reactivo": reactivo,
        "coste_recomendado": c_rec, "coste_sin_fauna": c_sin,
        # ahorro sobre no hacer nada con el plan que el sistema recomienda siempre, fauna e
        # hibrida
        "ahorro_recomendado": ah_rec,
        "ahorro_recomendado_pct": 100.0 * ah_rec / reactivo if reactivo else 0.0,
        # ahorro si ignorasemos la fauna, el min-coste sin restricciones, algo mayor en euros
        "ahorro_sin_fauna": ah_sin,
        "ahorro_sin_fauna_pct": 100.0 * ah_sin / reactivo if reactivo else 0.0,
        # lo que cuesta cuidar la fauna, el sobrecoste del recomendado sobre el min-coste
        # ciego, que es tambien ahorro_sin_fauna menos ahorro_recomendado
        "coste_fauna": c_rec - c_sin,
        "coste_fauna_pct_ahorro": 100.0 * (c_rec - c_sin) / ah_sin if ah_sin else 0.0,
        # a cambio de ese sobrecoste, el impacto ambiental en la misma escala f3 baja mucho
        "impacto_recomendado": recomendado.objetivos.impacto_ambiental,
        "impacto_sin_fauna": impacto_sin_real,
    }
