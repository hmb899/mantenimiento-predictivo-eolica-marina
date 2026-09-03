"""
formulacion del problema de scheduling (F3)

variables de decision, un dia de inicio por tarea (entero en su ventana), codificacion directa.
evaluar un plan es simularlo sobre N escenarios meteo, no lineal y estocastico, de ahi NSGA-II

f1 coste_esperado
    E[ dias-CTV usados x coste_dia_ctv + coste_intervencion x prima_urgencia(inicio)
       (prima si se interviene antes del lead de planificacion, expedita no es barata)
       + dias_derateada x derate_factor x coste_generacion_dia
       + dias_parada x coste_generacion_dia
       + p_fallo x coste_correctivo (p_fallo llega a 1.0 en la rotura, que el derating aleja) ]
f2 riesgo_esperado
    E[ suma de p_fallo acumulada hasta completar la intervencion, por tarea ]
    f1 embebe ese mismo p_fallo x coste_correctivo, luego f1 y f2 quedan correlados
f3 impacto_ambiental
    E[ sum_dias-trabajo ( W_TRAFICO x factor_trafico_buque x sens_trafico(dia)
       + W_RUIDO x factor_ruido_buque x sens_ruido(dia) )   (intervencion, jack-up >> CTV)
       + W_RUIDO_TURBINA x nivel_degradado x sum_dias-derate( sens_ruido(dia) ) ]

tension central, f3 contra f2 y f1. en verano reparar es barato y pronto pero el jack-up cae en
periodo sensible, posponer a otono baja f3 y sube f2, W_RUIDO_TURBINA va bajo para no anularlo

restricciones por penalty, metodo de penalizacion para MOO con restricciones. una tarea vetada,
una que no cabe en el horizonte o la sobre-capacidad diaria de CTV suman PENALTY a los tres y
quedan dominadas. el veto se mira en el dia de inicio, simplificacion registrada
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from src.agents.contracts import CandidatePlan, Objetivos, PlanRequest, ScheduledTask
from src.agents.tools.env_tools import _month_in, sensibilidad_dia
from src.envmod.acoustic import emission_level_db, emision_params

# f3 lleva sus dos terminos en la misma unidad y con pesos explicitos
W_TRAFICO = 1.0  # peso del trafico de embarcaciones de la intervencion en f3
W_RUIDO = 1.0  # peso del ruido de la intervencion (buque) en f3
# coste de una solucion infactible (veto violado, tarea sin caber, sobre-capacidad)
PENALTY = 1e9

# ruido continuo de la degradada esperando, [0,1] via acoustic.py en banda <1 kHz
# punto derateado, menos rotor y algo de multiplicadora, neto moderado
_ap = emision_params()
NIVEL_DEGRADADO_NORM = float((emission_level_db(0.45) - _ap["nivel_min_db"])
                             / (_ap["nivel_max_db"] - _ap["nivel_min_db"]))
# peso (assumption) del ruido continuo de la degradada frente a la intervencion
W_RUIDO_TURBINA = 0.08

# umbral de prob. de fallo al que se para la turbina, perilla de politica
# parada el riesgo se congela, queda acotado en 0.40 y nunca llega a 1.0
# a cambio se pierde toda la produccion hasta la reparacion
RIESGO_STOP = 0.40

# exponente de la curva de riesgo (assumption), exp=1 seria lineal, riesgo alto ya
# al principio y el optimo colapsa a 'actuar ya', exp>1 la hace convexa (S de fatiga)
# optimo interior, el frente no degenera y diferir sigue penalizando f2 sin meseta
RIESGO_EXP = 2.0


class SchedulingProblem:
    """adaptador de PlanRequest a simulador de los 3 objetivos, base del wrapper pymoo de
    nsga2.py."""

    def __init__(self, req: PlanRequest, weather_matrix: np.ndarray) -> None:
        """weather_matrix es (n_escenarios, horizonte_dias) bool de accesibilidad
        (scenarios.py)."""

        self.req = req
        # 'derate' (parcial + riesgo residual) | 'stop' (parada total)
        self.politica = req.politica_espera
        self.weather = np.asarray(weather_matrix, dtype=bool)
        # el numero de escenarios y el horizonte salen de la forma de la matriz, no de req
        self.n_esc, self.horizonte = self.weather.shape

        # sensibilidades por dia, solo dependen del calendario y se precomputan
        # lista vacia (con_restricciones=False) = sin fauna, no "KB", sensibilidad_dia([]) da 0
        periodos = req.periodos_ambientales
        self._sens_traf = np.array(
            [sensibilidad_dia(req.t0 + timedelta(days=i), "trafico_embarcaciones", periodos)
             for i in range(self.horizonte)])
        self._sens_ruido = np.array(
            [sensibilidad_dia(req.t0 + timedelta(days=i), "ruido_operacional", periodos)
             for i in range(self.horizonte)])

        # meses vetados por alcance, el VetoDuro tipado se aplana a conjuntos de meses
        self._veto = {"jackup": set(), "todas": set()}
        for v in req.vetos_duros:
            # _month_in resuelve los rangos que cruzan fin de ano, p.ej. nov a feb
            meses = {m for m in range(1, 13) if _month_in(m, v.mes_inicio, v.mes_fin)}
            self._veto[v.alcance] |= meses


    @classmethod
    def from_request(cls, req: PlanRequest) -> "SchedulingProblem":
        """construye el problema resolviendo la accesibilidad del provider indicado en el
        request."""

        # import diferido, construir el problema con una matriz ya dada no arrastra scenarios
        # ni ERA5
        from src.scheduler.scenarios import get_provider

        wm = get_provider(req.weather_provider).accessibility(
            req.t0, req.horizonte_dias, req.n_escenarios_meteo, req.seed)
        return cls(req, wm)


    def _dia_vetado(self, task, dia: int) -> bool:
        # solo se mira el mes del dia de inicio, es la simplificacion registrada en la cabecera
        mes = (self.req.t0 + timedelta(days=dia)).month
        return mes in self._veto["todas"] or (task.requiere_jackup and mes in self._veto["jackup"])


    @staticmethod
    def _deadline(task) -> float:
        """dia en que el riesgo llega al techo (rotura), es (techo-base)/crec, y crec se
        calibra en plan.py a factor_ventana_derating x urgencia, el derating lo aleja."""

        # sin crecimiento no hay rotura, el deadline es infinito y _p_fallo se queda en la base
        cr = task.riesgo_crecimiento_dia
        return (task.p_fallo_max - task.riesgo_fallo_base) / cr if cr > 0 else float("inf")


    def _p_fallo(self, task, dia: float) -> float:
        """prob. de fallo convexa en `dia` (dias derateados acumulados), de la base al techo
        1.0 en el deadline con exponente RIESGO_EXP. monotona y sin meseta hasta romper."""

        base, techo = task.riesgo_fallo_base, task.p_fallo_max
        D = self._deadline(task)
        if not np.isfinite(D):
            return min(base, techo)
        # frac es el avance hacia la rotura recortado a 1, pasado el deadline no sube mas
        frac = min(max(dia, 0.0) / D, 1.0)
        return min(base + (techo - base) * frac ** RIESGO_EXP, techo)


    def _dia_stop_safety(self, task) -> int:
        """dia desde t0 en que el riesgo derateado alcanza el umbral RIESGO_STOP, el inverso de
        _p_fallo. es la parada fija de 'hibrida' y el techo de la parada optimizada de
        'hibrida_opt', nunca se opera derateada mas alla de aqui."""

        base, techo = task.riesgo_fallo_base, task.p_fallo_max
        # si ya se nace por encima del umbral hay que parar desde el dia 0
        if RIESGO_STOP <= base:
            return 0
        D = self._deadline(task)
        # sin rotura a la vista, o con un techo que no llega al umbral, no hay parada forzosa
        if not np.isfinite(D) or RIESGO_STOP >= techo:
            return self.horizonte
        # se despeja frac de la curva convexa, de ahi la raiz RIESGO_EXP-esima
        frac = ((RIESGO_STOP - base) / (techo - base)) ** (1.0 / RIESGO_EXP)
        return int(max(0.0, min(D * frac, self.horizonte)))


    @property
    def n_var(self) -> int:
        """variables de decision del solver, 1 por tarea (dia de inicio) salvo en
        'hibrida_opt', que anade una 2a por tarea (dia de parada), o sea 2 x nº de tareas."""

        n = len(self.req.tasks)
        return 2 * n if self.politica == "hibrida_opt" else n


    def bounds(self):
        """(xl, xu) del vector de decision completo. en 'hibrida_opt' la 2a mitad son las
        paradas, en [0, _dia_stop_safety] por tarea, nunca se para despues del umbral."""

        xl = [t.ventana_inicio_dia for t in self.req.tasks]
        xu = [min(t.ventana_fin_dia, self.horizonte - 1) for t in self.req.tasks]
        if self.politica == "hibrida_opt":
            # las paradas se concatenan detras de los inicios, mismo orden de tareas en las dos
            # mitades
            xl += [0 for _ in self.req.tasks]
            xu += [max(0, min(self._dia_stop_safety(t), self.horizonte - 1)) for t in self.req.tasks]
        return np.array(xl), np.array(xu)


    def decision_ranges(self, span_cap: int):
        """rangos por variable para la enumeracion exacta de exact_check. el rango de inicio ya
        viene acotado por la ventana, recortada a span_cap en el sub-caso."""

        rr = [range(t.ventana_inicio_dia, min(t.ventana_fin_dia, self.horizonte - 1) + 1)
              for t in self.req.tasks]
        if self.politica == "hibrida_opt":
            # mismos rangos de parada que ve NSGA en bounds(), o el gap del gate no seria
            # honesto
            rr += [range(0, max(0, min(self._dia_stop_safety(t), self.horizonte - 1)) + 1)
                   for t in self.req.tasks]
        return rr


    def _simular_tarea(self, task, d: int, access: np.ndarray):
        """devuelve (dias_trabajo, dia_completado, completada) de una tarea que arranca >= d
        ocupando `duracion_dias` dias accesibles."""

        # se avanza dia a dia y solo cuentan los accesibles, los inaccesibles desplazan el
        # trabajo
        working = []
        dia = d
        while dia < self.horizonte and len(working) < task.duracion_dias:
            if access[dia]:
                working.append(dia)
            dia += 1
        # si el horizonte se acaba antes de juntar duracion_dias, la tarea queda sin completar
        completada = len(working) == task.duracion_dias
        completado = working[-1] if working else self.horizonte - 1
        return working, completado, completada


    def evaluate_plan(self, dias_inicio: np.ndarray) -> Objetivos:
        """simula el plan sobre los N escenarios y devuelve los 3 objetivos, ya como
        esperanzas."""

        # el vector es [inicio_0..inicio_n-1] y en 'hibrida_opt' sigue [parada_0..parada_n-1]
        # si greedy o exact no traen las paradas se usa el umbral y se comporta como 'hibrida'
        x = np.asarray(dias_inicio, dtype=int)
        n = len(self.req.tasks)
        dias = x[:n]
        paradas = x[n:2 * n] if (self.politica == "hibrida_opt" and len(x) >= 2 * n) else None
        rec = self.req.recursos
        coste_tot = riesgo_tot = amb_tot = 0.0

        # un escenario meteo por vuelta, al final se promedia, de ahi la esperanza de los 3
        # objetivos
        for s in range(self.n_esc):
            access = self.weather[s]
            coste = riesgo = amb = 0.0
            carga = {}  # dia, nº de tareas trabajando ese dia (capacidad)
            for idx, (task, d0) in enumerate(zip(self.req.tasks, dias)):
                # el dia propuesto se recorta a la ventana de la tarea y al horizonte
                d = int(np.clip(d0, task.ventana_inicio_dia,
                                min(task.ventana_fin_dia, self.horizonte - 1)))
                working, completado, completada = self._simular_tarea(task, d, access)
                # infactible (vetada o que no cabe), penaliza los tres objetivos y queda
                # dominada, solo con el coste colarian las de impacto bajo
                if self._dia_vetado(task, d) or not completada:
                    coste += PENALTY
                    riesgo += 1.0
                    amb += PENALTY
                    continue
                inicio = working[0]
                # prima por urgencia, antes del lead se moviliza expedito (un jack-up para
                # manana no sale a precio planificado), baja a 1.0 en el lead, antes no es
                # barato
                lead = task.lead_planificacion_dias
                prima = (1.0 + (task.prima_urgencia_max - 1.0) * max(0.0, (lead - inicio) / lead)
                         if lead > 0 else 1.0)
                coste += len(working) * rec.coste_dia_ctv + task.coste_intervencion * prima
                # parada total durante la reparacion
                coste += (completado - inicio + 1) * task.coste_generacion_dia
                if self.politica == "stop":
                    # parada total esperando, pierde el 100% de produccion, no emite ruido y el
                    # riesgo de rotura es ~0 (parada no revienta), se espera a la mejor ventana
                    coste += inicio * task.coste_generacion_dia
                    p_fallo = 0.0
                elif self.politica in ("hibrida", "hibrida_opt"):
                    # derateada (produccion parcial) hasta RIESGO_STOP y parada total hasta
                    # reparar. fija en _dia_stop_safety o variable (paradas[idx]) en
                    # 'hibrida_opt', antes baja riesgo y ruido y cuesta produccion
                    safety = self._dia_stop_safety(task)
                    dia_stop = (int(np.clip(paradas[idx], 0, safety))
                                if (self.politica == "hibrida_opt" and paradas is not None) else safety)
                    d_der = min(inicio, dia_stop)  # dias derateada (produccion parcial)
                    # dias parada antes de la reparacion (sin produccion)
                    d_par = max(0, inicio - dia_stop)
                    # el riesgo se congela al parar, se evalua en min(completado, dia_stop),
                    # nunca supera RIESGO_STOP ni llega a 1.0, la 'turbina perdida' es
                    # imposible
                    p_fallo = self._p_fallo(task, min(completado, dia_stop))
                    coste += d_der * task.derate_factor * task.coste_generacion_dia
                    coste += d_par * task.coste_generacion_dia
                    coste += p_fallo * task.coste_correctivo
                    if d_der > 0:  # ruido continuo solo mientras esta derateada
                        amb += float(W_RUIDO_TURBINA * NIVEL_DEGRADADO_NORM * self._sens_ruido[:d_der].sum())
                else:
                    # derateada esperando (research), pierde derate_factor, no todo, el
                    # derating frena el dano y aleja la rotura, riesgo convexo a 1.0, diferir
                    # mas rompe
                    p_fallo = self._p_fallo(task, completado)
                    coste += inicio * task.derate_factor * task.coste_generacion_dia
                    coste += p_fallo * task.coste_correctivo
                    # ruido continuo de la degradada (co-beneficio de reparar pronto)
                    if inicio > 0:
                        amb += float(W_RUIDO_TURBINA * NIVEL_DEGRADADO_NORM * self._sens_ruido[:inicio].sum())
                riesgo += p_fallo
                w = np.asarray(working)
                # impacto de la intervencion en ambas politicas, trafico mas ruido, ponderados
                # por el tipo de buque (jack-up >> CTV) y la sensibilidad de la fauna ese dia
                amb += float((W_TRAFICO * task.factor_trafico_buque * self._sens_traf[w]
                              + W_RUIDO * task.factor_ruido_buque * self._sens_ruido[w]).sum())
                for wd in working:
                    carga[wd] = carga.get(wd, 0) + 1
            # sobre-capacidad diaria de CTV, infactible igual que el veto, penaliza los tres
            # objetivos, si no colarian infactibles como extremos de min-riesgo o min-impacto
            for n in carga.values():
                if n > rec.max_tareas_por_dia:
                    # el exceso escala la penalizacion, dos tareas de mas pesan el doble que
                    # una
                    exceso = n - rec.max_tareas_por_dia
                    coste += PENALTY * exceso
                    riesgo += exceso
                    amb += PENALTY * exceso
            coste_tot += coste
            riesgo_tot += riesgo
            amb_tot += amb

        inv = 1.0 / self.n_esc
        return Objetivos(coste_esperado=coste_tot * inv, riesgo_esperado=riesgo_tot * inv,
                         impacto_ambiental=amb_tot * inv)


    def to_candidate(self, dias_inicio: np.ndarray) -> CandidatePlan:
        # empaqueta un vector de inicios como plan, sin dia_parada, lo usa el baseline greedy
        return CandidatePlan(
            schedule=[ScheduledTask(task_id=t.task_id, dia_inicio=int(d))
                      for t, d in zip(self.req.tasks, dias_inicio)],
            objetivos=self.evaluate_plan(dias_inicio))
