"""
sub-score Reliability (seccion 4.1.1 y algoritmo 1 del articulo), un F_beta por evento
(EF_beta) calculado con el Criticality Algorithm.

    crit[0] = 0
    for i = 1..N:
        if status[i] is NORMAL (in {0, 2}):
            crit[i] = crit[i-1] + 1   if prediction[i] == 1
                      max(crit[i-1] - 1, 0)  otherwise
        else (ABNORMAL):  crit[i] = crit[i-1]   # frozen
    alarm = (max(crit) >= tc)   with tc = 72 (= 12 h at 10-min resolution)

EF_beta = (1+beta^2)*TP_e / ((1+beta^2)*TP_e + beta^2*FN_e + FP_e), beta = 1/2.
"""
from __future__ import annotations

import numpy as np

from .labels import NORMAL_STATUS


def criticality_alarm(y_pred: np.ndarray, statuses: np.ndarray, tc: int = 72,
                      normal_status: frozenset[int] = NORMAL_STATUS) -> bool:
    """corre el Criticality Algorithm, True si el contador llega a tc."""

    crit = 0
    max_crit = 0
    # el contador sube con cada 1 y baja con cada 0 sin bajar de cero, asi una alarma aislada
    # no dispara y una sostenida si, que es lo que la metrica quiere premiar
    for st, pt in zip(statuses, y_pred):
        if int(st) in normal_status:
            crit = crit + 1 if int(pt) == 1 else max(crit - 1, 0)
            if crit > max_crit:
                max_crit = crit
        # un estado anormal deja crit congelado y el punto de datos se ignora
    return max_crit >= tc


def ef_beta(tp_e: int, fn_e: int, fp_e: int, beta: float = 0.5) -> float:
    """F_beta a nivel de evento a partir de los tres conteos de eventos."""

    # mismo F_beta que en Coverage, pero contando eventos en vez de puntos
    beta2 = beta ** 2
    denom = (1 + beta2) * tp_e + beta2 * fn_e + fp_e
    if denom == 0:
        return 0.0
    return (1 + beta2) * tp_e / denom


def reliability_score(anomaly_results: list[tuple[np.ndarray, np.ndarray]],
                      normal_results: list[tuple[np.ndarray, np.ndarray]], beta: float = 0.5,
                      tc: int = 72) -> float:
    """el sub-score EF_beta (Reliability) sobre los datasets de anomalia y los normales."""

    tp_e = 0
    fn_e = 0
    fp_e = 0

    # un dataset anomalo es acierto si llega a dar la alarma y fallo si no la da
    for y_pred, statuses in anomaly_results:
        alarm = criticality_alarm(
            np.asarray(y_pred, dtype=int),
            np.asarray(statuses, dtype=int),
            tc=tc,
        )
        if alarm:
            tp_e += 1
        else:
            fn_e += 1

    # en un dataset normal solo cabe el falso positivo, no hay evento que acertar ni que perder
    for y_pred, statuses in normal_results:
        alarm = criticality_alarm(
            np.asarray(y_pred, dtype=int),
            np.asarray(statuses, dtype=int),
            tc=tc,
        )
        if alarm:
            fp_e += 1

    return ef_beta(tp_e, fn_e, fp_e, beta=beta)
