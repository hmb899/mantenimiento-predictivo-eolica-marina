"""
sub-score Coverage (eq. 1 del articulo), un F_beta punto a punto sobre los datasets de
anomalia.

    F_beta(g, p) = (1 + beta^2) * tp / ((1 + beta^2)*tp + beta^2*fn + fp)

beta = 1/2, que penaliza mas un falso positivo que un falso negativo. el score es la media
aritmetica sobre los sub-datasets de anomalia. solo entran los puntos cuyo estado esta en
NORMAL_STATUS = {0, 2}, y en un dataset de anomalia la verdad de referencia g vale 1 en
todos esos puntos de estado normal conservados.
"""
from __future__ import annotations

import numpy as np


def f_beta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> float:
    """F_beta punto a punto entre dos arrays de 0/1."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # conteo fila a fila, que es lo que separa Coverage de la Reliability, que cuenta eventos
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))

    # con beta < 1 el fp pesa mas que el fn, entra sin factor mientras que el fn lleva beta^2
    beta2 = beta ** 2
    denom = (1 + beta2) * tp + beta2 * fn + fp
    # denominador 0, ni la verdad ni la prediccion tienen un solo 1 y no hay nada que puntuar
    if denom == 0:
        return 0.0
    return (1 + beta2) * tp / denom


def coverage_score(predictions_per_dataset: list[tuple[np.ndarray, np.ndarray]],
                   beta: float = 0.5) -> float:
    """F_beta medio sobre los datasets de anomalia, una media aritmetica por dataset."""

    if not predictions_per_dataset:
        return 0.0
    scores = [f_beta(gt, pred, beta) for gt, pred in predictions_per_dataset]
    return float(np.mean(scores))
