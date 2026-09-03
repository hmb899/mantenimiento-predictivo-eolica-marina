"""
sub-score Earliness (seccion 3.2.1, figura 1 y eq. 3 del articulo), un score pesado WS con
funcion de peso lineal.

    WS_i = sum_t { w(t) * (y_pred(t) == 1) } / sum_t { w(t) }

la Earliness final es mean(WS_i) sobre todos los datasets de anomalia, y un dataset sin
deteccion aporta WS_i = 0.
"""
from __future__ import annotations

import numpy as np

from .labels import compute_earliness_weights


def weighted_score_single(y_pred: np.ndarray, weights: np.ndarray) -> float:
    """WS de un dataset de anomalia. 0.0 si los pesos suman cero."""

    y_pred = np.asarray(y_pred, dtype=int)
    weights = np.asarray(weights, dtype=float)

    if len(y_pred) == 0 or weights.sum() == 0:
        return 0.0

    # el producto escalar pesa cada deteccion por su w(t), que decae segun se acerca el fallo
    return float(np.dot(weights, y_pred == 1) / weights.sum())


def earliness_score(predictions_and_weights: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """WS medio sobre los datasets de anomalia. 0.0 si la lista esta vacia."""

    if not predictions_and_weights:
        return 0.0
    # la media va sobre todos los eventos, uno sin detectar suma 0 y arrastra la Earliness
    scores = [
        weighted_score_single(pred, w) for pred, w in predictions_and_weights
    ]
    return float(np.mean(scores))
