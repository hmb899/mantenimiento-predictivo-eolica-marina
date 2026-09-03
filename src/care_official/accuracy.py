"""
sub-score Accuracy (eq. 2 del articulo), la fraccion de puntos de prediccion en estado normal
bien clasificados como normales, promediada sobre los sub-datasets normales.

el filtrado por estado ({0,2}) se hace antes, en prepare_normal_dataset_inputs(), asi que los
arrays que llegan aqui solo traen predicciones de estado normal y todo '1' de y_pred es por
tanto un falso positivo.
"""
from __future__ import annotations

import numpy as np


def accuracy_score_single(y_pred: np.ndarray) -> float:
    """Accuracy de un dataset normal, 1 menos la tasa de FP. una ventana vacia da 1.0."""

    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_pred) == 0:
        return 1.0
    # todo punto que llega aqui es de estado normal, asi que un 1 solo puede ser falso positivo
    return float(np.mean(y_pred == 0))


def accuracy_score(predictions_per_normal_dataset: list[np.ndarray]) -> float:
    """Accuracy medio sobre todos los sub-datasets normales. una lista vacia da 1.0."""

    if not predictions_per_normal_dataset:
        return 1.0
    # media por sub-dataset, sin ponderar por puntos, uno corto pesa igual que uno largo
    scores = [accuracy_score_single(pred) for pred in predictions_per_normal_dataset]
    return float(np.mean(scores))
