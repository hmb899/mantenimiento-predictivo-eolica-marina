"""
limpiador Isolation Forest, quita del conjunto de entrenamiento las ventanas anomalas

se aplica despues de MinMaxScaler + PCA, sobre la representacion aplanada (N, W*K), asi que
cada ventana es un solo vector y el outlier se juzga en todo el espacio feature-tiempo

solo se limpia el entrenamiento, las secuencias de test no se filtran nunca
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class IFCleaner:
    """ajusta un Isolation Forest sobre las ventanas de train y quita los outliers."""

    def __init__(self, contamination: float = 0.001, n_estimators: int = 100, random_state: int = 42, n_jobs: int = -1):
        """instancia el IsolationForest con los parametros dados."""

        # contamination 0.01 en los config, o sea descarta como mucho 1 de cada 100 ventanas de train
        self.if_ = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._fitted = False


    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """ajusta el Isolation Forest sobre las ventanas de train y devuelve el
        subconjunto limpio."""

        n, w, k = windows.shape
        # una ventana entera es un unico vector, el IF no ve la estructura temporal
        flat = windows.reshape(n, -1)  # (N, W*K)
        preds = self.if_.fit_predict(flat)  # +1 inlier  -1 outlier
        self._fitted = True
        mask = preds == 1
        n_removed = int((~mask).sum())
        frac = n_removed / n * 100
        logger.info(
            f"[IsolationForest] {n:,} windows removed {n_removed:,} "
            f"({frac:.2f}%) kept {mask.sum():,} clean windows"
        )
        return windows[mask]
