"""
escalado a nivel de fila + reduccion PCA para el orden pca-first del pipeline

WindowPCA reduce arrays ya ventaneados (N, W, F) paso a paso. RowReducer, en cambio, escala y
proyecta a nivel de fila, de (n_rows, F) a (n_rows, K), asi que el ventaneo va despues, sobre
las K componentes. eso es lo que deja al modelo usar muchas features, cientos de sensores
`_avg`, sin la explosion de memoria del window-first (N x W x F). lo usa la mejor config de
Farm C

acopla StandardScaler y PCA(variance_threshold) porque el orden pca-first los aplica juntos a
nivel de fila, y los dos quedan expuestos (`.scaler`, `.pca`) para volcar artefactos y para
explicabilidad
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RowReducer:
    """ajusta StandardScaler + PCA por fila sobre las filas de train y proyecta train
    y test."""

    def __init__(self, variance_threshold: float = 0.95, random_state: int = 42,
                 max_fit_samples: int = 200_000) -> None:
        """fija el objetivo de varianza del PCA y el tope de la submuestra de ajuste."""

        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.max_fit_samples = max_fit_samples
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.n_components_: int | None = None
        self._fitted = False


    def fit_transform(self, rows: np.ndarray) -> np.ndarray:
        """ajusta StandardScaler + PCA sobre las filas de train (sin NaN) y devuelve las
        K componentes."""

        if self._fitted:
            raise RuntimeError("RowReducer already fitted. Use transform() for new data.")
        self.scaler = StandardScaler().fit(rows)
        scaled = self.scaler.transform(rows)

        # el PCA se ajusta sobre un subconjunto, con millones de filas el fit completo no cabe
        rng = np.random.default_rng(self.random_state)
        sample = scaled if len(scaled) <= self.max_fit_samples else \
            scaled[rng.choice(len(scaled), self.max_fit_samples, replace=False)]
        # n_components float significa umbral de varianza, y eso exige el solver full
        self.pca = PCA(n_components=self.variance_threshold, random_state=self.random_state, svd_solver="full")
        self.pca.fit(sample)

        # K lo decide el umbral de varianza, no se fija a mano
        self.n_components_ = int(self.pca.n_components_)
        self._fitted = True
        logger.info(
            f"[RowReducer] {rows.shape[1]} features to {self.n_components_} components "
            f"({self.pca.explained_variance_ratio_.sum()*100:.2f}% variance)"
        )
        # el ajuste va con la muestra, la proyeccion siempre con todas las filas
        return self.pca.transform(scaled).astype(np.float32)


    def transform(self, rows: np.ndarray) -> np.ndarray:
        """escala las filas y las proyecta sobre las componentes ajustadas."""

        if not self._fitted:
            raise RuntimeError("RowReducer not fitted. Call fit_transform() first.")
        return self.pca.transform(self.scaler.transform(rows)).astype(np.float32)
