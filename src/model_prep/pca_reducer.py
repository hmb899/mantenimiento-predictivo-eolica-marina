"""
reduccion de dimension por PCA de las ventanas de entrada del modelo.

se ajusta sobre las ventanas de train (despues de MinMaxScaler).
reduce (N, W, F) a (N, W, K), donde K es el numero de componentes
que explican >= variance_threshold de la varianza total.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class WindowPCA:
    """ajusta el PCA con las ventanas de train y transforma los arrays de train y test."""

    def __init__(self, variance_threshold: float = 0.98, whiten: bool = False, random_state: int = 42,
                 max_fit_samples: int = 500_000) -> None:
        """arranca los parametros del PCA y el estado interno."""

        self.variance_threshold = variance_threshold
        self.whiten = whiten
        self.random_state = random_state
        self.max_fit_samples = max_fit_samples
        self.pca: PCA | None = None
        self._fitted = False
        self.n_components_: int | None = None


    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """ajusta el PCA con las ventanas de train y devuelve el resultado reducido."""

        if self._fitted:
            raise RuntimeError("WindowPCA already fitted. Use transform() for new data.")
        n, w, f = windows.shape
        flat = windows.reshape(-1, f)
        n_rows = flat.shape[0]

        rng = np.random.default_rng(self.random_state)

        # n_components se decide con una submuestra si el dataset es grande
        if n_rows > self.max_fit_samples:
            idx = rng.choice(n_rows, size=self.max_fit_samples, replace=False)
            sample = flat[idx]
        else:
            sample = flat

        pca_probe = PCA(
            n_components=self.variance_threshold,
            whiten=self.whiten,
            random_state=self.random_state,
            svd_solver="full",
        )
        pca_probe.fit(sample)
        k = pca_probe.n_components_

        # el PCA final se ajusta con n_components entero y SVD aleatorizado
        self.pca = PCA(
            n_components=k,
            whiten=self.whiten,
            random_state=self.random_state,
            svd_solver="randomized",
        )
        self.pca.fit(sample)

        self._fitted = True
        self.n_components_ = k
        evr = self.pca.explained_variance_ratio_.sum()
        logger.info(
            f"[PCA] {f} features to {self.n_components_} components "
            f"({evr*100:.2f}% variance explained)"
        )

        # el dataset entero se transforma de una pasada
        reduced = self.pca.transform(flat)
        return reduced.reshape(n, w, self.n_components_)


    def transform(self, windows: np.ndarray) -> np.ndarray:
        """proyecta las ventanas sobre las componentes del PCA y devuelve el array reducido."""

        if not self._fitted:
            raise RuntimeError("WindowPCA not fitted. Call fit_transform() first.")
        n, w, f = windows.shape
        flat = windows.reshape(-1, f)
        return self.pca.transform(flat).reshape(n, w, self.n_components_)


    def transform_sequences(self, test_sequences: list[dict]) -> list[dict]:
        """aplica el PCA a la clave windows de cada secuencia de test y devuelve la lista."""

        for seq in test_sequences:
            seq["windows"] = self.transform(seq["windows"])
        return test_sequences
