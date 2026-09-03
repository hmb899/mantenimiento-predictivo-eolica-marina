"""
MinMaxScaler de las ventanas de entrada del modelo

se ajusta sobre el array de entrenamiento (N, window_size, n_features). el scaler trabaja por
feature a lo largo de todos los pasos de tiempo y ventanas, asi que para ajustar ve la vista
2D (N*window_size, n_features)

global
    un scaler ajustado con todas las ventanas de train, el comportamiento original
per-turbine
    un scaler por asset_id, ajustado con las ventanas de esa turbina, lo que recorta la
    varianza entre turbinas y deja al AE aprender normales propios de cada una
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class WindowScaler:
    """ajusta un scaler MinMax o Standard con las ventanas de train y transforma train
    y test."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), per_turbine: bool = False,
                 method: str = "minmax") -> None:
        """arranca con el rango de feature objetivo, el modo per-turbine y el metodo
        (minmax/standard)."""

        self.feature_range = feature_range
        self.per_turbine = per_turbine
        self.method = method
        self._fitted = False
        self._global_scaler = None
        self._turbine_scalers: dict = {}


    def _make(self):
        """instancia el scaler configurado, StandardScaler ignora feature_range."""

        return StandardScaler() if self.method == "standard" else MinMaxScaler(feature_range=self.feature_range)


    def fit_transform(self, windows: np.ndarray, turbine_ids: np.ndarray | None = None) -> np.ndarray:
        """ajusta el scaler con las ventanas de train y devuelve el resultado escalado."""

        if self.per_turbine and turbine_ids is not None:
            assert len(windows) == len(turbine_ids), (
                f"WindowScaler: windows ({len(windows)}) and turbine_ids ({len(turbine_ids)}) length mismatch"
            )
            result = np.empty_like(windows)
            # un scaler por turbina, cada uno ajustado solo con las ventanas de esa turbina
            for tid in np.unique(turbine_ids):
                mask = turbine_ids == tid
                scaler = self._make()
                n, w, f = windows[mask].shape
                # aplanado a (N*W, F) para ajustar por feature, y devuelto a su forma original
                flat = windows[mask].reshape(-1, f)
                result[mask] = scaler.fit_transform(flat).reshape(n, w, f)
                self._turbine_scalers[int(tid)] = scaler
        else:
            n, w, f = windows.shape
            flat = windows.reshape(-1, f)
            self._global_scaler = self._make()
            result = self._global_scaler.fit_transform(flat).reshape(n, w, f)

        self._fitted = True
        return result


    def transform(self, windows: np.ndarray, turbine_ids: np.ndarray | None = None) -> np.ndarray:
        """transforma las ventanas con el scaler ajustado, o con los de cada turbina."""

        if not self._fitted:
            raise RuntimeError("WindowScaler not fitted. Call fit_transform() first.")

        if self._turbine_scalers and turbine_ids is not None:
            assert len(windows) == len(turbine_ids), (
                f"WindowScaler: windows ({len(windows)}) and turbine_ids ({len(turbine_ids)}) length mismatch"
            )
            result = np.empty_like(windows)
            # una turbina no vista en fit cae en el primer scaler ajustado, mejor que fallar
            fallback = next(iter(self._turbine_scalers.values()))
            for tid in np.unique(turbine_ids):
                mask = turbine_ids == tid
                tid_int = int(tid)
                if tid_int not in self._turbine_scalers:
                    logger.warning(f"WindowScaler: turbine {tid_int} not seen during fit, using fallback scaler")
                scaler = self._turbine_scalers.get(tid_int, fallback)
                n, w, f = windows[mask].shape
                result[mask] = scaler.transform(windows[mask].reshape(-1, f)).reshape(n, w, f)
            return result

        n, w, f = windows.shape
        return self._global_scaler.transform(windows.reshape(-1, f)).reshape(n, w, f)


    def transform_sequences(self, test_sequences: list[dict]) -> list[dict]:
        """escala in situ la clave windows de cada secuencia de test y devuelve la misma
        lista."""

        fallback = next(iter(self._turbine_scalers.values())) if self._turbine_scalers else None

        # aqui la turbina viene en el propio evento, no en un vector paralelo como en transform
        for seq in test_sequences:
            windows = seq["windows"]
            n, w, f = windows.shape
            if self._turbine_scalers:
                asset_id = int(seq["asset_id"])
                if asset_id not in self._turbine_scalers:
                    logger.warning(f"WindowScaler: turbine {asset_id} not seen during fit, using fallback scaler")
                scaler = self._turbine_scalers.get(asset_id, fallback)
                seq["windows"] = scaler.transform(windows.reshape(-1, f)).reshape(n, w, f)
            else:
                seq["windows"] = self._global_scaler.transform(windows.reshape(-1, f)).reshape(n, w, f)

        return test_sequences
