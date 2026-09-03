"""
adaptadores de ventana a grafo para CRGN

las ventanas llegan como (N, W, K), sensores al final, la disposicion que usan el harness, el
AE y GDN. CRGN lee cada ventana como K nodos cuya feature es la senal de longitud W de ese
sensor, asi que las features de nodo son (N, K, W). el grafo Spearman fijo se construye con
los timesteps de train-normal apilados (T, K)

to_node_major
    (N, W, K) a (N, K, W), un nodo por sensor
stack_timesteps
    (N, W, K) a (T, K), las filas con las que se construye el grafo
"""
from __future__ import annotations

import numpy as np


def to_node_major(windows: np.ndarray) -> np.ndarray:
    """(N, W, K) a (N, K, W), cada sensor pasa a ser un nodo y su senal de ventana la
    feature."""

    if windows.ndim != 3:
        raise ValueError(f"windows must be 3D (N, W, K), got shape {windows.shape}")
    # transpose solo devuelve una vista con los strides cambiados, la copia se materializa aqui
    return np.ascontiguousarray(np.transpose(windows, (0, 2, 1)))


def stack_timesteps(windows: np.ndarray, max_rows: int | None = None, seed: int = 0) -> np.ndarray:
    """(N, W, K) a (T, K), los timesteps apilados con los que se construye el grafo Spearman,
    opcionalmente submuestreados. las ventanas solapadas repiten timesteps, pero para el grafo
    una correlacion de rangos no se ve afectada por esa duplicacion, y `max_rows` acota el
    coste en las granjas grandes."""

    if windows.ndim != 3:
        raise ValueError(f"windows must be 3D (N, W, K), got shape {windows.shape}")
    # se apilan las N ventanas una detras de otra, cada timestep pasa a ser una fila
    n, w, k = windows.shape
    flat = windows.reshape(n * w, k)
    if max_rows is not None and flat.shape[0] > max_rows:
        # muestreo sin reemplazo y con semilla propia, para que el grafo sea reproducible
        rng = np.random.default_rng(seed)
        flat = flat[rng.choice(flat.shape[0], max_rows, replace=False)]
    return flat
