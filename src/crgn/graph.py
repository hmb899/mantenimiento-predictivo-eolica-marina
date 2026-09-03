"""
construccion del grafo para CRGN (Yang, Zhang & Yu, Energies 2025, nucleo AFS-CRGN)

las variables son nodos y la adyacencia es un grafo fijo de correlacion de Spearman, porque
las senales SCADA no pasan la normalidad y Pearson no aplica, que es la justificacion del
articulo. se construye una vez por granja y solo con datos de train-normal, y se inyecta en
los encoders GCN, externo como el de GDN para poder cambiar el grafo en las ablaciones sin
tocar el modelo

hermano de src/gdn/graph.py pero con otra salida. GDN usa un grafo top-k binario y dirigido
para su atencion, CRGN propaga con un GCN, asi que necesita el operador normalizado simetrico
Â = D^-1/2 (A+I) D^-1/2 conservando opcionalmente los pesos |corr|. NumPy puro, sin GPU y sin
torch

convenio, grafo no dirigido con pesos, A simetrica, A[i, j] es la fuerza de la arista que une
i y j
"""
from __future__ import annotations

import numpy as np


def spearman_matrix(x: np.ndarray) -> np.ndarray:
    """correlacion de rangos de Spearman entre columnas (sensores), x (T, N) a (N, N). se
    calcula como Pearson sobre los rangos de columna, con rangos medios en los empates via
    scipy.rankdata, y una columna constante da 0. misma definicion que
    src/gdn/graph.spearman_similarity."""

    from scipy.stats import rankdata

    if x.ndim != 2:
        raise ValueError(f"x must be 2D (T, N), got shape {x.shape}")
    # ranking por columna, y la correlacion de Pearson sobre los rangos es la de Spearman
    ranks = rankdata(x, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):  # columna constante, std 0, da NaN
        corr = np.corrcoef(ranks, rowvar=False)
    return np.nan_to_num(corr, nan=0.0)


def topk_graph(sim: np.ndarray, k: int, weighted: bool = True,
               self_loops: bool = True) -> np.ndarray:
    """dispersa una matriz de similitud a un grafo top-k simetrico de pesos no negativos.

    cada nodo conserva sus k vecinos mas fuertes por |corr|, el signo se tira porque la
    anticorrelacion tambien es estructura, y la simetrizacion por union deja la arista cuando
    cualquiera de los dos extremos coloca al otro en su top-k, asi que un nodo puede acabar
    con hasta 2k vecinos. el peso fuera de la diagonal es |corr| con `weighted` y 1 si no, y
    la diagonal vale 1 con `self_loops`, el +I de la renormalizacion del GCN, y 0 si no.
    """

    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"sim must be square (N, N), got shape {sim.shape}")
    n = sim.shape[0]
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, N-1]=[1, {n - 1}], got {k}")
    a = np.abs(sim).copy()
    np.fill_diagonal(a, 0.0)  # fuera del ranking uno mismo
    idx = np.argpartition(-a, kth=k - 1, axis=1)[:, :k]  # (N, k) k emisores top por nodo
    mask = np.zeros((n, n), dtype=bool)
    # se marcan de golpe los pares (fila repetida k veces, columna del vecino elegido)
    mask[np.repeat(np.arange(n), k), idx.ravel()] = True
    mask |= mask.T  # union simetrica
    w = np.where(mask, a if weighted else 1.0, 0.0)
    np.fill_diagonal(w, 1.0 if self_loops else 0.0)
    return w


def normalize_adjacency(a: np.ndarray) -> np.ndarray:
    """normalizacion simetrica del GCN Â = D^-1/2 A D^-1/2, A ya debe traer sus self-loops.
    un nodo aislado, de grado 0, da una fila y una columna de ceros en vez de NaN."""

    deg = a.sum(axis=1)
    with np.errstate(divide="ignore"):  # nodo aislado, grado 0, da 0 y no inf
        dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    # escalar por filas y por columnas equivale a D^-1/2 A D^-1/2 sin construir D
    return dinv[:, None] * a * dinv[None, :]


def build_adjacency(x: np.ndarray, k: int, weighted: bool = True,
                    self_loops: bool = True) -> np.ndarray:
    """de las filas de train-normal (T, N) a la adyacencia Spearman top-k normalizada Â
    (N, N), lista para el GCN."""

    return normalize_adjacency(topk_graph(spearman_matrix(x), k, weighted, self_loops))
