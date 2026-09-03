"""tests estaticos del grafo Spearman de CRGN, la adyacencia top-k normalizada de forma
simetrica. todo sintetico, sin artefactos de datos y sin GPU."""
import numpy as np
import pytest

from src.crgn.graph import (build_adjacency, normalize_adjacency, spearman_matrix, topk_graph)


def test_spearman_matrix_monotone_invariance():
    rng = np.random.default_rng(0)
    a = rng.random(200)
    x = np.column_stack([a, 2 * a + 1, np.exp(a), -a])  # transformaciones monotonas de a
    s = spearman_matrix(x)
    assert s.shape == (4, 4)
    assert s[0, 1] == pytest.approx(1.0, abs=1e-6)  # afin creciente da +1
    assert s[0, 2] == pytest.approx(1.0, abs=1e-6)  # exp es creciente, da +1
    assert s[0, 3] == pytest.approx(-1.0, abs=1e-6)  # la negacion da -1


def test_spearman_constant_column_is_zero():
    x = np.column_stack([np.arange(50.0), np.ones(50)])
    assert spearman_matrix(x)[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_topk_graph_is_symmetric_with_self_loops():
    rng = np.random.default_rng(1)
    sim = np.corrcoef(rng.random((300, 8)), rowvar=False)
    g = topk_graph(sim, k=3, weighted=True, self_loops=True)
    assert g.shape == (8, 8)
    assert np.allclose(g, g.T)  # union simetrica
    assert np.all(np.diag(g) == 1.0)  # self-loops (+I)
    assert np.all(g >= 0.0)  # pesos |corr|, no negativos


def test_topk_graph_degree_at_least_k():
    rng = np.random.default_rng(2)
    sim = np.corrcoef(rng.random((300, 10)), rowvar=False)
    g = topk_graph(sim, k=3, weighted=False, self_loops=False)
    off = g - np.diag(np.diag(g))
    assert set(np.unique(off).tolist()) <= {0.0, 1.0}  # binaria con weighted=False
    assert np.all((off > 0).sum(axis=1) >= 3)  # la union simetrica deja al menos k


def test_topk_graph_validation():
    sim = np.eye(5)
    with pytest.raises(ValueError):
        topk_graph(sim, k=0)
    with pytest.raises(ValueError):
        topk_graph(sim, k=5)  # k tiene que ser <= N-1
    with pytest.raises(ValueError):
        topk_graph(np.zeros((3, 4)), k=1)  # no cuadrada


def test_normalize_adjacency_symmetric_and_bounded():
    rng = np.random.default_rng(3)
    sim = np.corrcoef(rng.random((300, 6)), rowvar=False)
    a = topk_graph(sim, k=2, weighted=True, self_loops=True)
    norm = normalize_adjacency(a)
    assert np.allclose(norm, norm.T)
    assert np.isfinite(norm).all()
    assert np.all(np.abs(norm) <= 1.0 + 1e-9)  # entradas de D^-1/2 A D^-1/2 en [-1, 1]


def test_normalize_adjacency_isolated_node_no_nan():
    a = np.zeros((3, 3))  # todos aislados, grado 0
    norm = normalize_adjacency(a)
    assert np.isfinite(norm).all()
    assert np.all(norm == 0.0)


def test_build_adjacency_end_to_end():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((500, 12))  # filas (T, N) de train normal
    adj = build_adjacency(x, k=4)
    assert adj.shape == (12, 12)
    assert np.allclose(adj, adj.T)
    assert np.isfinite(adj).all()
    assert np.all(np.diag(adj) > 0)  # los self-loops sobreviven a la normalizacion
