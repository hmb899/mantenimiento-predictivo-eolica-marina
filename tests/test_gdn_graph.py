"""tests estaticos del aprendizaje de estructura de grafo de GDN, la adyacencia coseno top-k, y
del auxiliar de grafo fijo Spearman. todo sintetico, sin artefactos de datos y sin GPU."""
import numpy as np
import pytest
import torch

from src.gdn.graph import (cosine_similarity_matrix, learned_adjacency, spearman_similarity,
                           topk_adjacency)


def test_cosine_similarity_values():
    emb = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sim = cosine_similarity_matrix(emb)
    assert sim.shape == (4, 4)
    assert torch.allclose(sim.diagonal(), torch.ones(4), atol=1e-5)
    assert sim[0, 1].item() == pytest.approx(1.0, abs=1e-5)  # misma direccion
    assert sim[0, 2].item() == pytest.approx(0.0, abs=1e-5)  # ortogonales
    assert sim[0, 3].item() == pytest.approx(-1.0, abs=1e-5)  # opuestas


def test_cosine_similarity_requires_2d():
    with pytest.raises(ValueError):
        cosine_similarity_matrix(torch.zeros(3))


def test_topk_adjacency_picks_most_similar_excluding_self():
    sim = torch.tensor([[1.0, 0.9, 0.5, -0.2],
                        [0.9, 1.0, 0.1, 0.0],
                        [0.5, 0.1, 1.0, 0.3],
                        [-0.2, 0.0, 0.3, 1.0]])
    adj = topk_adjacency(sim, k=1, include_self=False)
    assert adj.shape == (4, 4)
    assert adj.diagonal().sum() == 0  # no se piden self-loops
    assert adj[0].sum() == 1 and adj[0, 1] == 1.0  # receptor 0, emisor 1 (0.9)
    assert adj[2, 0] == 1.0  # receptor 2, emisor 0 (0.5)


def test_topk_adjacency_rowwise_k_and_self_loops():
    torch.manual_seed(0)
    sim = torch.randn(10, 10)
    sim = sim + sim.t()
    adj = topk_adjacency(sim, k=3, include_self=True)
    assert (adj.sum(dim=1) == 4).all()  # 3 vecinos mas el propio por receptor
    assert (adj.diagonal() == 1).all()
    assert set(adj.unique().tolist()) <= {0.0, 1.0}


def test_topk_adjacency_is_directed():
    # similitudes con ranking asimetrico a proposito, la adyacencia no sale simetrica
    sim = torch.tensor([[1.0, 0.9, 0.8],
                        [0.2, 1.0, 0.9],
                        [0.1, 0.3, 1.0]])
    adj = topk_adjacency(sim, k=1, include_self=False)
    assert adj[0, 1] == 1.0 and adj[1, 0] == 0.0  # hay arista de 1 a 0, pero no de 0 a 1


def test_topk_adjacency_validation():
    sim = torch.eye(5)
    with pytest.raises(ValueError):
        topk_adjacency(sim, k=0)
    with pytest.raises(ValueError):
        topk_adjacency(sim, k=5)  # k tiene que ser <= N-1
    with pytest.raises(ValueError):
        topk_adjacency(torch.zeros(3, 4), k=1)  # no cuadrada


def test_learned_adjacency_shape_and_degree():
    torch.manual_seed(1)
    emb = torch.randn(8, 16)
    adj = learned_adjacency(emb, k=3)
    assert adj.shape == (8, 8)
    assert (adj.sum(dim=1) == 4).all()  # 3 vecinos mas el propio


def test_spearman_similarity_monotone_invariance():
    rng = np.random.default_rng(0)
    a = rng.random(200)
    x = np.column_stack([a, 2 * a + 1, np.exp(a), -a])  # transformaciones monotonas de a
    s = spearman_similarity(x)
    assert s.shape == (4, 4)
    assert s[0, 1] == pytest.approx(1.0, abs=1e-6)  # afin creciente da +1
    assert s[0, 2] == pytest.approx(1.0, abs=1e-6)  # exp es creciente, da +1
    assert s[0, 3] == pytest.approx(-1.0, abs=1e-6)  # la negacion da -1


def test_spearman_constant_column_is_zero():
    x = np.column_stack([np.arange(50.0), np.ones(50)])  # la segunda columna es constante
    s = spearman_similarity(x)
    assert s[0, 1] == pytest.approx(0.0, abs=1e-9)
