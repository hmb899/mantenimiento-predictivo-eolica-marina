"""tests estaticos del contrato de arnes de CRGNScorer, score_windows mas .percentile.
sintetico y sin GPU."""
import numpy as np
import pytest
import torch

from src.crgn.graph import build_adjacency
from src.crgn.model import CRGN
from src.crgn.scorer import CRGNScorer


def _setup(metric, n_nodes=8, w=27, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    adj = build_adjacency(rng.standard_normal((300, n_nodes)), k=3)
    model = CRGN(n_nodes=n_nodes, in_dim=w, embed_dim=16, n_layers=2)
    scorer = CRGNScorer(model, adj, torch.device("cpu"), metric=metric, knn_k=3, batch_size=32)
    val = rng.standard_normal((120, w, n_nodes)).astype(np.float32)
    return scorer.fit_reference(val), rng, w, n_nodes


def test_mahalanobis_score_contract():
    scorer, rng, w, n = _setup("mahalanobis")
    s = scorer.score_windows(rng.standard_normal((40, w, n)).astype(np.float32))
    assert s.shape == (40,)
    assert s.dtype == np.float64
    assert np.all(s >= 0.0)  # la Mahalanobis al cuadrado es >= 0
    assert np.isfinite(s).all()
    assert scorer.percentile == 99.0


def test_knn_score_contract():
    scorer, rng, w, n = _setup("knn")
    s = scorer.score_windows(rng.standard_normal((40, w, n)).astype(np.float32))
    assert s.shape == (40,)
    assert s.dtype == np.float64
    assert np.all(s >= 0.0)  # la distancia media al vecino es >= 0
    assert np.isfinite(s).all()


def test_score_requires_fit():
    adj = build_adjacency(np.random.default_rng(0).standard_normal((200, 8)), k=3)
    model = CRGN(n_nodes=8, in_dim=27, embed_dim=16, n_layers=2)
    scorer = CRGNScorer(model, adj, torch.device("cpu"), metric="mahalanobis")
    with pytest.raises(RuntimeError):
        scorer.score_windows(np.random.default_rng(1).standard_normal((5, 27, 8)).astype(np.float32))


def test_rejects_unknown_metric():
    adj = build_adjacency(np.random.default_rng(0).standard_normal((200, 8)), k=3)
    model = CRGN(n_nodes=8, in_dim=27, embed_dim=16, n_layers=2)
    with pytest.raises(ValueError):
        CRGNScorer(model, adj, torch.device("cpu"), metric="cosine")


def test_anomalous_windows_score_higher():
    # las ventanas lejos de la referencia normal tienen que puntuar por encima de las de dentro
    scorer, rng, w, n = _setup("mahalanobis", seed=3)
    normal = rng.standard_normal((50, w, n)).astype(np.float32)
    shifted = (rng.standard_normal((50, w, n)) + 6.0).astype(np.float32)
    assert scorer.score_windows(shifted).mean() > scorer.score_windows(normal).mean()
