"""tests del scorer de desviacion de GDN, contrato de arnes, normalizacion por sensor,
agregacion y exclusion de regimen. sintetico, en CPU."""
import numpy as np
import pytest
import torch

from src.gdn.model import GDN
from src.gdn.scorer import GDNScorer, aggregate_sensor_errors


def _scorer(n_nodes: int = 5, context: int = 8, **kw) -> GDNScorer:
    torch.manual_seed(0)
    model = GDN(n_nodes=n_nodes, context=context, embed_dim=8, out_layer_inter_dim=16, topk=2)
    return GDNScorer(model, context=context, device=torch.device("cpu"), **kw)


def _windows(n: int = 20, w: int = 9, k: int = 5, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).rand(n, w, k).astype(np.float32)


def test_aggregate_modes():
    err = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert aggregate_sensor_errors(err, "max")[0] == 4.0
    assert aggregate_sensor_errors(err, "mean")[0] == 2.5
    assert aggregate_sensor_errors(err, "topk", topk_q=0.5)[0] == 3.5  # media de los 2 mayores
    with pytest.raises(ValueError):
        aggregate_sensor_errors(err, "bogus")


def test_score_requires_fit():
    with pytest.raises(RuntimeError):
        _scorer().score_windows(_windows())


def test_score_shape_and_dtype():
    s = _scorer()
    w = _windows()
    s.fit_normalization(w)
    out = s.score_windows(w)
    assert out.shape == (20,) and out.dtype == np.float64
    assert np.isfinite(out).all()


def test_percentile_attr_exposed():
    assert _scorer(percentile=95.0).percentile == 95.0


def test_regime_exclusion_changes_score():
    w = _windows(seed=1)
    s_all = _scorer(aggregation="mean").fit_normalization(w)
    s_exo = _scorer(aggregation="mean", exo_idx=[0, 1]).fit_normalization(w)
    assert not np.allclose(s_all.score_windows(w), s_exo.score_windows(w))


def test_anomalous_window_scores_higher():
    # la normalizacion se ajusta sobre ventanas normales, y una ventana con un salto
    # grande en un solo sensor tiene que puntuar por encima
    w = _windows(n=40, seed=2)
    s = _scorer(aggregation="max").fit_normalization(w)
    normal = s.score_windows(w)
    spiked = w.copy()
    spiked[:, -1, 0] += 10.0  # rompe el objetivo del paso siguiente
    assert s.score_windows(spiked).mean() > normal.mean()
