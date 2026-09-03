"""tests del encuadre de pronostico de GDN, del contexto al paso siguiente, y de sus controles
canario. sintetico, sin artefactos de datos."""
import numpy as np
import pytest

from src.gdn.data import forecast_tensors, load_frozen_windows, time_permuted


def test_forecast_tensors_shapes_and_target():
    rng = np.random.default_rng(0)
    w = rng.random((5, 27, 4)).astype(np.float32)
    x, y = forecast_tensors(w, context=26)
    assert x.shape == (5, 4, 26) and y.shape == (5, 4)
    # el objetivo es el ultimo paso, y la historia va por nodo
    np.testing.assert_allclose(y.numpy(), w[:, 26, :], rtol=1e-6)
    np.testing.assert_allclose(x.numpy()[0, 1, :], w[0, 0:26, 1], rtol=1e-6)


def test_forecast_tensors_shorter_context():
    w = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    x, y = forecast_tensors(w, context=2)
    np.testing.assert_allclose(y.numpy(), w[:, 4, :])  # indice del objetivo = W-1 = 4
    np.testing.assert_allclose(x.numpy(), np.transpose(w[:, 2:4, :], (0, 2, 1)))  # pasos 2 y 3


def test_forecast_tensors_validation():
    w = np.zeros((3, 10, 2), np.float32)
    with pytest.raises(ValueError):
        forecast_tensors(w, context=0)
    with pytest.raises(ValueError):
        forecast_tensors(w, context=10)  # context+horizon > W
    with pytest.raises(ValueError):
        forecast_tensors(np.zeros((3, 4)), context=2)  # no es 3D
    with pytest.raises(ValueError):
        forecast_tensors(w, context=9, horizon=2)  # 9+2 > 10


def test_forecast_tensors_multistep():
    w = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    x, y = forecast_tensors(w, context=4, horizon=3)  # contexto [1:4], objetivos [5,6,7]
    assert x.shape == (2, 3, 4) and y.shape == (2, 3, 3)  # (N,K,ctx) y (N,K,horizon)
    np.testing.assert_allclose(y.numpy(), np.transpose(w[:, 5:8, :], (0, 2, 1)))
    np.testing.assert_allclose(x.numpy(), np.transpose(w[:, 1:5, :], (0, 2, 1)))


def test_time_permuted_preserves_marginals():
    rng = np.random.default_rng(1)
    w = rng.random((4, 27, 3)).astype(np.float32)
    p = time_permuted(w, np.random.default_rng(2))
    assert p.shape == w.shape
    for n in range(4):
        for k in range(3):  # el mismo multiconjunto, reordenado
            np.testing.assert_allclose(np.sort(p[n, :, k]), np.sort(w[n, :, k]), rtol=1e-6)


def test_load_frozen_windows_missing_path():
    with pytest.raises(FileNotFoundError):
        load_frozen_windows("nonexistent_dir_xyz", "B")
