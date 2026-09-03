"""tests de datos. LazyWindowDataset reproduce exactamente preprocessing.Windower, y la
mascara de retencion del Isolation Forest de Farm B reproduce las 407.136 ventanas de
entrenamiento del AE desde el npz congelado."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.imdiffusion.data import LazyWindowDataset, compute_if_retention_mask, segments_from_frame
from src.preprocessing.windower import Windower

FROZEN_B_NPZ = Path("data/processed/farm_b_train_windows.npz")
W_CFG = {"windowing": {"window_size_samples": 6, "stride_samples": 1,
                       "min_valid_ratio": 0.5, "max_gap_minutes": 30}}


def _synthetic_frame() -> tuple[pd.DataFrame, list[str]]:
    """dos turbinas con ids sin ordenar, rejilla de 10 min y un hueco de 60 min en cada una,
    o sea dos segmentos por turbina."""

    rng = np.random.default_rng(42)
    feats = ["f0", "f1", "f2"]
    frames = []
    for asset in (7, 3):
        ts = pd.date_range("2020-01-01", periods=120, freq="10min").to_numpy().copy()
        ts[60:] += np.timedelta64(60, "m")  # mete un hueco mayor que max_gap_minutes
        df = pd.DataFrame(rng.normal(size=(120, 3)).astype(np.float32), columns=feats)
        df["asset_id"] = asset
        df["time_stamp"] = ts
        frames.append(df)
    return pd.concat(frames, ignore_index=True), feats


def test_lazy_windows_equal_windower():
    df, feats = _synthetic_frame()
    expected, _, _ = Windower(W_CFG).create_train_windows(df, feats)
    rows, segments = segments_from_frame(df, feats, max_gap_minutes=30)
    ds = LazyWindowDataset(rows, segments, window_size=6, stride=1)
    assert len(ds) == len(expected) > 0
    actual = np.stack([ds[i].numpy() for i in range(len(ds))])
    np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_lazy_windows_respect_stride():
    df, feats = _synthetic_frame()
    rows, segments = segments_from_frame(df, feats, max_gap_minutes=30)
    ds1 = LazyWindowDataset(rows, segments, window_size=6, stride=1)
    ds3 = LazyWindowDataset(rows, segments, window_size=6, stride=3)
    assert 0 < len(ds3) < len(ds1)
    np.testing.assert_array_equal(ds3[0].numpy(), ds1[0].numpy())
    np.testing.assert_array_equal(ds3[1].numpy(), ds1[3].numpy())


def test_short_segments_yield_no_windows():
    rows = np.zeros((10, 2), np.float32)
    ds = LazyWindowDataset(rows, [(0, 4), (4, 10)], window_size=6, stride=1)
    assert len(ds) == 1  # solo el segmento de 6 filas da una ventana


def test_non_finite_rows_rejected():
    rows = np.zeros((10, 2), np.float32)
    rows[3, 1] = np.nan
    with pytest.raises(ValueError):
        LazyWindowDataset(rows, [(0, 10)], window_size=6)


@pytest.mark.slow
@pytest.mark.skipif(not FROZEN_B_NPZ.exists(), reason="frozen Farm B npz not on this machine")
def test_if_mask_reproduces_ae_train_count():
    """el AE de Farm B conservo 407.136 de 411.249 ventanas tras el IsolationForest en PCA
    (results/logs/_ae_ident_b_s42.log). la mascara deja a ImDiffusion entrenar antes de la
    PCA sobre ese mismo conjunto."""

    windows = np.load(FROZEN_B_NPZ)["windows"]  # sin cast, el dtype tiene que ser el del AE
    if windows.shape[0] != 411_249:
        pytest.skip(f"npz has {windows.shape[0]} windows; not the frozen artifact")
    assert windows.shape[1:] == (27, 30)
    mask, _, pca = compute_if_retention_mask(windows)
    assert mask.size == 411_249
    assert int(mask.sum()) == 407_136
    assert pca.n_components_ in (17, 18)
