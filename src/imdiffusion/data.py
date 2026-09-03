"""
window data access for ImDiffusion

LazyWindowDataset
    reproduces the sliding windows of preprocessing.Windower, same segmentation via
    find_continuous_segments, same window_size and stride, same ordering, without
    materializing (N, W, K), which Farm C pre-PCA needs because 1.12M windows x 27 x 100
    float32 would be ~12 GB of host RAM
    rows must be pre-imputed and finite, which the frozen pipeline guarantees and the
    constructor asserts, and under that condition Windower's min_valid_ratio drops nothing,
    so both produce identical window sets (tested in tests/test_imdiffusion_data.py)
compute_if_retention_mask
    replicates the Farm B model path, WindowScaler then WindowPCA then IsolationForest on
    the flattened PCA windows (see run_model.run_farm), to recover the retention mask of
    the windows the BiLSTM-AE actually trained on, so ImDiffusion can train pre-PCA on the
    same window set
    H0 check, mask.sum() == 407,136 on the frozen Farm B npz
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from torch.utils.data import Dataset

from src.model_prep.model_scaler import WindowScaler
from src.model_prep.pca_reducer import WindowPCA
from src.preprocessing.utils import find_continuous_segments

logger = logging.getLogger(__name__)


def segments_from_frame(df: pd.DataFrame, feature_cols: list[str],
                        max_gap_minutes: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """extract per-turbine continuous segments exactly like Windower.create_train_windows,
    returning the rows (R, K) float32 in segment order and the row bounds per segment."""

    # se agrupa por asset_id en orden de aparicion y cada grupo se ordena por time_stamp
    rows_list: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []
    cursor = 0
    for _, group in df.groupby("asset_id", sort=False):
        group = group.sort_values("time_stamp")
        # find_continuous_segments parte donde el hueco pasa de max_gap_minutes
        for seg_idx in find_continuous_segments(group, max_gap_minutes=max_gap_minutes):
            seg = group.loc[seg_idx, feature_cols].to_numpy(np.float32)
            rows_list.append(seg)
            # los limites son indices de fila del array concatenado, no del dataframe
            segments.append((cursor, cursor + len(seg)))
            cursor += len(seg)
    rows = np.concatenate(rows_list, axis=0) if rows_list else np.empty((0, len(feature_cols)), np.float32)
    return rows, segments


class LazyWindowDataset(Dataset):
    """sliding windows over pre-segmented rows, materialized one window at a time."""

    def __init__(self, rows: np.ndarray, segments: list[tuple[int, int]],
                 window_size: int, stride: int = 1) -> None:
        if rows.ndim != 2:
            raise ValueError(f"rows must be (R, K), got {rows.shape}")
        # la ventana perezosa no puede imputar, asi que las filas tienen que llegar finitas
        if not np.isfinite(rows).all():
            raise ValueError("rows contain non-finite values; impute before windowing "
                             "(the frozen pipeline guarantees finite rows)")
        self.rows = np.ascontiguousarray(rows, dtype=np.float32)
        self.window_size = window_size
        self.stride = stride
        # un inicio por ventana dentro de cada segmento, sin cruzar de un segmento al siguiente
        # un segmento mas corto que window_size no da ninguna ventana y se cuenta aparte
        starts = [np.arange(s, e - window_size + 1, stride)
                  for s, e in segments if e - s >= window_size]
        self._starts = np.concatenate(starts) if starts else np.empty(0, np.int64)
        n_skipped = sum(1 for s, e in segments if e - s < window_size)
        logger.info(f"[LazyWindowDataset] {len(self._starts):,} windows from "
                    f"{len(segments):,} segments ({n_skipped} too short), "
                    f"W={window_size} stride={stride} K={rows.shape[1]}")

    def __len__(self) -> int:
        return len(self._starts)


    def __getitem__(self, i: int) -> torch.Tensor:
        # aqui es donde la ventana se materializa, una sola cada vez, que es el punto de la
        # clase
        s = self._starts[i]
        return torch.from_numpy(self.rows[s:s + self.window_size])


def compute_if_retention_mask(train_windows: np.ndarray, variance_threshold: float = 0.95,
                              pca_seed: int = 42, contamination: float = 0.01,
                              n_estimators: int = 100, if_seed: int = 42,
                              n_jobs: int = -1) -> tuple[np.ndarray, WindowScaler, WindowPCA]:
    """replicate the Farm B model path to get the Isolation Forest retention mask.
    pass the npz windows without casting, the dtype must match what the AE run saw."""

    # espeja run_model.run_farm, WindowScaler(standard) y luego WindowPCA(0.95, seed)
    scaler = WindowScaler(per_turbine=False, method="standard")
    scaled = scaler.fit_transform(train_windows, None)
    pca = WindowPCA(variance_threshold=variance_threshold, random_state=pca_seed)
    comps = pca.fit_transform(scaled)
    # el IsolationForest mira la ventana entera aplanada (N, W*K), el espacio de IFCleaner
    flat = comps.reshape(len(comps), -1)
    forest = IsolationForest(contamination=contamination, n_estimators=n_estimators,
                             random_state=if_seed, n_jobs=n_jobs)
    # +1 es ventana retenida, la mascara se aplica luego a las ventanas PRE-PCA para que
    # ImDiffusion entrene sobre el conjunto exacto del AE
    mask = forest.fit_predict(flat) == 1
    logger.info(f"[IF retention mask] {len(mask):,} windows -> kept {int(mask.sum()):,} "
                f"(K={pca.n_components_})")
    return mask, scaler, pca
