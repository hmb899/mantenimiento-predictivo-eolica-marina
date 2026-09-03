"""
data access for GDN

the frozen-window loader and the forecast framing, shared by the canary, the trainer and the
scorer. the full representation arms (curated/soft/allavg/top100, on-the-fly windowing at
arbitrary W) reuse src/preprocessing utilities without touching baselines

forecast framing
    each preprocessed window (W, K) becomes a single next-step prediction, the last `context`
    steps before the final step predict the final step
    the sensors (K) are the graph nodes and each node's input is its own `context`-length
    history, so x is (N, K, context) and the target is (N, K)
    one score per window, which lines up 1:1 with the frozen labels
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_frozen_windows(processed_path: str | Path, farm: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """load the frozen training windows for a farm, giving (windows (N, W, K) float32,
    feature_cols list[str], turbine_ids (N,) int)."""

    fl = farm.lower()
    npz = Path(processed_path) / f"farm_{fl}_train_windows.npz"
    if not npz.exists():
        raise FileNotFoundError(f"frozen windows not found: {npz}")
    d = np.load(npz, allow_pickle=True)
    windows = d["windows"].astype(np.float32)
    feats = [str(c) for c in d["feature_cols"]]
    # la granja que no guarda turbina se trata como si todas las ventanas fueran de la misma
    tids = d["turbine_ids"] if "turbine_ids" in d.files else np.zeros(len(windows), dtype=int)
    return windows, feats, np.asarray(tids)


def forecast_tensors(windows: np.ndarray, context: int,
                     horizon: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """frame sliding windows (N, W, K) as an h-step-ahead forecast, where the `context` steps
    before the final `horizon` steps predict those steps. gives x (N, K, context) node-major
    and a target (N, K) for horizon=1, back-compatible, or (N, K, horizon) for horizon>1. a
    multi-step target keeps a steady fault persistently hard to forecast, so the error stays
    elevated (coverage/earliness)."""

    w = np.asarray(windows)
    if w.ndim != 3:
        raise ValueError(f"windows must be (N, W, K), got shape {w.shape}")
    n, win, k = w.shape
    if horizon < 1 or context < 1 or context + horizon > win:
        raise ValueError(f"need context>=1, horizon>=1, context+horizon<=W={win}; "
                         f"got context={context}, horizon={horizon}")
    # el corte se hace desde el final de la ventana, los pasos sobrantes del principio se tiran
    tgt = win - horizon  # first predicted step
    ctx = w[:, tgt - context:tgt, :]  # (N, context, K)
    # node-major, el sensor pasa a ser la dimension de nodo y su historia el vector de entrada
    x = np.ascontiguousarray(np.transpose(ctx, (0, 2, 1)), dtype=np.float32)  # (N, K, context)
    if horizon == 1:
        target = np.ascontiguousarray(w[:, tgt, :], dtype=np.float32)  # (N, K)
    else:
        target = np.ascontiguousarray(np.transpose(w[:, tgt:tgt + horizon, :], (0, 2, 1)),
                                      dtype=np.float32)  # (N, K, horizon)
    return torch.from_numpy(x), torch.from_numpy(target)


def time_permuted(windows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """per-window random permutation of the time axis, keeps each sensor's marginal and
    destroys the temporal dynamics. canary control, windows (N, W, K) keep their shape."""

    # cada ventana lleva su propia permutacion, si fuera una comun quedaria estructura
    # compartida
    w = np.asarray(windows)
    return np.stack([win[rng.permutation(win.shape[0])] for win in w]).astype(np.float32)


def load_test_sequences(processed_path: str | Path, farm: str,
                        feature_idx: list[int] | None = None) -> list[dict]:
    """load the frozen per-event test sequences for a farm in the dict format the harness
    expects (event_id, event_label, asset_id, windows (n,W,K) float32, labels, datetime64
    window_timestamps). `feature_idx` optionally selects or reorders the feature columns."""

    fl = farm.lower()
    seqs = Path(processed_path) / f"farm_{fl}_test_sequences.npz"
    meta_path = Path(processed_path) / f"farm_{fl}_test_metadata.json"
    if not seqs.exists() or not meta_path.exists():
        raise FileNotFoundError(f"frozen test sequences not found: {seqs} / {meta_path}")
    arrs = np.load(seqs, allow_pickle=True)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    out = []
    # las claves del npz son los event_id como cadena, y son las mismas claves del metadata
    for eid in arrs.files:
        m = meta[eid]
        w = arrs[eid].astype(np.float32)
        if feature_idx is not None:
            w = w[:, :, feature_idx]
        out.append({
            "event_id": int(eid),
            "event_label": m["event_label"],
            "asset_id": int(m["asset_id"]),
            "windows": w,
            "labels": np.asarray(m["labels"], dtype=int),
            "window_timestamps": np.asarray(m["window_timestamps"], dtype="datetime64[ns]"),
        })
    return out
