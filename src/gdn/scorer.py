"""
GDN deviation scorer

GDNScorer
    the harness contract, score_windows takes windows (N, W, K) and returns (N,) float64, plus
    the percentile attribute, so the untouched harness (run_model.evaluate, CUSUM, h-sweep, CV
    and official CARE) evaluates GDN exactly like the AE and ImDiffusion
    GDN scoring is one forward pass per window, so the scorer goes straight to the harness with
    no ReplayScorer precompute
    per the GDN paper the per-sensor forecast error is normalised by the validation-normal
    median and IQR and then aggregated over sensors, max in the paper, here max, mean or
    topk, configurable
    regime arm, the exogenous (wind/ambient) nodes stay in the graph but are left out of the
    aggregation, so high wind never raises the deviation score by itself
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.gdn.data import forecast_tensors
from src.gdn.model import GDN


def aggregate_sensor_errors(err: np.ndarray, mode: str = "max", topk_q: float = 0.9) -> np.ndarray:
    """aggregate the per-sensor errors (N, K) into (N,). topk is the mean of the top (1-topk_q)
    sensors."""

    if mode == "max":
        return err.max(axis=1)
    if mode == "mean":
        return err.mean(axis=1)
    if mode == "topk":
        # topk_q es un cuantil, no un numero de sensores, y siempre queda al menos uno
        k = max(1, int(round((1.0 - topk_q) * err.shape[1])))
        return np.partition(err, -k, axis=1)[:, -k:].mean(axis=1)
    raise ValueError(f"unknown aggregation '{mode}'")


class GDNScorer:
    """turn a trained GDN into a per-window deviation scorer for the harness."""

    def __init__(self, model: GDN, context: int, device: torch.device, aggregation: str = "max",
                 topk_q: float = 0.9, exo_idx: list[int] | None = None, batch_size: int = 512,
                 percentile: float = 99.0, eps: float = 1e-6, horizon: int = 1) -> None:
        self.model = model
        self.context = context
        self.horizon = horizon
        self.device = device
        self.aggregation = aggregation
        self.topk_q = topk_q
        self.batch_size = batch_size
        self.percentile = percentile
        self.eps = eps
        self.exo_idx = set(exo_idx or [])
        # la referencia por sensor se rellena en fit_normalization, con ventanas normales
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None


    @torch.no_grad()
    def _sensor_errors(self, windows: np.ndarray) -> np.ndarray:
        """per-window, per-sensor absolute forecast error, (N, W, K) to (N, K). for horizon>1
        the error is averaged over the h forecast steps."""

        x, target = forecast_tensors(windows, self.context, self.horizon)
        self.model.to(self.device).eval()
        # shuffle=False es obligatorio, la salida tiene que ir alineada con las ventanas
        loader = DataLoader(TensorDataset(x, target), batch_size=self.batch_size, shuffle=False)
        out = []
        for xb, yb in loader:
            # el gancho de ablacion de grafo, _adj_override a None deja el grafo aprendido, asi
            # que el scorer sigue siendo retro-compatible con quien no lo ponga
            err = (self.model(xb.to(self.device), adjacency=getattr(self, "_adj_override", None))
                   - yb.to(self.device)).abs()  # (n,K) o (n,K,h)
            out.append((err.mean(dim=2) if err.dim() == 3 else err).cpu().numpy())
        return np.concatenate(out)


    def fit_normalization(self, val_windows: np.ndarray) -> GDNScorer:
        """per-sensor median and IQR of the forecast error on validation-normal windows."""

        err = self._sensor_errors(val_windows)
        # mediana e IQR en vez de media y desviacion, para que un pico no descalibre el sensor
        self.median_ = np.median(err, axis=0)
        q25, q75 = np.percentile(err, [25, 75], axis=0)
        # el suelo eps evita dividir por cero en un sensor cuyo error apenas se mueve
        self.iqr_ = np.maximum(q75 - q25, self.eps)
        return self


    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("call fit_normalization(val_windows) before scoring")
        norm = (self._sensor_errors(windows) - self.median_) / self.iqr_  # (N, K)
        # los exogenos se apartan despues de normalizar, siguen estando en el grafo
        if self.exo_idx:
            keep = [i for i in range(norm.shape[1]) if i not in self.exo_idx]
            norm = norm[:, keep]
        return aggregate_sensor_errors(norm, self.aggregation, self.topk_q).astype(np.float64)
