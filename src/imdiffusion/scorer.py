"""
window scoring for ImDiffusion

ImDiffusionScorer
    the harness contract, score_windows takes windows (N,W,K) and returns (N,), plus the
    percentile attribute, consumed by run_model.evaluate through the ReplayScorer pattern
    (see sweep_timeaddm_scoring.py)
    two complementary grating passes impute every observed position exactly once and the
    per-position |x - x_hat| is aggregated to one scalar per window
    the model neither smooths nor thresholds nor decides events, CUSUM,
    threshold and event logic stay in the harness
StepEnsemble
    vote fraction over the per-DDIM-step errors, calibrated on train scores only
"""
from __future__ import annotations

import numpy as np
import torch

from .diffusion import ImDiffusionModel
from .masking import gratings_for_model

AGG_MODES = ("mean", "median", "topk_q")


def aggregate_window_errors(err: torch.Tensor, mode: str = "mean", topk_q: float = 0.9) -> torch.Tensor:
    """aggregate per-position errors (B, P) to one scalar per window (B,)."""

    if mode == "mean":
        # misma escala de comportamiento que el MAE por ventana del AE delante del CUSUM
        return err.mean(dim=1)
    if mode == "median":
        # robusta a unos pocos errores por posicion muy grandes
        return err.median(dim=1).values
    if mode == "topk_q":
        # media de los errores por encima del cuantil q, mira solo las bandas peores
        k = max(1, int(round((1.0 - topk_q) * err.shape[1])))
        return err.topk(k, dim=1).values.mean(dim=1)
    raise ValueError(f"unknown aggregation mode: {mode}")


class ImDiffusionScorer:
    """score windows by complementary-mask imputation error."""

    def __init__(self, model: ImDiffusionModel, split: int = 9, ddim_steps: int = 10,
                 eta: float = 1.0, n_samples: int = 1, aggregation: str = "mean",
                 topk_q: float = 0.9, batch_size: int = 256, percentile: float = 99.0,
                 device: str | torch.device | None = None, mask_mode: str = "time",
                 exo_idx: list[int] | None = None) -> None:
        self.model = model
        self.split = split
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.n_samples = n_samples
        self.aggregation = aggregation
        self.topk_q = topk_q
        self.batch_size = batch_size
        self.percentile = float(percentile)  # cosmetic, harness JSON field (contract)
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.mask_mode = mask_mode
        self.exo_idx = exo_idx
        # las dos rejillas se calculan una sola vez aqui y se reutilizan en cada lote
        pair = gratings_for_model(model.window_size, model.n_features, split, mask_mode, exo_idx)
        # (K, L) each
        self._gratings = tuple(torch.from_numpy(np.ascontiguousarray(g)) for g in pair)


    def _check_input(self, windows: np.ndarray) -> None:
        if windows.ndim != 3 or windows.shape[1] != self.model.window_size \
                or windows.shape[2] != self.model.n_features:
            raise ValueError(
                f"windows must be (N, {self.model.window_size}, {self.model.n_features}), "
                f"got {windows.shape}")
        # the frozen pipeline imputes NaN before windowing, a non-finite value here would
        # silently corrupt the score, so observed ≡ 1 is asserted
        if not np.isfinite(windows).all():
            raise ValueError("windows contain non-finite values; observed mask is assumed all-True")


    def _batch_errors(self, x: torch.Tensor, per_step: bool) -> torch.Tensor:
        """per-position imputation error for one batch x (B, K, L), returning (B, K, L), or
        (S, B, K, L) per DDIM step when per_step is True."""

        err = None
        # las dos estrategias son complementarias, cada posicion la imputa exactamente una
        for s, grating in enumerate(self._gratings):
            g = grating.to(self.device)[None, :, :]  # (1, K, L) model-layout mask
            # lo que una estrategia marca como objetivo es justo lo que la otra condiciona
            target = g.expand_as(x).to(x.dtype)
            cond = 1.0 - target
            acc = None
            for _ in range(self.n_samples):
                out = self.model.impute(x, cond, strategy=s, ddim_steps=self.ddim_steps,
                                        eta=self.eta, return_intermediates=per_step)
                if per_step:
                    _, inter = out
                    draw = torch.stack([(x - xh).abs() for xh in inter])  # (S, B, K, L)
                else:
                    draw = (x - out).abs()
                acc = draw if acc is None else acc + draw
            # se promedia sobre las tiradas y se anula fuera del objetivo de esta estrategia
            e = (acc / self.n_samples) * target
            err = e if err is None else err + e
        return err


    @torch.no_grad()
    def _score(self, windows: np.ndarray, per_step: bool) -> np.ndarray:
        self._check_input(windows)
        self.model.eval()
        chunks = []
        for start in range(0, len(windows), self.batch_size):
            batch = np.ascontiguousarray(windows[start:start + self.batch_size], dtype=np.float32)
            # el permute lleva la ventana del lado numpy (N, W, K) al layout del modelo (B, K,
            # L)
            x = torch.from_numpy(batch).to(self.device).permute(0, 2, 1)  # (B, K, L)
            err = self._batch_errors(x, per_step)
            # K y L se aplanan en un unico eje de posiciones antes de agregar
            flat = err.flatten(start_dim=-2)  # (..., B, K*L)
            agg = aggregate_window_errors(flat.reshape(-1, flat.shape[-1]), self.aggregation, self.topk_q)
            if per_step:
                agg = agg.reshape(flat.shape[0], -1).T  # (B, S)
            chunks.append(agg.cpu().numpy().astype(np.float64))
        return np.concatenate(chunks)


    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        """harness contract, takes (N, W, K) and returns (N,) float64 raw scores."""

        return self._score(windows, per_step=False)


    def score_windows_steps(self, windows: np.ndarray) -> np.ndarray:
        """per-DDIM-step scores (N, S) for the step ensemble (default off)."""

        return self._score(windows, per_step=True)


    @torch.no_grad()
    def score_windows_full(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        """one scoring pass giving every aggregation plus per-DDIM-step and per-feature scores,
        as {"mean"/"median"/"topk_q": (N,), "steps": (N, S), "perfeat": (N, K)}."""

        self._check_input(windows)
        self.model.eval()
        out: dict[str, list[np.ndarray]] = {m: [] for m in AGG_MODES}
        steps_chunks: list[np.ndarray] = []
        perfeat_chunks: list[np.ndarray] = []
        for start in range(0, len(windows), self.batch_size):
            batch = np.ascontiguousarray(windows[start:start + self.batch_size], dtype=np.float32)
            x = torch.from_numpy(batch).to(self.device).permute(0, 2, 1)
            err_steps = self._batch_errors(x, per_step=True)  # (S, B, K, L)
            # el error del ultimo paso DDIM es el mismo que da score_windows, con las mismas
            # tiradas del rng, asi el barrido posterior reproduce sin volver a puntuar
            final = err_steps[-1].flatten(start_dim=1)  # (B, K*L)
            for mode in AGG_MODES:
                agg = aggregate_window_errors(final, mode, self.topk_q)
                out[mode].append(agg.cpu().numpy().astype(np.float64))
            per_step = err_steps.flatten(start_dim=2).mean(dim=-1)  # (S, B)
            steps_chunks.append(per_step.T.cpu().numpy().astype(np.float64))
            # un fallo suele estar en pocos sensores y la media sobre las K features lo diluye,
            # guardar (N, K) deja probar luego agregaciones por feature (max, cuantiles, top-m)
            perfeat_chunks.append(err_steps[-1].mean(dim=-1).cpu().numpy().astype(np.float32))
        result: dict[str, np.ndarray] = {m: np.concatenate(v) for m, v in out.items()}
        result["steps"] = np.concatenate(steps_chunks)
        result["perfeat"] = np.concatenate(perfeat_chunks)
        return result


class StepEnsemble:
    """vote-fraction score over per-DDIM-step errors, calibrated on train scores only, a
    continuous score with all event thresholding left in the harness.
    # ref: IMDiffusion ensemble_proper.py (per-step normalisation idea)
    """

    def __init__(self, percentile: float = 99.0) -> None:
        self.percentile = float(percentile)
        self.step_mean: np.ndarray | None = None
        self.thresholds: np.ndarray | None = None


    def fit(self, train_step_scores: np.ndarray) -> "StepEnsemble":
        """train_step_scores (M, S), the per-step scores of the train windows."""

        if train_step_scores.ndim != 2:
            raise ValueError(f"expected (M, S), got {train_step_scores.shape}")
        # cada paso se normaliza por su residuo medio en train y se corta en un percentil de
        # train
        self.step_mean = train_step_scores.mean(axis=0).clip(min=1e-12)
        norm = train_step_scores / self.step_mean
        self.thresholds = np.percentile(norm, self.percentile, axis=0)
        return self


    def score(self, step_scores: np.ndarray) -> np.ndarray:
        """step_scores (N, S), returns the vote fraction (N,) in [0, 1]."""

        if self.thresholds is None:
            raise RuntimeError("StepEnsemble not fitted")
        norm = step_scores / self.step_mean
        # fraccion de pasos que votan anomalo, a proposito distinto de la eleccion oficial del
        # umbral por mejor-F1 y point-adjust sobre etiquetas de test, prohibida aqui
        return (norm > self.thresholds).mean(axis=1)
