"""
anomaly scoring for TimeADDM

multi-level diffusion reconstruction score
    for each configured noise level t_k the window is encoded by the LSTMAE, noised to t_k,
    denoised back with DDIM and then decoded, and the MSE between the original and the
    reconstructed window is the score at that level
    the final score is a weighted sum over all the levels, so an anomaly is penalised across
    several scales at once

after scoring, threshold calibration and CUSUM follow the same protocol as the base BiLSTM-AE
pipeline, on the same CARE evaluation infrastructure
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.timeaddm.diffusion_utils import ddim_denoise, noise_estimation_loss

logger = logging.getLogger(__name__)


def _cusum_event(window_errors: np.ndarray, threshold: float, alarm_threshold: int) -> dict:
    """CUSUM criticality counter over one event sequence, mirroring
    AnomalyDetector.cusum_event(), the same formula from Barnabei et al. (2024)."""

    n = len(window_errors)
    criticality = np.zeros(n, dtype=np.int32)
    c = 0
    alarm_idx: int | None = None

    for i, err in enumerate(window_errors):
        # el contador sube uno por ventana sobre umbral y baja uno por ventana debajo, con
        # suelo 0
        c = c + 1 if err > threshold else max(0, c - 1)
        criticality[i] = c
        # la alarma se queda con el primer cruce, aunque el contador siga subiendo despues
        if alarm_idx is None and c >= alarm_threshold:
            alarm_idx = i

    return {
        "criticality_trace": criticality,
        "max_criticality": int(criticality.max()),
        "alarm_raised": alarm_idx is not None,
        "alarm_window_idx": alarm_idx,
    }


class TimeADDMScorer:
    """compute per-window anomaly scores using multi-level diffusion reconstruction.

    workflow per window batch
        1. LSTMAE encode  → latent z  (batch, W, H)
        2. permute        → z_perm    (batch, H, W)  [channels-first for Conv1d]
        3. for each noise level t_k, add noise → z_noisy, then DDIM denoise → z_clean
        4. permute back   → (batch, W, H)
        5. LSTMAE decode  → x_recon  (batch, W, F)
        6. score_k = MSE(x_orig, x_recon).mean over (W, F)
        7. score = sum_k  weight_k * score_k
    """

    def __init__(self, lstmae: nn.Module, diffusion: nn.Module, betas: torch.Tensor,
                 config: dict, device: torch.device | str = "cpu") -> None:
        self.lstmae = lstmae
        self.diffusion = diffusion
        self.betas = betas
        self.device = torch.device(device)
        # los dos modelos van en modo evaluacion, aqui solo se puntua, nunca se entrena
        self.lstmae.to(self.device).eval()
        self.diffusion.to(self.device).eval()

        sc = config["scoring"]
        self.noise_levels: list[int] = sc["noise_levels"]
        self.noise_weights: list[float] = sc["noise_weights"]
        self.ddim_steps: int = sc["ddim_steps"]
        self.percentile: float = sc["threshold_percentile"]
        self.max_calibration_samples: int = sc.get("max_calibration_samples", 50_000)
        self.alarm_threshold: int = config["cusum"]["alarm_threshold"]
        self.batch_size: int = config["lstmae"]["batch_size"]

        # niveles y pesos van emparejados por posicion, y los pesos tienen que sumar uno para
        # que el score compuesto quede en la misma escala que el de un nivel suelto
        assert len(self.noise_levels) == len(self.noise_weights), (
            "scoring.noise_levels and scoring.noise_weights must have the same length"
        )
        assert abs(sum(self.noise_weights) - 1.0) < 1e-6, (
            "scoring.noise_weights must sum to 1.0"
        )

        self.threshold_: float | None = None


    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        """return a per-window anomaly score array of shape (N,)."""

        all_scores: list[np.ndarray] = []
        n = len(windows)

        # por lotes porque cada nivel de ruido hace su propio bucle DDIM sobre el lote entero
        for start in range(0, n, self.batch_size):
            batch = torch.tensor(
                windows[start : start + self.batch_size], dtype=torch.float32
            ).to(self.device)
            all_scores.append(self._score_batch(batch).cpu().numpy())

        return np.concatenate(all_scores, axis=0)


    def calibrate(self, train_windows: np.ndarray) -> float:
        """score a subsample of training windows and set the threshold at the given
        percentile."""

        n = len(train_windows)
        if n > self.max_calibration_samples:
            # semilla fija, la submuestra de calibracion tiene que ser la misma en cada
            # ejecucion
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=self.max_calibration_samples, replace=False)
            sample = train_windows[idx]
            logger.info(
                f"[TimeADDMScorer] Calibrating on {self.max_calibration_samples:,} / {n:,} windows"
            )
        else:
            sample = train_windows
        # el umbral sale solo de train, nunca de los eventos que luego se puntuan
        errors = self.score_windows(sample)
        self.threshold_ = float(np.percentile(errors, self.percentile))
        logger.info(
            f"[TimeADDMScorer] Threshold at {self.percentile}th percentile: {self.threshold_:.6f}"
        )
        return self.threshold_


    def score_sequence(self, seq: dict) -> dict:
        """score one test sequence dict and return a cusum result dict."""

        if self.threshold_ is None:
            raise RuntimeError("Scorer not calibrated. Call calibrate() first.")
        errors = self.score_windows(seq["windows"])
        cusum = _cusum_event(errors, self.threshold_, self.alarm_threshold)
        # el dict del CUSUM se enriquece con lo que el harness necesita por ventana
        cusum["window_predictions"] = (errors > self.threshold_).astype(np.int8)
        cusum["window_errors"] = errors
        return cusum


    def _score_batch(self, x: torch.Tensor) -> torch.Tensor:
        """compute the weighted multi-level MSE for a single batch tensor (b, W, F)."""

        with torch.no_grad():
            z = self.lstmae.encode(x)  # (b, W, H)
            z_perm = z.permute(0, 2, 1)  # (b, H, W)

            score = torch.zeros(x.size(0), device=self.device)
            for t_k, w_k in zip(self.noise_levels, self.noise_weights):
                # t_k es un nivel contado desde uno, y el tensor de pasos va desde cero
                t_vec = torch.full((x.size(0),), t_k - 1, dtype=torch.long, device=self.device)
                e = torch.randn_like(z_perm)
                z_noisy, _ = noise_estimation_loss(self.diffusion, z_perm, t_vec, e, self.betas)
                z_clean = ddim_denoise(z_noisy, t_k, self.diffusion, self.betas, self.ddim_steps)
                z_clean_seq = z_clean.permute(0, 2, 1)  # (b, W, H)
                # el error se mide en espacio original, no en el latente, por eso se decodifica
                x_recon = self.lstmae.decode(z_clean_seq)
                mse = (x - x_recon).pow(2).mean(dim=(1, 2))
                score = score + w_k * mse

        return score
