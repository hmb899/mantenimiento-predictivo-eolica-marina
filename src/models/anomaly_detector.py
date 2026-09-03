"""
deteccion de anomalias por umbral de error de reconstruccion mas contador de
criticidad CUSUM

la deteccion tiene dos pasos, uno a nivel de ventana y otro a nivel de evento

run_cusum
    traza del CUSUM estadistico sobre una secuencia de errores de ventana
fit_cusum_params
    calibra mu, k y h_base sobre los errores de entrenamiento
AnomalyDetector
    una ventana es anomala cuando su MAE pasa del umbral calibrado
    la alarma del evento salta cuando el contador de criticidad llega a alarm_threshold (tc)

referencias Barnabei et al. (2024) y Guck et al. (2024) CARE Score
"""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# barrido de h_scale para el CUSUM estadistico, el score MAE por ventana no esta en la
# escala original y la h por defecto sale demasiado conservadora, h = h_base * h_scale
H_SCALES = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]


def run_cusum(loss_seq: np.ndarray, mu: float, k: float) -> np.ndarray:
    """CUSUM estadistico, S[t] = max(0, S[t-1] + x_t - mu - k), devuelve la traza S."""

    S = np.empty(len(loss_seq), dtype=np.float64)
    s = 0.0
    for t, x in enumerate(loss_seq):
        # el suelo en cero borra lo acumulado cuando el error vuelve a la normalidad
        s = max(0.0, s + float(x) - mu - k)
        S[t] = s
    return S


def fit_cusum_params(train_losses: np.ndarray, slack: float, percentile: float) -> tuple[float, float, float]:
    """calibra (mu, k, h_base) sobre los errores de reconstruccion del train ya limpiado con
    IF. mu es la media, k es slack por la std y h_base un percentil del CUSUM del train."""

    mu = float(np.mean(train_losses))
    # k es la banda muerta, solo se acumula lo que pasa de mu + k
    k = slack * float(np.std(train_losses))
    # la traza del train se recorre en su orden, h_base sale de esa traza y no de un barajado
    h = float(np.percentile(run_cusum(train_losses, mu, k), percentile))
    return mu, k, h


class AnomalyDetector:
    """calcula errores de reconstruccion, calibra el umbral y corre el CUSUM."""

    def __init__(self, model: torch.nn.Module, config: dict, device: torch.device | str = "cpu") -> None:
        """fija el dispositivo del modelo, el percentil del umbral y el umbral de alarma
        del CUSUM."""

        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        # el detector no entrena, el modelo entra ya ajustado y se queda en eval
        self.model.eval()

        th_cfg = config["threshold"]
        self.percentile: float = th_cfg["percentile"]
        self.threshold_: float | None = None

        cusum_cfg = config["cusum"]
        self.alarm_threshold: int = cusum_cfg["alarm_threshold"]  # el tc del CARE, misma clave

        self.batch_size: int = config.get("autoencoder", {}).get("batch_size", 512)
        # readout del error, mae_window sobre la ventana entera o l2_last solo en el ultimo
        # paso
        self.score_mode: str = config.get("score_mode", "mae_window")


    def calibrate(self, train_windows: np.ndarray, batch_size: int | None = None) -> float:
        """calcula los errores sobre las ventanas de train limpias y pone el umbral en el
        percentil configurado."""

        # el umbral es un percentil de los errores del train limpio, nada del test entra aqui
        errors = self._compute_errors(train_windows, batch_size or self.batch_size)
        self.threshold_ = float(np.percentile(errors, self.percentile))
        logger.info(
            f"[AnomalyDetector] Threshold calibrated at {self.percentile}th percentile "
            f"of training errors {self.threshold_:.6f}"
        )
        return self.threshold_


    def score_windows(self, windows: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """devuelve el error de reconstruccion MAE por ventana como array de numpy."""

        return self._compute_errors(windows, batch_size or self.batch_size)


    def predict_windows(self, windows: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """devuelve 1 en las ventanas anomalas y 0 en las normales."""

        if self.threshold_ is None:
            raise RuntimeError("Threshold not calibrated. Call calibrate() first.")
        errors = self._compute_errors(windows, batch_size or self.batch_size)
        # comparacion estricta, una ventana justo en el umbral cuenta como normal
        return (errors > self.threshold_).astype(np.int8)


    def cusum_event(self, window_errors: np.ndarray) -> dict:
        """corre el contador de criticidad CUSUM en un evento y devuelve la info de alarma."""

        if self.threshold_ is None:
            raise RuntimeError("Threshold not calibrated. Call calibrate() first.")

        n = len(window_errors)
        criticality = np.zeros(n, dtype=np.int32)
        c = 0
        alarm_idx: int | None = None

        # el contador sube uno por ventana sobre umbral y baja uno por ventana normal, sin
        # bajar de cero, hace falta persistencia y no un pico suelto
        for i, err in enumerate(window_errors):
            if err > self.threshold_:
                c += 1
            else:
                c = max(0, c - 1)
            criticality[i] = c
            # se guarda la primera vez que se alcanza tc, no la ultima ni el maximo
            if alarm_idx is None and c >= self.alarm_threshold:
                alarm_idx = i

        return {
            "criticality_trace": criticality,
            "max_criticality": int(criticality.max()),
            "alarm_raised": alarm_idx is not None,
            "alarm_window_idx": alarm_idx,
        }


    def _compute_errors(self, windows: np.ndarray, batch_size: int) -> np.ndarray:
        """devuelve el error de reconstruccion MAE por ventana como array de numpy."""

        all_errors: list[np.ndarray] = []
        n = len(windows)

        # por lotes y sin gradiente, las ventanas de una granja entera no caben de una vez
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = torch.tensor(windows[start : start + batch_size], dtype=torch.float32).to(self.device)
                # el readout lo resuelve el modelo, aqui solo se le pasa el modo
                errors = self.model.reconstruction_error(batch, reduction="mean", mode=self.score_mode)
                all_errors.append(errors.cpu().numpy())

        # los lotes se concatenan en el orden en que salieron, el indice de ventana se conserva
        return np.concatenate(all_errors, axis=0)
