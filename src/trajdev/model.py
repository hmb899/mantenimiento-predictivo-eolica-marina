"""
TrajForecaster, GRU con cabeza directa multi-paso, sin autoregresion, estable y right-sized

la entrada es el contexto (B, C, K) de componentes PCA estandarizados y la salida el pronostico
(B, H, K) de los H pasos siguientes de la ventana. el score por ventana deriva del tensor de
error |pred - real|.

dep_h{h}
    media de |err| sobre los primeros h pasos por K, la desviacion de trayectoria a horizonte h
dep_end
    |err| del ultimo paso, extrapolacion pura, el mas "departure"
dep_z
    z por componente del error medio, con stats del train y leakage-free, promediado sobre
    componentes
dis
    desacuerdo de ensemble, la std entre los M modelos de la prediccion, al estilo FATE, y solo
    si M>1
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TrajForecaster(nn.Module):
    def __init__(self, k: int, context: int, horizon: int, hidden: int = 128, layers: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.k, self.context, self.horizon = k, context, horizon
        # PyTorch ignora dropout con una sola capa y avisa, de ahi el condicional
        self.gru = nn.GRU(k, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        # una sola lineal saca el horizonte entero, sin realimentar la prediccion
        self.head = nn.Linear(hidden, horizon * k)


    def forward(self, x_ctx: torch.Tensor) -> torch.Tensor:  # (B, C, K) a (B, H, K)
        # solo interesa el estado oculto de la ultima capa, no la secuencia de salidas
        _, h = self.gru(x_ctx)
        return self.head(h[-1]).view(-1, self.horizon, self.k)


@torch.no_grad()
def predict_windows(models: list[TrajForecaster], windows: np.ndarray, context: int, horizon: int,
                    device: str, batch: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """pronostica los H pasos finales de cada ventana, con la media de los M modelos, y por
    lotes, sin copiar el array entero, porque el train de C son ~1.1M ventanas. devuelve
    (err_abs (N,H,K), dis (N,) desacuerdo)."""

    for m in models:
        m.eval()
    n = len(windows)
    err = np.empty((n, horizon, windows.shape[2]), dtype=np.float32)
    dis = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch):
        wb = windows[i:i + batch]
        # el contexto son los primeros pasos y el objetivo los horizon siguientes, sin solapar
        xb = torch.from_numpy(np.ascontiguousarray(wb[:, :context, :], dtype=np.float32)).to(device)
        yb = np.ascontiguousarray(wb[:, context:context + horizon, :], dtype=np.float32)
        preds = np.stack([m(xb).cpu().numpy() for m in models], axis=0)   # (M, b, H, K)
        err[i:i + batch] = preds.mean(axis=0) - yb  # err con signo (real - pred negado)
        # con un solo modelo no hay desacuerdo que medir, y dis se queda en cero
        if len(models) > 1:
            dis[i:i + batch] = preds.std(axis=0).mean(axis=(1, 2))
    return err, dis


def dep_variants(err: np.ndarray, sub_h: list[int]) -> dict[str, np.ndarray]:
    """scores por ventana desde el tensor de error con signo (N, H, K). dep_persist es
    |media con signo sobre H|, y por eso sale alto solo si la desviacion es persistente en una
    direccion, la firma del drift incipiente, mientras que el ruido alterna y se cancela. los
    demas usan |err|."""

    a = np.abs(err)
    out = {f"dep_h{h}": a[:, :h, :].mean(axis=(1, 2)) for h in sub_h}
    out["dep_end"] = a[:, -1, :].mean(axis=1)
    # el valor absoluto va fuera de la media, que es justo lo que mide la persistencia
    out["dep_persist"] = np.abs(err.mean(axis=1)).mean(axis=1)
    return out
