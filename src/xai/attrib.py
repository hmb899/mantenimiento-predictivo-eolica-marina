"""
atribucion para los notebooks XAI

piramide metodologica de tres niveles, del mas exacto al mas general. no se elige uno, se
presentan juntos porque cada nivel valida al anterior.

ae_residual_sensor
    nivel exacto, residual por sensor via la inversa de PCA, exacto porque el mapa es lineal
    el z por canal nativo de TrajDev (loaders.trajdev_channel_z) es su equivalente alli
    el residual del AE sufre smearing (Westerhuis 2000), por eso se z-normaliza con stats de
    train y se contrasta con oclusion

ae_integrated_gradients, trajdev_integrated_gradients
    nivel gradiente, Integrated Gradients con la proyeccion PCA compuesta dentro del grafo,
    o sea la regla de la cadena exacta hasta el sensor
    IG es deviation-agnostic (Ide 2021), por eso se presenta siempre junto al residual
    cudnn no soporta backward de RNN en eval, se deshabilita al atribuir (Captum #564)

occlusion_per_channel, trajdev_counterfactual
    nivel perturbacion, model-agnostic, la validacion mas defendible ante quien decide
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.xai.loaders import DEVICE


# AE, residual por sensor (exacto)
@torch.no_grad()
def ae_residual_sensor(ae: dict, raw_windows: np.ndarray, batch: int = 1024) -> np.ndarray:
    """|residual| por sensor y ventana en espacio fisico, |raw - inverse(AE(transform(raw)))|
    promediado sobre la ventana, (N, F). contribution-plot clasico (MSPC/ARCANA) trasladado al
    AE, exacto porque PCA es lineal; z-normalizar con stats de train antes de rankear."""

    model = ae["model"]
    out = []
    for s in range(0, len(raw_windows), batch):
        raw = raw_windows[s:s + batch]
        k = ae["transform"](raw)  # espacio sensor al espacio del AE
        x = torch.tensor(k, dtype=torch.float32, device=DEVICE)
        xhat = model(x).cpu().numpy()
        recon_raw = ae["inverse"](xhat)  # y de vuelta a espacio sensor
        # axis=1 promedia los W pasos de la ventana y deja (B, F), una cifra por sensor
        out.append(np.abs(raw - recon_raw).mean(axis=1))
    return np.concatenate(out)


def residual_train_stats(res_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mu, sd) por sensor del residual de train (leakage-free) para z-normalizar
    atribuciones."""

    # el epsilon evita dividir por cero en un sensor plano durante todo el train
    return res_train.mean(axis=0), res_train.std(axis=0) + 1e-8


# AE, IG end-to-end (gradiente)
class _AEComposed(nn.Module):
    """sensor (N, W, F), luego scaler+PCA como capa afin congelada, luego el AE, y sale el MAE
    por ventana (N,). con la proyeccion dentro del grafo, IG atribuye en espacio sensor por la
    cadena exacta."""

    def __init__(self, ae: dict) -> None:
        super().__init__()
        self.model = ae["model"]
        af = ae["affine"]
        # buffers y no parametros, la proyeccion viaja con el modulo pero no se entrena
        self.register_buffer("W", torch.tensor(af["W"], dtype=torch.float32))  # (F, K)
        self.register_buffer("b", torch.tensor(af["b"], dtype=torch.float32))  # (K,)


    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        k = raw @ self.W + self.b
        # el score es el MAE de reconstruccion, media sobre pasos y componentes a la vez
        return (k - self.model(k)).abs().mean(dim=(1, 2))


def ae_integrated_gradients(ae: dict, raw_windows: np.ndarray, baseline_raw: np.ndarray,
                            steps: int = 50, batch: int = 64) -> np.ndarray:
    """IG del score MAE respecto de cada (paso, sensor) del input crudo, (N, W, F).
    baseline_raw es la mediana de ventanas normales de train en espacio sensor, con forma de
    ventana. presentar junto al residual, porque IG explica el modelo y no la desviacion."""

    from captum.attr import IntegratedGradients

    net = _AEComposed(ae).to(DEVICE).eval()
    ig = IntegratedGradients(net)
    base = torch.tensor(baseline_raw[None], dtype=torch.float32, device=DEVICE)  # (1, W, F)
    outs = []
    prev = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False  # backward de RNN en eval (Captum #564)
    try:
        for s in range(0, len(raw_windows), batch):
            x = torch.tensor(raw_windows[s:s + batch], dtype=torch.float32, device=DEVICE)
            att = ig.attribute(x, baselines=base, n_steps=steps)
            outs.append(att.detach().cpu().numpy())
    finally:
        # se restaura pase lo que pase, cudnn es estado global del proceso
        torch.backends.cudnn.enabled = prev
    return np.concatenate(outs)


# TrajDev, IG sobre el contexto
class _TrajTarget(nn.Module):
    """ventana (N, ctx+H, K) al |err| del canal objetivo agregado sobre el horizonte, (N,).
    err = y_true menos la media del ensemble, el mismo pronostico que el score de campana."""

    def __init__(self, bundle: dict, channel: int) -> None:
        super().__init__()
        self.models = nn.ModuleList(bundle["models"])
        self.ctx = bundle["arch"]["context"]
        self.h = bundle["arch"]["horizon"]
        self.k = bundle["arch"]["k"]
        self.channel = channel


    def forward(self, win: torch.Tensor) -> torch.Tensor:
        # la ventana llega concatenada, primero los ctx pasos de contexto y luego los h a
        # predecir
        ctx, y = win[:, :self.ctx, :], win[:, self.ctx:self.ctx + self.h, :]
        # cada miembro devuelve el horizonte aplanado, se reconstruye (B, h, k) y se promedia
        preds = torch.stack([m(ctx).reshape(-1, self.h, self.k) for m in self.models]).mean(dim=0)
        return (y[:, :, self.channel] - preds[:, :, self.channel]).abs().mean(dim=1)


def trajdev_integrated_gradients(bundle: dict, windows: np.ndarray, channel: int,
                                 baseline_window: np.ndarray, steps: int = 50,
                                 batch: int = 128) -> np.ndarray:
    """IG del error del canal disparador respecto del input completo, (N, ctx+H, K). el tramo
    [0:ctx] atribuye al contexto, o sea deriva relacional, el resto de sensores ya no predicen
    el canal; el tramo [ctx:] atribuye al propio tramo objetivo, el sensor se desvio."""

    from captum.attr import IntegratedGradients

    net = _TrajTarget(bundle, channel).to(DEVICE).eval()
    ig = IntegratedGradients(net)
    base = torch.tensor(baseline_window[None], dtype=torch.float32, device=DEVICE)
    outs = []
    prev = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False  # backward de RNN en eval (Captum #564)
    try:
        for s in range(0, len(windows), batch):
            x = torch.tensor(windows[s:s + batch], dtype=torch.float32, device=DEVICE)
            att = ig.attribute(x, baselines=base, n_steps=steps)
            outs.append(att.detach().cpu().numpy())
    finally:
        torch.backends.cudnn.enabled = prev
    return np.concatenate(outs)


# oclusion (perturbacion)
def occlusion_per_channel(score_fn, windows: np.ndarray, fill: np.ndarray,
                          channels: list[int] | None = None) -> np.ndarray:
    """delta del score al sustituir cada canal por su relleno, (N, F_sel). score_fn es el score
    exacto del modelo, de (N, W, K/F) a (N,), por ejemplo un lambda con ae_scores o trajdev
    max-z; fill es el vector (F,) de relleno, media o mediana de train por canal. responde a si
    quito este sensor la anomalia desaparece, la validacion mas defendible."""

    base = score_fn(windows)
    chans = channels if channels is not None else list(range(windows.shape[2]))
    deltas = np.empty((len(windows), len(chans)), dtype=np.float64)
    for j, c in enumerate(chans):
        # se ocluye un solo canal por pasada, sobre una copia, para no arrastrar oclusiones
        w = windows.copy()
        w[:, :, c] = fill[c]
        deltas[:, j] = base - score_fn(w)  # >0, el canal sostiene la anomalia
    return deltas


# counterfactual TrajDev
def trajdev_counterfactual(bundle: dict, windows: np.ndarray,
                           channel: int | list[int]) -> np.ndarray:
    """sustituye el tramo objetivo de uno o varios canales por su propio pronostico y devuelve
    las ventanas contrafactuales. test de suficiencia, si el score o el CUSUM cae, esos canales
    explican la alarma. es la perturbacion correcta para un score de pronostico porque preserva
    la dinamica, y un relleno constante rompe la trayectoria y genera su propio error de
    forecast."""

    from src.xai.loaders import trajdev_forecast_errors

    err, _ = trajdev_forecast_errors(bundle, windows)
    ctx, h = bundle["arch"]["context"], bundle["arch"]["horizon"]
    # admite un canal suelto o una lista, se normaliza a lista para tratar los dos igual
    chans = [channel] if isinstance(channel, (int, np.integer)) else list(channel)
    cf = windows.copy()
    # predict_windows define err = pred - y_true, luego el pronostico del ensemble es y_true +
    # err
    for c in chans:
        cf[:, ctx:ctx + h, c] = windows[:, ctx:ctx + h, c] + err[:, :, c]
    return cf
