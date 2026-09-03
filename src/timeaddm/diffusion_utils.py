"""
DDPM utilities for TimeADDM, beta schedule, noise estimation loss and DDIM denoising
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def get_beta_schedule(schedule: str, beta_start: float, beta_end: float, n_steps: int) -> np.ndarray:
    """return a 1-D array of beta values for the requested schedule."""

    if schedule == "linear":
        return np.linspace(beta_start, beta_end, n_steps, dtype=np.float64)
    if schedule == "quad":
        # se interpola en la raiz y se eleva al cuadrado, asi el ruido crece despacio al
        # principio
        return (np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_steps, dtype=np.float64)) ** 2
    if schedule == "sigmoid":
        # el rango [-6, 6] cubre la sigmoide casi entera, de 0.002 a 0.998
        x = np.linspace(-6.0, 6.0, n_steps)
        sig = 1.0 / (np.exp(-x) + 1.0)
        return sig * (beta_end - beta_start) + beta_start
    raise ValueError(f"Unknown beta schedule: {schedule!r}")


def noise_estimation_loss(model: nn.Module, x0: torch.Tensor, t: torch.Tensor, e: torch.Tensor,
                          betas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """add noise at timestep t and return (x_noisy, scalar MSE loss). x0 and e are
    (batch, channels, length) channels-first latents, t is a (batch,) long tensor of diffusion
    timesteps."""

    # alpha_bar(t) = prod(1 - beta) hasta t, con view a (batch, 1, 1) para difundir por canal
    a = (1.0 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
    # forma cerrada del proceso directo, se salta a t de una vez en lugar de iterar
    x_noisy = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    predicted = model(x_noisy, t.float())
    # se suma sobre canal y longitud y se promedia sobre el lote, no es un MSE por elemento
    loss = (e - predicted).square().sum(dim=(1, 2)).mean()
    return x_noisy, loss


def _alpha_cumprod(betas: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """cumulative product of (1 - beta) up to timestep t. t=-1 returns 1.0."""

    # el cero delante desplaza el indice, de modo que t=-1 cae en el 1.0 del producto vacio
    b_pad = torch.cat([torch.zeros(1, device=betas.device), betas], dim=0)
    return (1.0 - b_pad).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)


def ddim_denoise(x_noisy: torch.Tensor, t_max: int, model: nn.Module, betas: torch.Tensor,
                 n_steps: int = 20, eta: float = 0.0) -> torch.Tensor:
    """DDIM denoising from noise level t_max back to t=0. n_steps is how many reverse steps
    are taken, by uniform subsampling, and eta=0 gives deterministic DDIM while eta=1 recovers
    DDPM."""

    # submuestreo uniforme, y seq_next es la misma lista corrida un puesto, el destino de cada
    # paso
    skip = max(1, t_max // n_steps)
    seq = list(range(0, t_max, skip))
    seq_next = [-1] + seq[:-1]
    n = x_noisy.size(0)
    xt = x_noisy.clone()

    with torch.no_grad():
        # se recorre al reves, de mas ruido a menos, emparejando cada t con su t siguiente
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t_vec = torch.full((n,), i, dtype=torch.long, device=x_noisy.device)
            t_next_vec = torch.full((n,), j, dtype=torch.long, device=x_noisy.device)

            at = _alpha_cumprod(betas, t_vec)
            at_next = _alpha_cumprod(betas, t_next_vec)

            # se estima el ruido y se despeja x0, que es la prediccion limpia desde el paso
            # actual
            et = model(xt, t_vec.float())
            x0_pred = (xt - et * (1.0 - at).sqrt()) / at.sqrt().clamp(min=1e-8)

            # c1 es la desviacion del ruido que se reinyecta, cero cuando eta=0, y c2 la
            # completa para que la varianza total del paso siga siendo la que toca
            c1 = eta * (
                (1.0 - at / at_next.clamp(min=1e-8)) * (1.0 - at_next) / (1.0 - at).clamp(min=1e-8)
            ).sqrt()
            c2 = ((1.0 - at_next) - c1 ** 2).clamp(min=0.0).sqrt()
            xt = at_next.sqrt() * x0_pred + c2 * et + c1 * torch.randn_like(x_noisy)

    return xt
