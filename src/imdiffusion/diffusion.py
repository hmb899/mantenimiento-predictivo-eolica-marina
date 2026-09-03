"""
conditional diffusion for imputation, DDPM training loss and DDIM imputer

schedule T=50, quadratic betas from 1e-4 to 0.5. the high beta_end is a deliberate choice
of the official implementation for imputation, not the usual 0.02
# ref: IMDiffusion config/base.yaml

training input has 2 channels [cond_obs, noisy_target] (conditional mode) and the loss is
computed only on target = observed ∧ grating positions
# ref: IMDiffusion main_model.py:calc_loss
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .denoiser import ImDiffusionDenoiser


def quad_beta_schedule(num_steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """quadratic beta schedule, linspace on sqrt(beta) and then squared."""

    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2


def masked_noise_loss(noise: torch.Tensor, predicted: torch.Tensor,
                      target_mask: torch.Tensor) -> torch.Tensor:
    """MSE between true and predicted noise restricted to target positions, everything
    outside target_mask contributes exactly zero whatever its value.
    # ref: IMDiffusion main_model.py:calc_loss (sum of squared masked residual / count)
    """

    tm = target_mask.to(noise.dtype)
    residual = (noise - predicted) * tm
    # media sobre las posiciones target, con suelo 1 para no dividir entre cero
    return residual.square().sum() / tm.sum().clamp(min=1.0)


def time_embedding(positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """fixed sinusoidal embedding of intra-window positions, (L, d_model). the windows are a
    regular 10-min grid, so positions = arange(W) matches the official embedding."""

    pe = torch.zeros(len(positions), d_model)
    pos = positions.unsqueeze(1).float()
    # frecuencias geometricas, columnas pares en seno y columnas impares en coseno
    div = 1.0 / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ImDiffusionModel(nn.Module):
    """conditional diffusion imputer, side info, denoiser and DDPM/DDIM machinery. layout
    inside is (B, K, L), the window side (N, W, K) is permuted at the scorer boundary."""

    def __init__(self, n_features: int, window_size: int, num_steps: int = 50,
                 beta_start: float = 1e-4, beta_end: float = 0.5, channels: int = 64,
                 layers: int = 4, nheads: int = 8, diffusion_embedding_dim: int = 128,
                 feature_embedding_dim: int = 16, time_embedding_dim: int = 128) -> None:
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.num_steps = num_steps
        betas = quad_beta_schedule(num_steps, beta_start, beta_end)
        # buffers no persistentes, se recalculan del schedule al construir y no van al
        # state_dict
        self.register_buffer("betas", betas, persistent=False)
        # el acumulado alpha_bar es lo unico que hace falta para saltar directo a cualquier
        # paso t
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0), persistent=False)
        self.register_buffer(
            "time_emb", time_embedding(torch.arange(window_size), time_embedding_dim), persistent=False)
        self.feature_embedding = nn.Embedding(n_features, feature_embedding_dim)
        side_dim = time_embedding_dim + feature_embedding_dim + 1  # +1 = cond_mask channel
        self.denoiser = ImDiffusionDenoiser(
            num_steps=num_steps, side_dim=side_dim, channels=channels, layers=layers,
            nheads=nheads, diffusion_embedding_dim=diffusion_embedding_dim)


    def _side_info(self, cond_mask: torch.Tensor) -> torch.Tensor:
        """time and variable embeddings plus the cond_mask channel, as (B, side_dim, K, L)."""

        b, k, l = cond_mask.shape
        # los dos embeddings se difunden a todas las posiciones, el tiempo por K y la variable
        # por L
        time_part = self.time_emb[None, None].expand(b, k, l, -1)  # (B,K,L,Dt)
        # (B,K,L,Df)
        feat_part = self.feature_embedding.weight[None, :, None, :].expand(b, k, l, -1)
        side = torch.cat([time_part, feat_part], dim=-1).permute(0, 3, 1, 2)
        # la propia mascara de condicionamiento entra como un canal mas del side info
        return torch.cat([side, cond_mask.unsqueeze(1)], dim=1)


    @staticmethod
    def _check_masks(x: torch.Tensor, cond_mask: torch.Tensor, target_mask: torch.Tensor) -> None:
        if cond_mask.shape != x.shape or target_mask.shape != x.shape:
            raise ValueError(f"mask shapes {cond_mask.shape}/{target_mask.shape} != x {x.shape}")
        # structural guarantee, a position is conditioning XOR target XOR unobserved
        if bool((cond_mask * target_mask).any()):
            raise ValueError("cond_mask and target_mask overlap")


    @staticmethod
    def _as_strategy(strategy: int | torch.Tensor, b: int, device: torch.device) -> torch.Tensor:
        # la estrategia puede venir escalar, para todo el lote, o ya por muestra
        if isinstance(strategy, int):
            return torch.full((b,), strategy, dtype=torch.long, device=device)
        return strategy.to(device=device, dtype=torch.long)


    def calc_loss(self, x: torch.Tensor, cond_mask: torch.Tensor, target_mask: torch.Tensor,
                  strategy: int | torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """DDPM loss on target positions only. x (B, K, L) float, cond_mask and target_mask
        (B, K, L) float 0/1 and disjoint, strategy int or (B,) long, t optional for tests."""

        self._check_masks(x, cond_mask, target_mask)
        b = x.shape[0]
        strategy = self._as_strategy(strategy, b, x.device)
        # lo que no es ni cond ni target se pone a cero antes de entrar, asi no toca la perdida
        x = x * (cond_mask + target_mask)
        if t is None:
            # un paso de difusion al azar por muestra, el test lo fija para ser determinista
            t = torch.randint(0, self.num_steps, (b,), device=x.device)
        noise = torch.randn_like(x)
        # forma cerrada del DDPM, se salta directo al paso t sin iterar
        a = self.alpha_bar[t].view(b, 1, 1)
        x_noisy = a.sqrt() * x + (1.0 - a).sqrt() * noise
        # canal 0 lo condicionado limpio y canal 1 el objetivo con ruido, o sea el modo
        # condicional
        total_input = torch.stack([cond_mask * x, (1.0 - cond_mask) * x_noisy], dim=1)
        predicted = self.denoiser(total_input, self._side_info(cond_mask), t, strategy)
        return masked_noise_loss(noise, predicted, target_mask)


    @torch.no_grad()
    def impute(self, x: torch.Tensor, cond_mask: torch.Tensor, strategy: int | torch.Tensor,
               ddim_steps: int = 10, eta: float = 1.0,
               return_intermediates: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """DDIM imputation of all ¬cond positions, cond positions keep their values. returns
        x_hat (B, K, L), and with return_intermediates also the per-step x0 estimates, each
        combined with the conditioning values, used by the step-ensemble score.
        # ref: IMDiffusion main_model.py:ddim_impute / get_middle_impute_value
        """

        b = x.shape[0]
        strategy = self._as_strategy(strategy, b, x.device)
        x_cond = x * cond_mask  # the model never sees values outside cond
        # subconjunto de pasos del schedule, unique porque el redondeo puede repetir alguno
        ts = torch.linspace(0, self.num_steps - 1, min(ddim_steps, self.num_steps)).round().long().unique()
        # se arranca de ruido puro y se recorre el schedule hacia atras
        sample = torch.randn_like(x)
        intermediates: list[torch.Tensor] = []
        x0 = sample
        for i in range(len(ts) - 1, -1, -1):
            t = ts[i]
            a_t = self.alpha_bar[t]
            t_vec = torch.full((b,), int(t), dtype=torch.long, device=x.device)
            # la red ve lo condicionado limpio y la parte a imputar en su estado actual
            diff_input = torch.stack([x_cond, (1.0 - cond_mask) * sample], dim=1)
            eps = self.denoiser(diff_input, self._side_info(cond_mask), t_vec, strategy)
            # estimacion de x0 despejando el ruido predicho de la misma forma cerrada
            x0 = (sample - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()
            if return_intermediates:
                intermediates.append(x_cond + (1.0 - cond_mask) * x0)
            if i > 0:
                # paso DDIM, sigma dosifica el ruido que se reinyecta y con eta=0 seria
                # determinista
                a_prev = self.alpha_bar[ts[i - 1]]
                sigma = eta * ((1.0 - a_prev) / (1.0 - a_t)).sqrt() * (1.0 - a_t / a_prev).sqrt()
                dir_coeff = (1.0 - a_prev - sigma ** 2).clamp(min=0.0).sqrt()
                sample = a_prev.sqrt() * x0 + dir_coeff * eps + sigma * torch.randn_like(x)
            else:
                # ultimo paso, se toma la estimacion de x0 sin volver a meter ruido
                sample = x0
        # solo se sustituye lo no condicionado, las posiciones cond conservan su valor original
        x_hat = x_cond + (1.0 - cond_mask) * sample
        if return_intermediates:
            return x_hat, intermediates
        return x_hat
