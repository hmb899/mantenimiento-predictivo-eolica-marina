"""
CSDI-style noise-prediction network with temporal and variable attention

the variable (feature) attention is the top component of the ImDiffusion ablation and is
not simplified away. the architecture derives from CSDI (Tashiro et al. 2021,
MIT), and ImDiffusion adds a learned embedding of the masking strategy (0/1) on top
# ref: IMDiffusion diff_models.py

layout, x (B, 2, K, L) = [cond_obs, noisy_target] channels, side_info (B, side_dim, K, L)
and output (B, K, L) = predicted noise. K = variables, L = time steps
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1x1(in_channels: int, out_channels: int) -> nn.Conv1d:
    # kernel 1, o sea una proyeccion lineal sobre el eje de canales que respeta la longitud
    layer = nn.Conv1d(in_channels, out_channels, 1)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def _transformer_layer(channels: int, nheads: int, ff_dim: int = 64) -> nn.TransformerEncoder:
    # ref: IMDiffusion diff_models.py:get_torch_trans (1-layer encoder, GELU, ff=64)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=nheads, dim_feedforward=ff_dim,
        activation="gelu", batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)


class DiffusionEmbedding(nn.Module):
    """sinusoidal embedding of the diffusion step plus two SiLU linear projections."""

    def __init__(self, num_steps: int, embedding_dim: int = 128) -> None:
        super().__init__()
        half = embedding_dim // 2
        assert half >= 2, "embedding_dim too small"
        # la tabla es fija, no se entrena, y por eso va como buffer no persistente
        self.register_buffer("table", self._build(num_steps, half), persistent=False)
        self.proj1 = nn.Linear(embedding_dim, embedding_dim)
        self.proj2 = nn.Linear(embedding_dim, embedding_dim)


    @staticmethod
    def _build(num_steps: int, half_dim: int) -> torch.Tensor:
        # frecuencias en escala logaritmica de 1 a 1e4, una fila por paso de difusion
        steps = torch.arange(num_steps).unsqueeze(1)  # (T, 1)
        # (1, half)
        freqs = 10.0 ** (torch.arange(half_dim) / (half_dim - 1) * 4.0).unsqueeze(0)
        table = steps * freqs
        # seno y coseno concatenados, de ahi que half_dim sea la mitad de embedding_dim
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T, dim)


    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.table[t]
        x = F.silu(self.proj1(x))
        return F.silu(self.proj2(x))


class ResidualBlock(nn.Module):
    """gated residual block, temporal attention, variable attention and side-info
    conditioning."""

    def __init__(self, side_dim: int, channels: int, diffusion_embedding_dim: int, nheads: int) -> None:
        super().__init__()
        self.diffusion_proj = nn.Linear(diffusion_embedding_dim, channels)
        # las proyecciones de salida dan 2*channels porque luego se parten en dos mitades
        self.cond_proj = _conv1x1(side_dim, 2 * channels)
        self.mid_proj = _conv1x1(channels, 2 * channels)
        self.output_proj = _conv1x1(channels, 2 * channels)
        self.time_layer = _transformer_layer(channels, nheads)
        self.feature_layer = _transformer_layer(channels, nheads)


    def _attend_time(self, y: torch.Tensor, base_shape: torch.Size) -> torch.Tensor:
        # atencion en el tiempo, las K variables se meten en el lote y la secuencia son los L
        # pasos
        b, ch, k, l = base_shape
        y = y.reshape(b, ch, k, l).permute(0, 2, 3, 1).reshape(b * k, l, ch)
        y = self.time_layer(y)
        # se deshace el permute para devolver el (B, ch, K*L) plano que espera el bloque
        return y.reshape(b, k, l, ch).permute(0, 3, 1, 2).reshape(b, ch, k * l)


    def _attend_feature(self, y: torch.Tensor, base_shape: torch.Size) -> torch.Tensor:
        # atencion entre variables, ahora el lote lleva los L pasos y la secuencia son las K
        # variables
        b, ch, k, l = base_shape
        y = y.reshape(b, ch, k, l).permute(0, 3, 2, 1).reshape(b * l, k, ch)
        y = self.feature_layer(y)
        return y.reshape(b, l, k, ch).permute(0, 3, 2, 1).reshape(b, ch, k * l)


    def forward(self, x: torch.Tensor, side_info: torch.Tensor,
                diffusion_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, ch, k, l = x.shape
        base_shape = x.shape
        # el embedding del paso entra sumado a cada posicion, aplanando K y L en un solo eje
        y = x.reshape(b, ch, k * l) + self.diffusion_proj(diffusion_emb).unsqueeze(-1)
        y = self._attend_time(y, base_shape)
        y = self._attend_feature(y, base_shape)
        y = self.mid_proj(y)  # (B, 2ch, K*L)
        side = side_info.reshape(b, side_info.shape[1], k * l)
        y = y + self.cond_proj(side)
        # activacion con puerta, una mitad del tensor abre y la otra aporta el valor
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_proj(y)
        # la otra particion en dos, el residuo se queda en la cadena y el skip sale al
        # sumatorio
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(b, ch, k * l)
        # el /sqrt(2) evita que la varianza crezca al ir sumando residuos capa tras capa
        return ((x + residual) / math.sqrt(2.0)).reshape(base_shape), skip


class ImDiffusionDenoiser(nn.Module):
    """predict the diffusion noise on target strips given conditioning and side info."""

    def __init__(self, num_steps: int, side_dim: int, channels: int = 64, layers: int = 4,
                 nheads: int = 8, diffusion_embedding_dim: int = 128) -> None:
        super().__init__()
        self.channels = channels
        self.diffusion_embedding = DiffusionEmbedding(num_steps, diffusion_embedding_dim)
        # embedding of the masking strategy (0/1) added to the step embedding, the
        # ImDiffusion-specific conditioning on top of CSDI
        # ref: IMDiffusion diff_models.py
        self.strategy_embedding = nn.Embedding(2, diffusion_embedding_dim)
        self.input_proj = _conv1x1(2, channels)
        self.out_proj1 = _conv1x1(channels, channels)
        # la ultima proyeccion arranca en cero, la red predice ruido nulo antes de entrenar
        self.out_proj2 = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.out_proj2.weight)
        nn.init.zeros_(self.out_proj2.bias)
        self.residual_layers = nn.ModuleList(
            ResidualBlock(side_dim, channels, diffusion_embedding_dim, nheads) for _ in range(layers)
        )


    def forward(self, x: torch.Tensor, side_info: torch.Tensor,
                t: torch.Tensor, strategy: torch.Tensor) -> torch.Tensor:
        """x (B, 2, K, L); side_info (B, side_dim, K, L); t (B,) long; strategy (B,) long in
        {0,1}."""

        b, _, k, l = x.shape
        h = F.relu(self.input_proj(x.reshape(b, 2, k * l))).reshape(b, self.channels, k, l)
        # paso de difusion y estrategia de mascara comparten embedding, se suman antes de
        # entrar
        emb = self.diffusion_embedding(t) + self.strategy_embedding(strategy)
        skips = []
        for layer in self.residual_layers:
            h, skip = layer(h, side_info, emb)
            skips.append(skip)
        # los skips de todas las capas se suman y se normalizan por la raiz del numero de capas
        h = torch.stack(skips).sum(dim=0) / math.sqrt(len(self.residual_layers))
        h = F.relu(self.out_proj1(h))
        return self.out_proj2(h).reshape(b, k, l)
