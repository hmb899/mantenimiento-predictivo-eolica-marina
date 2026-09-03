"""
1-D U-Net noise predictor for DDPM on LSTMAE latent representations

input and output are (batch, hidden_size, seq_len), channels-first to match the Conv1d
conventions, so callers have to permute from and to (batch, seq_len, hidden_size)

config namespace fields consumed
    model.ch               base channel width (must be divisible by 32)
    model.out_ch           latent channels = LSTMAE hidden_size
    model.ch_mult          channel multipliers per encoder level, e.g. [1, 2]
    model.num_res_blocks   ResBlocks per encoder level
    model.dropout
    model.attn_resolutions resolutions at which to add self-attention ([] to disable)
    data.image_size        seq_len (spatial dimension seen by Conv1d layers)

references Hu et al. (ICASSP 2024), architecture adapted from Ho et al. (NeurIPS 2020)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# embedding sinusoidal del paso de difusion, mitad senos y mitad cosenos
def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    # frecuencias en progresion geometrica entre 1 y 1/10000, la escala de Ho et al.
    freq = math.log(10000.0) / (half - 1)
    emb = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))  # dim impar, se rellena la columna que falta
    return emb


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# 32 grupos fijos, de ahi que ch tenga que ser divisible por 32
def _groupnorm(n_ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=n_ch, eps=1e-6, affine=True)


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = _groupnorm(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.temb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2 = _groupnorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=1)
        # el atajo solo lleva convolucion cuando hay que cambiar de numero de canales
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None


    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(_swish(self.norm1(x)))
        # el paso de difusion entra como sesgo por canal, unsqueeze(-1) lo difunde por la
        # longitud
        h = h + self.temb_proj(_swish(temb)).unsqueeze(-1)
        h = self.conv2(self.drop(_swish(self.norm2(h))))
        return h + (self.skip(x) if self.skip is not None else x)


class _AttnBlock(nn.Module):
    def __init__(self, n_ch: int) -> None:
        super().__init__()
        self.norm = _groupnorm(n_ch)
        self.q = nn.Conv1d(n_ch, n_ch, kernel_size=1)
        self.k = nn.Conv1d(n_ch, n_ch, kernel_size=1)
        self.v = nn.Conv1d(n_ch, n_ch, kernel_size=1)
        self.proj = nn.Conv1d(n_ch, n_ch, kernel_size=1)
        self.scale = n_ch ** -0.5  # escala 1/sqrt(d) del producto escalar


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        # atencion entre posiciones de la ventana, cada paso atiende a todos los demas
        q = self.q(h).permute(0, 2, 1)  # (b, l, c)
        k = self.k(h)  # (b, c, l)
        w = torch.bmm(q, k) * self.scale  # (b, l, l)
        # el softmax va sobre la clave y luego se traspone, para poder multiplicar v por la
        # derecha
        w = w.softmax(dim=2).permute(0, 2, 1)
        h = torch.bmm(self.v(h), w)
        return x + self.proj(h)


class DiffusionUNet(nn.Module):
    """1-D U-Net noise predictor.

    encoder  conv_in → [ResBlocks + optional Attn + Downsample] × n_levels
    middle   ResBlock + Attn + ResBlock
    decoder  [ResBlocks with skip + optional Attn + Upsample] × n_levels → conv_out
    """

    def __init__(self, config) -> None:
        super().__init__()
        ch: int = config.model.ch
        out_ch: int = config.model.out_ch
        ch_mult: tuple[int, ...] = tuple(config.model.ch_mult)
        num_res: int = config.model.num_res_blocks
        attn_res: set[int] = set(config.model.attn_resolutions)
        dropout: float = config.model.dropout
        n_levels: int = len(ch_mult)
        # el ancho del embedding de tiempo es 4x el base, como en Ho et al.
        temb_ch: int = ch * 4
        resolution: int = config.data.image_size

        assert ch % 32 == 0, f"ch must be divisible by 32 for GroupNorm, got {ch}"

        self._n_levels = n_levels
        self._num_res = num_res
        self.ch = ch

        # el 1 delante da el numero de canales de entrada de cada nivel, corridos un puesto
        in_ch_mult = (1,) + ch_mult

        self.temb_dense = nn.ModuleList([
            nn.Linear(ch, temb_ch),
            nn.Linear(temb_ch, temb_ch),
        ])

        self.conv_in = nn.Conv1d(out_ch, ch, kernel_size=1)

        # encoder
        enc_block_levels: list[nn.ModuleList] = []
        enc_attn_levels: list[nn.ModuleList] = []
        downsampler_list: list[nn.Module] = []
        curr_res = resolution
        block_in = ch

        for i in range(n_levels):
            block_in = ch * in_ch_mult[i]
            block_out = ch * ch_mult[i]
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res):
                blocks.append(_ResBlock(block_in, block_out, temb_ch, dropout))
                # solo el primer bloque del nivel cambia de ancho, los demas ya van a block_out
                block_in = block_out
                # la atencion se pone solo en las resoluciones que pida la config
                attns.append(_AttnBlock(block_in) if curr_res in attn_res else nn.Identity())
            enc_block_levels.append(blocks)
            enc_attn_levels.append(attns)
            # el ultimo nivel no baja de resolucion, se le pone una identidad para indexar
            # igual
            if i < n_levels - 1:
                downsampler_list.append(nn.Conv1d(block_in, block_in, kernel_size=1))
                curr_res = curr_res // 2
            else:
                downsampler_list.append(nn.Identity())

        self.enc_blocks = nn.ModuleList(enc_block_levels)
        self.enc_attns = nn.ModuleList(enc_attn_levels)
        self.downsamplers = nn.ModuleList(downsampler_list)

        # middle
        self.mid_block1 = _ResBlock(block_in, block_in, temb_ch, dropout)
        self.mid_attn = _AttnBlock(block_in)
        self.mid_block2 = _ResBlock(block_in, block_in, temb_ch, dropout)

        # decoder, built in reverse order, stored forward for indexing by i
        dec_block_levels: list[nn.ModuleList] = [nn.ModuleList()] * n_levels
        upsampler_list: list[nn.Module] = [nn.Identity()] * n_levels

        for i in reversed(range(n_levels)):
            block_out = ch * ch_mult[i]
            blocks = nn.ModuleList()
            # num_res + 1 bloques, uno mas que en el encoder, para consumir tambien el skip de
            # entrada
            for j in range(num_res + 1):
                skip_in = ch * in_ch_mult[i] if j == num_res else ch * ch_mult[i]
                blocks.append(_ResBlock(block_in + skip_in, block_out, temb_ch, dropout))
                block_in = block_out
            dec_block_levels[i] = blocks
            if i > 0:
                upsampler_list[i] = nn.Conv1d(block_in, block_in, kernel_size=1)
                curr_res = curr_res * 2

        self.dec_blocks = nn.ModuleList(dec_block_levels)
        self.upsamplers = nn.ModuleList(upsampler_list)

        self.norm_out = _groupnorm(block_in)
        self.conv_out = nn.Conv1d(block_in, out_ch, kernel_size=1)


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """predict the noise added to x at timestep t. x is (batch, out_ch, seq_len), t is a
        (batch,) tensor of float or long timestep indices, the output has the shape of x."""

        temb = _sinusoidal_embedding(t.long(), self.ch)
        temb = _swish(self.temb_dense[0](temb))
        temb = self.temb_dense[1](temb)

        # hs es la pila de skips, se apila en el encoder y se desapila en el decoder por LIFO
        hs = [self.conv_in(x)]
        for i in range(self._n_levels):
            for j in range(self._num_res):
                h = self.enc_blocks[i][j](hs[-1], temb)
                h = self.enc_attns[i][j](h)
                hs.append(h)
            if i < self._n_levels - 1:
                hs.append(self.downsamplers[i](hs[-1]))

        h = self.mid_block1(hs[-1], temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for i in reversed(range(self._n_levels)):
            # cada bloque concatena por canal el skip que toca, y el orden LIFO es el contrato
            for j in range(self._num_res + 1):
                h = self.dec_blocks[i][j](torch.cat([h, hs.pop()], dim=1), temb)
            if i > 0:
                h = self.upsamplers[i](h)

        return self.conv_out(_swish(self.norm_out(h)))
