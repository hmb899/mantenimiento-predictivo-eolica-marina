"""
D3R (Dynamic Decomposition with Diffusion Reconstruction), Modelo A del Track B

puerto fiel de ForestsKing/D3R (NeurIPS 2023, "Drift doesn't Matter") adaptado a nuestro
pipeline. la entrada son ventanas de componentes PCA (B, W, K) con K canales, no sensores
crudos, y el condicionamiento temporal va por features de timestamp (B, W, time_num). se
consolidan en un solo fichero los modulos del repo original, embedding, attention, block,
subtraction, decomposition, diffusion, reconstruction y dddr.

por que podria batir o complementar al AE. la descomposicion separa la tendencia lenta, el
drift, del residuo estacionario; la difusion ruidea hacia el trend y no hacia 0, y la
reconstruccion resta el trend antes de denoise y lo re-suma. asi el error de reconstruccion se
concentra en la estructura estable, o sea en la rotura de la relacion normal entre
componentes, que es el fallo, y no en el drift operativo normal, que el AE reconstruye
trivialmente y por eso avisa tarde o da falsas alarmas por regimen.

adaptaciones respecto al repo original
    model_dim y ff_dim por defecto mas pequenos (transformer sobre W=27, K=55 y N grande) que
    los 512/2048 del paper, pensados para datasets chicos SMD/SWaT. parametrizable
    OffsetSubtraction, el shift `d` se acota a < W (el paper usa d=30 con W=64, con W=27 se
    clampa)
    forward() acepta un `noise` externo, para promediar draws en scoring de forma determinista

las formas y las formulas salen de los .py originales del repo ForestsKing/D3R citado arriba
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# embeddings
class PositionEmbedding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        # tabla sinusoidal precomputada hasta max_len, luego solo se recorta a la longitud real
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # frecuencias en progresion geometrica entre 1 y 1/10000, la escala de Vaswani et al.
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.norm = nn.LayerNorm(model_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.pe[:, : x.size(1), :])


class TimeEmbedding(nn.Module):
    """Conv1d sobre las features de timestamp, de time_num a model_dim."""


    def __init__(self, model_dim: int, time_num: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(time_num, model_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.norm = nn.LayerNorm(model_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d quiere el canal en medio, de ahi el permute de ida y el de vuelta
        return self.norm(self.conv(x.permute(0, 2, 1)).permute(0, 2, 1))


class DataEmbedding(nn.Module):
    """Conv1d sobre los canales de datos, de feature_num=K a model_dim."""


    def __init__(self, model_dim: int, feature_num: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(feature_num, model_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)


# attention
class OrdAttention(nn.Module):
    """self-attention estandar multi-cabeza, con o sin residual."""


    def __init__(self, model_dim: int, atten_dim: int, head_num: int, dropout: float, residual: bool) -> None:
        super().__init__()
        self.atten_dim, self.head_num, self.residual = atten_dim, head_num, residual
        self.W_Q = nn.Linear(model_dim, atten_dim * head_num)
        self.W_K = nn.Linear(model_dim, atten_dim * head_num)
        self.W_V = nn.Linear(model_dim, atten_dim * head_num)
        self.fc = nn.Linear(atten_dim * head_num, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)


    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        residual = Q
        # (B, L, heads*d) se abre a (B, heads, L, d), para que cada cabeza atienda por separado
        Q = self.W_Q(Q).view(Q.size(0), Q.size(1), self.head_num, self.atten_dim).transpose(1, 2)
        K = self.W_K(K).view(K.size(0), K.size(1), self.head_num, self.atten_dim).transpose(1, 2)
        V = self.W_V(V).view(V.size(0), V.size(1), self.head_num, self.atten_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        context = torch.matmul(torch.softmax(scores, dim=-1), V)
        # se vuelven a pegar las cabezas antes de la proyeccion de salida
        context = context.transpose(1, 2).reshape(residual.size(0), residual.size(1), -1)
        out = self.dropout(self.fc(context))
        return self.norm(out + residual) if self.residual else self.norm(out)


class MixAttention(nn.Module):
    """data-time mix-attention, los pesos salen de softmax(scores_data + scores_time) y los
    valores de la rama de tiempo. es el mecanismo que extrae la parte estable, guiada a la vez
    por el dato y por el timestamp."""


    def __init__(self, model_dim: int, atten_dim: int, head_num: int, dropout: float, residual: bool) -> None:
        super().__init__()
        self.atten_dim, self.head_num, self.residual = atten_dim, head_num, residual
        self.W_Q_data = nn.Linear(model_dim, atten_dim * head_num)
        self.W_Q_time = nn.Linear(model_dim, atten_dim * head_num)
        self.W_K_data = nn.Linear(model_dim, atten_dim * head_num)
        self.W_K_time = nn.Linear(model_dim, atten_dim * head_num)
        self.W_V_time = nn.Linear(model_dim, atten_dim * head_num)
        self.fc = nn.Linear(atten_dim * head_num, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)


    def forward(self, Q_data, Q_time, K_data, K_time, V_time) -> torch.Tensor:
        residual = Q_data
        b = Q_data.size(0)

        def split(lin, x):
            return lin(x).view(b, x.size(1), self.head_num, self.atten_dim).transpose(1, 2)

        Q_data, Q_time = split(self.W_Q_data, Q_data), split(self.W_Q_time, Q_time)
        K_data, K_time = split(self.W_K_data, K_data), split(self.W_K_time, K_time)
        V_time = split(self.W_V_time, V_time)
        scores_data = torch.matmul(Q_data, K_data.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        scores_time = torch.matmul(Q_time, K_time.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        # las dos ramas suman antes del softmax, que es lo que las mezcla de verdad
        attn = torch.softmax(scores_data + scores_time, dim=-1)
        context = torch.matmul(attn, V_time).transpose(1, 2).reshape(b, residual.size(1), -1)
        out = self.dropout(self.fc(context))
        return self.norm(out + residual) if self.residual else self.norm(out)


# transformer blocks
class TemporalTransformerBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, head_num, dropout) -> None:
        super().__init__()
        self.attention = OrdAttention(model_dim, atten_dim, head_num, dropout, True)
        self.conv1 = nn.Conv1d(model_dim, ff_dim, 1)
        self.conv2 = nn.Conv1d(ff_dim, model_dim, 1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x, x, x)
        residual = x
        x = F.gelu(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        return self.norm(x + residual)


class SpatialTransformerBlock(nn.Module):
    """atencion sobre el eje de canales, o sea espacial, con model_dim de la atencion igual a
    window_size."""


    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout) -> None:
        super().__init__()
        self.attention = OrdAttention(window_size, atten_dim, head_num, dropout, True)
        self.conv1 = nn.Conv1d(model_dim, ff_dim, 1)
        self.conv2 = nn.Conv1d(ff_dim, model_dim, 1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # el permute pone el canal como eje de secuencia, para que la atencion cruce canales
        x = x.permute(0, 2, 1)
        x = self.attention(x, x, x)
        x = x.permute(0, 2, 1)
        residual = x
        x = F.gelu(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        return self.norm(x + residual)


class SpatialTemporalTransformerBlock(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout) -> None:
        super().__init__()
        self.time_block = TemporalTransformerBlock(model_dim, ff_dim, atten_dim, head_num, dropout)
        self.feature_block = SpatialTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dropout)
        self.conv1 = nn.Conv1d(2 * model_dim, ff_dim, 1)
        self.conv2 = nn.Conv1d(ff_dim, model_dim, 1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(2 * model_dim)
        self.norm2 = nn.LayerNorm(model_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(torch.cat([self.time_block(x), self.feature_block(x)], dim=-1))
        x = F.gelu(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        return self.norm2(x)


class DecompositionBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, feature_num, head_num, dropout) -> None:
        super().__init__()
        self.mixed_attention = MixAttention(model_dim, atten_dim, head_num, dropout, False)
        self.ordinary_attention = OrdAttention(model_dim, atten_dim, head_num, dropout, True)
        self.conv1 = nn.Conv1d(model_dim, ff_dim, 1)
        self.conv2 = nn.Conv1d(ff_dim, model_dim, 1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, feature_num)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)


    def forward(self, trend: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stable = self.mixed_attention(trend, time, trend, time, time)
        stable = self.ordinary_attention(stable, stable, stable)
        residual = stable
        stable = F.gelu(self.conv1(stable.permute(0, 2, 1)))
        stable = self.dropout(self.conv2(stable).permute(0, 2, 1))
        stable = self.norm1(stable + residual)
        trend = self.norm2(trend - stable)  # lo no-estable pasa al siguiente bloque
        stable = self.fc2(F.gelu(self.fc1(stable)))  # proyeccion a feature_num
        return stable, trend


# offset subtraction
class OffsetSubtraction(nn.Module):
    """resta robusta a drift horizontal, que para cada posicion elige, en un rango +-d, la
    diferencia (residual - stable) de menor valor absoluto. `d` se acota a < window_size."""


    def __init__(self, window_size: int, feature_num: int, d: int) -> None:
        super().__init__()
        d = max(1, min(d, window_size - 1))
        self.d = d
        # el +window_size deja sitio al relleno de los bordes que anade forward, asi el indice
        # nunca se sale por la izquierda ni por la derecha
        init_index = (torch.arange(window_size) + window_size).unsqueeze(-1).unsqueeze(-1)
        init_index = init_index.repeat(1, feature_num, 2 * d + 1)
        # desplazamientos 0, +1..+d, -1..-d; el 0 va primero para que empate a favor del centro
        delay = torch.tensor([0] + list(range(1, d + 1)) + [-i for i in range(1, d + 1)]).int()
        delay = delay.unsqueeze(0).unsqueeze(0).repeat(window_size, feature_num, 1)
        self.register_buffer("index", init_index + delay)


    def forward(self, subed: torch.Tensor, sub: torch.Tensor) -> torch.Tensor:
        b = subed.shape[0]
        index = self.index.unsqueeze(0).repeat(b, 1, 1, 1).to(sub.device)
        # bordes replicados a los dos lados, para que un desplazamiento de +-d siempre caiga
        # dentro
        front = sub[:, 0:1, :].repeat(1, sub.shape[1], 1)
        end = sub[:, -1:, :].repeat(1, sub.shape[1], 1)
        sub = torch.cat([front, sub, end], dim=1)
        sub = torch.gather(sub.unsqueeze(-1).repeat(1, 1, 1, 2 * self.d + 1), dim=1, index=index)
        # de los 2d+1 desplazamientos se queda el de |diferencia| minima, uno por posicion y
        # canal
        res = subed.unsqueeze(-1).repeat(1, 1, 1, 2 * self.d + 1) - sub
        res = torch.gather(res, dim=-1, index=torch.argmin(torch.abs(res), dim=-1, keepdim=True))
        return res.reshape(subed.shape)


# decomposition, diffusion y reconstruction
class DataEncoder(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, block_num, head_num, dropout) -> None:
        super().__init__()
        self.data_embedding = DataEmbedding(model_dim, feature_num)
        self.position_embedding = PositionEmbedding(model_dim)
        # el ultimo bloque va sin dropout, para que la salida del encoder sea determinista
        self.encoder_blocks = nn.ModuleList(
            [SpatialTemporalTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num,
                                             0 if i == block_num - 1 else dropout) for i in range(block_num)]
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.data_embedding(x) + self.position_embedding(x)
        for block in self.encoder_blocks:
            x = block(x)
        return x


class TimeEncoder(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, time_num, block_num, head_num, dropout) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(model_dim, time_num)
        self.encoder_blocks = nn.ModuleList(
            [TemporalTransformerBlock(model_dim, ff_dim, atten_dim, head_num,
                                      0 if i == block_num - 1 else dropout) for i in range(block_num)]
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_embed(x)
        for block in self.encoder_blocks:
            x = block(x)
        return x


class DynamicDecomposition(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, time_num, block_num,
                 head_num, dropout, d) -> None:
        super().__init__()
        self.data_encoder = DataEncoder(window_size, model_dim, ff_dim, atten_dim, feature_num, block_num,
                                        head_num, dropout)
        self.time_encoder = TimeEncoder(model_dim, ff_dim, atten_dim, time_num, block_num, head_num, dropout)
        self.decomposition_blocks = nn.ModuleList(
            [DecompositionBlock(model_dim, ff_dim, atten_dim, feature_num, head_num,
                                0 if i == block_num - 1 else dropout) for i in range(block_num)]
        )
        self.minus = OffsetSubtraction(window_size, feature_num, d)


    def forward(self, data: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = data.clone()
        data = self.data_encoder(data)
        time = self.time_encoder(time)
        # cada bloque extrae un trozo de lo estable y devuelve el resto para el siguiente
        stable = torch.zeros_like(residual)
        for block in self.decomposition_blocks:
            tmp_stable, data = block(data, time)
            stable = stable + tmp_stable
        # el trend es un nivel constante en la ventana, de ahi la media y el repeat
        trend = torch.mean(self.minus(residual, stable), dim=1).unsqueeze(1).repeat(1, data.shape[1], 1)
        return stable, trend


class Diffusion:
    """difusion directa (q_sample) con la media desplazada hacia el trend, y paso t fijo desde
    fuera. no es un DDPM iterativo, la reconstruccion es un denoise de un solo paso."""


    def __init__(self, time_steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu") -> None:
        betas = torch.linspace(beta_start, beta_end, time_steps).float().to(device)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.one_minus_sqrt_alphas_cumprod = 1.0 - torch.sqrt(alphas_cumprod)


    @staticmethod
    def _extract(data, batch_t, shape):
        out = torch.gather(data, -1, batch_t)
        return out.reshape(batch_t.shape[0], *((1,) * (len(shape) - 1)))

    # el tercer termino es lo que separa esto de un DDPM, la media se va hacia el trend y no a
    # 0


    def q_sample(self, x_start, trend, batch_t, noise) -> torch.Tensor:
        sac = self._extract(self.sqrt_alphas_cumprod, batch_t, x_start.shape)
        somac = self._extract(self.sqrt_one_minus_alphas_cumprod, batch_t, x_start.shape)
        omsac = self._extract(self.one_minus_sqrt_alphas_cumprod, batch_t, x_start.shape)
        return sac * x_start + somac * noise + omsac * trend


class Reconstruction(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, time_num, block_num,
                 head_num, dropout) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(model_dim, time_num)
        self.data_embedding = DataEmbedding(model_dim, feature_num)
        self.position_embedding = PositionEmbedding(model_dim)
        self.decoder_blocks = nn.ModuleList(
            [SpatialTemporalTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num,
                                             0 if i == block_num - 1 else dropout) for i in range(block_num)]
        )
        self.fc1 = nn.Linear(model_dim, feature_num)


    def forward(self, noise: torch.Tensor, trend: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # se resta el trend antes de denoise y se re-suma al final, para que el error quede en
        # lo estable y no en el drift
        trend = self.data_embedding(trend)
        x = self.data_embedding(noise) - trend
        x = x + self.position_embedding(noise) + self.time_embed(time)
        for block in self.decoder_blocks:
            x = block(x)
        return self.fc1(x + trend)


# modelo de nivel superior
class DDDR(nn.Module):
    """modelo D3R completo, forward() devuelve (stable, trend, recon), los tres (B, W, K).

    `decompose=False` desactiva la descomposicion, con trend=0, y entonces la difusion y la
    reconstruccion operan sobre la senal cruda. es la ablacion que reduce D3R a un denoiser de
    difusion tipo TimeADDM-raw, que sirve de referencia.
    """


    def __init__(self, *, window_size, feature_num, time_num, model_dim=128, ff_dim=256, atten_dim=32,
                 block_num=2, head_num=4, dropout=0.1, d=13, t=500, time_steps=1000,
                 beta_start=1e-4, beta_end=0.02, decompose=True, device="cpu") -> None:
        super().__init__()
        self.window_size = window_size
        self.feature_num = feature_num
        self.t = t
        self.decompose = decompose
        self.device = device
        if decompose:
            self.dynamic_decomposition = DynamicDecomposition(
                window_size, model_dim, ff_dim, atten_dim, feature_num, time_num, block_num, head_num, dropout, d)
        self.diffusion = Diffusion(time_steps, beta_start, beta_end, device)
        self.reconstruction = Reconstruction(
            window_size, model_dim, ff_dim, atten_dim, feature_num, time_num, block_num, head_num, dropout)


    def to(self, *args, **kwargs):
        # Diffusion no es un nn.Module, asi que sus tensores no viajan solos y hay que moverlos
        obj = super().to(*args, **kwargs)
        dev = obj.reconstruction.fc1.weight.device
        for name in ("sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod", "one_minus_sqrt_alphas_cumprod"):
            setattr(obj.diffusion, name, getattr(obj.diffusion, name).to(dev))
        obj.device = dev
        return obj


    def forward(self, data: torch.Tensor, time: torch.Tensor, p: float = 0.0,
                noise: torch.Tensor | None = None):
        # aumento por perturbacion, un offset vertical constante por (batch, canal) ~ U(0,p) y
        # replicado en la ventana, que fuerza al trend a absorber shifts de nivel, o sea le da
        # robustez al drift
        if p > 0:
            disturb = (torch.rand(data.shape[0], data.shape[2], device=data.device) * p)
            disturb = disturb.unsqueeze(1).repeat(1, self.window_size, 1)
        else:
            disturb = torch.zeros(data.shape[0], self.window_size, data.shape[2], device=data.device)
        data = data + disturb

        if self.decompose:
            stable, trend = self.dynamic_decomposition(data, time)
        else:
            stable = torch.zeros_like(data)
            trend = torch.zeros_like(data)

        # el paso de ruido es el mismo para todo el lote, no se sortea
        bt = torch.full((data.shape[0],), self.t, device=data.device)
        if noise is None:
            noise = torch.randn_like(data)
        noise_data = self.diffusion.q_sample(data, trend, bt, noise)
        recon = self.reconstruction(noise_data, trend, time)
        # se descuenta la perturbacion de la salida, para que el score no la vea
        return stable, trend - disturb, recon - disturb


def build_dddr(feature_num: int, window_size: int, time_num: int, cfg: dict, device: str) -> DDDR:
    """construye el DDDR desde un dict de config, el subarbol 'driftdiff'."""

    m = DDDR(
        window_size=window_size, feature_num=feature_num, time_num=time_num,
        model_dim=cfg.get("model_dim", 128), ff_dim=cfg.get("ff_dim", 256),
        atten_dim=cfg.get("atten_dim", 32), block_num=cfg.get("block_num", 2),
        head_num=cfg.get("head_num", 4), dropout=cfg.get("dropout", 0.1),
        d=cfg.get("d", max(1, window_size // 2)), t=cfg.get("t", 500),
        time_steps=cfg.get("time_steps", 1000), beta_start=cfg.get("beta_start", 1e-4),
        beta_end=cfg.get("beta_end", 0.02), decompose=cfg.get("decompose", True), device=device)
    return m.to(device)
