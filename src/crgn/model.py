"""
nucleo CRGN (Yang, Zhang & Yu, Energies 2025), contrastive residual graph network

dos encoders con pesos compartidos, cada uno una pila de capas GCN residuales con LeakyReLU.
uno ve la ventana original y el otro una vista enmascarada con ruido, y un objetivo InfoNCE /
NT-Xent, donde los positivos son las dos vistas de una ventana y los negativos las demas
ventanas del lote, con temperatura y weight decay L2, lo entrena de forma auto-supervisada
solo con datos normales. sin pronostico y sin etiquetas de fallo

el embedding por ventana, un readout mean o max sobre los nodos tomado antes de la cabeza de
proyeccion, es lo que el scorer compara con la distribucion de validacion-normal. la cabeza de
proyeccion solo da forma al espacio contrastivo, el convenio de SimCLR

PyTorch a secas con una adyacencia densa (K, K) (K <= 100), la misma pila que GDN, sin PyTorch
Geometric
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualGCN(nn.Module):
    """`n_layers` capas GCN sobre una adyacencia fija normalizada, LeakyReLU, y un residuo en
    cuanto las dimensiones casan, o sea en toda capa a partir de la segunda. features de nodo
    (B, N, in_dim) a (B, N, hidden)."""

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, slope: float = 0.2) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.slope = slope
        # la primera capa cambia de dimension, de la segunda en adelante ya es hidden a hidden
        dims = [in_dim] + [hidden_dim] * n_layers
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers))


    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for lin in self.lins:
            # la j se suma, o sea cada nodo i recoge la media ponderada de sus vecinos
            z = torch.einsum("ij,bjf->bif", adj, lin(h))  # Â (H W), agrega sobre los vecinos
            z = F.leaky_relu(z, self.slope)
            h = z + h if z.shape == h.shape else z  # residuo cuando las dimensiones casan
        return h


class CRGN(nn.Module):
    """encoder GCN residual siames con cabeza InfoNCE. `encode` da el embedding de puntuacion
    y `contrastive_loss` el objetivo de entrenamiento, y la adyacencia se inyecta, construida
    por granja."""

    def __init__(self, n_nodes: int, in_dim: int, embed_dim: int = 64, n_layers: int = 2,
                 slope: float = 0.2, readout: str = "mean", proj_dim: int | None = None,
                 temperature: float = 0.5, noise_std: float = 0.1, mask_ratio: float = 0.2) -> None:
        super().__init__()
        if readout not in ("mean", "max"):
            raise ValueError(f"readout must be mean|max, got {readout}")
        self.n_nodes = n_nodes
        self.readout = readout
        self.temperature = temperature
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.encoder = ResidualGCN(in_dim, embed_dim, n_layers, slope)
        # la cabeza de proyeccion solo se usa en el loss, el score sale del embedding de antes
        proj_dim = proj_dim or embed_dim
        self.proj = nn.Sequential(nn.Linear(embed_dim, proj_dim), nn.LeakyReLU(slope),
                                  nn.Linear(proj_dim, proj_dim))


    def _readout(self, h: torch.Tensor) -> torch.Tensor:  # (B, N, d) a (B, d)
        return h.mean(dim=1) if self.readout == "mean" else h.max(dim=1).values


    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """de las features de nodo de la ventana (B, N, in_dim) al embedding (B, embed_dim)."""

        return self._readout(self.encoder(x, adj))


    def project(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # el espacio contrastivo, solo para entrenar
        return self.proj(self.encode(x, adj))


    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """enmascarado con ruido gaussiano (del articulo), una fraccion aleatoria de las
        entradas (nodo, tiempo) recibe ruido."""

        # la mascara se sortea entrada a entrada, no por nodo ni por paso completo
        mask = (torch.rand_like(x) < self.mask_ratio).to(x.dtype)
        return x + mask * torch.randn_like(x) * self.noise_std


    def contrastive_loss(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # las dos vistas de la misma ventana, la limpia y la enmascarada, son el par positivo
        z1 = F.normalize(self.project(x, adj), dim=1)
        z2 = F.normalize(self.project(self.augment(x), adj), dim=1)
        return nt_xent_loss(z1, z2, self.temperature)


    # el forward por defecto puntua
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.encode(x, adj)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """entropia cruzada normalizada y escalada por temperatura (SimCLR / InfoNCE). z1 y z2 son
    vistas (B, d) normalizadas en L2, el positivo de la fila i es su pareja en la otra vista y
    los negativos son todas las demas filas."""

    b = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, d)
    # con z ya normalizado el producto escalar es el coseno, y la temperatura afila el softmax
    sim = z @ z.t() / temperature  # (2B, 2B)
    # el -inf en la diagonal saca del softmax la similitud de cada fila consigo misma
    sim = sim + torch.diag(torch.full((2 * b,), float("-inf"), device=z.device))
    targets = torch.cat([torch.arange(b, device=z.device) + b,
                         torch.arange(b, device=z.device)], dim=0)  # la fila i va con la i+B
    return F.cross_entropy(sim, targets)
