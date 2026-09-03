"""
GraphDiff, encoder de grafo (CRGN), latente normal-gathering (VAE), difusion latente sobre
el eje de nodos y decoder

reuso de solo lectura de src/crgn (ResidualGCN, CRGN) y de src/timeaddm (DiffusionUNet, DDPM
utils). el score de anomalia lo calcula src/graphdiff/scorer.py.

piezas y por que
    backbone CRGN (ResidualGCN), da embeddings por nodo H (B, K, embed) sobre el grafo Spearman
    fijo. es el encoder que ya produce la senal ortogonal y fuerte del proyecto, la fusion
    AE+CRGN gana en C. se usa .encoder, por nodo y antes del readout, y se reutiliza
    .contrastive_loss como regularizador
    GatheringHead, un VAE por nodo con KL a N(0,I), el normal gathering latent space de
    NGLS-Diff en version simple y robusta; su ELBO por patches esta pensado para T=100, y aqui
    W=27 y el nodo es el sensor. compacta la clase normal, la difusion la modela mejor y el eps
    separa
    DiffusionUNet 1D (TimeADDM) sobre z permutado a (B, d_lat, K), que difunde sobre el eje de
    nodos; su self-attention mezcla nodos, o sea grafo blando, y el grafo duro entra por el
    encoder via la adyacencia Â
    NodeDecoder, de z a x_hat (B, K, W) para el score recon; es un hedge, los papers puntuan
    asi aunque aqui sea redundante
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.crgn.model import CRGN
from src.timeaddm.diffusion_model import DiffusionUNet
from src.timeaddm.diffusion_utils import get_beta_schedule, noise_estimation_loss


def _diff_config(ch: int, out_ch: int, ch_mult, num_res: int, dropout: float,
                 attn: bool, image_size: int) -> SimpleNamespace:
    """namespace que consume DiffusionUNet.__init__, con los mismos campos que
    run_timeaddm._make_diffusion_config. attn=True activa self-attention al nivel de resolucion
    K, que es lo que mezcla los nodos y hace de grafo blando."""

    # la resolucion de la U-Net es el numero de nodos, asi que pedir atencion ahi es pedirla
    # toda
    attn_res = [image_size] if attn else []
    return SimpleNamespace(
        model=SimpleNamespace(ch=ch, out_ch=out_ch, ch_mult=list(ch_mult), num_res_blocks=num_res,
                              dropout=dropout, attn_resolutions=attn_res),
        data=SimpleNamespace(image_size=image_size))


class GatheringHead(nn.Module):
    """VAE por nodo, de H (B,K,embed) a z (B,K,d_lat), con KL(N(mu,sigma)||N(0,I)) que reune lo
    normal."""

    def __init__(self, embed_dim: int, d_lat: int) -> None:
        super().__init__()
        self.mu = nn.Linear(embed_dim, d_lat)
        self.logvar = nn.Linear(embed_dim, d_lat)


    def forward(self, h: torch.Tensor, sample: bool = True):
        mu = self.mu(h)
        # el clamp del logvar evita que exp() se vaya a infinito o a cero en el truco de
        # reparam
        logvar = self.logvar(h).clamp(-8.0, 8.0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        # KL media por elemento (nodos x canales x batch), comparable a recon y diff, con un
        # peso interpretable
        kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()
        return z, mu, logvar, kl


class NodeDecoder(nn.Module):
    """de z (B,K,d_lat) a x_hat (B,K,W), reconstruye la ventana de cada nodo con un MLP
    compartido por nodo."""

    def __init__(self, d_lat: int, w: int, hidden: int = 64, slope: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_lat, hidden), nn.LeakyReLU(slope), nn.Linear(hidden, w))


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GraphDiff(nn.Module):
    """encoder CRGN, gathering VAE, difusion latente y decoder. el entrenamiento y el scoring
    van fuera."""

    def __init__(self, n_nodes: int, w: int, embed_dim: int = 64, gcn_layers: int = 2, d_lat: int = 64,
                 slope: float = 0.2, readout: str = "mean", proj_dim: int | None = 64,
                 temperature: float = 0.5, noise_std: float = 0.1, mask_ratio: float = 0.2,
                 diff_ch: int = 64, diff_ch_mult=(1, 2), diff_num_res: int = 2, diff_dropout: float = 0.1,
                 diff_attn: bool = True, n_steps: int = 500, beta_start: float = 1e-4,
                 beta_end: float = 0.02, beta_schedule: str = "linear") -> None:
        super().__init__()
        # el latente es el canal de la U-Net, y su GroupNorm usa 32 grupos fijos
        if d_lat % 32 != 0:
            raise ValueError(f"d_lat debe ser multiplo de 32 (GroupNorm de la U-Net), got {d_lat}")
        self.n_nodes = n_nodes
        self.w = w
        self.d_lat = d_lat
        self.n_steps = n_steps
        # el CRGN entra entero, aunque de el solo se usen .encoder y .contrastive_loss
        self.backbone = CRGN(n_nodes=n_nodes, in_dim=w, embed_dim=embed_dim, n_layers=gcn_layers,
                             slope=slope, readout=readout, proj_dim=proj_dim, temperature=temperature,
                             noise_std=noise_std, mask_ratio=mask_ratio)
        self.gather = GatheringHead(embed_dim, d_lat)
        self.decoder = NodeDecoder(d_lat, w, hidden=embed_dim, slope=slope)
        # image_size = n_nodes, o sea que el eje espacial de la U-Net es el de sensores
        self.diffusion = DiffusionUNet(
            _diff_config(diff_ch, d_lat, diff_ch_mult, diff_num_res, diff_dropout, diff_attn, n_nodes))
        betas = get_beta_schedule(beta_schedule, beta_start, beta_end, n_steps)
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))


    # piezas reutilizables por el trainer y el scorer


    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """de x node-major (B,K,W) a los embeddings por nodo H (B,K,embed)."""

        return self.backbone.encoder(x, adj)


    def latent(self, x: torch.Tensor, adj: torch.Tensor, sample: bool = True):
        return self.gather(self.encode(x, adj), sample=sample)


    def diffusion_loss(self, z: torch.Tensor) -> torch.Tensor:
        """perdida DDPM de prediccion de ruido sobre el latente z (B,K,d_lat)."""

        zt = z.permute(0, 2, 1)  # (B, d_lat, K)
        # un paso de ruido distinto por elemento del lote, que es el muestreo estandar de DDPM
        t = torch.randint(0, self.n_steps, (zt.size(0),), device=z.device)
        e = torch.randn_like(zt)
        # noise_estimation_loss suma sobre (canales, nodos), aqui se normaliza a media por
        # elemento
        _, loss = noise_estimation_loss(self.diffusion, zt, t, e, self.betas)
        return loss / (self.d_lat * self.n_nodes)


    def contrastive_loss(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.backbone.contrastive_loss(x, adj)


    def training_losses(self, x: torch.Tensor, adj: torch.Tensor, w_gather: float, w_recon: float,
                        w_contrast: float, with_contrast: bool, pairs=None, w_phys: float = 0.0) -> dict:
        """perdida conjunta. with_contrast=False es el modo encoder-congelado, con el backbone
        ya entrenado, y pairs con w_phys>0 activa la coherencia fisica, que reune los latentes
        de los sensores redundantes."""

        from src.graphdiff.physics import coherence_loss

        h = self.encode(x, adj)
        z, _mu, _logvar, kl = self.gather(h, sample=True)
        # cuatro terminos, todos ya en media por elemento, y los pesos deciden el reparto
        l_diff = self.diffusion_loss(z)
        l_recon = F.mse_loss(self.decoder(z), x)
        l_contrast = self.contrastive_loss(x, adj) if with_contrast else torch.zeros((), device=x.device)
        l_phys = (coherence_loss(z, pairs, str(x.device)) if pairs and w_phys > 0
                  else torch.zeros((), device=x.device))
        total = l_diff + w_gather * kl + w_recon * l_recon + w_contrast * l_contrast + w_phys * l_phys
        # solo el total lleva gradiente, los demas van detached porque son para el log
        return {"total": total, "diff": l_diff.detach(), "kl": kl.detach(),
                "recon": l_recon.detach(), "contrast": l_contrast.detach(), "phys": l_phys.detach()}


def build_graphdiff(n_nodes: int, w: int, cfg: dict, device: str) -> GraphDiff:
    """construye GraphDiff desde un dict de hiperparametros y lo mueve a device."""

    m = GraphDiff(
        n_nodes=n_nodes, w=w, embed_dim=cfg["embed_dim"], gcn_layers=cfg["gcn_layers"],
        d_lat=cfg["d_lat"], readout=cfg.get("readout", "mean"), proj_dim=cfg.get("proj_dim", 64),
        temperature=cfg.get("temperature", 0.5), noise_std=cfg.get("noise_std", 0.1),
        mask_ratio=cfg.get("mask_ratio", 0.2), diff_ch=cfg.get("diff_ch", 64),
        diff_ch_mult=cfg.get("diff_ch_mult", (1, 2)), diff_num_res=cfg.get("diff_num_res", 2),
        diff_dropout=cfg.get("diff_dropout", 0.1), diff_attn=cfg.get("diff_attn", True),
        n_steps=cfg.get("n_steps", 500), beta_start=cfg.get("beta_start", 1e-4),
        beta_end=cfg.get("beta_end", 0.02), beta_schedule=cfg.get("beta_schedule", "linear"))
    return m.to(device)
