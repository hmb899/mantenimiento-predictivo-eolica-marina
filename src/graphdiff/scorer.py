"""
scoring de anomalia para GraphDiff, un escalar por ventana para ReplayScorer, CUSUM y CARE

variantes de la ablacion entre score recon, eps y recon+eps
    eps         score nativo de difusion ||e - eps_theta(z_t, L)||^2 en el latente, sin
                decodificar. es el candidato ortogonal, porque al no pasar por el decoder no
                re-deriva el AE. es el primario
    recon       MSE(decode(mu), x) por ventana, la reconstruccion del grafo-VAE. hedge, porque
                los papers puntuan asi
    eps+recon   suma z-normalizada con stats del train (leakage-free)

`draws` promedia realizaciones de ruido, lo que reduce la varianza del score y sube la
reliability, igual que en TimeADDM y D3R
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def run_scores(model, x_nm: np.ndarray, adj: np.ndarray, device: str, levels: list[int],
               draws: int = 1, batch_size: int = 512) -> dict[str, np.ndarray]:
    """x_nm node-major (N,K,W) da {'eps{L}': (N,) por nivel, 'recon': (N,)}, con el latente
    determinista = mu."""

    model.eval()
    betas = model.betas
    # alpha_bar por paso, se calcula una vez y se indexa por nivel dentro del bucle
    acum = (1.0 - betas).cumprod(dim=0)
    adj_t = torch.as_tensor(adj, dtype=torch.float32, device=device)
    out_eps: dict[int, list] = {L: [] for L in levels}
    out_recon: list = []
    n = len(x_nm)
    for s in range(0, n, batch_size):
        x = torch.as_tensor(x_nm[s:s + batch_size], dtype=torch.float32, device=device)
        # sample=False fija el latente en la media, para que el score no dependa del sorteo del
        # VAE
        _z, mu, _lv, _kl = model.gather(model.encode(x, adj_t), sample=False)
        z = mu.permute(0, 2, 1)  # (B, d_lat, K), la difusion va sobre el eje de nodos
        x_hat = model.decoder(mu)
        out_recon.append((x - x_hat).pow(2).mean(dim=(1, 2)).cpu().numpy())
        for L in levels:
            # el nivel L se cuenta desde uno y el tensor de pasos desde cero, de ahi el L-1
            a = acum[L - 1].view(1, 1, 1)
            t_vec = torch.full((x.size(0),), L - 1, dtype=torch.long, device=device)
            acc = torch.zeros(x.size(0), device=device)
            for _ in range(max(1, draws)):
                # se ruidea el latente y se mide cuanto se equivoca la red al estimar ese ruido
                e = torch.randn_like(z)
                z_t = z * a.sqrt() + e * (1.0 - a).sqrt()
                pred = model.diffusion(z_t, t_vec.float())
                acc = acc + (e - pred).square().sum(dim=(1, 2))
            out_eps[L].append((acc / max(1, draws)).cpu().numpy())
    res: dict[str, np.ndarray] = {f"eps{L}": np.concatenate(v) for L, v in out_eps.items()}
    res["recon"] = np.concatenate(out_recon)
    return res


def _z(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (x - mu) / (sd + 1e-12)


def assemble(train_raw: dict, event_raws: list[dict], variant: str, level: int
             ) -> tuple[np.ndarray, list[np.ndarray]]:
    """combina las salidas crudas en el score por ventana, con stats del train
    (leakage-free)."""

    # el nivel elige cual de los eps precomputados se usa, los demas se quedan sin tocar
    key = f"eps{level}"
    tr_eps, ev_eps = train_raw[key], [e[key] for e in event_raws]
    tr_rec, ev_rec = train_raw["recon"], [e["recon"] for e in event_raws]
    if variant == "eps":
        return tr_eps, ev_eps
    if variant == "recon":
        return tr_rec, ev_rec
    if variant == "eps+recon":
        # los dos terminos se llevan a la misma escala antes de sumar, si no manda el mayor
        em, es = float(tr_eps.mean()), float(tr_eps.std())
        rm, rs = float(tr_rec.mean()), float(tr_rec.std())
        comb = lambda ep, rc: _z(ep, em, es) + _z(rc, rm, rs)
        return comb(tr_eps, tr_rec), [comb(ep, rc) for ep, rc in zip(ev_eps, ev_rec)]
    raise ValueError(f"variante de score desconocida: {variant}")
