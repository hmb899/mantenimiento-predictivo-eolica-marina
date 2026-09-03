"""
scoring de anomalia para D3R, un escalar por ventana que alimenta ReplayScorer, CUSUM y CARE

variantes de la ablacion residuo frente a residuo+drift-anomalo
    recon         MSE de reconstruccion por ventana, media sobre W y K. es el score nativo de
                  D3R y mide rotura de la estructura estable, porque el drift se ha quitado y
                  re-sumado y por tanto no cuenta
    trend         anomalia del trend por ventana, Mahalanobis diagonal del vector trend (B, K)
                  contra la distribucion del train. hedge por si un fallo lento se absorbe en
                  el trend
    recon+trend   suma de los dos z-normalizados con stats del train (leakage-free)

`draws` promedia realizaciones de ruido de la difusion, lo que reduce la varianza del score y
sube la reliability, igual que el harness de TimeADDM
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def run_model_scores(model, data: np.ndarray, time: np.ndarray, device: str,
                     draws: int = 1, batch_size: int = 512) -> dict[str, np.ndarray]:
    """pasa las ventanas por el modelo (p=0) y devuelve {'recon': (N,), 'trend_vec': (N,K)}."""

    model.eval()
    recon_out, trend_out = [], []
    n = len(data)
    for s in range(0, n, batch_size):
        x = torch.as_tensor(data[s:s + batch_size], dtype=torch.float32, device=device)
        tf = torch.as_tensor(time[s:s + batch_size], dtype=torch.float32, device=device)
        acc = torch.zeros(x.size(0), device=device)
        trend_b = None
        # solo el recon se promedia sobre draws, el trend es determinista y basta con el
        # primero
        for _ in range(max(1, draws)):
            noise = torch.randn_like(x)
            _stable, trend, recon = model(x, tf, p=0.0, noise=noise)
            acc = acc + (x - recon).pow(2).mean(dim=(1, 2))
            if trend_b is None:
                # trend es constante en W, se toma un paso y queda (b, K)
                trend_b = trend[:, 0, :]
        recon_out.append((acc / max(1, draws)).cpu().numpy())
        trend_out.append(trend_b.cpu().numpy())
    return {"recon": np.concatenate(recon_out), "trend_vec": np.concatenate(trend_out, axis=0)}


def _z(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (x - mu) / (sd + 1e-12)


def assemble_scores(train_out: dict, event_outs: list[dict], variant: str
                    ) -> tuple[np.ndarray, list[np.ndarray]]:
    """combina las salidas crudas en el score por ventana segun la variante, con stats del
    train."""

    tr_recon = train_out["recon"]
    ev_recon = [e["recon"] for e in event_outs]
    if variant == "recon":
        return tr_recon, ev_recon

    # media y desviacion por componente del trend en train, la referencia de normalidad
    mu = train_out["trend_vec"].mean(axis=0)
    sd = train_out["trend_vec"].std(axis=0) + 1e-8

    def tanom(tv: np.ndarray) -> np.ndarray:
        # Mahalanobis diagonal, o sea z al cuadrado promediado sobre las K componentes
        return (((tv - mu) / sd) ** 2).mean(axis=1)

    tr_tan = tanom(train_out["trend_vec"])
    ev_tan = [tanom(e["trend_vec"]) for e in event_outs]
    if variant == "trend":
        return tr_tan, ev_tan
    if variant == "recon+trend":
        # los dos terminos se llevan a la misma escala antes de sumar, si no manda el mayor
        rm, rs = float(tr_recon.mean()), float(tr_recon.std())
        tm, ts = float(tr_tan.mean()), float(tr_tan.std())
        comb = lambda r, t: _z(r, rm, rs) + _z(t, tm, ts)
        return comb(tr_recon, tr_tan), [comb(er, et) for er, et in zip(ev_recon, ev_tan)]
    raise ValueError(f"variante de score desconocida: {variant}")
