"""
fisica blanda relacional para GraphDiff, coherencia de sensores redundantes

los datos CARE estan anonimizados y escalados, asi que no hay fisica exacta (kW, m/s) pero si
relacional, que sobrevive al escalado. dos sensores fisicamente redundantes, medidas
duplicadas del mismo punto, estan casi perfectamente correlados en operacion normal
(|Spearman|~0.99), y un fallo incipiente en el sensor o en el componente que miden rompe esa
coherencia. eso da una
senal de anomalia ortogonal por construccion, que no es recon ni difusion, y con sentido de
dominio. los pares se descubren data-driven del Spearman train-normal, no de IDs memorizados.

coherence_score
    score de anomalia por ventana, model-free, la violacion de coherencia, para fusion y eval
coherence_loss
    regularizador de entrenamiento que reune los latentes de pares redundantes, y asi shapea la
    variedad normal para que el eps de la difusion detecte la rotura de coherencia
"""
from __future__ import annotations

import numpy as np
import torch


def find_redundant_pairs(train_rows: np.ndarray, thresh: float = 0.95) -> list[tuple[int, int, float]]:
    """descubre pares redundantes por |Spearman|>=thresh sobre timesteps train-normal apilados.
    train_rows (T, K), devuelve [(i, j, |corr|)] con i<j, ordenados por |corr| desc."""

    from src.crgn.graph import spearman_matrix

    s = np.abs(spearman_matrix(train_rows))
    # la diagonal a cero, un sensor consigo mismo correla 1 y no es un par
    np.fill_diagonal(s, 0.0)
    # solo el triangulo superior, para no devolver (i,j) y (j,i) como si fueran dos pares
    iu = np.triu_indices(s.shape[0], 1)
    m = s[iu] >= thresh
    pairs = [(int(i), int(j), float(s[i, j])) for i, j, keep in zip(iu[0], iu[1], m) if keep]
    return sorted(pairs, key=lambda p: -p[2])


def coherence_stats(train_nm: np.ndarray, pairs: list[tuple[int, int, float]], rate: bool = False) -> dict:
    """media y desviacion normales del residuo de coherencia r_ij = x_i - x_j por par, sobre
    todos los timesteps de train (leakage-free). train_nm es node-major (N, K, W). con
    rate=True se calcula sobre la derivada temporal de r, la rotura de la tasa de coherencia,
    que es aun mas ortogonal a AE y CRGN y da Earliness alta."""

    mu, sd = {}, {}
    for i, j, _w in pairs:
        r = train_nm[:, i, :] - train_nm[:, j, :]
        if rate:
            r = np.diff(r, axis=1)
        # se aplana el par entero, la referencia es una sola distribucion por par
        r = r.ravel()
        mu[(i, j)] = float(r.mean())
        sd[(i, j)] = float(r.std() + 1e-8)
    return {"mu": mu, "sd": sd}


def coherence_score(windows_nm: np.ndarray, pairs: list[tuple[int, int, float]], stats: dict,
                    agg: str = "mean", rate: bool = False) -> np.ndarray:
    """score de anomalia por ventana = violacion de coherencia, ponderada por |corr| y
    z-normalizada con stats del train. windows_nm es node-major (N, K, W) y sale (N,), donde
    alto quiere decir que se rompe la redundancia fisica.

    la agregacion del residuo r_t = x_i - x_j sobre la ventana puede ser 'mean', que da
    |media_t - mu|/sd, 'max', que da max_t|r_t - mu|/sd y capta la rotura transitoria sin que
    la media la lave, o 'std', que da std_t(r_t)/sd. con rate=True se trabaja sobre la derivada
    de r, la rotura de la tasa, que es la mas ortogonal. el campeon es agg='max' con
    rate=False, o sea phys95max, y rate con 'max' sale aun mas ⊥. stats tiene que venir de
    coherence_stats con el mismo rate, si no hay fuga.
    """

    mu, sd = stats["mu"], stats["sd"]
    num = np.zeros(len(windows_nm), dtype=np.float64)
    den = 0.0
    # media ponderada por |corr|, un par mas correlado pesa mas porque su rotura dice mas
    for i, j, w in pairs:
        r = windows_nm[:, i, :] - windows_nm[:, j, :]  # (N, W)
        if rate:
            r = np.diff(r, axis=1)
        if agg == "mean":
            dev = np.abs(r.mean(axis=1) - mu[(i, j)]) / sd[(i, j)]
        elif agg == "max":
            dev = np.abs(r - mu[(i, j)]).max(axis=1) / sd[(i, j)]
        elif agg == "std":
            dev = r.std(axis=1) / sd[(i, j)]
        else:
            raise ValueError(f"agg desconocido: {agg}")
        num += w * dev
        den += w
    return num / max(den, 1e-8)


def coherence_loss(z: torch.Tensor, pairs: list[tuple[int, int, float]], device: str) -> torch.Tensor:
    """regularizador que reune los latentes de pares redundantes, ||z_i - z_j||^2 ponderado por
    |corr| y promediado por elemento. z es (B, K, d_lat). shapea la variedad normal, y asi una
    rotura de coherencia da eps alto."""

    if not pairs:
        return torch.zeros((), device=device)
    # los pares se indexan de golpe con index_select, no en bucle, que hay cientos
    idx_i = torch.as_tensor([p[0] for p in pairs], device=device)
    idx_j = torch.as_tensor([p[1] for p in pairs], device=device)
    w = torch.as_tensor([p[2] for p in pairs], dtype=torch.float32, device=device)
    diff = z.index_select(1, idx_i) - z.index_select(1, idx_j)  # (B, P, d_lat)
    per_pair = diff.pow(2).mean(dim=(0, 2))  # (P,)
    return (w * per_pair).sum() / w.sum()
