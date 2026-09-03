"""
graph structure learning for GDN (Deng & Hooi, AAAI 2021, arXiv:2106.06947)

each sensor is a node with a learnable embedding, and the directed sparse graph is the top-k
cosine-similarity neighbourhood of those embeddings, so a node aggregates from its k most
similar sensors. the adjacency is computed here but kept injectable from the outside
(`GDN.forward(x, adjacency=...)`), so the learned graph can be swapped for a fixed correlation
graph, a full graph or no graph without touching the model.

convention, adjacency A is (N, N) with A[i, j] = 1 meaning an edge from j to i, so node i
receives messages from node j. rows index the target or receiver, columns the sender.
# ref: GDN models/GDN.py (get_batch_edge_index / cosine topk)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def cosine_similarity_matrix(emb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """pairwise cosine similarity of node embeddings, emb (N, d) to (N, N)."""

    if emb.dim() != 2:
        raise ValueError(f"emb must be 2D (N, d), got shape {tuple(emb.shape)}")
    # normalizando primero, el producto de la matriz por su traspuesta ya es el coseno
    normed = F.normalize(emb, p=2, dim=-1, eps=eps)
    return normed @ normed.t()


def topk_adjacency(sim: torch.Tensor, k: int, include_self: bool = True) -> torch.Tensor:
    """directed top-k adjacency from a similarity matrix.

    for each receiver node i, keep the k highest-similarity senders j != i, the edges from j to
    i. gives a float (N, N) matrix of 0/1, and self-loops are added when include_self is true,
    because GDN aggregates the node itself together with its neighbours.
    """

    if sim.dim() != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"sim must be square (N, N), got shape {tuple(sim.shape)}")
    n = sim.shape[0]
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, N-1]=[1, {n - 1}], got {k}")
    s = sim.clone()
    s.fill_diagonal_(float("-inf"))  # exclude self from the ranking
    idx = s.topk(k, dim=-1).indices  # (N, k) senders per receiver
    # scatter por filas, cada fila enciende sus k columnas de golpe
    adj = torch.zeros_like(sim)
    adj.scatter_(1, idx, 1.0)
    if include_self:
        adj.fill_diagonal_(1.0)
    return adj


def learned_adjacency(emb: torch.Tensor, k: int, include_self: bool = True) -> torch.Tensor:
    """top-k cosine-similarity graph from node embeddings, the GDN-learned graph."""

    return topk_adjacency(cosine_similarity_matrix(emb), k, include_self)


def spearman_similarity(x: np.ndarray) -> np.ndarray:
    """Spearman rank correlation between columns (sensors), x (T, N) to (N, N).

    SCADA signals fail normality, so the fixed-correlation graph arm of the ablation has to
    use Spearman and not Pearson. computed as Pearson on the column ranks, average ranks
    for ties via scipy.rankdata, and a constant column yields 0.
    """

    from scipy.stats import rankdata

    if x.ndim != 2:
        raise ValueError(f"x must be 2D (T, N), got shape {x.shape}")
    ranks = rankdata(x, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):  # constant column, std 0, gives NaN
        corr = np.corrcoef(ranks, rowvar=False)
    return np.nan_to_num(corr, nan=0.0)
