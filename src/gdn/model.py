"""
GDN forecasting network (Deng & Hooi, AAAI 2021, arXiv:2106.06947), reimplemented in plain
PyTorch, no PyTorch Geometric. the graph is small (<= ~100 sensor nodes) so the single
graph-attention layer is computed densely over an (N, N) adjacency.

four components in the paper, (1) a learnable embedding per sensor, (2) a directed top-k cosine
graph over those embeddings (src.gdn.graph), (3) an embedding-aware graph attention that
forecasts each sensor's next value from its neighbours, and (4) the per-sensor forecast error
as the deviation signal, normalised and aggregated in the scorer.

the adjacency is injectable (`forward(x, adjacency=...)`) for the graph ablation, pass an
external (N, N) matrix to use a fixed correlation, full or empty graph, or None to use the
learned top-k cosine graph. the embedding out-dim and the graph-attention out-dim are coupled
to a single `embed_dim`, because GDN concatenates feature and embedding in the attention.
# ref: GDN models/GDN.py and models/graph_layer.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gdn.graph import learned_adjacency


class GraphAttention(nn.Module):
    """single embedding-aware graph-attention layer, GDN's GraphLayer densified.

    it transforms each node window x_i (length `in_dim`) into g_i = W x_i (dim `out_dim`), then
    aggregates the neighbours with an attention that fuses the transformed feature and the node
    embedding, alpha_ij = softmax_j LeakyReLU(a_i.[g_i||v_i] + a_j.[g_j||v_j]) over the senders
    j of the edge from j to i. the output is z_i = sum_j alpha_ij g_j (+ bias).
    """

    def __init__(self, in_dim: int, out_dim: int, slope: float = 0.2) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.slope = slope
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        # cuatro vectores de atencion, dos por el lado del receptor y dos por el del emisor
        self.att_i = nn.Parameter(torch.empty(out_dim))  # weights over feature g_i
        self.att_j = nn.Parameter(torch.empty(out_dim))
        self.att_em_i = nn.Parameter(torch.empty(out_dim))  # weights over embedding v_i
        self.att_em_j = nn.Parameter(torch.empty(out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self._reset()


    def _reset(self) -> None:
        nn.init.kaiming_uniform_(self.lin.weight, a=self.slope)
        for p in (self.att_i, self.att_j, self.att_em_i, self.att_em_j):
            nn.init.uniform_(p, -0.1, 0.1)


    def forward(self, x: torch.Tensor, emb: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """x (B, N, in_dim), emb (N, out_dim), adjacency (N, N) with A[i, j]=1 for an edge from
        j to i. gives z (B, N, out_dim)."""

        g = self.lin(x)  # (B, N, out)
        # target term (depends on receiver i) and source term (depends on sender j)
        s_i = g @ self.att_i + emb @ self.att_em_i  # (B, N)  [+ (N,) broadcast]
        s_j = g @ self.att_j + emb @ self.att_em_j  # (B, N)
        # los unsqueeze cruzan los dos vectores en una matriz i por j sin materializar el
        # producto
        # (B, N, N) [i, j]
        scores = F.leaky_relu(s_i.unsqueeze(2) + s_j.unsqueeze(1), self.slope)
        # lo que no es arco se manda a -inf, asi el softmax le da exactamente cero
        mask = adjacency.unsqueeze(0) > 0  # (1, N, N)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)  # over senders j
        # una fila entera de -inf da NaN en el softmax, de ahi el saneado
        alpha = torch.nan_to_num(alpha, nan=0.0)  # receiver with no edge gives 0
        z = alpha @ g  # (B, N, out)
        return z + self.bias


class GDN(nn.Module):
    """Graph Deviation Network, a per-sensor next-step forecaster over a learned graph."""

    def __init__(self, n_nodes: int, context: int, embed_dim: int = 64,
                 out_layer_inter_dim: int = 128, out_layer_num: int = 1, topk: int = 15,
                 slope: float = 0.2, include_self_loops: bool = True, horizon: int = 1) -> None:
        super().__init__()
        if out_layer_num < 1:
            raise ValueError(f"out_layer_num must be >= 1, got {out_layer_num}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.n_nodes = n_nodes
        self.context = context
        self.embed_dim = embed_dim
        self.topk = topk
        self.horizon = horizon
        self.include_self_loops = include_self_loops
        # un embedding por sensor, es lo que aprende el grafo, no hay adyacencia parametrica
        self.embedding = nn.Embedding(n_nodes, embed_dim)
        nn.init.kaiming_uniform_(self.embedding.weight, a=slope)
        # la atencion entra con la historia de cada nodo y sale ya en el ancho del embedding
        self.gnn = GraphAttention(context, embed_dim, slope)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.out_mlp = self._build_out_mlp(embed_dim, out_layer_inter_dim, out_layer_num, horizon)


    @staticmethod
    def _build_out_mlp(dim: int, inter: int, num_layers: int, horizon: int) -> nn.Sequential:
        # con num_layers=1 el bucle no entra y queda una unica lineal a horizon
        layers: list[nn.Module] = []
        d = dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, inter), nn.BatchNorm1d(inter), nn.ReLU()]
            d = inter
        layers += [nn.Linear(d, horizon)]
        return nn.Sequential(*layers)


    def node_embeddings(self) -> torch.Tensor:
        idx = torch.arange(self.n_nodes, device=self.embedding.weight.device)
        return self.embedding(idx)


    def adjacency(self) -> torch.Tensor:
        """the learned top-k cosine graph from the current sensor embeddings."""

        return learned_adjacency(self.node_embeddings(), self.topk, self.include_self_loops)


    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        """x (B, N, context) gives (B, N) for horizon=1, else (B, N, horizon), the next
        `horizon` values per sensor. adjacency is an optional external (N, N) graph, and None
        means the learned one."""

        if x.dim() != 3 or x.shape[1] != self.n_nodes or x.shape[2] != self.context:
            raise ValueError(
                f"x must be (B, {self.n_nodes}, {self.context}), got shape {tuple(x.shape)}")
        emb = self.node_embeddings()
        # el grafo se recalcula en cada paso, porque depende de embeddings que estan
        # entrenandose
        a = self.adjacency() if adjacency is None else adjacency.to(x.device, x.dtype)
        z = self.gnn(x, emb, a)  # (B, N, out)
        z = z * emb.unsqueeze(0)  # fuse the sensor embedding (GDN)
        # BatchNorm1d espera (lote, canal), asi que nodo y lote se aplanan juntos y se deshace
        b, n, d = z.shape
        z = self.bn(z.reshape(b * n, d)).reshape(b, n, d)
        z = F.relu(z)
        out = self.out_mlp(z.reshape(b * n, d)).reshape(b, n, self.horizon)
        return out.squeeze(-1) if self.horizon == 1 else out
