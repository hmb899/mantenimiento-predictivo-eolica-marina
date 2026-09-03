"""
scorer de desviacion de CRGN

CRGNScorer
    el contrato del harness, score_windows recibe windows (N, W, K) y devuelve (N,) float64,
    mas el atributo percentile, asi el harness sin tocar (run_model.evaluate, CUSUM, h-sweep,
    CV y CARE oficial) evalua CRGN igual que al AE, GDN e ImDiffusion
    tras el entrenamiento auto-supervisado con datos normales, el score de anomalia de una
    ventana es lo lejos que cae su embedding CRGN de la distribucion de los embeddings de
    validacion-normal
    mahalanobis, distancia de Mahalanobis al cuadrado a la media, covarianza regularizada
    por shrinkage
    knn, distancia euclidea media a sus k embeddings de validacion-normal mas cercanos
    solo el score crudo, el suavizado, el umbral, el CUSUM y la logica de evento se quedan
    en el harness
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.crgn.data import to_node_major
from src.crgn.model import CRGN


class CRGNScorer:
    """convierte un CRGN entrenado y su adyacencia fija en un scorer de desviacion por
    ventana."""

    def __init__(self, model: CRGN, adjacency: np.ndarray, device: torch.device,
                 metric: str = "mahalanobis", knn_k: int = 5, shrinkage: float = 1e-3,
                 batch_size: int = 512, percentile: float = 99.0) -> None:
        if metric not in ("mahalanobis", "knn"):
            raise ValueError(f"metric must be mahalanobis|knn, got {metric}")
        self.model = model
        # la adyacencia se sube una vez y se reutiliza en todos los lotes
        self.adj = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
        self.device = device
        self.metric = metric
        self.knn_k = knn_k
        self.shrinkage = shrinkage
        self.batch_size = batch_size
        self.percentile = percentile
        # la referencia de normalidad se rellena en fit_reference, una rama u otra segun metric
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self._nn = None  # NearestNeighbors de sklearn ya ajustado (knn)


    @torch.no_grad()
    def _embed(self, windows: np.ndarray) -> np.ndarray:
        """embedding CRGN por ventana, (N, W, K) a (N, embed_dim) float64."""

        x = torch.from_numpy(to_node_major(windows).astype(np.float32))
        self.model.to(self.device).eval()
        # shuffle=False es obligatorio, la salida tiene que ir alineada con las ventanas
        loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
        out = [self.model.encode(xb.to(self.device), self.adj).cpu().numpy() for (xb,) in loader]
        return np.concatenate(out).astype(np.float64)


    def fit_reference(self, val_windows: np.ndarray) -> "CRGNScorer":
        """ajusta la referencia de embeddings normales con las ventanas de
        validacion-normal."""

        emb = self._embed(val_windows)
        if self.metric == "mahalanobis":
            self.mean_ = emb.mean(axis=0)
            # el shrinkage en la diagonal mantiene la covarianza invertible con pocos
            # embeddings
            cov = np.cov(emb, rowvar=False) + self.shrinkage * np.eye(emb.shape[1])
            self.precision_ = np.linalg.inv(cov)
        else:
            from sklearn.neighbors import NearestNeighbors

            self._nn = NearestNeighbors(n_neighbors=self.knn_k).fit(emb)
        return self


    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        if self.metric == "mahalanobis":
            if self.mean_ is None or self.precision_ is None:
                raise RuntimeError("call fit_reference(val_windows) before scoring")
            d = self._embed(windows) - self.mean_
            # Mahalanobis al cuadrado fila a fila, el einsum evita la matriz N x N intermedia
            return np.einsum("ni,ij,nj->n", d, self.precision_, d).astype(np.float64)
        if self._nn is None:
            raise RuntimeError("call fit_reference(val_windows) before scoring")
        dist, _ = self._nn.kneighbors(self._embed(windows))  # (N, k)
        return dist.mean(axis=1).astype(np.float64)
