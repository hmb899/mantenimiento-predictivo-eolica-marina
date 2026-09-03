"""
GDN (Deng & Hooi, AAAI 2021, arXiv:2106.06947) — reimplementation for Fase 2.

Forecast-based anomaly detection: a learnable embedding per sensor induces a directed
top-k cosine graph; an embedding-aware graph-attention network forecasts each sensor's
next value from its neighbours; the per-sensor forecast error (normalised by the
validation-normal median/IQR, aggregated over sensors) is the deviation score. The
official repo (d-ailin/GDN, no license) is the reference; the core is reimplemented in
plain PyTorch (no PyTorch Geometric) and the adjacency is injectable for the Fase 3 graph
ablation. Detection/threshold/CUSUM stay in the harness.
"""
from .data import forecast_tensors, load_frozen_windows, load_test_sequences, time_permuted
from .graph import (cosine_similarity_matrix, learned_adjacency, spearman_similarity,
                    topk_adjacency)
from .model import GDN, GraphAttention
from .scorer import GDNScorer, aggregate_sensor_errors

__all__ = [
    "cosine_similarity_matrix", "topk_adjacency", "learned_adjacency", "spearman_similarity",
    "GDN", "GraphAttention",
    "forecast_tensors", "load_frozen_windows", "load_test_sequences", "time_permuted",
    "GDNScorer", "aggregate_sensor_errors",
]
