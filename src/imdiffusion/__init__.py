"""
ImDiffusion (Chen et al., VLDB 2023, arXiv:2307.00754) — reimplementation for Fase 1b.

Anomaly detection by conditional diffusion IMPUTATION: complementary grating masks
hide alternating strips of each window; the model imputes them and the per-position
imputation error becomes the window anomaly score. The official repo has no license,
so the core is reimplemented here; ambiguities resolved
against the official code are documented with `# ref: IMDiffusion <file>` comments.
"""
from .masking import grating_mask, complementary_masks, build_cond_target
from .denoiser import ImDiffusionDenoiser
from .diffusion import ImDiffusionModel, masked_noise_loss, quad_beta_schedule
from .scorer import ImDiffusionScorer, StepEnsemble, aggregate_window_errors
from .data import LazyWindowDataset, segments_from_frame, compute_if_retention_mask

__all__ = [
    "grating_mask", "complementary_masks", "build_cond_target",
    "ImDiffusionDenoiser",
    "ImDiffusionModel", "masked_noise_loss", "quad_beta_schedule",
    "ImDiffusionScorer", "StepEnsemble", "aggregate_window_errors",
    "LazyWindowDataset", "segments_from_frame", "compute_if_retention_mask",
]
