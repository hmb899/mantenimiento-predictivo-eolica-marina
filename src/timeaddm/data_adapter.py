"""
adapt preprocessed FarmData for TimeADDM, MinMaxScaling without PCA

reuses ModelDataLoader and WindowScaler from the base pipeline
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.model_prep.data_loader import ModelDataLoader
from src.model_prep.model_scaler import WindowScaler

if TYPE_CHECKING:
    from src.preprocessing import TestSequenceDict

logger = logging.getLogger(__name__)


def load_and_scale(config: dict, farm: str,
                   feature_cols: list[str] | None = None) -> tuple[np.ndarray, list[dict], list[str], WindowScaler]:
    """load preprocessed farm data, optionally filter features, and apply MinMaxScaler [0, 1].
    no PCA is applied, the LSTMAE encoder serves as the dimensionality reduction. gives back
    train_scaled (N_train, W, F) float32, the test sequences with their windows already scaled,
    the feature names left after filtering, and the fitted WindowScaler, which eval-only mode
    needs."""

    data = ModelDataLoader(config).load(farm)

    train_windows = data.train_windows
    test_seqs = data.test_sequences
    feat_names = data.feature_cols

    if feature_cols is not None:
        # se filtra por posicion, y lo que se pida y no exista se avisa en vez de romper
        idx = [feat_names.index(f) for f in feature_cols if f in feat_names]
        missing = [f for f in feature_cols if f not in feat_names]
        if missing:
            logger.warning(f"[data_adapter] Features not found in Farm {farm} data: {missing}")
        train_windows = train_windows[:, :, idx]
        for seq in test_seqs:
            seq["windows"] = seq["windows"][:, :, idx]
        # los nombres se reordenan igual que las columnas, el orden es el contrato con el
        # modelo
        feat_names = [feat_names[i] for i in idx]

    # el scaler se ajusta solo con train y luego se aplica a los eventos, sin re-ajustar
    scaler = WindowScaler(feature_range=tuple(config["scaling"]["feature_range"]), per_turbine=False,
                          method=config["scaling"].get("method", "minmax"))
    train_scaled = scaler.fit_transform(train_windows)
    test_seqs = scaler.transform_sequences(test_seqs)

    logger.info(
        f"[data_adapter] Farm {farm}: {train_scaled.shape[0]:,} train windows "
        f"{train_scaled.shape[1]}×{train_scaled.shape[2]} "
        f"(W×F) range [{train_scaled.min():.3f}, {train_scaled.max():.3f}]"
    )

    return train_scaled, test_seqs, feat_names, scaler
