"""
carga las salidas del pipeline de preproceso (.npz, .json) para entrenar y evaluar modelos

ModelDataLoader.load
    recibe un codigo de granja, "B" o "C", y devuelve su FarmData
    train_windows (N_train, 18, F) y feature_cols list[str]
    test_sequences, un dict por evento con event_id, event_label, asset_id, windows
    (N_seq, 18, F), labels (N_seq,), window_timestamps, event_start y event_end
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.preprocessing import TestSequenceDict

logger = logging.getLogger(__name__)


@dataclass
class FarmData:
    """contenedor de los artefactos preprocesados de granja que usa el pipeline del modelo."""

    farm: str
    train_windows: np.ndarray  # (N_train, window_size, n_features)
    feature_cols: list[str]
    test_sequences: list[TestSequenceDict]
    turbine_ids: np.ndarray = None  # asset_id por ventana de entrenamiento
    scaler: object = None  # StandardScaler del preproceso ya ajustado (opcional)


def _load_config(config: dict | None = None) -> dict:
    """devuelve config tal cual, o carga el model_config.yaml por defecto si no llega."""

    if config is not None:
        return config
    # import perezoso, yaml solo hace falta si quien llama no pasa config
    import yaml
    cfg_path = Path("config/model_config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class ModelDataLoader:
    """carga los datos preprocesados de una granja para el pipeline del modelo."""

    def __init__(self, config: dict | None = None) -> None:
        """resuelve la ruta de los datos procesados respecto a la raiz del proyecto."""

        self.config = _load_config(config)
        # las rutas relativas salen de la raiz del proyecto, dos niveles sobre src/model_prep/
        _project_root = Path(__file__).resolve().parents[2]
        raw_path = Path(self.config["data"]["processed_path"])
        self.processed_path = raw_path if raw_path.is_absolute() else _project_root / raw_path


    def load(self, farm: str) -> FarmData:
        """carga todos los artefactos preprocesados de una granja y devuelve un FarmData."""

        farm_l = farm.lower()
        p = self.processed_path

        # los cuatro artefactos de una granja comparten prefijo farm_<b|c>_ y se leen por
        # nombre
        train_path = p / f"farm_{farm_l}_train_windows.npz"
        train_data = np.load(train_path, allow_pickle=False)
        train_windows = train_data["windows"].astype(np.float32)
        turbine_ids = train_data["turbine_ids"] if "turbine_ids" in train_data else None

        feat_path = p / f"farm_{farm_l}_feature_cols.txt"
        feature_cols = feat_path.read_text(encoding="utf-8").strip().splitlines()

        meta_path = p / f"farm_{farm_l}_test_metadata.json"
        with open(meta_path, encoding="utf-8") as f:
            metadata: dict = json.load(f)

        # las ventanas de test van indexadas por event_id como str
        seq_path = p / f"farm_{farm_l}_test_sequences.npz"
        seq_data = np.load(seq_path, allow_pickle=False)

        # el metadata manda, las ventanas se buscan en el npz por la clave del evento
        test_sequences: list[TestSequenceDict] = []
        for eid_str, meta in metadata.items():
            windows = seq_data[eid_str].astype(np.float32)  # (N, W, F)
            labels = np.array(meta["labels"], dtype=np.int8)
            ts = np.array(meta["window_timestamps"], dtype="datetime64[ns]")
            test_sequences.append(
                {
                    "event_id": int(eid_str),
                    "event_label": meta["event_label"],
                    "asset_id": int(meta["asset_id"]),
                    "windows": windows,
                    "labels": labels,
                    "window_timestamps": ts,
                    "event_start": meta["event_start"],
                    "event_end": meta["event_end"],
                    "event_description": meta.get("event_description", ""),
                }
            )

        # el scaler del preproceso es opcional, no todas las granjas lo dejan escrito
        scaler = None
        scaler_path = p / f"farm_{farm_l}_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f_pkl:
                scaler = pickle.load(f_pkl)

        n_anomaly = sum(1 for s in test_sequences if s["event_label"] == "anomaly")
        n_normal = sum(1 for s in test_sequences if s["event_label"] == "normal")
        logger.info(
            f"[ModelDataLoader] Farm {farm} "
            f"train {train_windows.shape} "
            f"{len(test_sequences)} test sequences "
            f"({n_anomaly} anomaly {n_normal} normal) "
            f"{len(feature_cols)} features"
        )

        return FarmData(
            farm=farm,
            train_windows=train_windows,
            feature_cols=feature_cols,
            test_sequences=test_sequences,
            turbine_ids=turbine_ids,
            scaler=scaler,
        )
