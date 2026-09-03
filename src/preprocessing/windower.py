"""
ventaneo temporal, ventanas deslizantes construidas sobre segmentos continuos de datos

el tamano optimo de ventana son 18 muestras, 3 horas a resolucion de 10 min, segun Jankauskas
et al. (2024)

ventanas de entrenamiento
    se construyen sobre cada segmento continuo de operacion normal, con stride 1 para maximo
    solape y cobertura densa; se descarta la ventana con menos valores validos (no NaN) que
    min_valid_ratio

secuencias de test
    una secuencia por evento, sobre las filas etiquetadas como test, ventaneadas tambien con
    stride 1 para que la puntuacion sea por ventana; el array de etiquetas vale 1 (anomalia) en
    los eventos anomalos y 0 en los normales
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd

from .utils import find_continuous_segments, get_logger, step_stats

logger = get_logger(__name__)


class TestSequenceDict(TypedDict):
    """esquema tipado de una secuencia de test ya cargada de disco, todo numpy y str, sin
    pd.Timestamp."""
    event_id: int
    event_label: str  # "anomaly" o "normal"
    asset_id: int
    windows: np.ndarray  # forma (N, window_size, n_features)
    labels: np.ndarray  # forma (N,) int8
    window_timestamps: np.ndarray  # forma (N,) datetime64[ns]
    event_start: str
    event_end: str
    event_description: str


@dataclass
class TestSequence:
    """guarda los datos de test ya ventaneados de un evento."""
    event_id: int
    event_label: str  # "anomaly" o "normal"
    asset_id: int
    windows: np.ndarray  # forma (N, window_size, n_features)
    labels: np.ndarray  # forma (N,) 1=ventana anomala
    window_timestamps: np.ndarray  # forma (N,) datetime64, marca de inicio por ventana
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    event_description: str = ""


class Windower:
    """construye arrays de ventana deslizante sobre segmentos temporales continuos."""

    def __init__(self, config: dict) -> None:
        cfg = config["windowing"]
        self.window_size: int = cfg["window_size_samples"]
        self.stride: int = cfg["stride_samples"]
        self.min_valid: float = cfg["min_valid_ratio"]
        self.max_gap_min: int = cfg["max_gap_minutes"]


    def create_train_windows(self, df_train: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
        """ventanas deslizantes sobre los datos de entrenamiento en operacion normal,
        devuelve las ventanas, los turbine_ids y las stats."""

        all_windows: list[np.ndarray] = []
        all_turbine_ids: list[int] = []
        groups = df_train.groupby("asset_id", sort=False)

        # turbina a turbina, y dentro de cada una segmento continuo a segmento continuo, para
        # que ninguna ventana cruce un hueco de datos
        for asset_id, group in groups:
            group = group.sort_values("time_stamp")
            segments = find_continuous_segments(group, max_gap_minutes=self.max_gap_min)

            for seg_idx in segments:
                seg_data = group.loc[seg_idx, feature_cols].values.astype(float)
                windows = self._slide(seg_data)
                all_windows.extend(windows)
                # una entrada de turbine_ids por ventana, el orden es el contrato con windows
                all_turbine_ids.extend([int(asset_id)] * len(windows))

        if not all_windows:
            logger.warning("Windower: no training windows created!")
            return np.empty((0, self.window_size, len(feature_cols))), np.empty(0, dtype=np.int32), {}

        windows_arr = np.stack(all_windows, axis=0)
        turbine_ids_arr = np.array(all_turbine_ids, dtype=np.int32)
        n_total = len(all_windows)
        logger.info(
            f"Windower (train): {n_total:,} windows "
            f"(shape {windows_arr.shape})"
        )

        stats = step_stats(
            "windowing_train",
            len(df_train),
            n_total,
            {
                "window_size_samples": self.window_size,
                "stride_samples": self.stride,
                "min_valid_ratio": self.min_valid,
                "n_features": len(feature_cols),
                "window_shape": list(windows_arr.shape),
            },
        )
        return windows_arr, turbine_ids_arr, stats


    def create_test_sequences(self, df_all: pd.DataFrame, event_info: pd.DataFrame,
                              feature_cols: list[str]) -> tuple[list[TestSequence], dict]:
        """una TestSequence ventaneada por evento, sobre las filas etiquetadas como test,
        devuelve las secuencias y las stats."""

        sequences: list[TestSequence] = []
        # CARE llama "prediction" a la parte de evaluacion, no "test"
        test_rows = df_all[df_all["train_test"] == "prediction"].copy()

        for _, evt in event_info.iterrows():
            event_id = evt["event_id"]
            event_label = evt["event_label"]
            asset_id = evt["asset_id"]

            # un evento se identifica por el par (event_id, asset_id), no por el event_id solo
            subset = test_rows[
                (test_rows["event_id"] == event_id)
                & (test_rows["asset_id"] == asset_id)
            ].sort_values("time_stamp")

            if subset.empty:
                logger.debug(f"  Event {event_id}: no test rows — skipped")
                continue

            timestamps = subset["time_stamp"].values  # numpy datetime64
            seg_data = subset[feature_cols].values.astype(float)
            windows, win_timestamps = self._slide_with_timestamps(seg_data, timestamps)

            if not windows:
                logger.debug(f"  Event {event_id}: segment too short for windowing")
                continue

            windows_arr = np.stack(windows, axis=0)
            win_ts_arr = np.array(win_timestamps, dtype="datetime64[ns]")

            # verdad de referencia por ventana. en un evento anomalo la ventana vale 1 si su
            # punto medio, win_start + floor(window_size/2) * sample_interval, cae en
            # [event_start, event_end), o sea solape mayoritario, ≥50% de sus muestras dentro
            if event_label == "anomaly":
                t_start = np.datetime64(evt["event_start"], "ns")
                t_end = np.datetime64(evt["event_end"], "ns")
                # el paso se infiere de la mediana de las diferencias, robusta a huecos;
                # con una sola marca no hay diferencia que medir y se cae a los 10 min de CARE
                if len(timestamps) > 1:
                    sample_ns = int(np.median(np.diff(timestamps.astype(np.int64))))
                    sample_interval = np.timedelta64(sample_ns, "ns")
                else:
                    sample_interval = np.timedelta64(10, "m")
                half_window = sample_interval * (self.window_size // 2)
                win_mid_arr = win_ts_arr + half_window
                # asi una ventana de frontera con 1-2 muestras anomalas no cuenta como anomala
                # entera, ni al reves, y las previas al fallo se quedan en 0
                labels = ((win_mid_arr >= t_start) & (win_mid_arr < t_end)).astype(np.int8)
            else:
                labels = np.zeros(len(windows_arr), dtype=np.int8)  # evento normal, sin fallo

            sequences.append(
                TestSequence(
                    event_id=int(event_id),
                    event_label=event_label,
                    asset_id=int(asset_id),
                    windows=windows_arr,
                    labels=labels,
                    window_timestamps=win_ts_arr,
                    event_start=evt["event_start"],
                    event_end=evt["event_end"],
                    event_description=str(evt.get("event_description", "")),
                )
            )

        n_anomaly = sum(1 for s in sequences if s.event_label == "anomaly")
        n_normal = sum(1 for s in sequences if s.event_label == "normal")
        total_windows = sum(len(s.windows) for s in sequences)
        logger.info(
            f"Windower (test): {len(sequences)} sequences "
            f"({n_anomaly} anomaly, {n_normal} normal), "
            f"{total_windows:,} total windows"
        )

        stats = {
            "step": "windowing_test",
            "n_sequences": len(sequences),
            "n_anomaly_sequences": n_anomaly,
            "n_normal_sequences": n_normal,
            "total_test_windows": total_windows,
        }
        return sequences, stats


    def _slide(self, data: np.ndarray) -> list[np.ndarray]:
        """desliza ventanas sobre un array 2-D y tira las que bajan de min_valid_ratio."""

        windows, _ = self._slide_with_timestamps(data, timestamps=None)
        return windows


    def _slide_with_timestamps(self, data: np.ndarray, timestamps: np.ndarray | None) -> tuple[list[np.ndarray], list]:
        """desliza ventanas sobre un array 2-D y, si recibe timestamps, devuelve tambien la
        marca de inicio de cada ventana."""

        n_samples, n_features = data.shape
        if n_samples < self.window_size:
            return [], []

        # el umbral es una cuenta de celdas, min_valid es fraccion de las W*F de la ventana
        min_valid_values = int(self.min_valid * self.window_size * n_features)
        windows: list[np.ndarray] = []
        win_timestamps: list = []

        # el ultimo arranque posible es n_samples - window_size, de ahi el +1 del rango
        for start in range(0, n_samples - self.window_size + 1, self.stride):
            window = data[start : start + self.window_size]
            valid = int(np.isfinite(window).sum())
            if valid >= min_valid_values:
                windows.append(window)
                if timestamps is not None:
                    win_timestamps.append(timestamps[start])

        return windows, win_timestamps
