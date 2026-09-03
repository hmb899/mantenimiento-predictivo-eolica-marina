"""suavizado de senal Savitzky-Golay

reduce el ruido de alta frecuencia del sensor conservando la forma de la tendencia de fondo,
sin desfase, a diferencia de una media movil. se aplica por turbina y por segmento temporal
continuo. las features de angulo (is_angle=True) se saltan, un dato circular no se presta a
un suavizado polinomico a traves de la frontera 0°/360°.

Barnabei et al. usan este filtro con window_length=24 (4 h) y poly_order=3. los valores de
aqui (5 / 2) son eleccion de este trabajo, fijada por un estudio de ablacion y no heredada
de ese articulo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .utils import col_to_sensor_name, find_continuous_segments, get_logger, step_stats

logger = get_logger(__name__)


class Smoother:
    """aplica el suavizado Savitzky-Golay a las features continuas y salta las de angulo."""

    def __init__(self, config: dict, sensor_registry: pd.DataFrame) -> None:
        cfg = config["smoothing"]
        self.enabled: bool = cfg["enabled"]
        self.window_length: int = cfg["window_length"]
        self.poly_order: int = cfg["poly_order"]
        self.max_gap_min: int = config["windowing"]["max_gap_minutes"]

        # los de angulo se saltan, el conjunto va por nombre base de sensor
        self._angle_sensors: frozenset[str] = frozenset(
            row["sensor_name"]
            for _, row in sensor_registry.iterrows()
            if row["is_angle"]
        )


    def transform(self, df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
        """suaviza las columnas de feature in situ, por turbina y segmento continuo.
        Devuelve (df, stats)."""

        if not self.enabled:
            logger.info("Smoother: disabled — skipping.")
            return df, step_stats("smoothing", len(df), len(df))

        # in situ para no quedarse sin memoria con el tamano de la Farm C

        # se excluyen los sensores de angulo
        smooth_cols = [
            c for c in feature_cols
            if col_to_sensor_name(c) not in self._angle_sensors
        ]
        skipped = len(feature_cols) - len(smooth_cols)

        # se ordena para que el orden de proceso de los grupos sea determinista
        df.sort_values(["asset_id", "time_stamp"], inplace=True)

        segments_processed = 0
        segments_skipped = 0
        # la escritura diferida evita modificar df mientras se itera
        updates: list[tuple[pd.Index, str, np.ndarray]] = []

        for _, group in df.groupby("asset_id", sort=True):
            # ya viene ordenado por time_stamp
            segments = find_continuous_segments(group, max_gap_minutes=self.max_gap_min)

            for seg_idx in segments:
                seg_len = len(seg_idx)
                # savgol necesita mas filas que la ventana, de ahi el + 1
                if seg_len < self.window_length + 1:
                    logger.debug(
                        f"Smoother: skipping segment of {seg_len} rows "
                        f"(minimum {self.window_length + 1} required)"
                    )
                    segments_skipped += 1
                    continue

                # la ventana tiene que ser impar y mayor que poly_order
                wl = self.window_length
                if wl % 2 == 0:
                    wl -= 1
                if wl <= self.poly_order:
                    logger.debug(
                        f"Smoother: skipping segment with effective window {wl} "
                        f"not greater than poly_order {self.poly_order}"
                    )
                    segments_skipped += 1
                    continue

                seg = df.loc[seg_idx, smooth_cols]

                for col in smooth_cols:
                    col_vals = seg[col].values.astype(float)
                    if np.isnan(col_vals).all():
                        continue
                    # savgol propaga un NaN a toda la ventana, la media es un
                    # relleno neutro que salva los valores de alrededor
                    nan_mask = np.isnan(col_vals)
                    if nan_mask.any():
                        col_vals[nan_mask] = np.nanmean(col_vals)

                    smoothed = savgol_filter(col_vals, window_length=wl, polyorder=self.poly_order)
                    # se restauran los NaN, el relleno no puede sobrevivir
                    smoothed[nan_mask] = np.nan
                    updates.append((seg_idx, col, smoothed))

                segments_processed += 1

        for seg_idx, col, smoothed in updates:
            df.loc[seg_idx, col] = smoothed

        logger.info(
            f"Smoother: processed {segments_processed} segments "
            f"({segments_skipped} skipped too short) "
            f"({len(smooth_cols)} features smoothed, {skipped} angle features skipped)"
        )

        stats = step_stats(
            "smoothing",
            len(df),
            len(df),
            {
                "window_length": self.window_length,
                "poly_order": self.poly_order,
                "n_smoothed_cols": len(smooth_cols),
                "n_skipped_cols": skipped,
                "segments_processed": segments_processed,
                "segments_skipped": segments_skipped,
            },
        )
        return df, stats
