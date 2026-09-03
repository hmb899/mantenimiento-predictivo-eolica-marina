"""deteccion y tratamiento de outliers

dos etapas, los limites fisicos duros, que cazan los valores fuera del rango plausible del
dominio, y un filtro IQR blando por tipo de feature, ajustado sobre los datos de
entrenamiento (Status-0).

tratamiento, los valores outlier pasan a NaN y despues se interpolan los huecos con la misma
logica que ZeroHandler, lo que conserva la estructura temporal.

referencia, Alagha et al. (2025), filtrado adaptativo de ruido para SCADA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    col_to_sensor_name,
    find_continuous_segments,
    get_logger,
    step_stats,
)

logger = get_logger(__name__)


# de unidad cruda al tipo que indexa iqr_multipliers y physical_limits
# los contadores (kWh, kvarh) caen bajo power a proposito, comparten rango
_UNIT_TO_TYPE: dict[str, str] = {
    "°C": "temperature",
    "celsius": "temperature",
    "kW": "power",
    "kVA": "power",
    "kWh": "power",
    "kvarh": "power",
    "m/s": "wind_speed",
    "rpm": "rpm",
    "1/min": "rpm",
    "rad/s": "rpm",
    "V": "voltage",
    "kV": "voltage",
    "Hz": "frequency",
    "A": "current",
    "bar": "pressure",
    "l/h": "pressure",
    "L/h": "pressure",
    "L/min": "pressure",
    "l/min": "pressure",
}


def _unit_to_type(unit: str) -> str:
    """traduce una unidad al tipo de sensor con el que se busca el multiplicador IQR."""

    u = unit.strip()
    result = _UNIT_TO_TYPE.get(u)
    if result:
        return result
    # reserva para los problemas de codificacion del grado
    # el tope de longitud frena una unidad larga acabada en c
    u_lower = u.lower()
    if "celsius" in u_lower or (len(u) <= 4 and u_lower.endswith("c")):
        return "temperature"
    return "default"


class OutlierHandler:
    """detecta y trata los outliers de las columnas de sensor, limites fisicos mas IQR."""

    def __init__(self, config: dict, sensor_registry: pd.DataFrame) -> None:
        cfg = config["outlier_detection"]
        self.iqr_multipliers: dict[str, float] = cfg["iqr_multipliers"]
        self.physical_limits: dict[str, list] = cfg["physical_limits"]
        # el presupuesto de interpolacion es el de ZeroHandler, misma config
        self.max_linear = config["zero_handling"]["interpolation"]["max_gap_linear"]
        self.max_ffill = config["zero_handling"]["interpolation"]["max_gap_ffill"]
        self.max_gap_min = config["windowing"]["max_gap_minutes"]

        # de nombre de sensor a tipo
        self._sensor_type: dict[str, str] = {
            row["sensor_name"]: _unit_to_type(row.get("unit_norm", row["unit"]))
            for _, row in sensor_registry.iterrows()
        }

        # lo rellena fit()
        self._bounds: dict[str, tuple[float, float]] = {}


    def fit(self, df_train: pd.DataFrame, feature_cols: list[str]) -> None:
        """calcula los limites IQR por columna a partir de los datos de entrenamiento."""

        self._bounds = {}
        for col in feature_cols:
            series = df_train[col].dropna()
            # sin datos no hay limites, transform dejara la columna tal cual
            if series.empty:
                continue
            base = col_to_sensor_name(col)
            # el multiplicador va por tipo, no es igual temperatura que potencia
            stype = self._sensor_type.get(base, "default")
            k = self.iqr_multipliers.get(stype, self.iqr_multipliers["default"])
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            self._bounds[col] = (q1 - k * iqr, q3 + k * iqr)
        logger.info(f"OutlierHandler fitted on {len(self._bounds)} features.")


    def transform(self, df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
        """aplica el tratamiento de limites fisicos e IQR, necesita fit() antes.
        Devuelve (df, stats)."""

        if not self._bounds:
            logger.warning("OutlierHandler.transform() called before fit() — skipping.")
            return df, step_stats("outlier_detection", len(df), len(df))

        rows_before = len(df)
        total_outliers = 0

        for col in feature_cols:
            if col not in df.columns:
                continue

            base = col_to_sensor_name(col)
            stype = self._sensor_type.get(base, "default")
            col_vals = df[col].to_numpy()
            # las dos etapas se acumulan, basta con que una marque el valor
            mask = np.zeros(len(df), dtype=bool)

            # limites fisicos duros
            phys = self.physical_limits.get(stype)
            if phys:
                lo, hi = phys
                mask |= col_vals < lo
                mask |= col_vals > hi

            # filtro IQR blando
            if col in self._bounds:
                lo_iqr, hi_iqr = self._bounds[col]
                mask |= col_vals < lo_iqr
                mask |= col_vals > hi_iqr

            n_out = int(mask.sum())
            if n_out:
                df.loc[df.index[mask], col] = np.nan
                total_outliers += n_out

        logger.info(f"Outlier handler: {total_outliers:,} values set to NaN")

        # aqui se rellenan los NaN recien escritos, el numero de filas no cambia
        self._interpolate(df, feature_cols)

        stats = step_stats("outlier_detection", rows_before, len(df), {"outliers_replaced": total_outliers})

        return df, stats


    def _interpolate(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        """interpola los huecos de NaN in situ, ordenando antes para que el orden de los
        grupos sea determinista."""

        df.sort_values(["asset_id", "time_stamp"], inplace=True)

        # la escritura diferida evita modificar df mientras se itera
        updates: list[tuple[pd.Index, np.ndarray]] = []

        for _, group in df.groupby("asset_id", sort=True):
            # ya viene ordenado por time_stamp
            segments = find_continuous_segments(group, max_gap_minutes=self.max_gap_min)
            for seg_idx in segments:
                seg_len = len(seg_idx)
                # con una sola fila no hay nada entre lo que interpolar
                if seg_len < 2:
                    continue

                seg = df.loc[seg_idx, feature_cols].copy()
                # primero lineal, con el tope acotado para que nunca supere al segmento
                lin_limit = min(self.max_linear, seg_len - 1)
                seg = seg.interpolate(method="linear", limit=lin_limit, limit_direction="forward")

                # el ffill cubre de max_gap_linear a max_gap_ffill, un hueco
                # mas largo se queda en NaN a proposito
                remaining = min(self.max_ffill - self.max_linear, seg_len - 1)
                if remaining > 0:
                    seg = seg.ffill(limit=remaining)
                updates.append((seg_idx, seg.values))

        # escritura posicional, seg.values respeta el orden de feature_cols
        for seg_idx, values in updates:
            df.loc[seg_idx, feature_cols] = values
