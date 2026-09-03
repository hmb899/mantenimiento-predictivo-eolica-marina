"""
tratamiento del cero como valor ausente en las granjas B y C de CARE

el articulo CARE (Guck et al., 2024) dice que las dos granjas codifican el valor ausente como
0, asi que este modulo clasifica cada sensor, sustituye por NaN los ceros fisicamente
imposibles e interpola los huecos pequenos. las reglas se aplican en orden y gana la primera

keep
    el 0 es fisicamente valido y no hay contraste fiable, solo los angulos
keep_cross_wind
    contraste entre turbinas, un 0 se marca si la mediana del parque supera cut_in
keep_cross_temp
    contraste entre turbinas, un 0 se marca si |mediana del parque| supera temp_threshold
keep_temporal
    contraste temporal, un 0 se marca si se queda congelado N muestras operativas
    consecutivas o mas
always
    cualquier 0 es fisicamente imposible, sea cual sea el estado de operacion
context
    un 0 solo es sospechoso mientras la turbina opera, con viento por encima del cut-in
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import col_to_sensor_name, find_continuous_segments, get_logger, step_stats

logger = get_logger(__name__)

# unidades que deciden la estrategia por defecto de un sensor
_ALWAYS_REPLACE_UNITS = frozenset({"°C", "V", "kV", "Hz"})
_CONTEXT_REPLACE_UNITS = frozenset({"rpm", "1/min", "rad/s", "kW", "kVA", "kvar",
                                    "A", "bar", "l/h", "L/h", "L/min", "l/min", "%"})
_AMBIENT_KEYWORDS = frozenset(["outside temperature", "ambient temperature", "nacelle outside temperature"])


def _classify_sensor(row: pd.Series) -> str:
    """da la estrategia de sustitucion de ceros de un sensor."""

    # los contadores y los angulos se resuelven por bandera del registro, antes que por unidad
    if row["is_counter"]:
        return "keep_temporal"  # detecta los ceros congelados en operacion
    if row["is_angle"]:
        return "keep"  # 0 grados es un angulo valido y no hay contraste fiable

    unit = row["unit_norm"].strip()
    desc = row["description"].lower().strip()

    if unit == "m/s" or "wind speed" in desc:
        return "keep_cross_wind"  # se contrasta entre turbinas en la misma marca de tiempo

    # la temperatura ambiente se contrasta entre turbinas, la de un componente no
    if unit == "°C":
        if any(kw in desc for kw in _AMBIENT_KEYWORDS):
            return "keep_cross_temp"  # temperatura exterior, se contrasta entre turbinas
        return "always"  # una temperatura de componente de 0 grados es imposible

    # tension y frecuencia, imposibles con la red conectada
    if unit in _ALWAYS_REPLACE_UNITS:
        return "always"

    # sospechoso durante la operacion
    if unit in _CONTEXT_REPLACE_UNITS:
        return "context"

    return "context"  # el default es el context-aware, no el keep


class ZeroHandler:
    """sustituye por NaN los ceros que son ausencia e interpola los huecos pequenos."""

    def __init__(self, config: dict, sensor_registry: pd.DataFrame) -> None:
        cfg = config["zero_handling"]
        self.cut_in: float = cfg["wind_cut_in_ms"]
        self.max_linear: int = cfg["interpolation"]["max_gap_linear"]
        self.max_ffill: int = cfg["interpolation"]["max_gap_ffill"]
        self.cross_turbine_temp_threshold: float = cfg["cross_turbine_temp_threshold"]
        self.counter_freeze_min_samples: int = cfg["counter_freeze_min_samples"]
        self.max_gap_min: int = config["windowing"]["max_gap_minutes"]

        # {base_sensor_name: strategy}, se resuelve una vez y luego se consulta por columna
        self._strategy: dict[str, str] = {
            row["sensor_name"]: _classify_sensor(row)
            for _, row in sensor_registry.iterrows()
        }


    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """sustituye por NaN los ceros sospechosos e interpola los huecos pequenos."""

        df = df.copy()
        rows_before = len(df)

        sensor_cols = self._get_sensor_columns(df)

        # la mascara de operacion se calcula una vez y la reusa cada columna "context"
        wind_cols = [c for c in sensor_cols if c.endswith("_avg")
                     and "wind_speed" in col_to_sensor_name(c)]
        # sin columna de viento no hay forma de saber si giraba, y se asume que si
        if wind_cols:
            operational = df[wind_cols].mean(axis=1) > self.cut_in
        else:
            operational = pd.Series(True, index=df.index)

        zeros_replaced = 0

        for col in sensor_cols:
            strategy = self._strategy.get(col_to_sensor_name(col), "context")

            # las cuatro estrategias "keep" no se resuelven aqui, o no tocan el 0 o lo miran
            # despues con el contraste entre turbinas o con el de estabilidad temporal
            if strategy in ("keep", "keep_cross_wind", "keep_cross_temp", "keep_temporal"):
                continue

            mask_zero = df[col] == 0.0

            if strategy == "always":
                replaced = mask_zero.sum()
                df.loc[mask_zero, col] = np.nan
            else:  # "context"
                suspect = mask_zero & operational
                replaced = suspect.sum()
                df.loc[suspect, col] = np.nan

            zeros_replaced += int(replaced)

        n_cross = self._cross_turbine_replace(df, sensor_cols)
        n_temporal = self._temporal_stability_replace(df, sensor_cols, operational)
        zeros_replaced += n_cross + n_temporal

        logger.info(
            f"Zero handler: replaced {zeros_replaced:,} zero values with NaN "
            f"(cross-turbine: {n_cross:,} temporal: {n_temporal:,})"
        )

        df = self._interpolate_gaps(df, sensor_cols)

        stats = step_stats("zero_handling", rows_before, len(df), {
            "zeros_replaced": zeros_replaced,
            "zeros_replaced_cross_turbine": n_cross,
            "zeros_replaced_temporal": n_temporal,
        })

        return df, stats


    def _get_sensor_columns(self, df: pd.DataFrame) -> list[str]:
        """da todas las columnas numericas de sensor, dejando fuera las de metadatos."""

        meta = {"time_stamp", "asset_id", "id", "train_test", "status_type_id",
                "event_id", "event_label", "event_start", "event_end","event_description"}

        return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


    def _cross_turbine_replace(self, df: pd.DataFrame, sensor_cols: list[str]) -> int:
        """sustituye los ceros de los sensores de viento y temperatura ambiente cuando la
        mediana entre turbinas se aleja lo bastante de ellos."""

        # con una sola turbina no hay con quien contrastar
        if df["asset_id"].nunique() < 2:
            return 0

        replaced = 0
        for col in sensor_cols:
            strat = self._strategy.get(col_to_sensor_name(col))
            if strat not in ("keep_cross_wind", "keep_cross_temp"):
                continue

            # la mediana del parque en ese mismo instante, que es el testigo del contraste
            cross_median = df.groupby("time_stamp")[col].transform("median")

            if strat == "keep_cross_wind":
                # viento, 0 m/s es sospechoso si la mediana del parque supera el cut-in
                suspect = (df[col] == 0.0) & (cross_median > self.cut_in)
            else:
                # temperatura ambiente, 0 grados es sospechoso si la mediana se aleja de 0
                suspect = (df[col] == 0.0) & (cross_median.abs() > self.cross_turbine_temp_threshold)

            n = int(suspect.sum())
            if n > 0:
                df.loc[suspect, col] = np.nan
                replaced += n

        return replaced


    def _temporal_stability_replace(self, df: pd.DataFrame, sensor_cols: list[str], operational: pd.Series) -> int:
        """sustituye los ceros congelados de los contadores cuando la turbina estuvo
        operativa N muestras consecutivas."""

        counter_cols = [
            c for c in sensor_cols
            if self._strategy.get(col_to_sensor_name(c)) == "keep_temporal"
        ]
        if not counter_cols:
            return 0

        replaced = 0
        for _, group in df.groupby("asset_id", sort=False):
            group_sorted = group.sort_values("time_stamp")
            op_group = operational.loc[group_sorted.index]

            for col in counter_cols:
                vals = group_sorted[col]
                is_frozen_zero = (vals == 0.0) & op_group

                # cada cambio de valor abre una racha nueva, asi que el cumsum de los cambios
                # numera las rachas y el sum por racha da su longitud
                run_id = (is_frozen_zero != is_frozen_zero.shift()).cumsum()
                run_len = is_frozen_zero.groupby(run_id).transform("sum")
                suspect_idx = is_frozen_zero.index[
                    is_frozen_zero & (run_len >= self.counter_freeze_min_samples)
                ]

                n = len(suspect_idx)
                if n > 0:
                    df.loc[suspect_idx, col] = np.nan
                    replaced += n

        return replaced


    def _interpolate_gaps(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """por turbina y por segmento, interpolacion lineal y luego ffill; un hueco mas ancho
        que max_ffill se queda en NaN."""

        for _, group in df.groupby("asset_id", sort=False):
            group_sorted = group.sort_values("time_stamp")
            segments = find_continuous_segments(group_sorted, max_gap_minutes=self.max_gap_min)

            for seg_idx in segments:
                seg_len = len(seg_idx)
                if seg_len < 2:
                    continue  # en un segmento de una sola fila no hay nada que interpolar

                seg = df.loc[seg_idx, sensor_cols]

                # el limite se recorta a la longitud del segmento para no cruzar su frontera
                lin_limit = min(self.max_linear, seg_len - 1)
                seg = seg.interpolate(method="linear", limit=lin_limit, limit_direction="forward")

                # el ffill solo cubre lo que va de max_linear a max_ffill, no repite tramo
                remaining_limit = min(self.max_ffill - self.max_linear, seg_len - 1)
                if remaining_limit > 0:
                    seg = seg.ffill(limit=remaining_limit)

                df.loc[seg_idx, sensor_cols] = seg.values

        return df.sort_values(["asset_id", "time_stamp"]).reset_index(drop=True)
