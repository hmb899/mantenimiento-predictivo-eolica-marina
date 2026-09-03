"""normalizacion de temperatura, T_norm = T_component - T_ambient

quita la variacion estacional de la temperatura ambiente para que el modelo detecte el
calentamiento propio de cada componente y no las tendencias del clima

referencia, Barnabei et al. (2024)
"""
from __future__ import annotations

import pandas as pd

from .utils import col_to_sensor_name, get_logger, step_stats

logger = get_logger(__name__)


class TemperatureNormalizer:
    """resta la temperatura ambiente a cada feature de temperatura que no sea ambiente."""

    def __init__(self, config: dict, sensor_registry: pd.DataFrame) -> None:
        cfg = config["temperature_normalization"]
        self.enabled: bool = cfg["enabled"]
        self.ambient_kws: list[str] = [k.lower() for k in cfg["ambient_keywords"]]
        self.ambient_exclude_kws: list[str] = [k.lower() for k in cfg.get("ambient_exclude_keywords", [])]

        # el registro va por nombre base, la columna anade el sufijo de stat
        self._desc: dict[str, str] = {}
        self._unit: dict[str, str] = {}
        for _, row in sensor_registry.iterrows():
            name = row["sensor_name"]
            self._desc[name] = row["description"].lower()
            self._unit[name] = row.get("unit_norm", row["unit"]).strip()


    def transform(self, df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
        """aplica T_norm = T_component - T_ambient a las features de temperatura.
        Devuelve (df, stats)."""

        if not self.enabled:
            logger.info("TemperatureNormalizer: disabled — skipping.")
            return df, step_stats("temperature_normalization", len(df), len(df))

        # in situ para no quedarse sin memoria con el tamano de la Farm C
        ambient_cols = [c for c in feature_cols if self._is_ambient(c)]
        # las ambiente son la referencia, restarlas de si mismas daria cero
        temp_cols = [
            c for c in feature_cols
            if self._is_temperature(c) and c not in ambient_cols
        ]

        if not ambient_cols:
            logger.warning("TemperatureNormalizer: no ambient temperature column found — skipping.")
            return df, step_stats("temperature_normalization", len(df), len(df))

        # se toma como referencia la columna ambiente con menos NaNs
        ambient_col = min(ambient_cols, key=lambda c: df[c].isna().sum())
        t_amb = df[ambient_col]

        for col in temp_cols:
            df[col] = df[col] - t_amb

        logger.info(
            f"TemperatureNormalizer: normalised {len(temp_cols)} temperature columns "
            f"using [{ambient_col}] as ambient reference"
        )

        stats = step_stats(
            "temperature_normalization",
            len(df),
            len(df),
            {
                "ambient_reference_col": ambient_col,
                "n_cols_normalised": len(temp_cols),
                "normalised_cols": temp_cols,
            },
        )
        return df, stats


    def _is_temperature(self, col: str) -> bool:
        """devuelve True si la unidad o la descripcion apuntan a un sensor de temperatura."""

        base = col_to_sensor_name(col)
        unit = self._unit.get(base, "").lower()
        desc = self._desc.get(base, "")
        # robusto a los problemas de codificacion del grado
        # el tope de longitud frena una unidad larga acabada en c
        unit_is_temp = (unit.endswith("c") and len(unit) <= 8) or "celsius" in unit
        desc_is_temp = "temperature" in desc
        return unit_is_temp or desc_is_temp


    def _is_ambient(self, col: str) -> bool:
        """devuelve True si la columna es un sensor de temperatura ambiente exterior."""

        if not self._is_temperature(col):
            return False
        base = col_to_sensor_name(col)
        desc = self._desc.get(base, "")
        if not any(kw in desc for kw in self.ambient_kws):
            return False
        # p.ej. "electrical cabinet ambient" no es ambiente exterior
        if any(kw in desc for kw in self.ambient_exclude_kws):
            return False
        return True
