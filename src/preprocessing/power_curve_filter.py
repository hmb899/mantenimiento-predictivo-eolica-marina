"""filtro de coherencia con la curva de potencia

quita las filas de Status-0 donde la velocidad de viento supera el cut-in pero la potencia
entregada es casi nula. esas filas suelen arrastrar una etiqueta de estado obsoleta y no
deberian usarse para entrenar el Normal Behavior Model

referencia, articulo CARE (Guck et al., 2024), paso de verificacion de la curva de potencia
"""
from __future__ import annotations

import pandas as pd

from .utils import col_to_sensor_name, get_logger, step_stats

logger = get_logger(__name__)


class PowerCurveFilter:
    """quita las filas con velocidad de viento alta y potencia activa casi nula."""

    def __init__(self, config: dict, sensor_registry: pd.DataFrame) -> None:
        cfg = config["power_curve_filter"]
        self.wind_threshold = cfg["wind_threshold_ms"]
        self.min_power_kw = cfg["min_power_kw"]
        self.use_relative = cfg.get("use_relative_threshold", False)
        self.min_power_pct = cfg.get("min_power_pct", 2.0)
        self._registry = sensor_registry


    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """quita las filas sospechosas, viento alto con potencia casi nula.
        Devuelve (df, stats)."""

        rows_before = len(df)

        wind_col = self._find_col(df, self._wind_candidates(df))
        power_col = self._find_col(df, self._power_candidates(df))

        # sin las dos columnas no hay curva, el paso deja pasar el frame
        if wind_col is None or power_col is None:
            logger.warning("PowerCurveFilter: could not identify wind / power columns — skipping.")
            return df.copy(), step_stats("power_curve_filter", rows_before, rows_before)

        high_wind = df[wind_col] > self.wind_threshold

        # el umbral relativo se adapta a la escala de los datos
        if self.use_relative:
            # p99 y no el maximo, un outlier no debe fijar la referencia
            p99 = df[power_col].quantile(0.99)
            min_power = p99 * (self.min_power_pct / 100.0)
            logger.info(
                f"  Relative threshold: {self.min_power_pct}% of p99 "
                f"({p99:.4f}) = {min_power:.4f}"
            )
        else:
            min_power = self.min_power_kw

        low_power = df[power_col] < min_power
        # sospechoso solo con las dos juntas, por separado cada una es normal
        suspect = high_wind & low_power

        df_out = df[~suspect].copy()
        df_out.reset_index(drop=True, inplace=True)

        rows_after = len(df_out)
        logger.info(
            f"Power-curve filter: {suspect.sum():,} rows removed "
            f"(wind > {self.wind_threshold} m/s AND power < {min_power:.4f}) "
            f"using [{wind_col}] vs [{power_col}]"
        )

        stats = step_stats(
            "power_curve_filter",
            rows_before,
            rows_after,
            {
                "wind_col": wind_col,
                "power_col": power_col,
                "wind_threshold_ms": self.wind_threshold,
                "min_power_threshold": min_power,
                "threshold_mode": "relative" if self.use_relative else "absolute",
            },
        )
        return df_out, stats


    def _wind_candidates(self, df: pd.DataFrame) -> list[str]:
        """devuelve las columnas _avg cuya descripcion de sensor contiene 'wind speed'."""

        return self._cols_by_keyword(df, ["wind speed"], stat="avg")


    def _power_candidates(self, df: pd.DataFrame) -> list[str]:
        """devuelve las columnas _avg de potencia activa, con el kW instantaneo por delante
        de los contadores de energia."""

        candidates = self._cols_by_keyword(df, ["active power"], stat="avg")
        # se descartan los contadores (kWh) en favor del kW instantaneo
        # un contador solo crece, nunca pareceria potencia casi nula
        kw_candidates = []
        for col in candidates:
            base = col_to_sensor_name(col)
            row = self._registry[self._registry["sensor_name"] == base]
            if row.empty:
                continue
            unit = row.iloc[0]["unit"].strip().lower()
            is_counter = bool(row.iloc[0]["is_counter"])
            if not is_counter and ("kwh" not in unit):
                kw_candidates.append(col)
        # mejor caer en la lista completa que saltarse el paso
        return kw_candidates if kw_candidates else candidates


    def _cols_by_keyword(self, df: pd.DataFrame, keywords: list[str], stat: str = "avg") -> list[str]:
        """devuelve las columnas terminadas en _{stat} cuya descripcion en el registro casa
        con alguna palabra clave."""

        results = []
        for col in df.columns:
            if not col.endswith(f"_{stat}"):
                continue
            # el registro va por nombre base, la columna lleva el sufijo
            base = col_to_sensor_name(col)
            row = self._registry[self._registry["sensor_name"] == base]
            if row.empty:
                continue
            desc = row.iloc[0]["description"].lower()
            if any(kw in desc for kw in keywords):
                results.append(col)
        return results


    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        """elige la mejor columna de la lista de candidatas, la de menor tasa de NaN."""

        if not candidates:
            return None
        available = [c for c in candidates if c in df.columns]
        if not available:
            return None
        return min(available, key=lambda c: df[c].isna().sum())
