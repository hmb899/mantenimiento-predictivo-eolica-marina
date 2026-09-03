"""
seleccion de features para el Normal Behavior Model

cuatro pasos en orden, restringir a las columnas del estadistico _avg, tirar las features con
mas de un 30% de NaN (el umbral min_valid_ratio), tirar las casi constantes (coeficiente de
variacion por debajo de 0.01) y quitar una de cada pareja muy correlada (|r| por encima del
umbral), conservando las features priority, que casan por una palabra clave de su descripcion.

segun Turnbull (2021), Barnabei (2024), Jankauskas (2024) y Castellani (2024).
"""
from __future__ import annotations

import re

import pandas as pd

from .utils import col_to_sensor_name, get_logger

logger = get_logger(__name__)

# columnas de metadatos, nunca pueden salir seleccionadas como feature
_META_COLS = frozenset(
    [
        "time_stamp", "asset_id", "id", "train_test", "status_type_id",
        "event_id", "event_label", "event_start", "event_end","event_description",
    ]
)


class FeatureSelector:
    """elige un conjunto de features compacto e informativo entre columnas de estadisticos."""

    def __init__(self, config: dict, sensor_registry: pd.DataFrame) -> None:
        cfg = config["feature_selection"]
        self.statistics: list[str] = cfg["statistics"]
        self.extra_statistics: list[str] = cfg.get("extra_statistics", [])
        self.extra_stat_keywords: list[str] = cfg.get("extra_statistics_keywords", [])
        self.priority_keywords: list[str] = cfg["priority_keywords"]
        self.corr_threshold: float = cfg["correlation_threshold"]
        self.min_valid: float = cfg["min_valid_ratio"]
        self.max_features: int | None = cfg.get("max_features", None)
        self.forced_features: list[str] = cfg.get("forced_features", [])

        # una sola pasada por el registro deja resueltas las dos consultas por sensor
        self._desc: dict[str, str] = {}
        _counters: set[str] = set()

        for _, row in sensor_registry.iterrows():
            self._desc[row["sensor_name"]] = row["description"].lower()

            if row["is_counter"]:
                _counters.add(row["sensor_name"])

        # los contadores se excluyen, su _avg es energia incremental, no senal de operacion
        self._counters: frozenset[str] = frozenset(_counters)

        self.selected_features_: list[str] = []


    def select(self, df_normal: pd.DataFrame) -> tuple[list[str], dict]:
        """corre la seleccion de features completa sobre las filas de operacion normal."""

        candidates = self._get_candidates(df_normal)
        n_start = len(candidates)

        candidates, dropped_nan = self._filter_nan(df_normal, candidates)
        candidates, dropped_const = self._filter_constant(df_normal, candidates)

        # las features priority sobreviven al podado por correlacion y al tope de max_features
        priority = {c for c in candidates if self._is_priority(c)}

        candidates, dropped_corr = self._filter_correlated(df_normal, candidates, priority)

        # las forced que haya tirado el podado por correlacion se reponen
        forced_readded: list[str] = []
        for col in self.forced_features:
            if col in df_normal.columns and col not in candidates:
                candidates.append(col)
                forced_readded.append(col)
                # se marcan tambien como priority, para que el tope no vuelva a tirarlas
                priority.add(col)
        if forced_readded:
            logger.info(f"Forced features re-injected after correlation pruning: {forced_readded}")

        # tope duro, max_features como mucho y las priority primero
        dropped_cap: list[str] = []
        if self.max_features is not None and len(candidates) > self.max_features:
            candidates, dropped_cap = self._apply_max_features(df_normal, candidates, priority & set(candidates))

        self.selected_features_ = candidates

        stats = {
            "step": "feature_selection",
            "n_start": n_start,
            "n_high_nan_dropped": len(dropped_nan),
            "n_constant_dropped": len(dropped_const),
            "n_correlated_dropped": len(dropped_corr),
            "n_forced_readded": len(forced_readded),
            "n_cap_dropped": len(dropped_cap),
            "n_selected": len(candidates),
            "priority_features": sorted(priority & set(candidates)),
            "forced_readded": sorted(forced_readded),
            "dropped_high_nan": sorted(dropped_nan),
            "dropped_constant": sorted(dropped_const),
            "dropped_correlated": sorted(dropped_corr),
            "dropped_cap": sorted(dropped_cap),
        }

        logger.info(
            f"Feature selector: {n_start} --> {len(candidates)} features "
            f"(removed {len(dropped_nan)} NaN + {len(dropped_const)} const + "
            f"{len(dropped_corr)} corr + {len(dropped_cap)} cap"
            + (f" | re-injected {len(forced_readded)} forced" if forced_readded else "")
            + f" | {len(priority & set(candidates))} priority kept)"
        )
        return candidates, stats


    def _get_candidates(self, df: pd.DataFrame) -> list[str]:
        """columnas candidatas iniciales, los estadisticos primarios de cada sensor mas los
        extra (max, min, std) de los sensores que casan con extra_stat_keywords."""

        cols = []
        for stat in self.statistics:
            cols += [
                c for c in df.columns
                if c not in _META_COLS
                and c.endswith(f"_{stat}")
                and pd.api.types.is_numeric_dtype(df[c])
                and col_to_sensor_name(c) not in self._counters
            ]

        # segunda pasada, mucho mas selectiva, solo para los sensores que casan por descripcion
        if self.extra_statistics and self.extra_stat_keywords:
            for stat in self.extra_statistics:
                for c in df.columns:
                    if c in _META_COLS or not c.endswith(f"_{stat}"):
                        continue
                    if not pd.api.types.is_numeric_dtype(df[c]):
                        continue
                    base = col_to_sensor_name(c)
                    if base in self._counters:
                        continue
                    desc = self._desc.get(base, "")
                    if any(re.search(kw, desc, re.IGNORECASE) for kw in self.extra_stat_keywords):
                        cols.append(c)

        return list(dict.fromkeys(cols))  # deduplica conservando el orden de aparicion


    def _is_priority(self, col: str) -> bool:
        """True si la descripcion del sensor de la columna casa con una palabra clave priority,
        o si es una columna de estadistico extra, el max, min o std de viento o potencia."""

        base = col_to_sensor_name(col)
        desc = self._desc.get(base, "")
        if any(re.search(kw, desc, re.IGNORECASE) for kw in self.priority_keywords):
            return True
        # una columna de estadistico extra tambien es priority, el podado no se la lleva
        if self.extra_stat_keywords:
            stat_suffix = col.rsplit("_", 1)[-1]
            if stat_suffix in self.extra_statistics:
                return any(re.search(kw, desc, re.IGNORECASE) for kw in self.extra_stat_keywords)
        return False


    def _filter_nan(self, df: pd.DataFrame, candidates: list[str]) -> tuple[list[str], list[str]]:
        """tira las candidatas con menos valores no NaN que min_valid_ratio."""

        kept, dropped = [], []
        for col in candidates:
            valid_ratio = 1.0 - df[col].isna().mean()
            if valid_ratio >= self.min_valid:
                kept.append(col)
            else:
                dropped.append(col)
        return kept, dropped


    def _filter_constant(self, df: pd.DataFrame, candidates: list[str]) -> tuple[list[str], list[str]]:
        """tira las candidatas casi constantes, coeficiente de variacion por debajo de 0.01."""

        kept, dropped = [], []
        for col in candidates:
            series = df[col].dropna()
            if series.empty:
                dropped.append(col)
                continue
            # con la media casi en cero el cociente se dispara, y se usa la std cruda
            std = series.std()
            mean = abs(series.mean())
            cv = std / mean if mean > 1e-9 else std
            if cv < 0.01:
                dropped.append(col)
            else:
                kept.append(col)
        return kept, dropped


    def _apply_max_features(self, df: pd.DataFrame, candidates: list[str], priority: set[str]) -> tuple[list[str], list[str]]:
        """recorta las candidatas a max_features, primero las priority y las
        max_features - n_priority plazas restantes con las mas completas del resto."""

        if len(priority) >= self.max_features:
            # ni las priority caben, se ordenan por NaN ascendente y se cortan; el nombre de
            # columna como segundo criterio hace deterministas los empates
            priority_sorted = sorted(priority, key=lambda c: (df[c].isna().sum(), c))
            kept = priority_sorted[: self.max_features]
            dropped = [c for c in candidates if c not in kept]

            logger.warning(
                f"max_features={self.max_features} is smaller than the number of "
                f"priority features ({len(priority)}); trimming priority features "
                f"by ascending NaN count."
            )

        else:
            # el resto entra por NaN ascendente, o sea el mas completo primero
            non_priority = [c for c in candidates if c not in priority]
            non_priority_sorted = sorted(non_priority, key=lambda c: (df[c].isna().sum(), c))

            n_extra = self.max_features - len(priority)

            kept_priority = [c for c in candidates if c in priority]
            kept_non_priority = non_priority_sorted[:n_extra]
            kept_set = set(kept_priority) | set(kept_non_priority)
            kept = [c for c in candidates if c in kept_set]  # conserva el orden original

            dropped = [c for c in candidates if c not in kept_set]

        return kept, dropped


    def _filter_correlated(self, df: pd.DataFrame, candidates: list[str], priority: set[str]) -> tuple[list[str], list[str]]:
        """podado voraz por parejas, de cada pareja muy correlada se tira una y se conservan
        las features priority."""

        if len(candidates) < 2:
            return candidates, []

        corr = df[candidates].corr().abs()
        to_drop: set[str] = set()

        # solo el triangulo superior, cada pareja se mira una vez
        for i, col_i in enumerate(candidates):
            if col_i in to_drop:
                continue
            for col_j in candidates[i + 1 :]:
                if col_j in to_drop:
                    continue
                if corr.loc[col_i, col_j] > self.corr_threshold:
                    # manda la marca de priority y, si empatan, la que tiene menos NaN
                    i_priority = col_i in priority
                    j_priority = col_j in priority

                    if i_priority and not j_priority:
                        to_drop.add(col_j)
                    elif j_priority and not i_priority:
                        to_drop.add(col_i)
                    else:
                        # empate de priority, se queda la de menos NaN
                        nan_i = df[col_i].isna().sum()
                        nan_j = df[col_j].isna().sum()
                        to_drop.add(col_j if nan_j >= nan_i else col_i)

        kept = [c for c in candidates if c not in to_drop]
        dropped = sorted(to_drop)
        return kept, dropped
