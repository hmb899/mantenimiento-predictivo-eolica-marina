"""exclusion de los periodos de transicion alrededor de los eventos de fallo

quita los datos temporalmente cercanos a un fallo, para que el NBM no aprenda las firmas de
degradacion previas a la averia ni la inestabilidad posterior a la reparacion

referencia, Barnabei et al. (2024), 24 h antes y 72 h despues del fallo. los 14 dias de
pre-exclusion del articulo CARE estan calibrados para la Farm A, que tiene fallos mas densos
y periodos anomalos mas largos, y resultan demasiado agresivos para las granjas B y C
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from .utils import get_logger, step_stats

logger = get_logger(__name__)


class TransitionExcluder:
    """quita las filas de la zona pre-fallo [event_start - hours_before, event_start) y de la
    post-reparacion (event_end, event_end + hours_after] de cada evento anomalo."""

    def __init__(self, config: dict) -> None:
        cfg = config["transition_exclusion"]
        self.td_before = timedelta(hours=cfg["hours_before_fault"])
        self.td_after = timedelta(hours=cfg["hours_after_fault"])


    def transform(self, df: pd.DataFrame, event_info: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """quita las filas del periodo de transicion de los fallos. Devuelve (df, stats)."""

        rows_before = len(df)
        # solo los eventos anomalos anclan exclusion, un normal no tiene fallo
        anomaly_events = event_info[event_info["event_label"] == "anomaly"].copy()

        if anomaly_events.empty:
            logger.info("TransitionExcluder: no anomaly events — nothing excluded.")
            return df.copy(), step_stats("transition_exclusion", rows_before, rows_before)

        exclude_mask, n_windows = self._build_exclude_mask(df, anomaly_events)

        df_out = df[~exclude_mask].copy()
        df_out.reset_index(drop=True, inplace=True)

        rows_after = len(df_out)
        h_before = int(self.td_before.total_seconds() // 3600)
        h_after = int(self.td_after.total_seconds() // 3600)
        logger.info(
            f"Transition exclusion: {rows_before - rows_after:,} rows removed "
            f"across {n_windows} fault events "
            f"({h_before}h before / {h_after}h after each fault)"
        )

        stats = step_stats(
            "transition_exclusion",
            rows_before,
            rows_after,
            {
                "hours_before_fault": self.td_before.total_seconds() / 3600,
                "hours_after_fault": self.td_after.total_seconds() / 3600,
                "anomaly_events_processed": int(len(anomaly_events)),
            },
        )
        return df_out, stats


    def _build_exclude_mask(self, df: pd.DataFrame, anomaly_events: pd.DataFrame) -> tuple[pd.Series, int]:
        """construye la mascara booleana de filas a excluir y cuenta los eventos afectados."""

        exclude_mask = pd.Series(False, index=df.index)
        n_windows = 0
        ts = df["time_stamp"]

        for _, evt in anomaly_events.iterrows():
            asset_id = evt["asset_id"]
            # las dos zonas se limitan a la turbina averiada, un fallo no viaja
            asset_mask = df["asset_id"] == asset_id

            # la fila del fallo se queda, la zona previa es abierta por la
            # derecha y la posterior por la izquierda
            pre_zone = asset_mask & (ts >= evt["event_start"] - self.td_before) & (ts < evt["event_start"])
            post_zone = asset_mask & (ts > evt["event_end"]) & (ts <= evt["event_end"] + self.td_after)

            n_pre = pre_zone.sum()
            n_post = post_zone.sum()
            # solo cuenta como afectado si quito filas, asi que n_windows puede
            # quedar por debajo de los eventos anomalos
            if n_pre + n_post > 0:
                n_windows += 1
                logger.info(
                    f"  Event {evt['event_id']} (asset {asset_id}): "
                    f"excluded {n_pre} pre-fault + {n_post} post-fault rows"
                )

            exclude_mask |= pre_zone | post_zone

        return exclude_mask, n_windows
