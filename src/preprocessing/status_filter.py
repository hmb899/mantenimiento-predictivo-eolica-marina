"""filtra las filas por Status_ID y deja solo los estados normales de operacion para entrenar

articulo CARE (Guck et al., 2024)
    status 0, operacion normal, se conserva
    status 2, marcha en vacio, ausente en las granjas B y C, confirmado por el EDA
    status 1, operacion limitada, se excluye
    status 3, servicio o mantenimiento, se excluye
    status 4, parada por fallo, se excluye y sirve de ancla a la exclusion de transitorios
    status 5, otros o sin definir, se excluye
"""
from __future__ import annotations

import pandas as pd

from .utils import get_logger, step_stats

logger = get_logger(__name__)


class StatusFilter:
    """conserva solo las filas con un status_type_id normal, para entrenar el NBM."""

    def __init__(self, config: dict) -> None:
        self.normal_statuses: list[int] = config["status_filtering"]["normal_statuses"]


    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """conserva las filas cuyo status_type_id esta en normal_statuses.
        Devuelve (df, stats)."""

        rows_before = len(df)
        # la distribucion se toma antes de filtrar, va al informe
        status_dist = df["status_type_id"].value_counts().to_dict()

        df_filtered = df[df["status_type_id"].isin(self.normal_statuses)].copy()
        df_filtered.reset_index(drop=True, inplace=True)

        rows_after = len(df_filtered)
        logger.info(
            f"Status filter: {rows_before:,} → {rows_after:,} rows "
            f"({rows_before - rows_after:,} removed, "
            f"kept statuses {self.normal_statuses})"
        )

        stats = step_stats(
            "status_filtering",
            rows_before,
            rows_after,
            {"status_distribution_before": status_dist},
        )
        return df_filtered, stats
