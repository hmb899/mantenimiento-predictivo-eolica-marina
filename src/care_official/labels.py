"""
constantes de status-id y extraccion de las ventanas de evento y de prediccion.

la definicion de estado normal viene directa del articulo (seccion 2.2), el status 0 (Normal
Operation) y el status 2 (Idling) son los unicos estados que cuentan como operacion normal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# seccion 2.2 del articulo y README, solo los status 0 y 2 cuentan como normal
NORMAL_STATUS: frozenset[int] = frozenset({0, 2})

# etiquetas de estado legibles, para documentacion y figuras
STATUS_LABELS: dict[int, str] = {
    0: "Normal Operation",
    1: "Derated Operation",
    2: "Idling",
    3: "Service",
    4: "Downtime",
    5: "Other",
}


@dataclass(frozen=True)
class EventWindow:
    """metadatos de la ventana de prediccion y de evento de un dataset."""

    event_id: int
    farm: str
    asset_id: int
    event_label: str
    event_start_id: int
    event_end_id: int
    event_description: str

    @property
    def is_anomaly(self) -> bool:
        return self.event_label == "anomaly"


def is_normal_status(status_series: pd.Series) -> pd.Series:
    """mascara booleana, True donde el status esta en NORMAL_STATUS."""

    return status_series.isin(NORMAL_STATUS)


def extract_prediction_split(df: pd.DataFrame, split_col: str = "train_test",
                             prediction_value: str = "prediction") -> pd.DataFrame:
    """solo las filas de prediction de un DataFrame de sub-dataset."""

    mask = df[split_col] == prediction_value
    return df.loc[mask].copy()


def extract_event_rows(pred_df: pd.DataFrame, event: EventWindow,
                       row_id_col: str = "id") -> pd.DataFrame:
    """las filas dentro de la ventana del evento [event_start_id, event_end_id]."""

    mask = (pred_df[row_id_col] >= event.event_start_id) & (
        pred_df[row_id_col] <= event.event_end_id
    )
    return pred_df.loc[mask].copy()


def get_normal_mask_in_prediction(pred_df: pd.DataFrame,
                                  status_col: str = "status_type_id") -> np.ndarray:
    """array booleano, True en las filas de prediction con estado normal (0 o 2)."""

    return is_normal_status(pred_df[status_col]).to_numpy()


def compute_earliness_weights(n_points: int) -> np.ndarray:
    """funcion de peso lineal del sub-score Earliness (WS), seccion 3.2.1 y figura 1 del
    articulo. la primera mitad de la ventana del evento pesa 1.0 y la segunda cae
    linealmente de 1.0 a 0.0, porque el final de la ventana, event_end_id, coincide con el
    arranque del fallo.
    """

    if n_points == 0:
        return np.array([], dtype=float)

    weights = np.ones(n_points, dtype=float)

    if n_points == 1:
        return weights

    # con n impar el punto que sobra cae en la segunda mitad, la que decae
    half = n_points // 2
    second_half_len = n_points - half

    # con un solo punto en la segunda mitad linspace no tiene rango, se le da el 0.0 del final
    if second_half_len == 1:
        weights[half] = 0.0
    else:
        weights[half:] = np.linspace(1.0, 0.0, second_half_len, endpoint=True)

    return weights
