"""
metrica CARE, Coverage, Accuracy, Reliability y Earliness.

score combinado (seccion 4.1.1 y eq. 4 del articulo)
    WA = (1 / sum_i w_i) * (w1*F_beta + w2*WS + w3*EF_beta + w4*Acc)
con pesos w1 = 1 (Coverage), w2 = 1 (Earliness), w3 = 1 (Reliability), w4 = 2 (Accuracy).

CARE a trozos (eq. 5 del articulo)
    CARE := 0,    si no se detecto ninguna anomalia en ningun dataset de anomalia
             Acc,  si Acc < 0.5
             WA,   en otro caso
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coverage import coverage_score
from .accuracy import accuracy_score
from .reliability import reliability_score
from .earliness import earliness_score, weighted_score_single
from .labels import compute_earliness_weights, NORMAL_STATUS


@dataclass
class CAREResult:
    """contenedor de los cuatro sub-scores de CARE y del score combinado."""

    coverage: float
    earliness: float
    reliability: float
    accuracy: float
    care: float

    def __str__(self) -> str:
        return (
            f"CARE={self.care:.4f} "
            f"(Cov={self.coverage:.4f}, "
            f"Earl={self.earliness:.4f}, "
            f"Rel={self.reliability:.4f}, "
            f"Acc={self.accuracy:.4f})"
        )


def weighted_average(coverage: float, earliness: float, reliability: float, accuracy: float,
                     w_coverage: float = 1.0, w_earliness: float = 1.0,
                     w_reliability: float = 1.0, w_accuracy: float = 2.0) -> float:
    """media ponderada de los cuatro sub-scores (eq. 4)."""

    # el peso 2 de Accuracy hace que la metrica castigue la falsa alarma sobre lo demas
    total_w = w_coverage + w_earliness + w_reliability + w_accuracy
    return float(
        (w_coverage * coverage
         + w_earliness * earliness
         + w_reliability * reliability
         + w_accuracy * accuracy) / total_w
    )


def compute_care(anomaly_datasets: list[dict], normal_datasets: list[dict], beta: float = 0.5,
                 w_coverage: float = 1.0, w_earliness: float = 1.0, w_reliability: float = 1.0,
                 w_accuracy: float = 2.0, tc: int = 72) -> CAREResult:
    """el CARE Score completo a partir de entradas de dataset que llegan ya filtradas."""

    # Coverage, F_beta sobre los puntos de prediccion en estado normal
    cov_pairs = [
        (ds["y_true_normal_pred"], ds["y_pred_normal_pred"])
        for ds in anomaly_datasets
    ]
    cov = coverage_score(cov_pairs, beta=beta)

    # Earliness, WS sobre todas las marcas de tiempo de la ventana del evento
    earl_pairs = [
        (ds["y_pred_event"], ds["weights_event"])
        for ds in anomaly_datasets
    ]
    earl = earliness_score(earl_pairs)

    # Reliability, EF_beta con el Criticality Algorithm
    rel_anomaly = [
        (ds["y_pred_all_pred"], ds["statuses_all_pred"])
        for ds in anomaly_datasets
    ]
    rel_normal = [
        (ds["y_pred_all_pred"], ds["statuses_all_pred"])
        for ds in normal_datasets
    ]
    rel = reliability_score(rel_anomaly, rel_normal, beta=beta, tc=tc)

    # Accuracy, los puntos de estado normal de los datasets normales
    normal_preds = [ds["y_pred_normal_pred"] for ds in normal_datasets]
    acc = accuracy_score(normal_preds)

    # CARE a trozos (eq. 5)
    # sin una deteccion el score es 0 aunque Accuracy sea perfecta, cierra al detector mudo
    any_anomaly_detected = any(
        np.any(np.asarray(ds["y_pred_all_pred"]) == 1)
        for ds in anomaly_datasets
    )
    if not any_anomaly_detected:
        return CAREResult(coverage=cov, earliness=earl, reliability=rel,
                          accuracy=acc, care=0.0)

    # por debajo de 0.5 de Accuracy la media ponderada no se usa, manda la Accuracy sola
    if acc < 0.5:
        return CAREResult(coverage=cov, earliness=earl, reliability=rel,
                          accuracy=acc, care=acc)

    wa = weighted_average(cov, earl, rel, acc,
                          w_coverage, w_earliness, w_reliability, w_accuracy)
    return CAREResult(coverage=cov, earliness=earl, reliability=rel,
                      accuracy=acc, care=wa)


def prepare_dataset_inputs(pred_df, event_start_id: int, event_end_id: int,
                           y_pred_all: np.ndarray, status_col: str = "status_type_id",
                           row_id_col: str = "id") -> dict:
    """dict de entrada de un dataset de anomalia."""

    # y_pred_all llega alineado fila a fila con pred_df, ese contrato lo garantiza quien llama
    ids = pred_df[row_id_col].to_numpy()
    statuses = pred_df[status_col].to_numpy()

    # dos poblaciones distintas, Coverage mira las filas de estado normal de toda la particion
    # de prediccion y Earliness mira la ventana del evento entera, estado incluido
    normal_mask = np.isin(statuses, list(NORMAL_STATUS))
    event_mask = (ids >= event_start_id) & (ids <= event_end_id)

    # en un dataset anomalo toda fila de estado normal cuenta como anomala, de ahi el 1
    normal_pred_idx = np.where(normal_mask)[0]
    y_pred_normal_pred = y_pred_all[normal_pred_idx]
    y_true_normal_pred = np.ones(len(y_pred_normal_pred), dtype=int)

    event_idx = np.where(event_mask)[0]
    y_pred_event = y_pred_all[event_idx]
    weights_event = compute_earliness_weights(len(y_pred_event))

    return {
        "y_pred_normal_pred": y_pred_normal_pred,
        "y_true_normal_pred": y_true_normal_pred,
        "event_mask_in_pred": event_mask,
        "y_pred_all_pred": y_pred_all,
        "statuses_all_pred": statuses,
        "weights_event": weights_event,
        "y_pred_event": y_pred_event,
    }


def prepare_normal_dataset_inputs(y_pred_all: np.ndarray, statuses: np.ndarray) -> dict:
    """dict de entrada de un dataset normal."""

    y_pred_all = np.asarray(y_pred_all, dtype=int)
    statuses = np.asarray(statuses, dtype=int)

    # sin ventana de evento, el normal solo da falsos positivos, a Accuracy y a Reliability
    normal_mask = np.isin(statuses, list(NORMAL_STATUS))
    y_pred_normal_pred = y_pred_all[normal_mask]

    return {
        "y_pred_all_pred": y_pred_all,
        "statuses_all_pred": statuses,
        "y_pred_normal_pred": y_pred_normal_pred,
    }
