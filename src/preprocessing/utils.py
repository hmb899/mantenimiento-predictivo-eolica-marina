"""utilidades compartidas, log, informe por paso, deteccion de segmentos y E/S de ficheros"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """devuelve un logger de modulo con un formato uniforme."""

    logger = logging.getLogger(name)
    # la guarda evita que un reimport apile un segundo handler
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def step_stats(step: str, rows_before: int, rows_after: int, extra: dict[str, Any] | None = None) -> dict:
    """construye el diccionario de estadisticas estandar de un paso del pipeline."""

    stats: dict[str, Any] = {
        "step": step,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        # el max(.., 1) deja definido el porcentaje con un frame vacio
        "pct_removed": round(100.0 * (rows_before - rows_after) / max(rows_before, 1), 3),
    }
    if extra:
        stats.update(extra)
    return stats


def find_continuous_segments(df: pd.DataFrame, max_gap_minutes: int = 15,
                             timestamp_col: str = "time_stamp") -> list[pd.Index]:
    """parte un DataFrame ya ordenado en segmentos temporales continuos, cortando
    donde el hueco supera max_gap_minutes."""

    if df.empty:
        return []

    times = df[timestamp_col]
    gaps = times.diff().dt.total_seconds() / 60  # minutos
    # los centinelas 0 y len(df) cierran los cortes en pares inicio-fin
    breaks = [0] + list((gaps > max_gap_minutes).values.nonzero()[0]) + [len(df)]

    segments = []
    for start, end in zip(breaks[:-1], breaks[1:]):
        idx = df.index[start:end]
        if len(idx) > 0:
            segments.append(idx)

    return segments


def col_to_sensor_name(col: str) -> str:
    """quita el sufijo de estadistico para dejar el nombre base del sensor,
    sensor_32_avg da sensor_32."""

    parts = col.split("_")
    return "_".join(parts[:-1])


def save_report(report: dict, path: Path) -> None:
    """serializa report a un fichero JSON y crea los directorios padre si hacen falta."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """serializador de reserva para los tipos que json.dump no sabe tratar por si solo."""

    # numpy y pandas llegan aqui, json solo conoce los tipos nativos
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
