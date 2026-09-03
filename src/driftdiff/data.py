"""
helpers de datos para D3R (Modelo A), funciones puras de features de timestamp y de target

el montaje (preprocess_rows + RowReducer + Windower del repo) vive en
scripts/run_driftdiff_c.py, con el patron de get_components de run_timeaddm_c.py; aqui solo va
lo que no depende de run_model, para mantener src/ limpio de imports de scripts

convenio D3R (data/preprocess.py del repo original)
    time features, minute/hour/weekday/day/month normalizados ~[-0.5, 0.5]
    target stable, la serie menos su mediana movil centrada. en el original la mediana es
    global por columna; aqui se calcula por turbina y por segmento continuo, respetando huecos,
    con la misma segmentacion que el Windower via find_continuous_segments, para no puentear
    gaps. `period` va en muestras de 10 min
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.utils import find_continuous_segments


def time_features(timestamps) -> np.ndarray:
    """(N,) datetime a (N, 5) features de calendario normalizadas ~[-0.5, 0.5].

    nota de fidelidad, el repo D3R usa month/365, que es una rareza; aqui month/12, mas
    sensato. el resto va igual.
    """

    t = pd.to_datetime(pd.Series(np.asarray(timestamps).ravel()))
    # cada campo se divide por su maximo y se centra, para que todos caigan en el mismo rango
    feats = np.stack([
        t.dt.minute.to_numpy() / 59.0 - 0.5,
        t.dt.hour.to_numpy() / 23.0 - 0.5,
        t.dt.weekday.to_numpy() / 6.0 - 0.5,
        t.dt.day.to_numpy() / 30.0 - 0.5,
        t.dt.month.to_numpy() / 12.0 - 0.5,
    ], axis=1)
    return feats.astype(np.float32)


def compute_stable_target(comps: np.ndarray, meta_df: pd.DataFrame, period: int,
                          max_gap_minutes: int) -> np.ndarray:
    """target de descomposicion, comps menos su mediana movil centrada, por turbina y segmento.

    comps (N, K) son los componentes a nivel fila, alineados con meta_df por posicion; meta_df
    lleva las columnas 'asset_id' y 'time_stamp' con RangeIndex 0..N-1; period es la ventana de
    la mediana movil en muestras, por ejemplo 144 = un dia a 10 min, y max_gap_minutes es el
    mismo umbral de hueco que usa el Windower para los segmentos continuos. devuelve stable
    (N, K) float32 y sin NaN, porque min_periods=1.
    """

    n, k = comps.shape
    stable = np.zeros((n, k), dtype=np.float32)
    cols = [f"_c{i}" for i in range(k)]
    # turbina a turbina y segmento a segmento, para que la mediana movil no cruce un hueco
    for _asset, g in meta_df.groupby("asset_id", sort=False):
        g = g.sort_values("time_stamp")
        for seg in find_continuous_segments(g, max_gap_minutes=max_gap_minutes):
            # seg trae posiciones del DataFrame, que valen tal cual como indice de comps
            pos = np.asarray(seg)
            sub = pd.DataFrame(comps[pos], columns=cols)
            trend = sub.rolling(period, center=True, min_periods=1).median()
            stable[pos] = (sub - trend).to_numpy(np.float32)
    return stable


def split_channels(windows: np.ndarray, sizes: dict[str, int]) -> dict[str, np.ndarray]:
    """divide un array ventaneado (N, W, sum(sizes)) en bloques nombrados por canal."""

    # el orden del dict es el contrato, los bloques se cortan uno tras otro sin solaparse
    out, off = {}, 0
    for name, s in sizes.items():
        out[name] = windows[:, :, off:off + s]
        off += s
    return out
