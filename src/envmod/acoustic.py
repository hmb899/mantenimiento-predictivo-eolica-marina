"""
modelo acustico simplificado de la emision operacional

fisica, de he_editorial_2023 y de las mediciones de Yoon et al. en una turbina de 3 MW. el
ruido submarino operacional se concentra por debajo de 1 kHz, con nivel de 80 a 150 dB re
1 uPa a 1 m, correlacion >=0.95 con el viento y la velocidad de rotor, y +20 dB en el tonal
de 198 Hz al
alcanzar la velocidad nominal de rotacion. se modela como una curva monotona del viento
normalizado, con una rampa lineal de nivel_min a (nivel_max - incremento) hasta el regimen
nominal, mas un salto sigmoide de amplitud incremento centrado en el nominal. los parametros
viven en el bloque emision_operacional de kb/especies.yaml, y los que no son citables van
marcados assumption.

convencion de wind_norm, viento normalizado [0, 1] del generador de escenarios, donde 0 es
calma y 1 es cut-out ~25 m/s, con el nominal en ~0.5, unos 12 m/s, que es assumption.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

KB_PATH = Path(__file__).resolve().parent / "kb" / "especies.yaml"


@lru_cache(maxsize=2)
def emision_params(path: Path = KB_PATH) -> dict:
    """bloque emision_operacional de la KB, la fuente de verdad de los parametros."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["emision_operacional"]


def emission_level_db(wind_norm, params: dict | None = None):
    """nivel de emision (dB re 1 uPa @ 1 m) en funcion del viento normalizado [0, 1].

    acepta escalar o array y devuelve lo mismo. es monotona no decreciente por construccion,
    con emission_level_db(0) ~ nivel_min y emission_level_db(1) ~ nivel_max, y el +incremento
    se completa justo al superar viento_nominal_norm, que es el tonal de 198 Hz.
    """

    p = params if params is not None else emision_params()
    w = np.clip(np.asarray(wind_norm, dtype=float), 0.0, 1.0)
    lo, hi = p["nivel_min_db"], p["nivel_max_db"]
    inc, v_nom = p["incremento_nominal_db"], p["viento_nominal_norm"]
    # la rampa se satura en el nominal, de ahi el minimo con 1.0, y deja sitio al salto
    base = lo + (hi - inc - lo) * np.minimum(w / v_nom, 1.0)
    # sigmoide centrada en el nominal, la anchura decide como de brusco es el salto del tonal
    salto = inc / (1.0 + np.exp(-(w - v_nom) / p["anchura_transicion_norm"]))
    out = base + salto
    # se devuelve escalar si entro escalar, para que el llamador no tenga que desempaquetar
    return float(out) if np.isscalar(wind_norm) else out


def noise_hours_in_band(scada_wind_norm_series, sensibilidad_periodo, dt_horas: float = 1.0,
                        params: dict | None = None) -> float:
    """horas-ruido ponderadas de un tramo de operacion, sum( dt * nivel_normalizado * sens ).

    es el termino de ruido del objetivo ambiental f3 (scheduler/problem.py). el
    nivel_normalizado es (nivel - nivel_min) / (nivel_max - nivel_min) en [0, 1], y
    sensibilidad_periodo puede ser un escalar, si el periodo es homogeneo, o un array por
    muestra (env_tools.sensibilidad_dia). la unidad son horas-equivalentes de ruido,
    adimensionalizadas y comparables entre planes.
    """

    p = params if params is not None else emision_params()
    lo, hi = p["nivel_min_db"], p["nivel_max_db"]
    niveles = emission_level_db(np.asarray(scada_wind_norm_series, dtype=float), p)
    # el nivel se lleva a [0, 1] antes de ponderar, para que dB y sensibilidad sean
    # multiplicables
    nivel_norm = (niveles - lo) / (hi - lo)
    return float(np.sum(nivel_norm * np.asarray(sensibilidad_periodo, dtype=float) * dt_horas))
