"""
escenarios meteorologicos de accesibilidad (F3)

interfaz unica WeatherProvider, matriz (n_escenarios, horizonte_dias) de bool donde True es dia
accesible para CTV. dos implementaciones, ambas deterministas dada la seed (reproducibilidad)

Era5Provider
    el elegido, ERA5 real (German Bight 2014-2023, viento 10m y Hs) con bootstrap estacional
    por bloques, copia dias historicos consecutivos (persistencia real) anclados al mes y dia
    (estacionalidad) de un ano al azar (diversidad), sin forma parametrica, datos reales
SyntheticSeasonalProvider
    fallback citable, degrada sin ERA5, Markov de 2 estados con acceso estacional (mas ventanas
    en primavera y verano), persistencia de papadopoulos_seizing_2022 (~53% inacc, rachas ~6 d)
    la fuente es Papadopoulos, Coit y Ezzat, IEEE Trans. Sustain. Energy 13(1):111-121, 2022,
    del New York/New Jersey Bight y no del Mar del Norte, el titular no depende de esos valores

un dia es accesible si >= min_window_h horas cumplen Hs<hs_max y viento<wind_max, la ventana
de trabajo CTV, y por eso se descargo la serie horaria
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np

from src.scheduler.era5_io import ERA5_DIR, HS_MAX_M, WIND_MAX_MS, load_hourly

MIN_WINDOW_H = 8  # horas accesibles/dia para declarar el dia trabajable (turno CTV)
# longitud media de bloque del bootstrap (persistencia real dentro del bloque)
MEAN_BLOCK_DAYS = 14


class WeatherProvider(Protocol):
    def accessibility(self, t0: date, horizonte_dias: int, n_escenarios: int,
                      seed: int) -> np.ndarray:
        """(n_escenarios, horizonte_dias) de bool, True es dia accesible para CTV."""

        ...


def daily_accessibility(df, hs_max: float = HS_MAX_M, wind_max: float = WIND_MAX_MS,
                        min_window_h: int = MIN_WINDOW_H):
    """de un DataFrame horario (wind, hs) a serie diaria bool de accesibilidad CTV, con
    indice diario contiguo en el que un dia sin dato queda en False."""

    import pandas as pd

    # se cuentan las horas del dia bajo los dos umbrales y el dia vale si llegan a min_window_h
    ok = (df["hs"] < hs_max) & (df["wind"] < wind_max)
    horas_ok = ok.groupby(ok.index.normalize()).sum()
    acc = horas_ok >= min_window_h
    # reindexado a un rango diario continuo, los dias que falten en el historico entran como
    # False
    idx = pd.date_range(acc.index.min(), acc.index.max(), freq="D")
    return acc.reindex(idx, fill_value=False)


class Era5Provider:
    """accesibilidad desde ERA5 real (bootstrap estacional por bloques). carga perezosa."""

    def __init__(self, hs_max: float = HS_MAX_M, wind_max: float = WIND_MAX_MS,
                 min_window_h: int = MIN_WINDOW_H, mean_block_days: int = MEAN_BLOCK_DAYS,
                 data_dir: Path = ERA5_DIR) -> None:
        self.hs_max, self.wind_max = hs_max, wind_max
        self.min_window_h, self.mean_block_days = min_window_h, mean_block_days
        self.data_dir = data_dir
        self._acc = None  # serie diaria (pandas), lazy
        self._arr = None  # np.bool_ contiguo
        self._ord0 = None  # ordinal del primer dia
        self._years = None


    def _ensure(self) -> None:
        # carga perezosa, la serie horaria se lee una sola vez y se cachea en el propio
        # provider
        if self._arr is not None:
            return
        acc = daily_accessibility(load_hourly(self.data_dir), self.hs_max, self.wind_max,
                                  self.min_window_h)
        self._acc = acc
        self._arr = acc.values.astype(bool)
        # el ordinal del primer dia convierte cualquier fecha en indice del array con una resta
        self._ord0 = acc.index.min().toordinal()
        self._years = sorted({ts.year for ts in acc.index})


    def accessibility(self, t0: date, horizonte_dias: int, n_escenarios: int,
                      seed: int) -> np.ndarray:
        self._ensure()
        arr, ord0, years, n = self._arr, self._ord0, self._years, len(self._arr)
        rng = np.random.default_rng(seed)
        out = np.empty((n_escenarios, horizonte_dias), dtype=bool)
        base = t0.toordinal()
        for s in range(n_escenarios):
            i = 0
            # el escenario se rellena a bloques, no dia a dia, y asi conserva las rachas reales
            while i < horizonte_dias:
                cal = date.fromordinal(base + i)
                # mismo mes y dia del calendario pero de un ano historico al azar, eso ancla
                # la estacionalidad y a la vez da diversidad entre escenarios
                y = int(years[rng.integers(len(years))])
                try:
                    anchor = date(y, cal.month, cal.day)
                except ValueError:  # 29-feb en ano no bisiesto
                    anchor = date(y, cal.month, 28)
                idx = anchor.toordinal() - ord0
                # longitud geometrica de media mean_block_days, la persistencia sale de los
                # datos
                bloque = int(rng.geometric(1.0 / self.mean_block_days))
                for _ in range(bloque):
                    if i >= horizonte_dias:
                        break
                    # el modulo cierra el historico en circulo si el bloque se sale por el
                    # final
                    out[s, i] = arr[idx % n]
                    idx += 1
                    i += 1
        return out


    def monthly_inaccessibility(self) -> np.ndarray:
        """% de dias inaccesibles por mes (1..12) en el historico, validacion y figura de
        memoria."""

        self._ensure()
        acc = self._acc
        return np.array([100.0 * (1.0 - acc[acc.index.month == m].mean()) for m in range(1, 13)])


class SyntheticSeasonalProvider:
    """fallback sin ERA5, Markov de 2 estados estacional, escenario adverso asumido (ver
    cabecera)."""

    def __init__(self, inaccesibilidad_media: float = 0.53, racha_inacc_dias: float = 6.0,
                 amplitud_estacional: float = 0.20) -> None:
        self.p_inacc = inaccesibilidad_media  # objetivo anual
        self.racha = racha_inacc_dias  # duracion media de racha inaccesible
        self.amp = amplitud_estacional  # +-modulacion estacional del acceso


    def _p_acceso_mes(self, mes: int) -> float:
        # pico de acceso en verano (jul ~ mes 7), minimo en invierno
        estacional = self.amp * np.cos(2 * np.pi * (mes - 7) / 12)
        # recortado a [0.05, 0.95] para que ningun mes quede totalmente cerrado ni totalmente
        # libre
        return float(np.clip((1 - self.p_inacc) + estacional, 0.05, 0.95))


    def accessibility(self, t0: date, horizonte_dias: int, n_escenarios: int,
                      seed: int) -> np.ndarray:
        # Markov de 2 estados con estacionaria p_acc(mes), la racha media fija p_ia = 1/racha
        # y el balance detallado P(A)*p_ai = P(I)*p_ia da p_ai = (1-p_acc)/p_acc * p_ia
        rng = np.random.default_rng(seed)
        out = np.empty((n_escenarios, horizonte_dias), dtype=bool)
        base = t0.toordinal()
        p_ia = 1.0 / self.racha
        for s in range(n_escenarios):
            accesible = None
            for i in range(horizonte_dias):
                # p_acc depende del mes del calendario, la cadena es no homogenea en el tiempo
                mes = date.fromordinal(base + i).month
                p_acc = self._p_acceso_mes(mes)
                p_ai = min((1.0 - p_acc) / max(p_acc, 1e-6) * p_ia, 1.0)
                if accesible is None:
                    # el primer dia se sortea con la estacionaria, no arrastra estado previo
                    accesible = rng.random() < p_acc
                elif accesible:
                    accesible = not (rng.random() < p_ai)
                else:
                    accesible = rng.random() < p_ia
                out[s, i] = accesible
        return out


_PROVIDERS: dict[str, WeatherProvider] = {}


def get_provider(name: str) -> WeatherProvider:
    # cacheado por nombre, Era5Provider carga la serie horaria una sola vez por proceso
    # los resultados no cambian, accessibility crea su propio rng con la seed en cada llamada
    if name not in _PROVIDERS:
        _PROVIDERS[name] = {"synthetic": SyntheticSeasonalProvider, "era5": Era5Provider}[name]()
    return _PROVIDERS[name]
