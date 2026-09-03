"""
io y configuracion de ERA5 para el scheduler (F3), fuente de verdad de la peticion al CDS, del
lector zip-aware y del ensamblado de la serie horaria

build_request
    la peticion al CDS, dataset, caja del German Bight, variables y formato
open_datasets
    el lector zip-aware, el CDS entrega 2 .nc en rejillas distintas, atmosferica 0.25 deg y
    oleaje 0.5 deg, dentro de un zip aunque se pida 'unarchived'
load_hourly
    el ensamblado de la serie horaria (viento m/s, Hs m) de todos los anos descargados

una sonda de viabilidad previa, ya retirada porque la decision que motivaba ya esta
tomada, valido este flujo con enero de 2023, descarga real en ~2 min, viento med 9.6 m/s, Hs
med 2.0 m, 0% NaN, 63% inaccesible. decision meteo tomada a la vista de esa sonda, ERA5 real
el script de descarga completa tampoco viaja aqui, este modulo documenta la peticion que hacia
"""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ERA5_DIR = ROOT / "data" / "era5"

DATASET = "reanalysis-era5-single-levels"
# punto generico del Mar del Norte aleman (German Bight), caja ~0.5 deg en orden [N, W, S, E]
AREA = [54.25, 6.25, 53.75, 6.75]
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "significant_height_of_combined_wind_waves_and_swell",
]
YEARS = tuple(range(2014, 2024))  # 10 anos completos y recientes
# el CDS acota el coste por peticion, 1 ano (~26k campos) lo excede y 1 mes (~2k) pasa
# tercios de 4 meses (~8.6k campos, con margen), 3 peticiones por ano, 30 en total
CHUNKS = ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12])

# umbral CTV de referencia, el definitivo lo fija scenarios.py. el 1.5 m sale de la banda UE de
# papadopoulos_seizing_2022 (1.20-1.75 m), viento y 8 h son eleccion propia, no de un articulo
HS_MAX_M = 1.5
WIND_MAX_MS = 12.0


def chunk_path(year: int, months: list[int], data_dir: Path = ERA5_DIR) -> Path:
    # el nombre codifica ano y tramo, load_hourly los recoge luego con el glob era5_gb_*.zip
    return data_dir / f"era5_gb_{year}_{months[0]:02d}-{months[-1]:02d}.zip"


def build_request(year: int, months: list[int]) -> dict:
    """peticion CDS de un tramo de meses de un ano, un punto, horario. trozos de 4 meses
    mantienen el request bajo el limite de coste del CDS (ver CHUNKS) y lo hacen reanudable."""

    # los 31 dias y las 24 horas van fijos, el tramo de meses es lo unico que cambia entre
    # trozos
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def open_datasets(path: Path) -> list:
    """abre el resultado del CDS como lista de datasets xarray (descomprime el zip de 2
    rejillas)."""

    import xarray as xr

    if zipfile.is_zipfile(path):
        # el zip trae un .nc por rejilla, se extraen a un temporal y se abren todos
        tmp = Path(tempfile.mkdtemp(prefix="era5_"))
        dsets = []
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith(".nc"):
                    z.extract(name, tmp)
                    dsets.append(xr.open_dataset(tmp / name))
        return dsets
    return [xr.open_dataset(path)]


def _var(dsets, *nombres):
    # cada variable llega con nombre corto o largo segun la version, se acepta cualquier alias
    quiero = {n.lower() for n in nombres}
    for ds in dsets:
        for v in ds.data_vars:
            if v.lower() in quiero:
                return ds[v]
    raise KeyError(f"ninguna de {nombres} en {[list(d.data_vars) for d in dsets]}")


def _spatial_mean(da):
    # la caja cubre varios puntos de rejilla, se promedia todo lo que no sea el eje temporal
    dims = [d for d in da.dims if d not in ("time", "valid_time")]
    return da.mean(dim=dims) if dims else da


def _time_values(da):
    # el eje temporal se llama valid_time o time segun la version, se prueban en ese orden
    for c in ("valid_time", "time"):
        if c in da.coords:
            return da[c].values
    raise KeyError("sin coordenada temporal (valid_time/time)")


def load_hourly(data_dir: Path = ERA5_DIR):
    """serie horaria ensamblada de todos los trozos era5_gb_*.zip presentes, como DataFrame
    indexado por tiempo, con columnas 'wind' (m/s) y 'hs' (m), media espacial de la caja."""

    import pandas as pd
    import xarray as xr

    # se ignora lo aun no descargado, por eso el flujo es reanudable
    # esta serie es la entrada del WeatherProvider de scenarios.py
    frames = []
    for p in sorted(data_dir.glob("era5_gb_*.zip")):
        dsets = open_datasets(p)
        try:
            u, v = _var(dsets, "u10", "10u"), _var(dsets, "v10", "10v")
            swh = _var(dsets, "swh", "significant_height_of_combined_wind_waves_and_swell")
            # el viento viene en componentes u y v, el modulo es la raiz de la suma de
            # cuadrados
            wind = _spatial_mean(np.sqrt(u**2 + v**2))
            hs = _spatial_mean(swh)
            # viento y oleaje llegan en rejillas distintas, align los deja en el mismo eje
            # temporal
            wind, hs = xr.align(wind, hs)
            idx = pd.DatetimeIndex(_time_values(wind))
            frames.append(pd.DataFrame(
                {"wind": np.asarray(wind.values), "hs": np.asarray(hs.values)}, index=idx))
        finally:
            # los datasets del zip se cierran pase lo que pase, si no el temporal queda
            # retenido
            for ds in dsets:
                ds.close()
    if not frames:
        raise FileNotFoundError(f"no hay ficheros era5_gb_*.zip en {data_dir}")
    # si dos trozos se pisan, se ordena por tiempo y se queda la primera aparicion de cada hora
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]
