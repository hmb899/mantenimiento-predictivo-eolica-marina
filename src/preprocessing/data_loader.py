# carga de una granja CARE, los CSV crudos mas event_info y sensor_registry
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import get_logger

logger = get_logger(__name__)


class CareDataLoader:
    """carga los CSV crudos, event_info y sensor_registry de una granja CARE."""

    def __init__(self, config: dict) -> None:
        # tres niveles sobre src/preprocessing/ esta la raiz, que ancla un
        # raw_path relativo para que la config valga desde cualquier sitio
        _project_root = Path(__file__).resolve().parent.parent.parent
        raw = Path(config["data"]["raw_path"])
        self.raw_path = raw if raw.is_absolute() else _project_root / raw
        self.sep = config["data"].get("separator", ";")
        self.encoding = config["data"].get("encoding", "latin-1")


    def load_farm(self, farm: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """carga todos los datos de una granja, los CSV, event_info y sensor_registry."""

        farm_path = self.raw_path / f"Wind Farm {farm}"
        if not farm_path.exists():
            raise FileNotFoundError(f"Farm directory not found: {farm_path}")

        logger.info(f"Loading Farm {farm} from {farm_path}")

        # event_info va primero, los datasets necesitan sus metadatos para el merge
        event_info = self._load_event_info(farm_path)
        sensor_registry = self._load_sensor_registry(farm_path)
        df = self._load_datasets(farm_path, event_info)

        logger.info(
            f"Farm {farm} loaded — {len(df):,} rows, "
            f"{df['asset_id'].nunique()} turbines, "
            f"{df['event_id'].nunique()} events"
        )
        return df, event_info, sensor_registry


    def _load_event_info(self, farm_path: Path) -> pd.DataFrame:
        """lee event_info.csv y parsea las marcas de tiempo de inicio y de fin."""

        path = farm_path / "event_info.csv"
        df = pd.read_csv(path, sep=self.sep, encoding=self.encoding)
        # las cabeceras CARE traen espacios, sin strip no casa ningun nombre
        df.columns = df.columns.str.strip()
        df["event_start"] = pd.to_datetime(df["event_start"])
        df["event_end"] = pd.to_datetime(df["event_end"])

        logger.debug(
            f"  event_info: {len(df)} events "
            f"({(df['event_label']=='anomaly').sum()} anomaly, "
            f"{(df['event_label']=='normal').sum()} normal)"
        )
        return df


    def _load_sensor_registry(self, farm_path: Path) -> pd.DataFrame:
        """lee feature_description.csv, los metadatos de sensor con unidades, banderas
        y descripciones."""

        path = farm_path / "feature_description.csv"
        df = pd.read_csv(path, sep=self.sep, encoding=self.encoding)
        df.columns = df.columns.str.strip()

        for col in ["sensor_name", "description", "unit"]:
            df[col] = df[col].str.strip()

        # las dos banderas llegan como el texto true / false, no como booleanos
        df["is_angle"] = df["is_angle"].astype(str).str.strip().str.lower() == "true"
        df["is_counter"] = df["is_counter"].astype(str).str.strip().str.lower() == "true"
        # unit_norm da una sola grafia a la unidad, el fichero crudo mezcla Celsius y °C
        df["unit_norm"] = df["unit"].str.replace("Celsius", "°C", case=False)

        logger.debug(f"  sensor_registry: {len(df)} sensors")
        return df


    def _load_datasets(self, farm_path: Path, event_info: pd.DataFrame) -> pd.DataFrame:
        """lee todos los CSV de evento, pega los metadatos y devuelve un DataFrame ordenado."""

        datasets_path = farm_path / "datasets"
        dfs: list[pd.DataFrame] = []
        event_ids: list[int] = []
        # ordenados por el numero del nombre, el 2 va antes que el 10
        csv_files = sorted(datasets_path.glob("*.csv"), key=lambda p: int(p.stem))

        for csv_file in csv_files:
            # el nombre del fichero es el id del evento, el CSV no lo lleva dentro
            event_id = int(csv_file.stem)
            df = pd.read_csv(csv_file, sep=self.sep, encoding=self.encoding, low_memory=False)
            df["time_stamp"] = pd.to_datetime(df["time_stamp"])
            dfs.append(df)
            # pares (id, filas), que es lo que necesita el np.repeat de abajo
            event_ids.append((event_id, len(df)))
            logger.info(f"  Event {event_id}: {len(df):,} rows")

        # el concat consolida los bloques, aqui ya se pueden anadir columnas sin fragmentar
        combined = pd.concat(dfs, ignore_index=True)
        # repetir cada id por sus filas evita recorrer fichero a fichero,
        # y solo vale porque el concat respeto el orden de dfs
        combined["event_id"] = np.repeat(
            [eid for eid, _ in event_ids],
            [n for _, n in event_ids],
        )

        meta_cols = [c for c in ["event_id", "event_label", "event_start", "event_end", "event_description"]
                     if c in event_info.columns]
        combined = combined.merge(event_info[meta_cols], on="event_id", how="left")
        # sin fila en event_info el evento se marca, no se descarta
        combined["event_label"] = combined["event_label"].fillna("unlabeled")
        if "event_description" in combined.columns:
            combined["event_description"] = combined["event_description"].fillna("")

        # los pasos posteriores dan por hecho este orden, ventaneo incluido
        combined.sort_values(["asset_id", "time_stamp"], inplace=True)
        combined.reset_index(drop=True, inplace=True)

        # sin zona horaria las marcas comparan con las fechas de los eventos
        if hasattr(combined["time_stamp"].dtype, "tz") and combined["time_stamp"].dt.tz is not None:
            combined["time_stamp"] = combined["time_stamp"].dt.tz_localize(None)

        return combined
