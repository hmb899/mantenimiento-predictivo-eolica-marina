"""
adapter que calcula el CARE Score oficial (src.care_official) para un modelo que produce
predicciones por ventana, el BiLSTM-AE del proyecto y TimeADDM.

por que existe. la metrica oficial es punto a punto sobre las filas de prediccion y necesita el
`status_type_id` de cada fila, que se pierde al ventanear, y las fronteras `event_start_id` y
`event_end_id`, que son ids de fila. este modulo recarga los CSV crudos por evento para
recuperarlas y realinea a filas las predicciones por ventana del modelo.

mapeo de ventana a fila, una eleccion interpretativa. cada ventana conservada (stride=1) va a
la fila cruda de su posicion de referencia, y el `ref="mid"` por defecto es `window_start +
window_size // 2`, igual que el proyecto etiqueta sus ventanas (`windower.py`, etiqueta por
punto medio). una fila a la que no llega ninguna ventana marcada se queda en 0, sin
deteccion. `"start"` y `"end"` se exponen para analisis de sensibilidad. la metrica oficial
corre luego su propio filtrado por estado, el Criticality Algorithm (tc=72) y el CARE a trozos
sobre estas predicciones por fila.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .care import (
    CAREResult,
    compute_care,
    prepare_dataset_inputs,
    prepare_normal_dataset_inputs,
)
from .earliness import weighted_score_single

# solo estas cinco columnas, el CSV crudo de un evento pesa cientos de MB
_PRED_COLS = ("time_stamp", "asset_id", "id", "train_test", "status_type_id")


class OfficialCAREAdapter:
    """mapea predicciones por ventana a predicciones por fila y puntua el CARE oficial."""

    def __init__(self, raw_path: str | Path, farm: str, window_size: int, sep: str = ";",
                 encoding: str = "latin-1", tc: int = 72, beta: float = 0.5,
                 ref: str = "mid") -> None:
        self.farm_dir = Path(raw_path) / f"Wind Farm {farm}"
        self.datasets_dir = self.farm_dir / "datasets"
        self.window_size = int(window_size)
        self.sep = sep
        self.encoding = encoding
        self.tc = int(tc)
        self.beta = float(beta)
        if ref not in ("mid", "start", "end"):
            raise ValueError(f"ref must be 'mid'|'start'|'end', got {ref!r}")
        self.ref = ref
        # cache por (evento, turbina), una ejecucion vuelve al mismo CSV una vez por metrica
        self._pred_cache: dict[tuple[int, int], pd.DataFrame] = {}
        self.event_info = self._load_event_info()


    # carga de los CSV crudos y mapeo de ventana a fila
    def _load_event_info(self) -> pd.DataFrame:
        ev = pd.read_csv(self.farm_dir / "event_info.csv", sep=self.sep, encoding=self.encoding)
        ev.columns = ev.columns.str.strip()
        # algunos volcados del dataset nombran la columna asset en vez de asset_id
        if "asset_id" not in ev.columns and "asset" in ev.columns:
            ev = ev.rename(columns={"asset": "asset_id"})
        return ev.set_index("event_id")


    def _load_pred_rows(self, event_id: int, asset_id: int) -> pd.DataFrame:
        """las filas de la particion de prediction ordenadas por id, con id, status y
        time_stamp."""

        key = (int(event_id), int(asset_id))
        if key in self._pred_cache:
            return self._pred_cache[key]
        path = self.datasets_dir / f"{event_id}.csv"
        df = pd.read_csv(
            path,
            sep=self.sep,
            encoding=self.encoding,
            usecols=lambda c: c.strip() in _PRED_COLS,
            low_memory=False,
        )
        df.columns = df.columns.str.strip()
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        # fuera la zona horaria, el timestamp de ventana del modelo es naive y hay que casarlo
        if getattr(df["time_stamp"].dt, "tz", None) is not None:
            df["time_stamp"] = df["time_stamp"].dt.tz_localize(None)
        df = df[(df["train_test"] == "prediction") & (df["asset_id"] == asset_id)].copy()
        # el orden por id es el contrato con prepare_dataset_inputs, que indexa por posicion
        df = df.sort_values("id").reset_index(drop=True)
        self._pred_cache[key] = df
        return df


    def _offset(self) -> int:
        if self.ref == "start":
            return 0
        if self.ref == "end":
            return self.window_size - 1
        return self.window_size // 2  # mid


    def _map_windows_to_rows(self, pred_df: pd.DataFrame, window_ts: np.ndarray,
                             window_preds: np.ndarray) -> np.ndarray:
        """prediccion 0/1 por fila alineada con pred_df, que llega ordenado por id."""

        n = len(pred_df)
        y = np.zeros(n, dtype=int)
        if n == 0:
            return y
        # los timestamps pasan a entero de nanosegundos para poder indexarlos por diccionario
        ts_ns = pred_df["time_stamp"].to_numpy().astype("datetime64[ns]").view("int64")
        idx_by_ts = {int(t): i for i, t in enumerate(ts_ns)}
        w_ns = np.asarray(window_ts).astype("datetime64[ns]").view("int64")
        w_pred = np.asarray(window_preds, dtype=int)
        off = self._offset()
        for t, p in zip(w_ns, w_pred):
            if p != 1:
                continue
            # window_ts trae el inicio de la ventana, el offset lleva a la fila de referencia
            start_idx = idx_by_ts.get(int(t))
            if start_idx is None:
                continue
            # una ventana marca una sola fila, no su tramo, y la del borde cae fuera de la
            # particion
            row_idx = start_idx + off
            if 0 <= row_idx < n:
                y[row_idx] = 1
        return y


    # API publica, de predicciones por ventana a CARE oficial
    def build_inputs(self, per_event: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        """las entradas de la metrica oficial a partir de las predicciones por ventana de
        cada evento.

        cada elemento de per_event trae event_id:int, asset_id:int, event_label:str,
        window_timestamps, un array datetime64[ns] con la marca de inicio de cada ventana,
        y window_predictions, un 0/1 por ventana.
        """

        anomaly_datasets: list[dict] = []
        normal_datasets: list[dict] = []
        diagnostics: list[dict] = []

        for ev in per_event:
            eid = int(ev["event_id"])
            asset = int(ev["asset_id"])
            label = ev["event_label"]
            pred_df = self._load_pred_rows(eid, asset)
            y = self._map_windows_to_rows(
                pred_df, ev["window_timestamps"], ev["window_predictions"]
            )
            info = self.event_info.loc[eid]

            # el dataset anomalo necesita su ventana de evento, el normal no la tiene
            if label == "anomaly":
                ds = prepare_dataset_inputs(
                    pred_df,
                    int(info["event_start_id"]),
                    int(info["event_end_id"]),
                    y,
                )
                anomaly_datasets.append(ds)
            else:
                ds = prepare_normal_dataset_inputs(y, pred_df["status_type_id"].to_numpy())
                normal_datasets.append(ds)

            # diagnostico por evento, cuantas filas quedan marcadas de cuantas ventanas habia
            diagnostics.append(
                {
                    "event_id": eid,
                    "event_label": label,
                    "n_pred_rows": int(len(pred_df)),
                    "n_windows": int(len(ev["window_predictions"])),
                    "n_flagged_rows": int(y.sum()),
                }
            )

        return anomaly_datasets, normal_datasets, diagnostics


    def evaluate(self, per_event: list[dict]) -> tuple[CAREResult, list[dict]]:
        """el CAREResult oficial y los diagnosticos por evento."""

        anomaly_datasets, normal_datasets, diagnostics = self.build_inputs(per_event)
        result = compute_care(
            anomaly_datasets, normal_datasets, beta=self.beta, tc=self.tc
        )
        return result, diagnostics


    def conditional_earliness(self, per_event: list[dict]) -> dict:
        """descriptiva, fuera de CARE, earliness solo sobre los eventos de anomalia detectados.

        usa el score pesado punto a punto del articulo (eq. 3) pero promedia WS_i solo sobre
        los eventos de anomalia con WS_i > 0, aquellos donde el modelo marco algo dentro de la
        ventana del evento. responde a cuando detecta el modelo, con cuanta antelacion, y es
        interpretable pero no comparable con el benchmark, cuya earliness promedia sobre todos
        los eventos y cuenta como 0 el evento no detectado.
        """

        anomaly_datasets, _, _ = self.build_inputs(per_event)
        # WS_i > 0 es la definicion operativa de evento detectado dentro de su ventana
        ws = [weighted_score_single(ds["y_pred_event"], ds["weights_event"])
              for ds in anomaly_datasets]
        detected = [w for w in ws if w > 0]
        return {
            "conditional_earliness": float(np.mean(detected)) if detected else 0.0,
            "official_earliness": float(np.mean(ws)) if ws else 0.0,
            "n_detected": len(detected),
            "n_anomaly": len(ws),
        }
