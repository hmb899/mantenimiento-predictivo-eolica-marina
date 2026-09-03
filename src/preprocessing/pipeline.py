"""
pipeline principal de preproceso, orquesta todos los pasos de punta a punta

load_config construye la configuracion y PreprocessingPipeline(config).run("B") o run("C")
devuelve un PipelineResult con train_windows (N, 27, n_features) y test_sequences
(list[TestSequence]), las dos con los NaN imputados y sin escalar, col_means (n_features,) con
las medias de entrenamiento por feature, feature_cols con los nombres elegidos, el
sensor_registry (pd.DataFrame) y el report, el rastro de auditoria completo, tambien en JSON.

en el preproceso no se escala nada, el escalado pertenece al pipeline del modelo (Barnabei
et al., 2024), que es escalado, PCA, IsolationForest y BiLSTM AE; el proyecto escala con
StandardScaler donde el articulo usa MinMaxScaler.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data_loader import CareDataLoader
from .feature_selector import FeatureSelector
from .normalizer import TemperatureNormalizer
from .outlier_handler import OutlierHandler
from .power_curve_filter import PowerCurveFilter
from .smoother import Smoother
from .status_filter import StatusFilter
from .transition_excluder import TransitionExcluder
from .windower import TestSequence, Windower
from .zero_handler import ZeroHandler
from .utils import get_logger, save_report

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    farm: str
    train_windows: np.ndarray  # (N_train, window_size, n_features), NaN imputados, sin escalar
    test_sequences: list[TestSequence]  # cada .windows va con NaN imputados y sin escalar
    col_means: np.ndarray  # (n_features,) medias de entrenamiento por feature para imputar NaN
    feature_cols: list[str]
    sensor_registry: pd.DataFrame
    report: dict = field(default_factory=dict)


class PreprocessingPipeline:
    """preproceso CARE completo, carga, limpia, selecciona, suaviza, ventanea y guarda."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        _project_root = Path(__file__).resolve().parent.parent.parent
        _proc = Path(self.config["data"]["processed_path"])
        self.processed_path = _proc if _proc.is_absolute() else _project_root / _proc


    def run(self, farm: str) -> PipelineResult:
        """ejecuta el pipeline completo de una granja y devuelve un PipelineResult."""

        logger.info(f"  PIPELINE START — Farm {farm}")

        report: dict = {
            "farm": farm,
            "pipeline_start": datetime.now().isoformat(),
            "steps": [],
        }

        logger.info("[1/10] Loading data …")
        loader = CareDataLoader(self.config)
        df, event_info, sensor_registry = loader.load_farm(farm)
        report["n_rows_raw"] = len(df)
        report["n_turbines"] = int(df["asset_id"].nunique())
        report["n_events"] = int(df["event_id"].nunique())

        logger.info("[2/10] Zero-as-missing handling …")
        zero_handler = ZeroHandler(self.config, sensor_registry)
        df, stats = zero_handler.transform(df)
        report["steps"].append(stats)

        logger.info("[3/10] Feature selection …")
        df_normal_status = df[
            df["status_type_id"].isin(self.config["status_filtering"]["normal_statuses"])
        ]
        selector = FeatureSelector(self.config, sensor_registry)
        feature_cols, stats = selector.select(df_normal_status)
        report["steps"].append(stats)
        report["feature_cols"] = feature_cols
        report["n_features"] = len(feature_cols)

        if not feature_cols:
            raise RuntimeError("Feature selection returned zero features — aborting.")

        # tirar las columnas de sensor sin usar lleva el dataframe de ~950 a ~140 columnas y
        # la Farm C de ~22 GB a ~3.5 GB, lo que evita el OOM al consolidar bloques en pandas
        _meta_cols = [
            "time_stamp", "asset_id", "status_type_id", "train_test",
            "event_id", "event_label", "event_start", "event_end", "event_description",
        ]
        # las columnas de velocidad de viento se conservan aunque no esten en feature_cols,
        # porque PowerCurveFilter las busca en el registro en tiempo de ejecucion
        _wind_cols = [
            c for c in df.columns
            if c.endswith("_avg") and "wind_speed" in c
        ]
        _keep = [c for c in _meta_cols if c in df.columns] + feature_cols
        _keep += [c for c in _wind_cols if c not in _keep]
        df = df[_keep].copy()
        # df acaba de ser sustituido, la vista de status normal se rehace sobre el podado
        df_normal_status = df[
            df["status_type_id"].isin(self.config["status_filtering"]["normal_statuses"])
        ]
        logger.info(
            f"Slimmed df to {len(_keep)} columns "
            f"({len(feature_cols)} features + {len(_keep) - len(feature_cols)} meta/wind)"
        )

        # los dos guardas valen True por defecto, asi que la B de produccion (la baseline del
        # AE) sale identica byte a byte; a false dan el preproceso "blando/estandar", cercano
        # al pipeline minimo de solo z-score de los benchmarks de imputacion por difusion
        if self.config.get("outlier_detection", {}).get("enabled", True):
            logger.info("[4/10] Outlier detection & treatment …")
            outlier_handler = OutlierHandler(self.config, sensor_registry)
            outlier_handler.fit(df_normal_status, feature_cols)  # solo ajusta con Status 0
            df, stats = outlier_handler.transform(df, feature_cols)
            report["steps"].append(stats)
        else:
            logger.info("[4/10] Outlier detection SKIPPED (enabled=false)")

        if self.config.get("temperature_normalization", {}).get("enabled", True):
            logger.info("[5/10] Temperature normalisation …")
            normalizer = TemperatureNormalizer(self.config, sensor_registry)
            df, stats = normalizer.transform(df, feature_cols)
            report["steps"].append(stats)
        else:
            logger.info("[5/10] Temperature normalisation SKIPPED (enabled=false)")

        logger.info("[6/10] Savitzky-Golay smoothing …")
        smoother = Smoother(self.config, sensor_registry)
        df, stats = smoother.transform(df, feature_cols)
        report["steps"].append(stats)

        logger.info("[7/10] Building training data (status filter + exclusion zones) …")

        # CARE etiqueta la particion "train" y "prediction", no "test"
        df_train = df[df["train_test"] == "train"].copy()

        # dedup opcional de las filas de train agrupadas por (asset_id, time_stamp), los 95 CSV
        # de evento de CARE se solapan mucho, la misma linea temporal de turbina aparece en
        # varios, sin dedup el train sobrepesa las turbinas recurrentes; solo se dedup el train
        if self.config.get("data", {}).get("dedup_train", False):
            _n0 = len(df_train)
            df_train = df_train.drop_duplicates(["asset_id", "time_stamp"], keep="first")
            logger.info(f"Dedup train (asset_id,time_stamp): {_n0:,} -> {len(df_train):,} rows")
            report["dedup_train"] = {"before": int(_n0), "after": int(len(df_train))}

        # filtro de status
        status_filter = StatusFilter(self.config)
        df_train, stats = status_filter.transform(df_train)
        report["steps"].append(stats)

        # filtro de curva de potencia
        logger.info("       Power-curve sanity filter …")
        pc_filter = PowerCurveFilter(self.config, sensor_registry)
        df_train, stats = pc_filter.transform(df_train)
        report["steps"].append(stats)

        # exclusion de transiciones
        logger.info("       Transition-period exclusion …")
        excluder = TransitionExcluder(self.config)
        df_train, stats = excluder.transform(df_train, event_info)
        report["steps"].append(stats)

        report["n_train_rows"] = len(df_train)

        logger.info("[8/10] Temporal windowing …")
        windower = Windower(self.config)
        train_windows, turbine_ids, stats = windower.create_train_windows(df_train, feature_cols)
        report["steps"].append(stats)

        test_sequences, stats = windower.create_test_sequences(df, event_info, feature_cols)
        report["steps"].append(stats)

        logger.info("[9/10] NaN imputation (no scaling — MinMaxScaler applied in model pipeline) …")
        train_windows, test_sequences, col_means = self._impute_nan(train_windows, test_sequences)
        report["scaling"] = {
            "method": self.config["scaling"]["method"],  # "none"
            "note": "MinMaxScaler applied in model pipeline (Barnabei et al., 2024)",
        }

        logger.info("[10/10] Saving outputs …")
        self._save(farm, train_windows, turbine_ids, test_sequences, col_means, feature_cols)

        report["pipeline_end"] = datetime.now().isoformat()
        report_path = self.processed_path / f"farm_{farm.lower()}_preprocessing_report.json"
        save_report(report, report_path)
        logger.info(f"Report saved: {report_path}")

        logger.info(
            f"PIPELINE COMPLETE — Farm {farm} | "
            f"{train_windows.shape[0]:,} train windows | "
            f"{len(test_sequences)} test sequences"
        )

        return PipelineResult(
            farm=farm,
            train_windows=train_windows,
            test_sequences=test_sequences,
            col_means=col_means,
            feature_cols=feature_cols,
            sensor_registry=sensor_registry,
            report=report,
        )


    def run_all(self) -> dict[str, PipelineResult]:
        """ejecuta el pipeline para cada granja listada en la configuracion."""

        results = {}
        for farm in self.config["data"]["farms"]:
            results[farm] = self.run(farm)
        return results


    def _impute_nan(self, train_windows: np.ndarray,
                    test_sequences: list[TestSequence]) -> tuple[np.ndarray, list[TestSequence], np.ndarray]:
        """imputa los NaN residuales con las medias de entrenamiento por feature y devuelve
        los tres, train_windows, test_sequences y col_means."""

        # se aplana a (N*W, F) para que la media y la imputacion sean por columna de feature,
        # y al final se devuelve a (N, W, F) con las mismas dimensiones
        n, w, f = train_windows.shape
        train_2d = train_windows.reshape(-1, f)
        col_means = np.nanmean(train_2d, axis=0)  # forma (f,)

        # imputa los NaN del train
        nan_mask = ~np.isfinite(train_2d)
        if nan_mask.any():
            n_imputed = int(nan_mask.sum())
            logger.info(f"Imputing {n_imputed:,} non-finite values in train windows with feature means")
            train_2d = train_2d.copy()
            nan_rows, nan_cols = np.where(nan_mask)
            train_2d[nan_rows, nan_cols] = col_means[nan_cols]
        train_windows = train_2d.reshape(n, w, f)

        # imputa los NaN de las secuencias de test con las medias de train, sin fuga
        for seq in test_sequences:
            n_s, w_s, f_s = seq.windows.shape
            seq_2d = seq.windows.reshape(-1, f_s)
            nan_mask_s = ~np.isfinite(seq_2d)
            if nan_mask_s.any():
                seq_2d = seq_2d.copy()
                nr, nc = np.where(nan_mask_s)
                seq_2d[nr, nc] = col_means[nc]
                seq.windows = seq_2d.reshape(n_s, w_s, f_s)

        return train_windows, test_sequences, col_means


    def _save(self, farm: str, train_windows: np.ndarray, turbine_ids: np.ndarray, test_sequences: list[TestSequence],
              col_means: np.ndarray, feature_cols: list[str]) -> None:
        """guarda en disco las salidas del pipeline."""

        farm_l = farm.lower()
        out_dir = self.processed_path
        out_dir.mkdir(parents=True, exist_ok=True)

        # ventanas de entrenamiento
        if self.config["export"]["compress_numpy"]:
            np.savez_compressed(
                out_dir / f"farm_{farm_l}_train_windows.npz",
                windows=train_windows,
                feature_cols=np.array(feature_cols),
                turbine_ids=turbine_ids,
            )
        else:
            np.save(out_dir / f"farm_{farm_l}_train_windows.npy", train_windows)

        # secuencias de test, primero se juntan enteras y luego se parten en dos artefactos
        test_data = {
            seq.event_id: {
                "windows": seq.windows,
                "labels": seq.labels,
                "window_timestamps": seq.window_timestamps,
                "event_label": seq.event_label,
                "asset_id": seq.asset_id,
                "event_start": str(seq.event_start),
                "event_end": str(seq.event_end),
                "event_description": seq.event_description,
            }
            for seq in test_sequences
        }
        # el npz solo lleva los arrays de ventanas, con el event_id de clave
        np.savez_compressed(
            out_dir / f"farm_{farm_l}_test_sequences.npz",
            **{
                str(eid): seq["windows"]
                for eid, seq in test_data.items()
            },
        )

        # el resto va a JSON, que es lo que CARE Score necesita sin recargar los datos crudos
        test_metadata = {
            str(eid): {
                "event_label": seq["event_label"],
                "asset_id": str(seq["asset_id"]),
                "event_start": seq["event_start"],
                "event_end": seq["event_end"],
                "event_description": seq["event_description"],
                "n_windows": int(seq["windows"].shape[0]),
                "labels": seq["labels"].tolist(),  # 0=normal 1=anomalia
                # marcas de tiempo ISO, una por inicio de ventana
                "window_timestamps": [
                    str(t) for t in seq.get("window_timestamps", [])
                ],
            }
            for eid, seq in test_data.items()
        }
        meta_path = out_dir / f"farm_{farm_l}_test_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f_meta:
            json.dump(test_metadata, f_meta, indent=2)

        # solo las medias, aqui no se aplica StandardScaler, asi que no hay scaler que guardar
        if self.config["export"]["save_scaler"]:
            np.save(out_dir / f"farm_{farm_l}_col_means.npy", col_means)

        # lista de features, un nombre por linea y en el orden que espera el modelo
        with open(out_dir / f"farm_{farm_l}_feature_cols.txt", "w") as f_txt:
            f_txt.write("\n".join(feature_cols))

        logger.info(
            f"Saved: train_windows {train_windows.shape}, "
            f"{len(test_sequences)} test sequences, col_means, feature list"
        )
