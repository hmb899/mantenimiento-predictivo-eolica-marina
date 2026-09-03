"""puntua un modelo ya entrenado bajo las dos metricas, sin reentrenarlo.

el CARE del proyecto, que es CAREEvaluator en src/models/evaluator.py, va a nivel de ventana
y es indulgente; el CARE oficial, en src/care_official, es fiel al articulo, va punto a
punto y mira el status. correrlos uno al lado del otro es lo que hace visible la distancia
entre los dos.

carga los artefactos guardados, scaler, PCA, pesos del modelo y umbral, igual que hace
run_model.py con --eval-only, vuelve a puntuar las ventanas de test y despues lleva las
predicciones por ventana de vuelta a filas con OfficialCAREAdapter, que recarga los CSV
crudos porque la metrica oficial necesita status_type_id y los limites de evento, que las
ventanas no llevan.

    python run_official_eval.py                 # las dos granjas
    python run_official_eval.py --farm B
    python run_official_eval.py --raw-path data/raw/CARE_To_Compare --ref mid
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import pickle
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/, imports hermanos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz, para from src.X

from src.models.bilstm_autoencoder import BiLSTMAutoencoder
from src.models.anomaly_detector import AnomalyDetector
from src.models.evaluator import CAREEvaluator
from src.care_official.adapter import OfficialCAREAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _np_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _load_test_sequences(processed_path: Path, farm_l: str) -> tuple[list[dict], list[str]]:
    """carga ligera para eval: secuencias de test y feature cols, sin las ventanas de train."""

    feature_cols = (
        (processed_path / f"farm_{farm_l}_feature_cols.txt").read_text(encoding="utf-8").strip().splitlines()
    )
    with open(processed_path / f"farm_{farm_l}_test_metadata.json", encoding="utf-8") as f:
        metadata: dict = json.load(f)
    seq_data = np.load(processed_path / f"farm_{farm_l}_test_sequences.npz", allow_pickle=False)
    test_seqs = [
        {
            "event_id": int(eid),
            "event_label": meta["event_label"],
            "asset_id": int(meta["asset_id"]),
            "windows": seq_data[eid].astype(np.float32),
            "labels": np.array(meta["labels"], dtype=np.int8),
            "window_timestamps": np.array(meta["window_timestamps"], dtype="datetime64[ns]"),
            "event_start": meta["event_start"],
            "event_end": meta["event_end"],
            "event_description": meta.get("event_description", ""),
        }
        for eid, meta in metadata.items()
    ]
    return test_seqs, feature_cols


def _apply_feature_subset(feature_cols, target_cols, test_seqs):
    # target_cols fija el orden, para que la columna k siga siendo la feature con la que se
    # entreno el modelo guardado; un set perderia justo eso
    idx = [feature_cols.index(f) for f in target_cols if f in feature_cols]
    for seq in test_seqs:
        seq["windows"] = seq["windows"][:, :, idx]
    return [feature_cols[i] for i in idx], test_seqs


def run_farm(farm: str, config: dict, device: str, raw_path: str, ref: str) -> dict:
    farm_l = farm.lower()
    project_root = Path(__file__).resolve().parent
    proc = Path(config["data"]["processed_path"])
    processed_path = proc if proc.is_absolute() else project_root / proc
    models_dir = Path(config["output"]["models_dir"])
    tables_dir = Path(config["output"]["tables_dir"])

    logger.info(f"=== Farm {farm} — dual-metric evaluation (no retraining) ===")

    test_seqs, feature_cols = _load_test_sequences(processed_path, farm_l)
    n_anom = sum(1 for s in test_seqs if s["event_label"] == "anomaly")
    logger.info(f"  {len(test_seqs)} test sequences ({n_anom} anomaly / {len(test_seqs)-n_anom} normal)")

    # el mismo subset de features con el que se entreno el modelo guardado
    model_feat_path = models_dir / f"farm_{farm_l}_model_feature_cols.txt"
    if model_feat_path.exists():
        target_cols = model_feat_path.read_text(encoding="utf-8").strip().splitlines()
        if target_cols != feature_cols:
            feature_cols, test_seqs = _apply_feature_subset(feature_cols, target_cols, test_seqs)
            logger.info(f"  feature subset applied: {len(feature_cols)} features")

    # el scaler y el PCA vienen ya ajustados sobre train; aqui solo se aplican
    with open(models_dir / f"farm_{farm_l}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / f"farm_{farm_l}_pca.pkl", "rb") as f:
        pca = pickle.load(f)
    W = int(test_seqs[0]["windows"].shape[1])  # tamano de ventana, para el desfase a fila
    test_seqs = scaler.transform_sequences(test_seqs)
    test_seqs = pca.transform_sequences(test_seqs)
    K = pca.n_components_

    # el umbral no se recalcula: se relee de la ejecucion que entreno el modelo
    with open(tables_dir / f"farm_{farm_l}_results.json", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])
    ae_cfg = config["autoencoder"]
    model = BiLSTMAutoencoder(
        input_size=K,
        encoder_units=ae_cfg["encoder_units"],
        decoder_units=ae_cfg["decoder_units"],
        dropout=ae_cfg["dropout"],
        # con escalado standard la salida tiene que poder ser negativa; con minmax basta relu
        out_activation=("linear" if getattr(scaler, "method", "minmax") == "standard" else "relu"),
    )
    model.load_state_dict(
        torch.load(models_dir / f"farm_{farm_l}_best.pt", map_location=device, weights_only=True)
    )
    detector = AnomalyDetector(model=model, config=config, device=device)
    detector.threshold_ = threshold
    logger.info(f"  K={K} PCA comps | threshold={threshold:.6f} | window_size={W}")

    # las ventanas se puntuan una vez y se reusan: las dos metricas leen las mismas
    # predicciones, que es lo que hace justa la comparacion entre ellas
    cusum_results = []
    per_event = []
    for seq in test_seqs:
        errors = detector.score_windows(seq["windows"])
        cusum = detector.cusum_event(errors)
        win_pred = (errors > threshold).astype(np.int8)
        cusum["window_predictions"] = win_pred
        cusum["window_errors"] = errors
        cusum_results.append(cusum)
        per_event.append(
            {
                "event_id": seq["event_id"],
                "asset_id": seq["asset_id"],
                "event_label": seq["event_label"],
                "window_timestamps": seq["window_timestamps"],
                "window_predictions": win_pred,
            }
        )

    # metrica del proyecto, indulgente y a nivel de ventana
    project = CAREEvaluator(config).evaluate(test_seqs, cusum_results)

    # metrica oficial, punto a punto; el adaptador es quien devuelve las ventanas a filas
    adapter = OfficialCAREAdapter(raw_path=raw_path, farm=farm, window_size=W, ref=ref)
    official, diag = adapter.evaluate(per_event)

    logger.info(
        f"  PROJECT  CARE={project['care_score']:.4f} "
        f"(Cov={project['coverage_fbeta']:.4f} Earl={project['earliness_ws']:.4f} "
        f"Rel={project['reliability_efbeta']:.4f} Acc={project['accuracy']:.4f})"
    )
    logger.info(
        f"  OFFICIAL CARE={official.care:.4f} "
        f"(Cov={official.coverage:.4f} Earl={official.earliness:.4f} "
        f"Rel={official.reliability:.4f} Acc={official.accuracy:.4f})"
    )

    out = {
        "farm": farm,
        "window_size": W,
        "ref": ref,
        "threshold": threshold,
        "project_metric": {
            "care": project["care_score"],
            "coverage": project["coverage_fbeta"],
            "earliness": project["earliness_ws"],
            "reliability": project["reliability_efbeta"],
            "accuracy": project["accuracy"],
        },
        "official_metric": {
            "care": official.care,
            "coverage": official.coverage,
            "earliness": official.earliness,
            "reliability": official.reliability,
            "accuracy": official.accuracy,
        },
        "per_event_diag": diag,
    }
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / f"farm_{farm_l}_care_comparison.json", "w") as f:
        json.dump(out, f, indent=2, default=_np_default)
    logger.info(f"  saved -> {tables_dir / f'farm_{farm_l}_care_comparison.json'}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm", choices=["B", "C"], default=None)
    parser.add_argument("--raw-path", default="data/raw/CARE_To_Compare")
    parser.add_argument("--ref", choices=["mid", "start", "end"], default="mid",
                        help="fila de referencia de la ventana; mid es el defecto y casa con el windower")
    args = parser.parse_args()

    with open("config/model_config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PyTorch {torch.__version__} | Device: {device}")

    farms = [args.farm] if args.farm else config["data"]["farms"]
    summary = {}
    for farm in farms:
        summary[farm] = run_farm(farm, config, device, args.raw_path, args.ref)

    # tabla de cierre: el delta es lo que cuantifica cuanto indulge la metrica del proyecto
    logger.info("=" * 64)
    logger.info(f"{'Farm':<6}{'PROJECT CARE':>16}{'OFFICIAL CARE':>16}{'delta':>10}")
    for farm, r in summary.items():
        p = r["project_metric"]["care"]
        o = r["official_metric"]["care"]
        logger.info(f"{farm:<6}{p:>16.4f}{o:>16.4f}{o - p:>10.4f}")
