"""reentrenamiento identico del BiLSTM-AE de Farm C, documentado exacto.

replica al 100% el camino de entrenamiento de run_model.py que produjo farm_c_best.pt, el
del 0.7432:

    representacion : el cache pca-first top-100 reducido a 55 componentes con semilla 42,
                     y W=27
    constructor    : BiLSTMAutoencoder con encoder [64,16], decoder [16,64], dropout 0.1
                     y salida lineal
    Trainer        : el de src/models/trainer.py, con perdida L1, que esta puesta a mano
                     en el codigo aunque el yaml diga "mse", lr=1e-4, batch=2048, maximo
                     100 epochs, patience 5 y un 10% de val por corte temporal
    datos          : el train completo y sin IsolationForest, que per_farm C deshabilita
    evaluacion     : la oficial, AnomalyDetector mas run_model.evaluate, calibrando sobre
                     el train completo

lo unico que varia es la semilla de entrenamiento, y con ella el no-determinismo de cuDNN,
que es inevitable: dos ejecuciones con la misma semilla ya divergieron. lo que se mide asi es
la distribucion del CARE-CV de un AE entrenado igual igual, y donde cae el 0.7432 dentro de
ella.

los checkpoints salen con la etiqueta de granja "c_ident_s{seed}", o sea
farm_c_ident_s{seed}_best.pt, y no tocan farm_c_best.pt.

    python train_ae_identical_c.py --seed 42
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/, imports hermanos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz, para from src.X

from src.models.bilstm_autoencoder import BiLSTMAutoencoder
from src.models.trainer import Trainer
from src.models.anomaly_detector import AnomalyDetector
from src.care_official.adapter import OfficialCAREAdapter
from run_timeaddm_c import get_components
import run_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite-frozen", action="store_true",
                    help="permite sobrescribir los artefactos congelados de results/models/")
    args = ap.parse_args()
    seed = args.seed
    np.random.seed(seed); torch.manual_seed(seed)
    logger.info(f"Device: {DEVICE} | Farm C | IDENTICAL AE retrain (official path) | seed={seed}")

    # la misma config fundida que usa el main de run_model para Farm C: si esto se desviara,
    # el reentrenamiento ya no seria identico y la distribucion no diria nada
    base = yaml.safe_load(open("config/model_config.yaml", encoding="utf-8"))
    cfg = run_model._deep_merge(base, base.get("per_farm", {}).get("C", {}))
    cfg["score_mode"] = cfg.get("score_mode", "mae_window")
    cfg["output"]["overwrite_frozen"] = args.overwrite_frozen
    assert not cfg["isolation_forest"].get("enabled", True), "official C runs WITHOUT IF-clean"

    train_windows, test_seqs, K = get_components()  # representacion identica, cache s42
    W = train_windows.shape[1]
    if train_windows.dtype != np.float32:
        train_windows = train_windows.astype(np.float32)
    logger.info(f"train {train_windows.shape} (full, no IF) | W={W} K={K}")

    ae_cfg = cfg["autoencoder"]
    # la salida tiene que poder ser negativa si el escalado es standard; con minmax basta relu
    out_act = "linear" if cfg["scaling"].get("method", "standard") == "standard" else "relu"
    model = BiLSTMAutoencoder(input_size=K, encoder_units=ae_cfg["encoder_units"],
                              decoder_units=ae_cfg["decoder_units"], dropout=ae_cfg["dropout"],
                              out_activation=out_act)
    logger.info(f"[AE] BiLSTM ({sum(p.numel() for p in model.parameters()):,} params) | "
                f"L1Loss lr={ae_cfg['learning_rate']} batch={ae_cfg['batch_size']} "
                f"max_epochs={ae_cfg['max_epochs']} patience={ae_cfg['early_stopping']['patience']}")

    t0 = time.time()
    # la etiqueta de granja solo afecta a nombres de checkpoint y de log, asi que los pesos
    # desplegados de farm_c_best.pt no se tocan
    history = Trainer(model=model, config=cfg, device=DEVICE).fit(train_windows, farm=f"c_ident_s{seed}")
    train_secs = time.time() - t0
    best_val = float(min(history["val_loss"]))
    logger.info(f"[AE] trained | {train_secs:.0f}s | epochs={len(history['val_loss'])} best_val={best_val:.6f}")

    t0 = time.time()
    detector = AnomalyDetector(model=model, config=cfg, device=DEVICE)
    adapter = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm="C", window_size=W)
    ev = run_model.evaluate(detector, train_windows, test_seqs, adapter, seed)
    b, cv = ev["best"], ev["cv"]
    logger.info(f"[Eval] CARE-CV={cv['care']:.4f} (oracle={b['care']:.4f}, optim={cv['optimism']:+.4f}) | "
                f"cov={b['cov']:.3f} earl={b['earl']:.3f} rel={b['rel']:.3f} acc={b['acc']:.3f} "
                f"ROC-ds={b['roc_ds']:.3f} | {time.time()-t0:.0f}s")
    logger.info("=" * 70)
    logger.info(f"IDENTICAL-AE seed {seed}: CARE-CV={cv['care']:.4f}  vs farm_c_best.pt 0.7432  "
                f"(delta {cv['care'] - 0.7432:+.4f})")

    out = {"farm": "C", "experiment": "identical_ae_retrain", "seed": seed, "K": K, "window_size": int(W),
           "epochs": len(history["val_loss"]), "best_val_loss": best_val, "train_seconds": round(train_secs),
           "care_cv": cv["care"], "optimism": cv["optimism"], "best": b, "roc_window": ev["roc_window"],
           "ae_ckpt": f"results/models/farm_c_ident_s{seed}_best.pt"}
    p = Path(f"results/tables/ae_identical_c_s{seed}.json")
    json.dump(out, open(p, "w"), indent=2, default=run_model._np_default)
    logger.info(f"saved -> {p}")


if __name__ == "__main__":
    main()
