"""reentrenamiento identico del BiLSTM-AE de Farm B, espejo del de Farm C.

replica el camino oficial que produjo farm_b_best.pt, el del 0.7211:

    representacion : la congelada, window-first con 30 features, scaler standard y PCA con
                     K=18, de los artefactos de results/models que ya reprodujeron el
                     0.7211 exacto
    datos          : el train limpiado con IsolationForest, porque el camino oficial de B
                     entrena sobre train_clean
    Trainer        : el oficial, con perdida L1 puesta a mano, lr 1e-4, batch 2048,
                     patience 5 y un 10% de val por corte temporal
    evaluacion     : la oficial

lo unico que varia es la semilla de entrenamiento, y lo que se mide asi es la distribucion
del CARE-CV del AE de B.

los checkpoints salen con la etiqueta de granja "b_ident_s{seed}", o sea
farm_b_ident_s{seed}_best.pt, y no tocan farm_b_best.pt.

    python train_ae_identical_b.py --seed 42
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
from compute_lead_times import load_farm_b
from run_driftdiff_c import _keep_awake
import run_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# el CARE-CV del AE-B desplegado, contra el que se compara cada reentrenamiento
FROZEN_CARE = 0.7211


def main() -> None:
    _keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite-frozen", action="store_true",
                    help="permite sobrescribir los artefactos congelados de results/models/")
    args = ap.parse_args()
    seed = args.seed
    np.random.seed(seed); torch.manual_seed(seed)
    logger.info(f"Device: {DEVICE} | Farm B | IDENTICAL AE retrain (official path) | seed={seed}")

    # la misma config fundida que usa el main de run_model para Farm B: si esto se desviara,
    # el reentrenamiento ya no seria identico y la distribucion no diria nada
    base = yaml.safe_load(open("config/model_config.yaml", encoding="utf-8"))
    cfg = run_model._deep_merge(base, base.get("per_farm", {}).get("B", {}))
    cfg["score_mode"] = cfg.get("score_mode", "mae_window")
    cfg["output"]["overwrite_frozen"] = args.overwrite_frozen

    # representacion congelada e IF-clean, que es de donde viene exactamente el 0.7211: el
    # camino oficial de B entrena el AE sobre train_clean
    train_windows, test_seqs, _ = load_farm_b()
    train_windows = train_windows.astype(np.float32, copy=False)
    W, K = train_windows.shape[1], train_windows.shape[2]
    logger.info(f"train {train_windows.shape} (IF-cleaned, frozen representation) | W={W} K={K}")

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
    # desplegados de farm_b_best.pt no se tocan
    history = Trainer(model=model, config=cfg, device=DEVICE).fit(train_windows, farm=f"b_ident_s{seed}")
    best_val = float(min(history["val_loss"]))
    logger.info(f"[AE] trained | {time.time()-t0:.0f}s | epochs={len(history['val_loss'])} best_val={best_val:.6f}")

    detector = AnomalyDetector(model=model, config=cfg, device=DEVICE)
    adapter = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm="B", window_size=W)
    # la cv-seed va fija a 42, que es el estandar del harness: si se pasara la semilla de
    # entrenamiento, cada ejecucion partiria los folds distinto y dejarian de ser comparables
    ev = run_model.evaluate(detector, train_windows, test_seqs, adapter, 42)
    b, cv = ev["best"], ev["cv"]
    logger.info("=" * 70)
    logger.info(f"IDENTICAL-AE-B seed {seed}: CARE-CV={cv['care']:.4f}  vs farm_b_best.pt {FROZEN_CARE}  "
                f"(delta {cv['care'] - FROZEN_CARE:+.4f}) | optim={cv['optimism']:+.4f} "
                f"cov={b['cov']:.3f} earl={b['earl']:.3f} rel={b['rel']:.3f} acc={b['acc']:.3f}")

    out = {"farm": "B", "experiment": "identical_ae_retrain", "seed": seed, "K": K, "window_size": int(W),
           "epochs": len(history["val_loss"]), "best_val_loss": best_val,
           "care_cv": cv["care"], "optimism": cv["optimism"], "best": b, "roc_window": ev["roc_window"],
           "ae_ckpt": f"results/models/farm_b_ident_s{seed}_best.pt"}
    p = Path(f"results/tables/ae_identical_b_s{seed}.json")
    json.dump(out, open(p, "w"), indent=2, default=run_model._np_default)
    logger.info(f"saved -> {p}")


if __name__ == "__main__":
    main()
