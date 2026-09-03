"""difusion TimeADDM para Farm C bajo el protocolo oficial, sobre la representacion ganadora.

esa representacion son las ventanas de componentes pca-first que van de top-100 a 55, la
entrada exacta que usa el modelo BiLSTM-AE de C (CARE-CV 0.7432). se reusa
run_model._prep_pca_first para construirlas desde el crudo, cacheadas, y sobre ellas se
entrenan el LSTM-AE (salida lineal, las componentes vienen de un StandardScaler) y la difusion.

despues se barre la config de puntuacion de la difusion, la palanca que llevo Farm B de 0.62 a
0.68, y se reporta el CARE-CV oficial por config, comparable con el BiLSTM-AE de C (0.7432).

    python run_timeaddm_c.py
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (imports hermanos)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz del repo (from src.X)

from src.timeaddm.lstmae import LSTMAE
from src.timeaddm.diffusion_model import DiffusionUNet
from src.timeaddm.trainer import train_lstmae, train_diffusion
from src.model_prep.isolation_forest_cleaner import IFCleaner
from src.care_official.adapter import OfficialCAREAdapter
from run_timeaddm import _build_betas, _make_diffusion_config
from sweep_timeaddm_scoring import score_per_level, ReplayScorer, CONFIGS, LEVELS
import run_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEED = 42  # semilla de entrenamiento por defecto, se cambia con --seed
REPR_SEED = 42  # la representacion va fija a la semilla 42, componentes cacheadas
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = Path("data/processed/farm_c_pcafirst_timeaddm.pkl")
SMOOTH = [1, 3, 5, 7]  # ventanas de media movil sobre los scores, antes del CUSUM


def get_components():
    """ventanas de componentes pca-first de C (train + test), la representacion ganadora del
    BiLSTM de C. va cacheada."""
    if CACHE.exists():
        logger.info(f"Loading cached C components from {CACHE}")
        d = pickle.load(open(CACHE, "rb"))
        return d["train"], d["test"], d["K"]
    logger.info("Generating C pca-first components from raw (run_model._prep_pca_first) — ~20 min ...")
    base = yaml.safe_load(open("config/model_config.yaml", encoding="utf-8"))
    pcfg = run_model.load_preproc_config()
    cfg = run_model._deep_merge(base, base.get("per_farm", {}).get("C", {}))
    train_w, test_seqs, _raw, _feats, K, _sc, _pca = run_model._prep_pca_first("C", cfg, pcfg, REPR_SEED)
    pickle.dump({"train": train_w, "test": test_seqs, "K": int(K)}, open(CACHE, "wb"))
    logger.info(f"  cached -> {CACHE}")
    return train_w, test_seqs, int(K)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-model", action="store_true", help="carga el modelo de C guardado en vez de reentrenar")
    ap.add_argument("--draws", type=int, default=1, help="promedia K extracciones de ruido por ventana (baja la varianza del score y sube la reliability)")
    ap.add_argument("--ddim", type=int, default=0, help="pasos DDIM para puntuar (0 = el valor por defecto de la config)")
    ap.add_argument("--hidden", type=int, default=0, help="tamano oculto del LSTM-AE (0 = el 64 por defecto de la config)")
    ap.add_argument("--if-clean", action="store_true", help="limpia el train con Isolation Forest (normal mas puro)")
    ap.add_argument("--long-diffusion", action="store_true", help="mas epocas y mas paciencia en la difusion")
    ap.add_argument("--seed", type=int, default=SEED, help="semilla de entrenamiento (la representacion va fija a la semilla 42)")
    ap.add_argument("--ema-decay", type=float, default=0.0, help="decaimiento EMA sobre los pesos de la difusion (0=apagado, p.ej. 0.999)")
    ap.add_argument("--exp-tag", type=str, default="", help="etiqueta de experimento que se anade a los nombres de modelo y JSON (deja los experimentos uno al lado del otro)")
    args = ap.parse_args()
    seed = args.seed
    tag = f"_s{seed}" + (f"_{args.exp_tag}" if args.exp_tag else "")  # sufijo de modelo/JSON
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Device: {DEVICE} | Farm C | pca-first components (BiLSTM-C winning representation) | seed={seed}")

    train_windows, test_seqs, K = get_components()
    W = train_windows.shape[1]
    logger.info(f"train {train_windows.shape} | W={W} K(components)={K} | "
                f"range [{train_windows.min():.2f}, {train_windows.max():.2f}]")

    config = yaml.safe_load(open("config/timeaddm_config.yaml", encoding="utf-8"))
    config["scaling"]["method"] = "standard"  # las componentes salen de StandardScaler
    config["lstmae"]["batch_size"] = 512
    config["diffusion"]["batch_size"] = 512
    if args.hidden:
        config["lstmae"]["hidden_size"] = args.hidden
    if args.long_diffusion:
        config["diffusion"]["max_epochs"] = 100
        config["diffusion"]["early_stopping"]["patience"] = 12
    if args.ema_decay > 0:
        config["diffusion"]["ema_decay"] = args.ema_decay
    hidden = config["lstmae"]["hidden_size"]
    if args.if_clean and not args.load_model:
        n0 = len(train_windows)
        train_windows = IFCleaner(contamination=0.01, n_estimators=100, random_state=42,
                                  n_jobs=-1).fit_transform(train_windows)
        logger.info(f"IF-clean train: {n0:,} -> {len(train_windows):,} windows")

    # LSTM-AE input = K components; linear output (signed, unbounded components).
    mdir = Path(config["output"]["models_dir"])
    lstmae = LSTMAE(input_size=K, hidden_size=hidden, dropout=config["lstmae"]["dropout"], linear_output=True)
    betas = _build_betas(config, torch.device(DEVICE))
    diffusion = DiffusionUNet(_make_diffusion_config(config, hidden, W))
    if args.load_model:
        logger.info(f"[1-2/3] Loading saved C model (no retrain) tag={tag} ...")
        lstmae.load_state_dict(torch.load(mdir / f"timeaddm_farm_c{tag}_lstmae.pt", map_location=DEVICE))
        diffusion.load_state_dict(torch.load(mdir / f"timeaddm_farm_c{tag}_diffusion.pt", map_location=DEVICE))
    else:
        t0 = time.time()
        logger.info("[1/3] Training LSTM-AE on components ...")
        train_lstmae(lstmae, train_windows, config, DEVICE, "C", ckpt_tag=tag)
        logger.info(f"  LSTM-AE done | {time.time()-t0:.0f}s")
        t0 = time.time()
        logger.info(f"[2/3] Training Diffusion U-Net ({sum(p.numel() for p in diffusion.parameters()):,} params) ...")
        train_diffusion(diffusion, lstmae, betas, train_windows, config, DEVICE, "C", ckpt_tag=tag)
        logger.info(f"  Diffusion done | {time.time()-t0:.0f}s")
    lstmae.to(DEVICE).eval()
    diffusion.to(DEVICE).eval()

    # [3/3] barrido de puntuacion bajo el protocolo oficial, precomputo por nivel y repesado
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_windows), min(30000, len(train_windows)), replace=False)
    train_calib = train_windows[idx]
    ddim = args.ddim or config["scoring"]["ddim_steps"]
    t0 = time.time()
    logger.info(f"[3/3] Per-level precompute (levels {LEVELS}, ddim={ddim}, draws={args.draws}) + scoring sweep ...")
    train_pl = score_per_level(lstmae, diffusion, betas, train_calib, LEVELS, ddim, DEVICE, draws=args.draws)
    event_pl = [score_per_level(lstmae, diffusion, betas, s["windows"], LEVELS, ddim, DEVICE, draws=args.draws)
                for s in test_seqs]
    logger.info(f"  precompute done | {time.time()-t0:.0f}s")

    def _smooth(s, w):
        return s if w <= 1 else np.convolve(s, np.ones(w) / w, mode="same")

    adapter = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm="C", window_size=W)
    rows = []
    for name, wd in CONFIGS:
        lv = list(wd.keys())
        train_score = sum(wd[l] * train_pl[l] for l in lv)
        ev_raw = [sum(wd[l] * epl[l] for l in lv) for epl in event_pl]
        for sw in SMOOTH:  # MA temporal por ventana, gratis
            ev_scores = [_smooth(s, sw) for s in ev_raw]
            scorer = ReplayScorer([train_score] + ev_scores, percentile=75.0)
            ev = run_model.evaluate(scorer, train_calib, test_seqs, adapter, seed)
            b, cv = ev["best"], ev["cv"]
            rows.append({"config": name, "smooth": sw, "care_cv": cv["care"], "oracle": b["care"],
                         "optim": cv["optimism"], "roc_ds": b["roc_ds"], "roc_win": ev["roc_window"],
                         "f1": b["f1"], "cov": b["cov"], "earl": b["earl"], "rel": b["rel"], "acc": b["acc"],
                         "h_scale": b["h_scale"]})
            logger.info(f"  {name:<18} MA{sw}  CARE-CV={cv['care']:.4f}  optim={cv['optimism']:+.4f}  "
                        f"ROC-ds={b['roc_ds']:.3f}  F1={b['f1']:.3f}")

    rows.sort(key=lambda r: -r["care_cv"])
    logger.info("=" * 70)
    logger.info(f"BEST C diffusion: {rows[0]['config']} MA{rows[0]['smooth']} -> CARE-CV={rows[0]['care_cv']:.4f} "
                f"(optim {rows[0]['optim']:+.4f}, ROC-ds {rows[0]['roc_ds']:.3f})  | BiLSTM-C ref 0.7432")
    out = {"farm": "C", "model": "timeaddm_diffusion", "representation": "pca_first_top100_components",
           "K": K, "window_size": int(W), "seed": seed, "draws": args.draws, "ddim": ddim,
           "ema_decay": args.ema_decay, "exp_tag": args.exp_tag, "hidden": hidden, "sweep": rows}
    # el nombre del JSON lleva dentro la semilla y las palancas de puntuacion (draws y ddim),
    # asi dos repuntuaciones de la fase 1 no se pisan entre si
    jtag = f"{tag}_d{args.draws}_ddim{ddim}"
    p = Path(f"results/tables/timeaddm_official_farm_c_sweep{jtag}.json")
    json.dump(out, open(p, "w"), indent=2, default=run_model._np_default)
    logger.info(f"saved -> {p}")


if __name__ == "__main__":
    main()
