"""cierre de TimeADDM en Farm B, difusor sobre la representacion ganadora de B.

esa representacion es window-first 30 feats, WindowScaler standard y PCA K=18, la del
BiLSTM-AE 0.7211, y nunca se habia probado con difusion: los runs vainilla antiguos de
B (0.59-0.68) iban sobre los 30 features crudos y a una sola semilla.

dos backbones, espejo de los de run_timeaddm_c.py:

    vanilla : LSTM-AE generico (TimeADDM original, linear_output) sobre los componentes
    bfrozen : el BiLSTM-AE ganador (farm_b_best.pt) congelado, y el difusor modela su
              cuello (b,W,32). ese checkpoint se perdio y no esta en el repositorio, asi
              que --backbone bfrozen no funciona: la copia que queda en el historial de
              git (355001c, 3c12b3a) es de la linea antigua K=14 y AE_CKPT revienta con
              size mismatch contra el K=18 que monta load_farm_b hoy. --backbone vanilla
              si corre, y es la limitacion que declara el README

mismo protocolo de sweep CONFIGS x SMOOTH y CARE-CV oficial que run_timeaddm_c.py, con
modelos y JSON por semilla. los datos entran por compute_lead_times.load_farm_b, la ruta
que reproduce el 0.7211 exacto.

de las tres invocaciones solo corre la primera, las de bfrozen quedan de referencia:

    python run_timeaddm_b.py --backbone vanilla --seed 42
    python run_timeaddm_b.py --backbone bfrozen --seed 42
    python run_timeaddm_b.py --backbone bfrozen --seed 42 --load-diffusion --score-mode eps
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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (imports hermanos)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz del repo (from src.X)

from src.models.bilstm_autoencoder import BiLSTMAutoencoder
from src.timeaddm.lstmae import LSTMAE
from src.timeaddm.bilstm_adapter import BiLSTMAEAdapter
from src.timeaddm.diffusion_model import DiffusionUNet
from src.timeaddm.trainer import train_lstmae, train_diffusion
from src.care_official.adapter import OfficialCAREAdapter
from run_timeaddm import _build_betas, _make_diffusion_config
from sweep_timeaddm_scoring import score_per_level, score_per_level_eps, ReplayScorer, CONFIGS, LEVELS
from compute_lead_times import load_farm_b
import run_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTH = [1, 3, 5, 7]
AE_CKPT = "results/models/farm_b_best.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["vanilla", "bfrozen"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score-mode", choices=["recon", "eps"], default="recon")
    ap.add_argument("--load-diffusion", action="store_true", help="carga los modelos guardados de esta semilla y backbone, y solo puntua")
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--ddim", type=int, default=0)
    args = ap.parse_args()
    seed = args.seed
    # el tag entra en el nombre de los checkpoints y del JSON, asi que --load-diffusion
    # solo reencuentra lo que se guardo con la misma semilla y el mismo backbone
    tag = f"_s{seed}_{args.backbone}"
    np.random.seed(seed); torch.manual_seed(seed)
    logger.info(f"Device: {DEVICE} | Farm B (winning representation K=18) | backbone={args.backbone} "
                f"| seed={seed} | score={args.score_mode}")

    # ventanas de componentes limpiadas con IF mas las secuencias de test, la procedencia
    # exacta del 0.7211 oficial
    train_windows, test_seqs, _bounds = load_farm_b()
    train_windows = train_windows.astype(np.float32, copy=False)
    W, K = train_windows.shape[1], train_windows.shape[2]
    logger.info(f"train {train_windows.shape} | {len(test_seqs)} events | W={W} K={K}")

    config = yaml.safe_load(open("config/timeaddm_config.yaml", encoding="utf-8"))
    config["scaling"]["method"] = "standard"
    config["lstmae"]["batch_size"] = 512
    config["diffusion"]["batch_size"] = 512
    # los mismos ajustes de difusion larga que run_timeaddm_c.py
    config["diffusion"]["max_epochs"] = 100
    config["diffusion"]["early_stopping"]["patience"] = 12

    mdir = Path(config["output"]["models_dir"])
    if args.backbone == "vanilla":
        backbone = LSTMAE(input_size=K, hidden_size=config["lstmae"]["hidden_size"],
                          dropout=config["lstmae"]["dropout"], linear_output=True)
        if args.load_diffusion:
            backbone.load_state_dict(torch.load(mdir / f"timeaddm_farm_b{tag}_lstmae.pt", map_location=DEVICE))
        else:
            t0 = time.time()
            logger.info("[1/3] Training vanilla LSTM-AE on components ...")
            train_lstmae(backbone, train_windows, config, DEVICE, "B", ckpt_tag=tag)
            logger.info(f"  LSTM-AE done | {time.time()-t0:.0f}s")
    else:
        ae = BiLSTMAutoencoder(input_size=K, encoder_units=[64, 16], decoder_units=[16, 64],
                               dropout=0.1, out_activation="linear")
        # strict=False solo tolera claves de mas o de menos, no dimensiones que no
        # cuadran: un checkpoint de otra linea revienta aqui mismo
        ae.load_state_dict(torch.load(AE_CKPT, map_location=DEVICE), strict=False)
        backbone = BiLSTMAEAdapter(ae)  # frozen
        logger.info(f"[1/3] Frozen BiLSTM-AE backbone ({AE_CKPT}) | H={backbone.hidden_size}")
    backbone.to(DEVICE).eval()

    betas = _build_betas(config, torch.device(DEVICE))
    diffusion = DiffusionUNet(_make_diffusion_config(config, backbone.hidden_size, W))
    if args.load_diffusion:
        diffusion.load_state_dict(torch.load(mdir / f"timeaddm_farm_b{tag}_diffusion.pt", map_location=DEVICE))
        logger.info(f"[2/3] loaded diffusion (tag={tag})")
    else:
        t0 = time.time()
        logger.info(f"[2/3] Training Diffusion U-Net out_ch={backbone.hidden_size} "
                    f"({sum(p.numel() for p in diffusion.parameters()):,} params) ...")
        train_diffusion(diffusion, backbone, betas, train_windows, config, DEVICE, "B", ckpt_tag=tag)
        logger.info(f"  Diffusion done | {time.time()-t0:.0f}s")
    diffusion.to(DEVICE).eval()

    # submuestra de calibracion: del train solo se necesita el percentil del umbral, y
    # puntuarlo entero es la parte cara
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_windows), min(30000, len(train_windows)), replace=False)
    train_calib = train_windows[idx]
    ddim = args.ddim or config["scoring"]["ddim_steps"]  # 0 en la CLI significa el del config
    t0 = time.time()
    logger.info(f"[3/3] mode={args.score_mode} per-level precompute (levels {LEVELS}, draws={args.draws}) ...")
    if args.score_mode == "eps":
        train_pl = score_per_level_eps(backbone, diffusion, betas, train_calib, LEVELS, DEVICE, draws=args.draws)
        event_pl = [score_per_level_eps(backbone, diffusion, betas, s["windows"], LEVELS, DEVICE, draws=args.draws)
                    for s in test_seqs]
    else:
        train_pl = score_per_level(backbone, diffusion, betas, train_calib, LEVELS, ddim, DEVICE, draws=args.draws)
        event_pl = [score_per_level(backbone, diffusion, betas, s["windows"], LEVELS, ddim, DEVICE, draws=args.draws)
                    for s in test_seqs]
    logger.info(f"  precompute done | {time.time()-t0:.0f}s")

    # correlacion del score eps con la reconstruccion del backbone; solo con bfrozen hay
    # un AE-B contra el que medirla, con vanilla el backbone es el LSTM-AE generico
    pearson = None
    if args.backbone == "bfrozen" and args.score_mode == "eps":
        @torch.no_grad()
        def _ae_recon(windows):
            outs = []
            for s in range(0, len(windows), 512):
                x = torch.tensor(windows[s:s + 512], dtype=torch.float32, device=DEVICE)
                outs.append((x - backbone(x)).pow(2).mean(dim=(1, 2)).cpu().numpy())
            return np.concatenate(outs)
        ae_all = np.concatenate([_ae_recon(s["windows"]) for s in test_seqs])
        # 500 es num_diffusion_timesteps entero, o sea el extremo de ruido puro
        diff_all = np.concatenate([epl[500] for epl in event_pl])
        pearson = float(np.corrcoef(ae_all, diff_all)[0, 1])
        logger.info(f"[Corr] eps(L500) vs AE-recon: Pearson={pearson:+.3f}")

    def _smooth(s, w):
        return s if w <= 1 else np.convolve(s, np.ones(w) / w, mode="same")

    adapter_care = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm="B", window_size=W)
    # los scores por nivel ya estan precomputados, asi que cada peso y cada suavizado es
    # una recombinacion lineal gratis mas una evaluacion CARE-CV barata
    rows = []
    for name, wd in CONFIGS:
        lv = list(wd.keys())
        train_score = sum(wd[l] * train_pl[l] for l in lv)
        ev_raw = [sum(wd[l] * epl[l] for l in lv) for epl in event_pl]
        for sw in SMOOTH:
            ev_scores = [_smooth(s, sw) for s in ev_raw]
            scorer = ReplayScorer([train_score] + ev_scores, percentile=75.0)
            ev = run_model.evaluate(scorer, train_calib, test_seqs, adapter_care, seed)
            b, cv = ev["best"], ev["cv"]
            rows.append({"config": name, "smooth": sw, "care_cv": cv["care"], "oracle": b["care"],
                         "optim": cv["optimism"], "roc_ds": b["roc_ds"], "roc_win": ev["roc_window"],
                         "f1": b["f1"], "cov": b["cov"], "earl": b["earl"], "rel": b["rel"], "acc": b["acc"],
                         "h_scale": b["h_scale"]})
            logger.info(f"  {name:<18} MA{sw}  CARE-CV={cv['care']:.4f}  optim={cv['optimism']:+.4f}  "
                        f"ROC-ds={b['roc_ds']:.3f}  F1={b['f1']:.3f}")

    rows.sort(key=lambda r: -r["care_cv"])
    logger.info("=" * 70)
    logger.info(f"BEST B {args.backbone}-{args.score_mode}: {rows[0]['config']} MA{rows[0]['smooth']} -> "
                f"CARE-CV={rows[0]['care_cv']:.4f} (optim {rows[0]['optim']:+.4f})  | BiLSTM-AE B solo 0.7211")
    # solo los modos distintos de recon llevan sufijo en el nombre del JSON
    sm = "" if args.score_mode == "recon" else f"_{args.score_mode}"
    out = {"farm": "B", "model": f"timeaddm_{args.backbone}", "representation": "window_first_30feats_K18",
           "K": K, "window_size": int(W), "seed": seed, "score_mode": args.score_mode,
           "corr_diff_vs_ae": pearson, "draws": args.draws, "ddim": ddim, "sweep": rows}
    p = Path(f"results/tables/timeaddm_official_farm_b_sweep{tag}{sm}_d{args.draws}_ddim{ddim}.json")
    json.dump(out, open(p, "w"), indent=2, default=run_model._np_default)
    logger.info(f"saved -> {p}")


if __name__ == "__main__":
    main()
