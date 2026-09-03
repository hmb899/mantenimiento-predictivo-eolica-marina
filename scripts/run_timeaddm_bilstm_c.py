"""BiLSTM-AE, el modelo ganador de C, como backbone del difusor TimeADDM.

el LSTM-AE vainilla de TimeADDM difunde un unico vector de cuello repetido W veces; aqui
el encoder y el decoder son el BiLSTM-AE de C, cuyo cuello es una secuencia real
(b, W, H=32) que el difusor Conv1d modela directamente. tres modos del AE en --ae-mode:

    frozen   : carga results/models/farm_c_best.pt, lo congela y entrena solo el difusor
    retrain  : BiLSTM-AE nuevo desde cero por reconstruccion, se congela y va al difusor
    finetune : parte de farm_c_best.pt y entrena AE y difusor a la vez, con la perdida
               ruido_difusion + lambda * MSE_reconstruccion y el AE a LR bajo

mismo protocolo de scoring y CARE-CV oficial que run_timeaddm_c.py, del que reusa
score_per_level, CONFIGS, ReplayScorer y run_model.evaluate.

    python run_timeaddm_bilstm_c.py --ae-mode frozen   --seed 42 --if-clean
    python run_timeaddm_bilstm_c.py --ae-mode finetune --seed 42 --if-clean \\
        --lam 1.0 --ae-lr 1e-5
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
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (imports hermanos)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz del repo (from src.X)

from src.models.bilstm_autoencoder import BiLSTMAutoencoder
from src.timeaddm.bilstm_adapter import BiLSTMAEAdapter
from src.timeaddm.diffusion_model import DiffusionUNet
from src.timeaddm.diffusion_utils import noise_estimation_loss
from src.timeaddm.trainer import train_lstmae, train_diffusion, EarlyStopping
from src.model_prep.isolation_forest_cleaner import IFCleaner
from src.care_official.adapter import OfficialCAREAdapter
from run_timeaddm import _build_betas, _make_diffusion_config
from run_timeaddm_c import get_components, REPR_SEED
from sweep_timeaddm_scoring import score_per_level, score_per_level_eps, ReplayScorer, CONFIGS, LEVELS
import run_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTH = [1, 3, 5, 7]
# arquitectura del BiLSTM-AE de Farm C (config/model_config.yaml), salida lineal sobre
# componentes de StandardScaler
ENC_UNITS, DEC_UNITS, DROPOUT = [64, 16], [16, 64], 0.1


def build_bilstm_ae(K: int) -> BiLSTMAutoencoder:
    return BiLSTMAutoencoder(input_size=K, encoder_units=ENC_UNITS, decoder_units=DEC_UNITS,
                             dropout=DROPOUT, bidirectional=True, out_activation="linear")


def train_diffusion_joint(diffusion, adapter, betas, train_windows, config, device, ckpt_tag, lam, ae_lr):
    """fine-tuning end-to-end del BiLSTM-AE (adapter.ae) y la difusion a la vez, el termino de
    recon ancla el latente para que siga siendo informativo y el termino de difusion le deja
    volverse mas facil de desruidar."""

    cfg = config["diffusion"]
    device = torch.device(device)
    diffusion.to(device); adapter.to(device)
    x_full = torch.tensor(train_windows, dtype=torch.float32)
    # el corte de validacion es la cola por posicion, no un sorteo, para que sea el mismo
    # que usa train_diffusion en los otros dos modos
    n = len(x_full); n_val = max(1, int(n * cfg["val_fraction"])); n_train = n - n_val
    train_loader = DataLoader(TensorDataset(x_full[:n_train]), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(x_full[n_train:]), batch_size=cfg["batch_size"], shuffle=False)

    # AE at ae_lr (low), diffusion at the configured LR: el AE ya venia entrenado y solo
    # se le deja moverse un poco
    optimizer = torch.optim.Adam([
        {"params": diffusion.parameters(), "lr": cfg["learning_rate"]},
        {"params": adapter.ae.parameters(), "lr": ae_lr},
    ])
    n_steps = betas.shape[0]
    recon_crit = nn.MSELoss()
    es_cfg = cfg["early_stopping"]
    models_dir = Path(config["output"]["models_dir"])
    ckpt_d = models_dir / f"timeaddm_farm_c{ckpt_tag}_diffusion.pt"
    ckpt_ae = models_dir / f"timeaddm_farm_c{ckpt_tag}_bilstmae.pt"

    best_val, best_d, best_ae, bad = float("inf"), None, None, 0
    logger.info(f"[Joint] Farm C: {n_train:,} train / {n - n_train:,} val | lambda={lam} ae_lr={ae_lr} "
                f"batch={cfg['batch_size']} max_epochs={cfg['max_epochs']} T={n_steps}")

    for epoch in range(1, cfg["max_epochs"] + 1):
        diffusion.train(); adapter.ae.train()
        td, tr = [], []
        for (batch,) in train_loader:
            x = batch.to(device)
            z = adapter.encode(x)
            recon = adapter.decode(z)
            rloss = recon_crit(recon, x)
            # el cuello sale (b, W, H) y Conv1d quiere el canal en medio, o sea (b, H, W)
            z_perm = z.permute(0, 2, 1)
            t = torch.randint(0, n_steps, (x.size(0),), device=device)
            e = torch.randn_like(z_perm)
            _, dloss = noise_estimation_loss(diffusion, z_perm, t, e, betas)
            # loss = diffusion_noise_loss(z) + lam * MSE(decode(encode(x)), x)
            loss = dloss + lam * rloss
            optimizer.zero_grad()
            loss.backward()
            if cfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(list(diffusion.parameters()) + list(adapter.ae.parameters()), cfg["grad_clip"])
            optimizer.step()
            td.append(dloss.item()); tr.append(rloss.item())

        diffusion.eval(); adapter.ae.eval()
        vd, vr = [], []
        with torch.no_grad():
            for (batch,) in val_loader:
                x = batch.to(device)
                z = adapter.encode(x)
                vr.append(recon_crit(adapter.decode(z), x).item())
                z_perm = z.permute(0, 2, 1)
                t = torch.randint(0, n_steps, (x.size(0),), device=device)
                e = torch.randn_like(z_perm)
                _, dloss = noise_estimation_loss(diffusion, z_perm, t, e, betas)
                vd.append(dloss.item())
        vdl, vrl = float(np.mean(vd)), float(np.mean(vr))
        combined = vdl + lam * vrl
        logger.info(f"  [Joint] epoch {epoch:>3}/{cfg['max_epochs']} train_d {np.mean(td):.4f} recon {np.mean(tr):.5f} "
                    f"| val_d {vdl:.4f} recon {vrl:.5f} combined {combined:.4f}")

        # para pronto segun la perdida de validacion combinada, y guarda el mejor de los dos
        if combined < best_val - es_cfg["min_delta"]:
            best_val, bad = combined, 0
            best_d = {k: v.cpu().clone() for k, v in diffusion.state_dict().items()}
            best_ae = {k: v.cpu().clone() for k, v in adapter.ae.state_dict().items()}
        else:
            bad += 1
            if bad >= es_cfg["patience"]:
                logger.info(f"  [Joint] early stop at epoch {epoch}, best combined {best_val:.4f}")
                break

    if best_d is not None:
        diffusion.load_state_dict(best_d); adapter.ae.load_state_dict(best_ae)
    torch.save(diffusion.state_dict(), ckpt_d)
    torch.save(adapter.ae.state_dict(), ckpt_ae)
    logger.info(f"  [Joint] saved diffusion -> {ckpt_d} | ae -> {ckpt_ae}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-mode", choices=["frozen", "retrain", "finetune"], required=True)
    ap.add_argument("--seed", type=int, default=42, help="semilla de entrenamiento (la representacion va fija a la semilla 42)")
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--ddim", type=int, default=0, help="0 = el valor por defecto de la config")
    ap.add_argument("--if-clean", action="store_true", help="limpia con IF el conjunto de train de la difusion")
    ap.add_argument("--long-diffusion", action="store_true", help="mas epocas y mas paciencia en la difusion")
    ap.add_argument("--lam", type=float, default=1.0, help="finetune, peso de la perdida de reconstruccion")
    ap.add_argument("--ae-lr", type=float, default=1e-5, help="finetune, LR de los parametros del BiLSTM-AE")
    ap.add_argument("--ae-ckpt", default="results/models/farm_c_best.pt", help="pesos del BiLSTM-AE ya entrenado")
    ap.add_argument("--load-diffusion", action="store_true", help="se salta el entrenamiento, carga la difusion guardada de esta semilla y modo, y solo puntua")
    ap.add_argument("--score-mode", choices=["recon", "eps"], default="recon",
                    help="recon=MSE de decodificacion (redundante con el AE); eps=error nativo de prediccion de ruido de la difusion (decorrelado)")
    ap.add_argument("--exp-tag", type=str, default="", help="etiqueta extra para los nombres de modelo y JSON (evita pisar otras ejecuciones)")
    args = ap.parse_args()
    seed = args.seed
    mode = args.ae_mode
    # el tag nombra checkpoints y JSON: semilla y modo lo separan todo salvo dos ejecuciones
    # del mismo par, que es para lo que esta --exp-tag
    tag = f"_s{seed}_bilstm_{mode}" + (f"_{args.exp_tag}" if args.exp_tag else "")
    np.random.seed(seed); torch.manual_seed(seed)
    logger.info(f"Device: {DEVICE} | Farm C | BiLSTM-AE backbone | ae_mode={mode} | seed={seed}")

    train_windows, test_seqs, K = get_components()
    W = train_windows.shape[1]
    logger.info(f"train {train_windows.shape} | W={W} K={K} | range [{train_windows.min():.2f}, {train_windows.max():.2f}]")

    config = yaml.safe_load(open("config/timeaddm_config.yaml", encoding="utf-8"))
    config["scaling"]["method"] = "standard"
    config["lstmae"]["batch_size"] = 512
    config["diffusion"]["batch_size"] = 512
    if args.long_diffusion:
        config["diffusion"]["max_epochs"] = 100
        config["diffusion"]["early_stopping"]["patience"] = 12

    if args.if_clean:
        n0 = len(train_windows)
        train_windows = IFCleaner(contamination=0.01, n_estimators=100, random_state=42,
                                  n_jobs=-1).fit_transform(train_windows)
        logger.info(f"IF-clean train: {n0:,} -> {len(train_windows):,} windows")

    # se construye el backbone del BiLSTM-AE segun el modo
    ae = build_bilstm_ae(K)
    if mode in ("frozen", "finetune"):
        # strict=False no exige que las claves casen exactamente, asi que la carga puede
        # salir a medias; de ahi que se loguee cuantas faltan y cuantas sobran
        sd = torch.load(args.ae_ckpt, map_location=DEVICE)
        miss, unexp = ae.load_state_dict(sd, strict=False)
        logger.info(f"[AE] loaded {args.ae_ckpt} (missing={len(miss)} unexpected={len(unexp)})")
    elif mode == "retrain":
        if args.load_diffusion:
            ae.load_state_dict(torch.load(Path(config["output"]["models_dir"]) / f"timeaddm_farm_c{tag}_lstmae.pt",
                                          map_location=DEVICE))
            logger.info("[AE] loaded saved retrained BiLSTM-AE (no retrain)")
        else:
            logger.info("[AE] retraining BiLSTM-AE from scratch (reconstruction) ...")
            t0 = time.time()
            # bucle de reconstruccion por MSE
            train_lstmae(ae, train_windows, config, DEVICE, "C", ckpt_tag=tag)
            logger.info(f"[AE] retrain done | {time.time()-t0:.0f}s")

    adapter = BiLSTMAEAdapter(ae, freeze=(mode != "finetune"))
    hidden = adapter.hidden_size
    logger.info(f"[AE] adapter hidden_size(H)={hidden} input_size(K)={adapter.input_size}")

    betas = _build_betas(config, torch.device(DEVICE))
    diffusion = DiffusionUNet(_make_diffusion_config(config, hidden, W))
    logger.info(f"[Diffusion] U-Net out_ch={hidden} ({sum(p.numel() for p in diffusion.parameters()):,} params)")

    # se entrena la difusion, o se carga
    mdir = Path(config["output"]["models_dir"])
    if args.load_diffusion:
        diffusion.load_state_dict(torch.load(mdir / f"timeaddm_farm_c{tag}_diffusion.pt", map_location=DEVICE))
        if mode == "finetune":  # finetune tambien persistio su AE co-entrenado
            adapter.ae.load_state_dict(torch.load(mdir / f"timeaddm_farm_c{tag}_bilstmae.pt", map_location=DEVICE))
        logger.info(f"[Diffusion] loaded saved model (tag={tag}, no retrain)")
    else:
        t0 = time.time()
        if mode == "finetune":
            train_diffusion_joint(diffusion, adapter, betas, train_windows, config, DEVICE, tag, args.lam, args.ae_lr)
        else:
            train_diffusion(diffusion, adapter, betas, train_windows, config, DEVICE, "C", ckpt_tag=tag)
        logger.info(f"  Diffusion done | {time.time()-t0:.0f}s")
    adapter.to(DEVICE).eval()
    diffusion.to(DEVICE).eval()

    # scoring sweep under the official protocol, el mismo que run_timeaddm_c.py

    # submuestra de calibracion: del train solo se necesita el percentil del umbral, y
    # puntuarlo entero es la parte cara
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_windows), min(30000, len(train_windows)), replace=False)
    train_calib = train_windows[idx]
    ddim = args.ddim or config["scoring"]["ddim_steps"]
    t0 = time.time()
    logger.info(f"[Scoring] mode={args.score_mode} per-level precompute (levels {LEVELS}, draws={args.draws}) ...")
    if args.score_mode == "eps":
        train_pl = score_per_level_eps(adapter, diffusion, betas, train_calib, LEVELS, DEVICE, draws=args.draws)
        event_pl = [score_per_level_eps(adapter, diffusion, betas, s["windows"], LEVELS, DEVICE, draws=args.draws)
                    for s in test_seqs]
    else:
        train_pl = score_per_level(adapter, diffusion, betas, train_calib, LEVELS, ddim, DEVICE, draws=args.draws)
        event_pl = [score_per_level(adapter, diffusion, betas, s["windows"], LEVELS, ddim, DEVICE, draws=args.draws)
                    for s in test_seqs]
    logger.info(f"  precompute done | {time.time()-t0:.0f}s")

    # correlacion del score de difusion con la reconstruccion del AE; decorrelar es requisito
    # previo para que la fusion sume, y solo en modo eps es una medida real: en recon el score
    # ES esa reconstruccion y se correlacionaria consigo mismo
    pearson = None
    if args.score_mode == "eps":
        @torch.no_grad()
        def _ae_recon(windows):
            outs = []
            for s in range(0, len(windows), 512):
                x = torch.tensor(windows[s:s + 512], dtype=torch.float32, device=DEVICE)
                outs.append((x - adapter(x)).pow(2).mean(dim=(1, 2)).cpu().numpy())
            return np.concatenate(outs)
        ae_all = np.concatenate([_ae_recon(s["windows"]) for s in test_seqs])
        # nivel 50 de 500, el extremo de poco ruido del rango precomputado
        diff_all = np.concatenate([event_pl[i][50] for i in range(len(test_seqs))])
        pearson = float(np.corrcoef(ae_all, diff_all)[0, 1])
        logger.info(f"[Corr] diffusion(eps,L50) vs AE-recon over {len(ae_all):,} test windows: Pearson={pearson:+.3f} "
                    f"(|r| bajo => decorrelado => fusion util)")

    def _smooth(s, w):
        return s if w <= 1 else np.convolve(s, np.ones(w) / w, mode="same")

    adapter_care = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm="C", window_size=W)
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
    logger.info(f"BEST C BiLSTM-{mode}: {rows[0]['config']} MA{rows[0]['smooth']} -> CARE-CV={rows[0]['care_cv']:.4f} "
                f"(optim {rows[0]['optim']:+.4f}, ROC-ds {rows[0]['roc_ds']:.3f})  | BiLSTM-AE solo 0.7432, baseline 0.7247")
    out = {"farm": "C", "model": f"timeaddm_bilstm_{mode}", "representation": "pca_first_top100_components",
           "K": K, "window_size": int(W), "seed": seed, "ae_mode": mode, "score_mode": args.score_mode,
           "corr_diff_vs_ae": pearson, "draws": args.draws, "ddim": ddim,
           "lam": args.lam if mode == "finetune" else None, "hidden": hidden, "sweep": rows}
    # solo los modos distintos de recon llevan sufijo en el nombre del JSON
    sm = "" if args.score_mode == "recon" else f"_{args.score_mode}"
    jtag = f"{tag}{sm}_d{args.draws}_ddim{ddim}"
    p = Path(f"results/tables/timeaddm_official_farm_c_sweep{jtag}.json")
    json.dump(out, open(p, "w"), indent=2, default=run_model._np_default)
    logger.info(f"saved -> {p}")


if __name__ == "__main__":
    main()
