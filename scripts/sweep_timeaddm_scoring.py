"""barre la config de puntuacion de TimeADDM, los niveles de ruido de la difusion y sus pesos,
sobre un modelo ya entrenado y bajo el protocolo CARE oficial, sin reentrenar.

los scores de reconstruccion por nivel se calculan una sola vez para un conjunto candidato de
niveles; cada pesado es despues una recombinacion lineal gratis mas una evaluacion CUSUM/CV
barata, reusando run_model.evaluate. pone a prueba la hipotesis de que la config por defecto,
peso 0.4 sobre el nivel de ruido 500 de 500, o sea ruido casi puro, es peor que cargar el peso
en niveles de ruido moderado o bajo.

carga results/models/timeaddm_farm_{f}_{lstmae,diffusion}.pt, el ultimo entrenado para esa
granja y ese scaler.

    python sweep_timeaddm_scoring.py --farm B --scaler standard
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

from src.timeaddm.data_adapter import load_and_scale
from src.timeaddm.lstmae import LSTMAE
from src.timeaddm.diffusion_model import DiffusionUNet
from src.timeaddm.diffusion_utils import noise_estimation_loss, ddim_denoise
from src.care_official.adapter import OfficialCAREAdapter
from run_timeaddm import _build_betas, _make_diffusion_config
import run_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# candidate noise levels to precompute; el 25 se descarto por inutil en las dos granjas
LEVELS = [50, 100, 250, 500]
# ventanas de media movil temporal sobre los scores por ventana, antes del CUSUM
SMOOTH = [1, 3, 5, 7]

# candidate scoring configs, (name, {level: weight}); los pesos no tienen por que sumar 1,
# CARE es invariante a la escala del score
CONFIGS = [
    ("baseline_default", {50: 0.1, 100: 0.2, 250: 0.3, 500: 0.4}),
    ("drop500_mid",      {50: 0.3, 100: 0.4, 250: 0.3}),
    ("invert_lowhigh",   {50: 0.4, 100: 0.3, 250: 0.2, 500: 0.1}),
    ("mid_100_250",      {100: 0.5, 250: 0.5}),
    ("low_50_100",       {50: 0.5, 100: 0.5}),
    ("single_50",        {50: 1.0}),
    ("single_100",       {100: 1.0}),
    ("single_250",       {250: 1.0}),
    ("single_500",       {500: 1.0}),
]


@torch.no_grad()
def score_per_level(lstmae, diffusion, betas, windows, levels, ddim_steps, device, batch_size=512, draws=1):
    """MSE de reconstruccion por nivel de ruido, como dict {nivel: scores (N,)}. es espejo de
    TimeADDMScorer._score_batch pero deja cada nivel por separado, para poder recombinar los
    pesados gratis."""

    out = {L: [] for L in levels}
    n = len(windows)
    for s in range(0, n, batch_size):
        x = torch.tensor(windows[s:s + batch_size], dtype=torch.float32, device=device)
        # el cuello sale (b, W, H) y el difusor Conv1d quiere el canal en medio, (b, H, W)
        z_perm = lstmae.encode(x).permute(0, 2, 1)
        for L in levels:
            # los pasos del difusor van indexados desde 0, de ahi el L-1
            t_vec = torch.full((x.size(0),), L - 1, dtype=torch.long, device=device)
            acc = torch.zeros(x.size(0), device=device)
            # draws promedia K realizaciones del ruido y baja la varianza del score: menos
            # picos espurios en ventanas normales, o sea mas reliability
            for _ in range(draws):
                e = torch.randn_like(z_perm)
                z_noisy, _ = noise_estimation_loss(diffusion, z_perm, t_vec, e, betas)
                z_clean = ddim_denoise(z_noisy, L, diffusion, betas, ddim_steps)
                x_recon = lstmae.decode(z_clean.permute(0, 2, 1))
                # el score es el MSE por ventana entre la entrada y su reconstruccion
                acc = acc + (x - x_recon).pow(2).mean(dim=(1, 2))
            out[L].append((acc / draws).cpu().numpy())
    return {L: np.concatenate(v) for L, v in out.items()}


@torch.no_grad()
def score_per_level_eps(lstmae, diffusion, betas, windows, levels, device, batch_size=512, draws=1):
    """score de anomalia nativo de la difusion, el error de prediccion de ruido por ventana
    ||e - eps_theta(z_t, L)|| ^2 a cada nivel de ruido fijo L, en el espacio latente y sin
    decodificar. la difusion se entreno sobre latentes normales, asi que los anomalos caen
    fuera de la variedad y se predicen peor, lo que sube el error. no decodificar nunca es lo
    que mantiene este score decorrelado de la reconstruccion del propio AE, al reves que
    score_per_level, que pasa por el decode y acaba siendo redundante con el AE."""

    out = {L: [] for L in levels}
    n = len(windows)
    # alfa acumulada del proceso directo, el factor que escala senal y ruido en cada paso
    acum = (1.0 - betas).cumprod(dim=0)
    for s in range(0, n, batch_size):
        x = torch.tensor(windows[s:s + batch_size], dtype=torch.float32, device=device)
        z = lstmae.encode(x).permute(0, 2, 1)  # (b, H, W)
        for L in levels:
            # (1,1,1) para que multiplique por broadcast contra el latente entero
            a = acum[L - 1].view(1, 1, 1)
            t_vec = torch.full((x.size(0),), L - 1, dtype=torch.long, device=device)
            acc = torch.zeros(x.size(0), device=device)
            # draws promedia realizaciones del ruido, igual que en score_per_level
            for _ in range(draws):
                e = torch.randn_like(z)
                # forma cerrada del proceso directo: ruidar z hasta el paso L de una vez
                z_t = z * a.sqrt() + e * (1.0 - a).sqrt()
                pred = diffusion(z_t, t_vec.float())
                # el score es cuanto se equivoca la red al predecir el ruido que se metio
                acc = acc + (e - pred).square().sum(dim=(1, 2))
            out[L].append((acc / draws).cpu().numpy())
    return {L: np.concatenate(v) for L, v in out.items()}


class ReplayScorer:
    """feeds precomputed per-window scores to run_model.evaluate in call order, el train
    primero y despues cada evento."""

    def __init__(self, scores: list[np.ndarray], percentile: float = 75.0) -> None:
        self._scores = scores
        self._i = 0
        self.percentile = percentile


    # windows se ignora a proposito: los scores ya vienen calculados y se sirven en orden,
    # asi que este contador es lo unico que empareja cada llamada con su serie
    def score_windows(self, windows):  # noqa: ARG002
        s = self._scores[self._i]
        self._i += 1
        return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", choices=["B", "C"], required=True)
    ap.add_argument("--scaler", choices=["minmax", "standard"], default="standard")
    ap.add_argument("--raw-path", default="data/raw/CARE_To_Compare")
    ap.add_argument("--ddim-steps", type=int, default=40)
    ap.add_argument("--calib-sample", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open("config/timeaddm_config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["scaling"]["method"] = args.scaler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    farm, fl = args.farm, args.farm.lower()
    # con StandardScaler los componentes no quedan acotados en [0,1], asi que el AE que
    # corresponde es el linear_output
    linear = args.scaler == "standard"
    logger.info(f"Device: {device} | Farm {farm} | scaler={args.scaler} | ddim_steps={args.ddim_steps}")

    train_scaled, test_seqs, feat_names, _ = load_and_scale(config, farm, None)
    W, F = train_scaled.shape[1], train_scaled.shape[2]

    # se carga el modelo entrenado, la ultima ejecucion de esta granja
    mdir = Path(config["output"]["models_dir"])
    lstmae = LSTMAE(input_size=F, hidden_size=config["lstmae"]["hidden_size"],
                    dropout=config["lstmae"]["dropout"], linear_output=linear)
    lstmae.load_state_dict(torch.load(mdir / f"timeaddm_farm_{fl}_lstmae.pt", map_location=device))
    lstmae.to(device).eval()
    diffusion = DiffusionUNet(_make_diffusion_config(config, config["lstmae"]["hidden_size"], W))
    diffusion.load_state_dict(torch.load(mdir / f"timeaddm_farm_{fl}_diffusion.pt", map_location=device))
    diffusion.to(device).eval()
    betas = _build_betas(config, torch.device(device))

    # precompute per-level scores once, la parte cara; sobre una submuestra del train,
    # que es de donde sale el percentil del umbral
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_scaled), min(args.calib_sample, len(train_scaled)), replace=False)
    train_calib = train_scaled[idx]
    t0 = time.time()
    logger.info(f"Precomputing per-level scores (levels {LEVELS}) on {len(train_calib):,} train + "
                f"{sum(len(s['windows']) for s in test_seqs):,} test windows ...")
    train_pl = score_per_level(lstmae, diffusion, betas, train_calib, LEVELS, args.ddim_steps, device)
    event_pl = [score_per_level(lstmae, diffusion, betas, s["windows"], LEVELS, args.ddim_steps, device)
                for s in test_seqs]
    logger.info(f"  per-level precompute done | {time.time()-t0:.0f}s")

    def _smooth(s, w):
        return s if w <= 1 else np.convolve(s, np.ones(w) / w, mode="same")

    adapter = OfficialCAREAdapter(raw_path=args.raw_path, farm=farm, window_size=W)
    rows = []
    for name, wd in CONFIGS:
        lv = list(wd.keys())
        # cada config es una suma ponderada de los niveles ya precomputados, gratis
        train_score = sum(wd[l] * train_pl[l] for l in lv)
        ev_raw = [sum(wd[l] * epl[l] for l in lv) for epl in event_pl]
        # MA temporal sobre los scores por ventana, gratis
        for sw in SMOOTH:
            ev_scores = [_smooth(s, sw) for s in ev_raw]
            # el orden importa: evaluate pide primero el train y despues cada evento, que
            # es como ReplayScorer los va sirviendo
            scorer = ReplayScorer([train_score] + ev_scores, percentile=75.0)
            ev = run_model.evaluate(scorer, train_calib, test_seqs, adapter, seed)
            b, cv = ev["best"], ev["cv"]
            rows.append({"config": name, "smooth": sw, "levels": lv, "care_cv": cv["care"], "oracle": b["care"],
                         "optim": cv["optimism"], "roc_ds": b["roc_ds"], "roc_win": ev["roc_window"],
                         "f1": b["f1"], "h_scale": b["h_scale"]})
            logger.info(f"  {name:<18} MA{sw}  CARE-CV={cv['care']:.4f}  optim={cv['optimism']:+.4f}  "
                        f"ROC-ds={b['roc_ds']:.3f}  F1={b['f1']:.3f}")

    # se ordena por el CARE de validacion cruzada, no por el ajustado sobre el test
    rows.sort(key=lambda r: -r["care_cv"])
    logger.info("=" * 70)
    logger.info(f"BEST scoring (Farm {farm}, {args.scaler}): {rows[0]['config']} MA{rows[0]['smooth']} "
                f"-> CARE-CV={rows[0]['care_cv']:.4f} (optim {rows[0]['optim']:+.4f}, ROC-ds {rows[0]['roc_ds']:.3f})")
    out = {"farm": farm, "scaler": args.scaler, "ddim_steps": args.ddim_steps, "sweep": rows}
    p = Path(config["output"]["tables_dir"]) / f"timeaddm_scoring_sweep_farm_{fl}_{args.scaler}.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=run_model._np_default)
    logger.info(f"saved -> {p}")


if __name__ == "__main__":
    main()
