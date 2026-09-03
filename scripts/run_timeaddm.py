"""corre el pipeline de TimeADDM sobre Farm B y Farm C.

dos variantes de features, la A coge todas las del dataset preprocesado y la B el
subconjunto guiado por SHAP ("D" en Farm B, "B" en Farm C).

    python run_timeaddm.py                        # las dos granjas, variante A
    python run_timeaddm.py --farm B               # solo Farm B, variante A
    python run_timeaddm.py --farm B --variant B   # Farm B, subconjunto SHAP
    python run_timeaddm.py --farm C --variant B   # Farm C, subconjunto SHAP
    python run_timeaddm.py --eval-only --farm B   # evalua artefactos guardados, sin reentrenar
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

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (imports hermanos)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz del repo (from src.X)

from src.timeaddm.data_adapter import load_and_scale
from src.timeaddm.lstmae import LSTMAE
from src.timeaddm.diffusion_model import DiffusionUNet
from src.timeaddm.diffusion_utils import get_beta_schedule
from src.timeaddm.trainer import train_lstmae, train_diffusion
from src.timeaddm.anomaly_scorer import TimeADDMScorer
from src.models.evaluator import CAREEvaluator
from src.models.diagnostics import plot_score_vs_time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# gancho de json.dump: los escalares de numpy no son serializables de serie
def _np_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _make_diffusion_config(config: dict, hidden_size: int, window_size: int):
    """construye un argparse.Namespace con el formato de config que espera DiffusionUNet."""

    mc = config["model"]

    # DiffusionUNet lee atributos anidados (cfg.model.ch), no claves de dict
    class _NS:
        pass

    cfg = _NS()
    cfg.model = _NS()
    cfg.model.ch = mc["ch"]
    cfg.model.out_ch = hidden_size  # el difusor genera el cuello del AE, no features crudas
    cfg.model.ch_mult = mc["ch_mult"]
    cfg.model.num_res_blocks = mc["num_res_blocks"]
    cfg.model.dropout = mc["dropout"]
    cfg.model.resamp_with_conv = mc["resamp_with_conv"]
    cfg.model.attn_resolutions = mc["attn_resolutions"]
    cfg.data = _NS()
    cfg.data.image_size = window_size  # el latente va como imagen 1-D de largo W
    return cfg


def _build_betas(config: dict, device: torch.device) -> torch.Tensor:
    dc = config["diffusion"]
    betas = get_beta_schedule(
        schedule=dc["beta_schedule"],
        beta_start=dc["beta_start"],
        beta_end=dc["beta_end"],
        n_steps=dc["num_diffusion_timesteps"],
    )
    return torch.from_numpy(betas).float().to(device)


def _score_sequences(scorer: TimeADDMScorer, test_seqs: list[dict]) -> list[dict]:
    return [scorer.score_sequence(seq) for seq in test_seqs]


def _log_results(farm: str, care: dict, std: dict, label: str = "") -> None:
    tag = f"Farm {farm}" + (f" ({label})" if label else "")
    logger.info(f"CARE Score — {tag}")
    logger.info(f"  Coverage Fβ  : {care['coverage_fbeta']:.4f}  (P={care['coverage_precision']:.4f}  R={care['coverage_recall']:.4f})")
    logger.info(f"  Accuracy     : {care['accuracy']:.4f}")
    logger.info(f"  Reliability  : {care['reliability_efbeta']:.4f}")
    logger.info(f"  Earliness WS : {care['earliness_ws']:.4f}")
    logger.info(f"  CARE Score   : {care['care_score']:.4f}")
    logger.info(f"  AUC-ROC      : {std['auc_roc']:.4f}  AUC-PR: {std['auc_pr']:.4f}")
    logger.info(f"  Precision    : {std['precision']:.4f}  Recall: {std['recall']:.4f}")


def run_farm(farm: str, config: dict, device: str, feature_subset: list[str] | None,
             variant_label: str) -> dict:
    SEED = config["reproducibility"]["random_seed"]
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    farm_l = farm.lower()
    models_dir = Path(config["output"]["models_dir"])
    tables_dir = Path(config["output"]["tables_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info(f"[1/5] Loading and scaling data (Farm {farm}, variant {variant_label}) ...")
    train_scaled, test_seqs, feat_names, scaler = load_and_scale(config, farm, feature_subset)
    # train_scaled es (N, W, F): ventanas, largo de ventana y features por paso
    W = train_scaled.shape[1]
    F = train_scaled.shape[2]
    logger.info(f"  {len(train_scaled):,} windows | W={W} F={F} | done in {time.time()-t0:.1f}s")

    dev = torch.device(device)
    betas = _build_betas(config, dev)

    hidden_size = config["lstmae"]["hidden_size"]
    diff_cfg = _make_diffusion_config(config, hidden_size, W)

    t0 = time.time()
    logger.info("[2/5] Training LSTMAE ...")
    lstmae = LSTMAE(input_size=F, hidden_size=hidden_size, dropout=config["lstmae"]["dropout"])
    train_lstmae(lstmae, train_scaled, config, device, farm)
    logger.info(f"  Done in {time.time()-t0:.1f}s")

    # el difusor modela el cuello del AE, asi que el AE tiene que ir entrenado ya
    t0 = time.time()
    logger.info("[3/5] Training Diffusion model ...")
    diffusion = DiffusionUNet(diff_cfg)
    n_params = sum(p.numel() for p in diffusion.parameters())
    logger.info(f"  DiffusionUNet: {n_params:,} parameters")
    train_diffusion(diffusion, lstmae, betas, train_scaled, config, device, farm)
    logger.info(f"  Done in {time.time()-t0:.1f}s")

    t0 = time.time()
    logger.info("[4/5] Calibrating threshold and evaluating ...")
    scorer = TimeADDMScorer(lstmae, diffusion, betas, config, device)
    threshold = scorer.calibrate(train_scaled)
    cusum_results = _score_sequences(scorer, test_seqs)
    evaluator = CAREEvaluator(config)
    care = evaluator.evaluate(test_seqs, cusum_results)
    std = evaluator.standard_metrics(test_seqs, cusum_results)
    logger.info(f"  Done in {time.time()-t0:.1f}s")
    _log_results(farm, care, std, label=f"variant {variant_label}")

    logger.info("[5/5] Saving artefacts ...")
    # mismo tag que run_farm_eval_only, que es como reencuentra los artefactos
    tag = f"timeaddm_farm_{farm_l}_v{variant_label.lower()}"

    for name, obj in [("lstmae", lstmae.state_dict()), ("diffusion", diffusion.state_dict())]:
        torch.save(obj, models_dir / f"{tag}_{name}.pt")
    with open(models_dir / f"{tag}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    (models_dir / f"{tag}_feature_cols.txt").write_text("\n".join(feat_names), encoding="utf-8")

    summary = {
        "farm": farm,
        "variant": variant_label,
        "n_train_windows": int(train_scaled.shape[0]),
        "n_features": F,
        "window_size": W,
        "threshold": float(threshold),
        "care_score": care["care_score"],
        "coverage_fbeta": care["coverage_fbeta"],
        "accuracy": care["accuracy"],
        "reliability_efbeta": care["reliability_efbeta"],
        "earliness_ws": care["earliness_ws"],
        **std,
    }
    with open(tables_dir / f"{tag}_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=_np_default)
    pd.DataFrame(care["per_event"]).to_csv(tables_dir / f"{tag}_per_event.csv", index=False)

    plot_score_vs_time(
        farm=farm,
        test_sequences=test_seqs,
        cusum_results=cusum_results,
        threshold=threshold,
        figures_dir=config["output"]["figures_dir"],
        model_label=f"ADDM-v{variant_label}",
    )

    return summary


def run_farm_eval_only(farm: str, config: dict, device: str, feature_subset: list[str] | None,
                       variant_label: str) -> dict:
    """evalua artefactos guardados sin reentrenar."""

    farm_l = farm.lower()
    tag = f"timeaddm_farm_{farm_l}_v{variant_label.lower()}"
    models_dir = Path(config["output"]["models_dir"])
    tables_dir = Path(config["output"]["tables_dir"])

    logger.info(f"Farm {farm} variant {variant_label} — eval only")

    t0 = time.time()
    logger.info("[1/3] Loading test sequences ...")
    _, test_seqs, feat_names, scaler = load_and_scale(config, farm, feature_subset)
    # todas las secuencias comparten forma, basta con mirar la primera
    W = next(iter(test_seqs))["windows"].shape[1]
    F = next(iter(test_seqs))["windows"].shape[2]
    logger.info(f"  W={W} F={F} | done in {time.time()-t0:.1f}s")

    # el scaler se pisa con el guardado, asi las transformaciones casan exactas con las del
    # entrenamiento
    scaler_path = models_dir / f"{tag}_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            saved_scaler = pickle.load(f)
        # el scaler es global sobre features, hay que aplanar las ventanas y rearmarlas
        for seq in test_seqs:
            n, w, fv = seq["windows"].shape
            seq["windows"] = saved_scaler._global_scaler.transform(
                seq["windows"].reshape(-1, fv)
            ).reshape(n, w, fv).astype(np.float32)

    t0 = time.time()
    logger.info("[2/3] Loading model weights ...")
    dev = torch.device(device)
    betas = _build_betas(config, dev)
    hidden_size = config["lstmae"]["hidden_size"]
    diff_cfg = _make_diffusion_config(config, hidden_size, W)

    lstmae = LSTMAE(input_size=F, hidden_size=hidden_size, dropout=config["lstmae"]["dropout"])
    lstmae.load_state_dict(torch.load(models_dir / f"{tag}_lstmae.pt", map_location=device, weights_only=True))

    diffusion = DiffusionUNet(diff_cfg)
    diffusion.load_state_dict(torch.load(models_dir / f"{tag}_diffusion.pt", map_location=device, weights_only=True))

    with open(tables_dir / f"{tag}_results.json") as f:
        threshold = float(json.load(f)["threshold"])
    logger.info(f"  Threshold: {threshold:.6f} | done in {time.time()-t0:.1f}s")

    t0 = time.time()
    logger.info("[3/3] Scoring and evaluating ...")
    scorer = TimeADDMScorer(lstmae, diffusion, betas, config, device)
    scorer.threshold_ = threshold
    cusum_results = _score_sequences(scorer, test_seqs)
    evaluator = CAREEvaluator(config)
    care = evaluator.evaluate(test_seqs, cusum_results)
    std = evaluator.standard_metrics(test_seqs, cusum_results)
    logger.info(f"  Done in {time.time()-t0:.1f}s")
    _log_results(farm, care, std, label=f"variant {variant_label} eval-only")

    return {
        "farm": farm,
        "variant": variant_label,
        "threshold": threshold,
        "care_score": care["care_score"],
        "coverage_fbeta": care["coverage_fbeta"],
        "accuracy": care["accuracy"],
        "reliability_efbeta": care["reliability_efbeta"],
        "earliness_ws": care["earliness_ws"],
        **std,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm", choices=["B", "C"], default=None)
    parser.add_argument(
        "--variant", choices=["A", "B"], default="A",
        help="A = todas las features; B = subconjunto SHAP (subconjunto D en Farm B, B en Farm C)",
    )
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    with open("config/timeaddm_config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PyTorch {torch.__version__} | Device: {device}")
    if device == "cuda":
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    farms = [args.farm] if args.farm else config["data"]["farms"]
    all_results: dict[str, dict] = {}
    total_t0 = time.time()

    for farm in farms:
        feature_subset: list[str] | None = None
        if args.variant == "B":
            # el subconjunto SHAP se eligio por granja, el config guarda solo su clave
            subset_key = config["data"]["variant_b_subset"][farm]
            subsets_path = Path(f"results/tables/farm_{farm.lower()}_feature_subsets.json")
            with open(subsets_path, encoding="utf-8") as f:
                feature_subset = json.load(f)[subset_key]
            logger.info(
                f"Variant B — Farm {farm}: subset '{subset_key}' ({len(feature_subset)} features)"
            )

        if args.eval_only:
            all_results[farm] = run_farm_eval_only(farm, config, device, feature_subset, args.variant)
        else:
            all_results[farm] = run_farm(farm, config, device, feature_subset, args.variant)

    # tabla resumen, los anchos van fijos para que cuadren cabecera y filas
    logger.info(
        f"{'Farm':<6} {'Var':<4} {'CARE':>6} {'Coverage':>10} {'Accuracy':>10} "
        f"{'Reliability':>12} {'Earliness':>10} {'AUC-ROC':>8}"
    )
    for farm, r in all_results.items():
        logger.info(
            f"{farm:<6} {r.get('variant','?'):<4} {r['care_score']:>6.4f} "
            f"{r['coverage_fbeta']:>10.4f} {r['accuracy']:>10.4f} "
            f"{r['reliability_efbeta']:>12.4f} {r['earliness_ws']:>10.4f} "
            f"{r['auc_roc']:>8.4f}"
        )
    logger.info(f"Total time: {time.time()-total_t0:.1f}s")
