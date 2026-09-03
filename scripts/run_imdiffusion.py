"""runner de ImDiffusion, entrena, puntua y evalua con el harness.

    python run_imdiffusion.py --canary [--canary-epochs 400]

el canario es un control de sobreajuste, menos de media hora de GPU. entrena el modelo a
tamano completo sobre 64 ventanas de Farm B hasta que la perdida toca suelo, y despues
comprueba que el error de imputacion sobre esas ventanas es mucho menor que sobre copias
permutadas en el tiempo, que es lo que distingue estructura aprendida de marginales
aprendidas, y pinta figuras de imputado contra real.

    python run_imdiffusion.py --farm B --seed 42 [--epochs 10] [--split 9]

una ejecucion completa por granja y por fila de tabla, sobre la representacion previa al PCA.
Farm B son las ventanas npz congeladas (27, 30) filtradas por la mascara de retencion del
Isolation Forest del AE, 407.136 en total, y escaladas con el StandardScaler del AE. cada
agregacion y cada paso DDIM se puntuan en una sola pasada y se guardan en npz, asi que un
barrido post-hoc se replica gratis, y la evaluacion pasa por el harness sin tocar,
run_model.evaluate con CUSUM, h-sweep, CV y el CARE oficial, via ReplayScorer. valida ademas
el adelgazado de calibracion, stride-1 contra stride-4, para Farm C.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (imports hermanos)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz del repo (from src.X)

import run_model
from run_model import _deep_merge
from src.care_official.adapter import OfficialCAREAdapter
from src.imdiffusion.data import compute_if_retention_mask
from src.imdiffusion.diffusion import ImDiffusionModel
from src.imdiffusion.masking import complementary_masks, gratings_for_model
from src.imdiffusion.scorer import ImDiffusionScorer
from src.model_prep.model_scaler import WindowScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/imdiffusion_config.yaml")


def load_farm_config(farm: str) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    return _deep_merge(base, base.get("per_farm", {}).get(farm, {}))


def build_model(cfg: dict, n_features: int) -> ImDiffusionModel:
    m, d = cfg["model"], cfg["diffusion"]
    return ImDiffusionModel(
        n_features=n_features, window_size=cfg["windowing"]["window_size"],
        num_steps=d["num_steps"], beta_start=d["beta_start"], beta_end=d["beta_end"],
        channels=m["channels"], layers=m["layers"], nheads=m["nheads"],
        diffusion_embedding_dim=m["diffusion_embedding_dim"],
        feature_embedding_dim=m["feature_embedding_dim"],
        time_embedding_dim=m["time_embedding_dim"])


def train_model(model: ImDiffusionModel, dataset, cfg: dict, device: torch.device,
                seed: int, epochs: int | None = None, exo_idx: list[int] | None = None) -> list[float]:
    """bucle de entrenamiento generico, una estrategia aleatoria por muestra que elige cual de
    las dos mascaras de rejilla complementarias es el objetivo de imputacion. devuelve la
    perdida media por epoca."""

    # exo_idx ancla las features exogenas de regimen como condicionantes siempre visibles
    tr = cfg["training"]
    epochs = epochs or tr["epochs"]
    pair = gratings_for_model(model.window_size, model.n_features, cfg["masking"]["split"],
                              cfg["masking"].get("mode", "time"), exo_idx)
    gratings = torch.from_numpy(np.ascontiguousarray(pair)).to(device)  # (2, K, L) bool
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=tr["batch_size"], shuffle=True,
                        generator=gen, num_workers=tr.get("num_workers", 0), drop_last=False,
                        pin_memory=(device.type == "cuda"))
    opt = torch.optim.Adam(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    # los milestones vienen en fraccion de las epocas, no en numero absoluto
    milestones = [max(1, int(epochs * frac)) for frac in tr["lr_decay_milestones"]]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
    model.train()
    epoch_losses: list[float] = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        total, count = 0.0, 0
        for (batch,) in loader:
            x = batch.to(device).permute(0, 2, 1)  # (B, K, L)
            b = x.shape[0]
            # cada muestra del batch elige una de las dos rejillas complementarias
            strategy = torch.randint(0, 2, (b,), device=device)
            target = gratings[strategy].to(x.dtype)  # (B, K, L)
            # las dos mascaras son complementarias, lo que no es objetivo es condicion
            cond = 1.0 - target
            loss = model.calc_loss(x, cond, target, strategy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach().item() * b
            count += b
        sched.step()
        epoch_losses.append(total / count)
        # se registra la primera epoca y luego una de cada diez, sea cual sea la ejecucion
        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            logger.info(f"  epoch {epoch:>4}/{epochs} | loss {epoch_losses[-1]:.5f} "
                        f"| {time.time()-t0:.0f}s")
    return epoch_losses


def _stitched_imputation(model: ImDiffusionModel, x: torch.Tensor, split: int,
                         ddim_steps: int, seed: int) -> torch.Tensor:
    """serie imputada completa, las dos pasadas complementarias cosidas para que cada posicion
    se impute exactamente una vez."""
    g0, g1 = complementary_masks(model.window_size, split)
    out = torch.zeros_like(x)
    for s, g in enumerate((g0, g1)):
        target = torch.from_numpy(g).to(x.device)[None, None, :].expand_as(x).to(x.dtype)
        torch.manual_seed(seed + s)
        out = out + model.impute(x, 1.0 - target, strategy=s, ddim_steps=ddim_steps) * target
    return out


def run_canary(cfg: dict, device: torch.device, epochs: int, seed: int = 42) -> dict:
    """sobreajusta 64 ventanas de Farm B; el error de imputacion ahi tiene que quedar muy por
    debajo del error sobre copias permutadas en el tiempo de esas mismas ventanas. escribe
    las cifras y una figura."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    npz = Path(cfg["data"]["processed_path"]) / "farm_b_train_windows.npz"
    windows = np.load(npz)["windows"].astype(np.float32)
    rng = np.random.default_rng(seed)
    sel = windows[rng.choice(len(windows), 64, replace=False)]
    scaled = WindowScaler(per_turbine=False, method="standard").fit_transform(sel, None).astype(np.float32)
    logger.info(f"Canary: 64/{len(windows):,} Farm B windows {scaled.shape[1:]} | device {device}")

    model = build_model(cfg, n_features=scaled.shape[2]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} params | T={model.num_steps} split={cfg['masking']['split']}")
    canary_cfg = _deep_merge(cfg, {"training": {"batch_size": 16}})
    t0 = time.time()
    losses = train_model(model, TensorDataset(torch.from_numpy(scaled)), canary_cfg,
                         device, seed=seed, epochs=epochs)
    train_seconds = time.time() - t0

    perm_rng = np.random.default_rng(123)
    # control 1, permutar el eje de tiempo dentro de cada ventana deja las marginales y se
    # lleva la dinamica; es una rotura debil para sensores lentos
    permuted = np.stack([w[perm_rng.permutation(w.shape[0])] for w in scaled])
    # control 2, ruido blanco N(0,1) a la escala de las features estandarizadas, sin estructura
    # temporal ni entre variables; es la rotura dura
    noise = perm_rng.normal(size=scaled.shape).astype(np.float32)
    sc = cfg["scoring"]
    scorer = ImDiffusionScorer(model, split=cfg["masking"]["split"], ddim_steps=sc["ddim_steps"],
                               eta=sc["eta"], aggregation="mean", batch_size=64,
                               percentile=cfg["threshold"]["percentile"], device=device)
    # se resiembra antes de cada scoring para que las tres tiradas de ruido sean la misma
    torch.manual_seed(seed)
    err_train = scorer.score_windows(scaled)
    torch.manual_seed(seed)
    err_perm = scorer.score_windows(permuted)
    torch.manual_seed(seed)
    err_noise = scorer.score_windows(noise)
    ratio = float(err_perm.mean() / err_train.mean())
    ratio_noise = float(err_noise.mean() / err_train.mean())
    logger.info(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} | err train {err_train.mean():.4f}±"
                f"{err_train.std():.4f} | permuted {err_perm.mean():.4f} (x{ratio:.2f}) | "
                f"white noise {err_noise.mean():.4f} (x{ratio_noise:.2f}) | train {train_seconds:.0f}s")

    fig_path = _canary_figure(cfg, model, scaled, losses, err_train, err_perm, err_noise, seed)
    out = {"milestone": "H1_canary", "n_windows": 64, "epochs": epochs, "n_params": int(n_params),
           "split": cfg["masking"]["split"], "ddim_steps": sc["ddim_steps"],
           "loss_first": losses[0], "loss_last": losses[-1],
           "err_train_mean": float(err_train.mean()), "err_train_std": float(err_train.std()),
           "err_permuted_mean": float(err_perm.mean()), "err_permuted_std": float(err_perm.std()),
           "err_noise_mean": float(err_noise.mean()), "err_noise_std": float(err_noise.std()),
           "ratio_permuted_over_train": ratio, "ratio_noise_over_train": ratio_noise,
           "train_seconds": round(train_seconds, 1),
           "device": str(device), "figure": str(fig_path)}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "imdiffusion_h1_canary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"saved {tables / 'imdiffusion_h1_canary.json'} + {fig_path}")
    return out


def _canary_figure(cfg: dict, model: ImDiffusionModel, scaled: np.ndarray, losses: list[float],
                   err_train: np.ndarray, err_perm: np.ndarray, err_noise: np.ndarray,
                   seed: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    device = next(model.parameters()).device
    x = torch.from_numpy(scaled[:2]).to(device).permute(0, 2, 1)  # 2 windows (B, K, L)
    x_hat = _stitched_imputation(model, x, cfg["masking"]["split"],
                                 cfg["scoring"]["ddim_steps"], seed).cpu().numpy()
    x_np = x.cpu().numpy()
    feats = np.argsort(scaled[0].std(axis=0))[-2:]  # 2 most dynamic features

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    ax.semilogy(range(1, len(losses) + 1), losses)
    ax.set_title("H1 canary: training loss (64 windows, Farm B)")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked noise MSE")
    for ax, (w, k) in zip([axes[0, 1], axes[1, 0]], [(0, feats[-1]), (1, feats[-2])]):
        ax.plot(x_np[w, k], "o-", label="real", lw=1.5, ms=3)
        ax.plot(x_hat[w, k], "x--", label="imputed (stitched)", lw=1.2, ms=5)
        ax.set_title(f"window {w}, feature idx {k}")
        ax.set_xlabel("t (10-min steps)"); ax.legend()
    ax = axes[1, 1]
    ax.boxplot([err_train, err_perm, err_noise],
               tick_labels=["trained", "time-permuted", "white noise"])
    ax.set_ylabel("per-window mean |imputation error|")
    ax.set_title(f"structure check: permuted x{err_perm.mean()/err_train.mean():.1f}, "
                 f"noise x{err_noise.mean()/err_train.mean():.1f}")
    fig.tight_layout()
    fig_dir = Path(cfg["output"]["figures_dir"]) / "imdiffusion"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "h1_canary_imputation.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


class ReplayScorer:
    """scores por ventana ya calculados y servidos en el orden en que los pide
    run_model.evaluate, train primero y luego cada evento, con un assert de consumo total."""

    # mismo patron con el que se evaluo toda la campanya de TimeADDM, sweep_timeaddm_scoring

    def __init__(self, scores: list[np.ndarray], percentile: float = 99.0) -> None:
        self._scores = scores
        self._i = 0
        self.percentile = percentile

    # el argumento windows se ignora a proposito, los scores ya vienen calculados y en orden
    def score_windows(self, windows):  # noqa: ARG002
        s = self._scores[self._i]
        self._i += 1
        return s

    def assert_consumed(self) -> None:
        assert self._i == len(self._scores), f"ReplayScorer consumed {self._i}/{len(self._scores)}"


def _smooth(s: np.ndarray, w: int) -> np.ndarray:
    """media movil centrada sobre los scores por ventana, heredada de la linea de TimeADDM."""
    return s if w <= 1 else np.convolve(s, np.ones(w) / w, mode="same")


ALLAVG_B_DIR = Path("data/processed/experiments/processed_imdiffusion_allavgB")
SOFT_B_DIR = Path("data/processed/experiments/processed_imdiffusion_softB")
_B_FEATCOL_DIR = {"best": Path("data/processed"), "allavg": ALLAVG_B_DIR, "soft": SOFT_B_DIR}


def compute_exo_idx(feats: list[str], keywords: list[str]) -> list[int]:
    """indices de las features exogenas que anclan el regimen, aquellas cuyo nombre empieza por
    un prefijo wind_speed o casa exactamente con un sensor de ambiente."""

    # el == exacto del sensor ambiente evita casar sensor_80 cuando el ancla es sensor_8_avg
    out = []
    for i, f in enumerate(feats):
        if any((f == k) or (k.endswith("_") and f.startswith(k)) or f.startswith(k + "_")
               for k in keywords):
            out.append(i)
    return out


def _ensure_soft_b_processed(cfg: dict) -> Path:
    """representacion soft de Farm B, todos los sensores `_avg` con la limpieza pesada de
    dominio apagada, sin outlier-IQR, sin temp-norm y sin suavizado."""

    # se quedan solo los pasos que hacen falta para dejar los fallos fuera del train, o sea
    # status-0, curva de potencia y exclusion de transiciones, mas dedup y W=27. es lo mas
    # cerca que llega el pipeline del preprocesado de solo z-score que usan los benchmarks de
    # imputacion por difusion, y sirve para probar si la limpieza curada estaba destruyendo
    # senal para el score de imputacion. las correlaciones crudas entre sensores quedan
    # intactas, que es de lo que tira la atencion entre variables
    if (SOFT_B_DIR / "farm_b_train_windows.npz").exists():
        return SOFT_B_DIR
    from src.preprocessing.config import load_config as load_preproc_config
    from src.preprocessing.pipeline import PreprocessingPipeline
    logger.info("soft B npz missing -> regenerating (outlier+temp-norm OFF, all _avg) (~10 min)")
    pcfg = load_preproc_config()
    pcfg = _deep_merge(pcfg, pcfg.get("per_farm", {}).get("B", {}))
    pcfg = _deep_merge(pcfg, {
        "feature_selection": {"statistics": ["avg"], "extra_statistics": [],
                              "correlation_threshold": 1.01, "max_features": 100_000},
        "outlier_detection": {"enabled": False},
        "temperature_normalization": {"enabled": False},
        "smoothing": {"enabled": False},
        "data": {"processed_path": str(SOFT_B_DIR)},
    })
    PreprocessingPipeline(pcfg).run("B")
    return SOFT_B_DIR


def _ensure_allavg_b_processed(cfg: dict) -> Path:
    """regenera la representacion de todos los `_avg` de Farm B por la cadena congelada de la
    granja, sobrescribiendo solo la seleccion de features, al estilo de C."""

    # la cadena es PreprocessingPipeline, con la misma limpieza, dedup, no-smoothing y
    # ventaneo que produjo el npz curado; el selector sigue excluyendo los contadores. se
    # persiste una vez en data/processed/experiments/processed_imdiffusion_allavgB y se reusa.
    # aviso: el directorio antiguo data/processed/experiments/processed_allavgB, el avgonly de
    # TimeADDM, salio de la config raiz, con smoothing y sin merge per-farm, y no es comparable
    if (ALLAVG_B_DIR / "farm_b_train_windows.npz").exists():
        return ALLAVG_B_DIR
    from src.preprocessing.config import load_config as load_preproc_config
    from src.preprocessing.pipeline import PreprocessingPipeline
    logger.info("all-avg B npz missing -> regenerating through the frozen pipeline (~10 min)")
    pcfg = load_preproc_config()
    pcfg = _deep_merge(pcfg, pcfg.get("per_farm", {}).get("B", {}))
    pcfg = _deep_merge(pcfg, {
        "feature_selection": {"statistics": ["avg"], "extra_statistics": [],
                              "correlation_threshold": 1.01, "max_features": 100_000},
        "data": {"processed_path": str(ALLAVG_B_DIR)},
    })
    PreprocessingPipeline(pcfg).run("B")
    return ALLAVG_B_DIR


def load_farm_b_representation(cfg: dict, features: str = "best") -> tuple[np.ndarray, list[dict]]:
    """representacion de Farm B previa al PCA, las ventanas npz pasadas por la mascara de
    retencion del IF en el espacio del PCA y un StandardScaler ajustado con todas las ventanas
    crudas de train."""

    # la receta de escalado es la de run_model._prep_window_first
    # arm "best", el npz curado y congelado (27, 30), con asertos anti-deriva de 411.249
    # ventanas crudas, 407.136 tras la mascara IF del AE y 70.540 de test
    if features == "best":
        p = Path(cfg["data"]["processed_path"])
        raw = np.load(p / "farm_b_train_windows.npz")["windows"]  # float64, sin cast
        if len(raw) != 411_249:
            raise RuntimeError(f"frozen npz drifted: {len(raw):,} windows != 411,249")
    # arm "allavg", todos los sensores `_avg` por la misma cadena congelada. el conjunto de
    # ventanas cambia un poco, porque min_valid descarta otras al haber 54 columnas, y la misma
    # receta, scaler mas PCA 0.95 mas IF 1%, se recalcula aqui en float32, sin bit-exactitud
    elif features == "allavg":
        p = _ensure_allavg_b_processed(cfg)
        raw = np.load(p / "farm_b_train_windows.npz")["windows"].astype(np.float32)
    elif features == "soft":
        p = _ensure_soft_b_processed(cfg)
        raw = np.load(p / "farm_b_train_windows.npz")["windows"].astype(np.float32)
    else:
        raise ValueError(f"unknown features arm: {features}")
    mask, scaler, _ = compute_if_retention_mask(raw)
    if features == "best" and int(mask.sum()) != 407_136:
        raise RuntimeError(f"IF retention mask drifted: kept {int(mask.sum()):,} != 407,136")
    train = np.empty(raw.shape, np.float32)
    for i in range(0, len(raw), 50_000):  # por trozos, para no disparar la RAM del host
        train[i:i + 50_000] = scaler.transform(raw[i:i + 50_000])
    train = train[mask]
    del raw

    with open(p / "farm_b_test_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    seq_npz = np.load(p / "farm_b_test_sequences.npz", allow_pickle=False)
    test_seqs = []
    for eid, meta in metadata.items():
        windows = scaler.transform(seq_npz[eid].astype(np.float32))
        test_seqs.append({"event_id": int(eid), "event_label": meta["event_label"],
                          "asset_id": int(meta["asset_id"]), "windows": windows,
                          "labels": np.array(meta["labels"], np.int8),
                          "window_timestamps": np.array(meta["window_timestamps"], dtype="datetime64[ns]")})
    # 70,540 es la cuenta de ventanas de test del run congelado de B, en
    # results/logs/_sanity_bc.log
    n_test = sum(len(s["windows"]) for s in test_seqs)
    if features == "best" and (n_test != 70_540 or len(test_seqs) != 15):
        raise RuntimeError(f"test sequences drifted: {len(test_seqs)} events / {n_test:,} windows "
                           f"!= 15 / 70,540")
    if len(test_seqs) != 15:
        raise RuntimeError(f"Farm B must have 15 events, got {len(test_seqs)}")
    logger.info(f"Farm B repr [{features}]: train {train.shape} (IF-masked, recipe scaler) | "
                f"{len(test_seqs)} events / {n_test:,} test windows")
    # el train sale en float32 y las secuencias de test ya escaladas con el mismo scaler
    return train, test_seqs


def load_farm_c_representation(seed: int, features: str = "top100",
                              stride_train: int = 1) -> tuple[np.ndarray, list[dict]]:
    """representacion de Farm C previa al PCA, reconstruida desde el crudo por el mismo camino
    pca-first que usa el AE de C, pero parando antes del PCA."""

    # el camino es el de run_model._prep_pca_first, filas de limpieza minima, filtros de
    # status, curva de potencia y transiciones, y dedup, pero sin aplicar la PCA. no hay IF
    from sklearn.preprocessing import StandardScaler
    from src.preprocessing.config import load_config as load_preproc_config
    from src.preprocessing.status_filter import StatusFilter
    from src.preprocessing.power_curve_filter import PowerCurveFilter
    from src.preprocessing.transition_excluder import TransitionExcluder
    from src.preprocessing.windower import Windower

    pcfg = load_preproc_config()
    pcfg = _deep_merge(pcfg, pcfg.get("per_farm", {}).get("C", {}))
    pp = pcfg.get("preproc", {})
    df, event_info, feats, reg = run_model.preprocess_rows(
        "C", pcfg, pp.get("outlier", False), pp.get("temp_norm", False), pp.get("smooth", False))
    # "top100" es el subset XAI ganador del AE (d=100) y "allavg" son todos los `_avg` (d=238)
    if features == "top100":
        sub = set(Path("config/farm_c_top100_features.txt").read_text(encoding="utf-8").strip().splitlines())
        feats = [c for c in feats if c in sub]
    elif features != "allavg":
        raise ValueError(f"unknown C features arm: {features}")
    logger.info(f"Farm C repr [{features}]: {len(feats)} sensors")

    df_train = df[df["train_test"] == "train"].copy()
    df_train, _ = StatusFilter(pcfg).transform(df_train)
    df_train, _ = PowerCurveFilter(pcfg, reg).transform(df_train)
    df_train, _ = TransitionExcluder(pcfg).transform(df_train, event_info)
    if pcfg["data"].get("dedup_train", True):
        df_train = df_train.drop_duplicates(["asset_id", "time_stamp"]).reset_index(drop=True)
    df_pred = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)
    col_means = df_train[feats].mean()

    def arr(src):
        return src[feats].fillna(col_means).fillna(0.0).to_numpy(np.float32)

    # el StandardScaler se ajusta sobre las filas de train y lo que se ventanea son los
    # sensores escalados, para que la atencion entre variables vea sensores reales correlados
    scaler = StandardScaler().fit(arr(df_train))

    def to_wdf(src, scaled):
        meta = src[[c for c in run_model.META if c in src.columns]].reset_index(drop=True)
        # todas las columnas de sensor se construyen de golpe, asi se evita el camino de
        # insercion una a una que fragmenta el frame
        return pd.concat([meta, pd.DataFrame(scaled, columns=feats)], axis=1)

    # stride_train adelgaza las 1,12M ventanas de train y es la palanca de coste; el stride de
    # test se queda fijo en 1 por protocolo
    wtrain = Windower({"windowing": {**pcfg["windowing"], "stride_samples": stride_train}})
    wtest = Windower({"windowing": {**pcfg["windowing"], "stride_samples": 1}})
    train_windows, _, _ = wtrain.create_train_windows(to_wdf(df_train, scaler.transform(arr(df_train))), feats)
    test_obj, _ = wtest.create_test_sequences(to_wdf(df_pred, scaler.transform(arr(df_pred))), event_info, feats)
    test_seqs = [{"event_id": s.event_id, "event_label": s.event_label, "asset_id": s.asset_id,
                  "windows": s.windows.astype(np.float32), "labels": np.asarray(s.labels, np.int8),
                  "window_timestamps": s.window_timestamps} for s in test_obj]
    n_test = sum(len(s["windows"]) for s in test_seqs)
    logger.info(f"Farm C repr [{features}]: train {train_windows.shape} (stride_train={stride_train}) | "
                f"{len(test_seqs)} events / {n_test:,} test windows")
    return train_windows.astype(np.float32), test_seqs, feats


def _evaluate_replay(train_scores: np.ndarray, ev_scores: list[np.ndarray], ma: int,
                     train_windows: np.ndarray, test_seqs: list[dict],
                     adapter: OfficialCAREAdapter, seed: int, percentile: float) -> dict:
    """una evaluacion del harness sobre scores ya calculados, con la media movil aplicada solo
    a los scores de evento."""
    rs = ReplayScorer([train_scores] + [_smooth(s, ma) for s in ev_scores], percentile)
    ev = run_model.evaluate(rs, train_windows, test_seqs, adapter, seed)
    rs.assert_consumed()
    best, cv = ev["best"], ev["cv"]
    return {"ma": ma, "care_cv": cv["care"], "care_oracle": best["care"], "optimism": cv["optimism"],
            "fold_h_scale": cv["fold_h_scale"], "coverage": best["cov"], "earliness": best["earl"],
            "earliness_conditional": best["earl_conditional"], "reliability": best["rel"],
            "accuracy": best["acc"], "f1": best["f1"], "roc_dataset": best["roc_ds"],
            "roc_window": ev["roc_window"]}


def run_full(farm: str, seed: int, cfg: dict, device: torch.device,
             epochs: int | None = None, features: str = "best", score_only: bool = False) -> dict:
    """ejecucion completa de una granja, semilla y arm de features, con representacion,
    entreno, puntuacion de todas las agregaciones en una pasada y evaluacion del harness."""

    # el barrido de medias moviles es informativo, y ademas se valida la calibracion con
    # stride-4. las features de B son best, allavg o soft, y las de C top100 o allavg.
    # con score_only se salta el entrenamiento, se recarga el .pt y se vuelve a puntuar con
    # los ddim_steps y n_samples configurados, un barrido barato de la palanca de scoring
    np.random.seed(seed)
    torch.manual_seed(seed)
    fl = farm.lower()
    t0 = time.time()
    if farm == "B":
        train, test_seqs = load_farm_b_representation(cfg, features)
        feat_names = (_B_FEATCOL_DIR[features] / "farm_b_feature_cols.txt").read_text(
            encoding="utf-8").strip().splitlines()
    else:
        train, test_seqs, feat_names = load_farm_c_representation(
            seed, features if features in ("top100", "allavg") else "top100",
            cfg["training"].get("stride_train", 1))
    t_repr = time.time() - t0

    mask_mode = cfg["masking"].get("mode", "time")
    exo_idx = None
    regime_tag = ""
    if cfg["masking"].get("regime_conditioning"):
        # anclas por defecto de cada granja, viento mas el sensor de ambiente que le toca
        default_kw = ["wind_speed", "sensor_8_avg"] if farm == "B" else ["wind_speed", "sensor_177_avg"]
        kw = cfg["masking"].get("exo_keywords") or default_kw
        exo_idx = compute_exo_idx(feat_names, kw)
        if not exo_idx:
            raise RuntimeError(f"regime conditioning: no exogenous anchors matched {kw} in {farm} features")
        regime_tag = "_regime"
        logger.info(f"REGIME conditioning: {len(exo_idx)} exo anchors (never masked): "
                    f"{[feat_names[i] for i in exo_idx]}")
    model = build_model(cfg, n_features=train.shape[2]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model {n_params:,} params | split={cfg['masking']['split']} "
                f"mask={mask_mode} T={model.num_steps} | seed {seed}")
    models_dir = Path(cfg["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    # los tags de la variante por defecto van vacios, y asi no renombran lo ya escrito
    mode_tag = "" if mask_mode == "time" else "_offset"
    split_tag = "" if cfg["masking"]["split"] == 9 else f"_split{cfg['masking']['split']}"
    tag = f"imdiffusion_farm_{fl}_{features}{mode_tag}{regime_tag}{split_tag}_s{seed}"

    if score_only:
        # re-score de un modelo ya entrenado con otras palancas de scoring, sin reentrenar
        # el npz de scores lleva sufijo para no pisar el del run original
        ckpt = torch.load(models_dir / f"{tag}.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        losses = ckpt.get("losses", [])
        t_train = 0.0
        logger.info(f"[score-only] loaded {tag}.pt | ddim={cfg['scoring']['ddim_steps']} "
                    f"n_samples={cfg['scoring']['n_samples']}")
        tag = f"{tag}_ddim{cfg['scoring']['ddim_steps']}_ns{cfg['scoring']['n_samples']}"
    else:
        t0 = time.time()
        losses = train_model(model, TensorDataset(torch.from_numpy(train)), cfg, device,
                             seed=seed, epochs=epochs, exo_idx=exo_idx)
        t_train = time.time() - t0
        torch.save({"state_dict": model.state_dict(), "config": cfg, "losses": losses},
                   models_dir / f"{tag}.pt")

    sc = cfg["scoring"]
    scorer = ImDiffusionScorer(model, split=cfg["masking"]["split"], ddim_steps=sc["ddim_steps"],
                               eta=sc["eta"], n_samples=sc["n_samples"], aggregation=sc["aggregation"],
                               topk_q=sc["topk_q"], batch_size=sc["batch_size"],
                               percentile=cfg["threshold"]["percentile"], device=device,
                               mask_mode=mask_mode, exo_idx=exo_idx)
    t0 = time.time()
    # dos offsets distintos sobre la semilla, para que el ruido de los eventos y el del train
    # no sea el mismo y la calibracion no herede la tirada del scoring
    torch.manual_seed(10_000 + seed)
    ev_full = [scorer.score_windows_full(s["windows"]) for s in test_seqs]
    torch.manual_seed(20_000 + seed)
    train_full = scorer.score_windows_full(train)
    t_score = time.time() - t0
    n_scored = len(train) + sum(len(s["windows"]) for s in test_seqs)
    arrays = {f"train_{k}": v for k, v in train_full.items()}
    for s, full in zip(test_seqs, ev_full):
        arrays.update({f"ev{s['event_id']}_{k}": v for k, v in full.items()})
    np.savez_compressed(models_dir / f"{tag}_scores.npz", **arrays)
    logger.info(f"scoring: {n_scored:,} windows in {t_score/60:.1f} min "
                f"({n_scored/t_score:.0f} win/s) -> {tag}_scores.npz")

    adapter = OfficialCAREAdapter(raw_path=cfg["data"]["raw_path"], farm=farm,
                                  window_size=cfg["windowing"]["window_size"])
    agg = sc["aggregation"]
    train_scores = train_full[agg]
    ev_scores = [f[agg] for f in ev_full]
    percentile = cfg["threshold"]["percentile"]
    evaluation = [_evaluate_replay(train_scores, ev_scores, ma, train, test_seqs,
                                   adapter, seed, percentile) for ma in (1, 3, 5, 7)]
    for r in evaluation:
        logger.info(f"  MA{r['ma']}: CARE-CV={r['care_cv']:.4f} (oracle {r['care_oracle']:.4f}, "
                    f"optim {r['optimism']:+.4f}) | Cov={r['coverage']:.3f} Earl={r['earliness']:.3f} "
                    f"Rel={r['reliability']:.3f} Acc={r['accuracy']:.3f}")

    # valida en B el atajo de calibracion de C: el CUSUM calibrado con todos los scores de
    # train y el calibrado con el adelgazado de stride-4 tienen que dar el mismo CARE-CV
    thinned = _evaluate_replay(train_scores[::4], ev_scores, 1, train, test_seqs,
                               adapter, seed, percentile)
    calib_validation = {"care_cv_stride1": evaluation[0]["care_cv"],
                        "care_cv_stride4": thinned["care_cv"],
                        "delta": thinned["care_cv"] - evaluation[0]["care_cv"]}
    logger.info(f"calib stride-1 vs stride-4: {calib_validation['care_cv_stride1']:.4f} vs "
                f"{calib_validation['care_cv_stride4']:.4f} (delta {calib_validation['delta']:+.4f})")

    out = {"milestone": "H2+", "farm": farm, "seed": seed, "n_params": int(n_params),
           "features": features, "n_features": int(train.shape[2]), "mask_mode": mask_mode,
           "config": {"epochs": epochs or cfg["training"]["epochs"],
                      "batch_size": cfg["training"]["batch_size"],
                      "split": cfg["masking"]["split"], "ddim_steps": sc["ddim_steps"],
                      "eta": sc["eta"], "aggregation": agg, "n_samples": sc["n_samples"]},
           "losses": losses, "timings": {"repr_s": round(t_repr, 1), "train_s": round(t_train, 1),
                                         "score_s": round(t_score, 1),
                                         "min_per_epoch": round(t_train / 60 / len(losses), 2),
                                         "score_windows_per_s": round(n_scored / t_score, 1)},
           "evaluation_ma": evaluation, "calib_validation": calib_validation,
           "device": str(device)}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=run_model._np_default)
    logger.info(f"saved {tables / (tag + '.json')}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", choices=["B", "C"], default="B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="sobrescribe training.epochs")
    ap.add_argument("--split", type=int, default=None, help="sobrescribe masking.split")
    ap.add_argument("--features", choices=["best", "allavg", "soft", "top100"], default=None,
                    help="B: best=curated del AE / allavg=todos los _avg / soft=todos los _avg sin outlier+temp-norm. "
                         "C: top100=subconjunto XAI del AE (d=100) / allavg=todos los _avg (d=238). "
                         "por defecto, best en B y top100 en C.")
    ap.add_argument("--stride-train", type=int, default=None,
                    help="solo C: adelgaza las 1.12M ventanas de train (palanca de coste; el stride de test sigue en 1)")
    ap.add_argument("--score-only", action="store_true",
                    help="se salta el entrenamiento, recarga el .pt y vuelve a puntuar (para barridos de ddim y n_samples)")
    ap.add_argument("--ddim-steps", type=int, default=None, help="sobrescribe scoring.ddim_steps")
    ap.add_argument("--n-samples", type=int, default=None, help="sobrescribe scoring.n_samples (extracciones de ruido)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="sobrescribe training.batch_size (p.ej. 128 con d=54: un batch de 256 "
                         "desborda los 12 GB de VRAM y el volcado de WDDM va a paso de tortuga)")
    ap.add_argument("--mask-mode", choices=["time", "feature_offset"], default=None,
                    help="variante de rejilla: las bandas temporales oficiales, o un desfase "
                         "por feature (variante nuestra, permite imputar entre sensores)")
    ap.add_argument("--regime", action="store_true",
                    help="condicionado por regimen (B): ancla viento y ambiente como exogenos "
                         "siempre observados, e imputa los sensores endogenos dado el regimen limpio")
    ap.add_argument("--canary", action="store_true", help="canario de sobreajuste H1 (64 ventanas de Farm B)")
    ap.add_argument("--canary-epochs", type=int, default=400)
    args = ap.parse_args()

    cfg = load_farm_config(args.farm)
    if args.split is not None:
        cfg["masking"]["split"] = args.split
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.mask_mode is not None:
        cfg["masking"]["mode"] = args.mask_mode
    if args.regime:
        cfg["masking"]["regime_conditioning"] = True
    if args.stride_train is not None:
        cfg["training"]["stride_train"] = args.stride_train
    if args.ddim_steps is not None:
        cfg["scoring"]["ddim_steps"] = args.ddim_steps
    if args.n_samples is not None:
        cfg["scoring"]["n_samples"] = args.n_samples
    # el arm por defecto depende de la granja, best en B y top100 en C
    features = args.features or ("best" if args.farm == "B" else "top100")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch {torch.__version__} | device {device}")
    if args.canary:
        run_canary(cfg, device, epochs=args.canary_epochs, seed=args.seed)
        return
    run_full(args.farm, args.seed, cfg, device, epochs=args.epochs, features=features,
             score_only=args.score_only)


if __name__ == "__main__":
    main()
