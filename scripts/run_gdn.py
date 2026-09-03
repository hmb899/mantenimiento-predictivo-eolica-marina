"""runner de GDN, entrena, puntua y evalua con el harness.

    python scripts/run_gdn.py --selftest --farm B

el selftest construye el modelo de la granja sobre datos sinteticos y hace una pasada
adelante, sin datos y sin GPU, asi que el cableado queda probado por si solo. las ejecuciones
por granja son espejo de run_imdiffusion.py y evaluan con el harness sin tocar,
run_model.evaluate, via ReplayScorer.
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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (imports hermanos)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz del repo (from src.X)

import run_model
from src.care_official.adapter import OfficialCAREAdapter
from src.gdn.data import (forecast_tensors, load_frozen_windows, load_test_sequences,
                          time_permuted)
from src.gdn.model import GDN
from src.gdn.scorer import GDNScorer, aggregate_sensor_errors
from src.imdiffusion.data import compute_if_retention_mask
from src.model_prep.model_scaler import WindowScaler

# los nodos exogenos de viento y ambiente se quedan fuera del score de desviacion, las mismas
# anclas que uso ImDiffusion. un exo_keywords nulo en la config cae a estos valores por granja
_DEFAULT_EXO = {"B": ["wind_speed", "sensor_8_avg"], "C": ["wind_speed", "sensor_177_avg"]}

# arms de representacion de Farm B, reusando los directorios de ImDiffusion. None es el
# curated congelado, soft es todo `_avg` sin outlier ni temp-norm, allavg todo `_avg` con la
# limpieza pesada de curated. mismo formato que data/processed, sin tocar ninguna baseline
_B_ARM_DIR = {"curated": None,
              "soft": "data/processed/experiments/processed_imdiffusion_softB",
              "allavg": "data/processed/experiments/processed_imdiffusion_allavgB"}


class _ReplayScorer:
    """sirve scores por ventana ya calculados en el orden en que los pide run_model.evaluate,
    train y val primero y luego cada evento, asi el sweep reusa una sola pasada adelante para
    todo agg y regime."""

    def __init__(self, scores: list[np.ndarray], percentile: float = 99.0) -> None:
        self._scores = scores
        self._i = 0
        self.percentile = percentile

    # las ventanas llegan ignoradas, los scores vienen ya calculados; el contador avanza una
    # posicion por llamada, que es lo que ata este scorer al orden de evaluate
    def score_windows(self, windows):  # noqa: ARG002
        s = self._scores[self._i]
        self._i += 1
        return s


def _smooth(s: np.ndarray, w: int) -> np.ndarray:
    """media movil centrada de una secuencia de scores por bloque, la SMA sobre el score de
    anomalia del articulo de GDN. w<=1 no hace nada; sostener el error a picos compra
    coverage y earliness."""

    return s if w <= 1 else np.convolve(s, np.ones(w) / w, mode="same")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/gdn_config.yaml")
# nodos nominales para --selftest (curated Farm B / top100 Farm C), lo real sale de feats
_NOMINAL_NODES = {"B": 30, "C": 100}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_farm_config(farm: str) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    return _deep_merge(base, base.get("per_farm", {}).get(farm, {}))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, n_nodes: int, context: int | None = None,
                horizon: int | None = None) -> GDN:
    m, g, w = cfg["model"], cfg["graph"], cfg["windowing"]
    return GDN(n_nodes=n_nodes, context=context or w["context"],
               horizon=horizon if horizon is not None else w.get("horizon", 1),
               embed_dim=m["embed_dim"], out_layer_inter_dim=m["out_layer_inter_dim"],
               out_layer_num=m["out_layer_num"], topk=m["topk"], slope=m["leaky_relu_slope"],
               include_self_loops=g["include_self_loops"])


def selftest(farm: str, seed: int) -> None:
    cfg = load_farm_config(farm)
    set_seed(seed)
    n_nodes = _NOMINAL_NODES[farm]
    context = cfg["windowing"]["context"]
    model = build_model(cfg, n_nodes)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(8, n_nodes, context)
    out = model(x)
    adj = model.adjacency()
    edges_per_node = int(adj.sum(dim=1).float().mean().item())
    logger.info(f"[selftest farm {farm}] nodes={n_nodes} context={context} "
                f"topk={cfg['model']['topk']} params={n_params:,}")
    logger.info(f"  forward {tuple(x.shape)} -> {tuple(out.shape)} | adjacency {tuple(adj.shape)} "
                f"| edges/node={edges_per_node} | out finite={bool(torch.isfinite(out).all())}")


def _loss_fn(name: str):
    return {"mse": F.mse_loss, "mae": F.l1_loss, "huber": F.smooth_l1_loss}[name]


def train_gdn(model: GDN, x: torch.Tensor, target: torch.Tensor, cfg: dict,
              device: torch.device, seed: int, epochs: int | None = None) -> list[float]:
    """bucle minimo de entrenamiento de pronostico a un paso, devuelve la perdida media por
    epoca."""

    # se resiembra aqui para que el barajado del DataLoader dependa solo de esta semilla y
    # no del estado que traiga el proceso
    torch.manual_seed(seed)
    t = cfg["training"]
    epochs = epochs or t["epochs"]
    loss_fn = _loss_fn(t.get("loss", "mse"))
    loader = DataLoader(TensorDataset(x, target), batch_size=t["batch_size"], shuffle=True,
                        num_workers=t.get("num_workers", 0))
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    model.to(device).train()
    losses: list[float] = []
    for _ in range(epochs):
        tot, nb = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        losses.append(tot / max(nb, 1))
    return losses


@torch.no_grad()
def forecast_error(model: GDN, x: torch.Tensor, target: torch.Tensor, device: torch.device,
                   batch_size: int = 512) -> np.ndarray:
    """error de pronostico por ventana = media sobre sensores de |pred - target|, devuelve
    (N,)."""

    model.to(device).eval()
    loader = DataLoader(TensorDataset(x, target), batch_size=batch_size, shuffle=False)
    errs = []
    for xb, yb in loader:
        pred = model(xb.to(device))
        errs.append((pred - yb.to(device)).abs().mean(dim=1).cpu().numpy())
    return np.concatenate(errs)


def run_canary(cfg: dict, device: torch.device, epochs: int, seed: int = 42) -> dict:
    """sobreajusta 64 ventanas de Farm B, donde el error de pronostico tiene que caer muy por
    debajo del error sobre copias permutadas en el tiempo (sin dinamica) y sobre ruido blanco
    (sin estructura). mas una figura."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    windows, _, _ = load_frozen_windows(cfg["data"]["processed_path"], "B")
    rng = np.random.default_rng(seed)
    sel = windows[rng.choice(len(windows), 64, replace=False)]
    scaled = WindowScaler(per_turbine=False, method="standard").fit_transform(sel, None).astype(np.float32)
    context, k = cfg["windowing"]["context"], scaled.shape[2]
    logger.info(f"Canary: 64/{len(windows):,} Farm B windows {scaled.shape[1:]} (K={k}) | device {device}")

    cfg_canary = _deep_merge(cfg, {"training": {"batch_size": 16}})
    model = build_model(cfg_canary, n_nodes=k).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} params | embed_dim={cfg['model']['embed_dim']} topk={cfg['model']['topk']}")
    x, target = forecast_tensors(scaled, context)
    t0 = time.time()
    losses = train_gdn(model, x, target, cfg_canary, device, seed=seed, epochs=epochs)
    train_seconds = time.time() - t0

    # el control 1 permuta el tiempo en cada ventana, mismas marginales, sin dinamica y flojo
    # en sensores lentos; el control 2 es ruido blanco N(0,1), sin estructura temporal ni
    # entre sensores
    perm_rng = np.random.default_rng(123)
    xp, yp = forecast_tensors(time_permuted(scaled, perm_rng), context)
    xn, yn = forecast_tensors(perm_rng.normal(size=scaled.shape).astype(np.float32), context)
    err_train = forecast_error(model, x, target, device)
    err_perm = forecast_error(model, xp, yp, device)
    err_noise = forecast_error(model, xn, yn, device)
    ratio = float(err_perm.mean() / err_train.mean())
    ratio_noise = float(err_noise.mean() / err_train.mean())
    # umbrales flojos a proposito, al canario solo se le pide ensenar que aprendio algo
    # de dinamica (>1.2 sobre lo permutado) y mucho de estructura (>2 sobre el ruido)
    verdict = "PASS" if ratio > 1.2 and ratio_noise > 2.0 else "CHECK"
    logger.info(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} | err train {err_train.mean():.4f}±"
                f"{err_train.std():.4f} | permuted {err_perm.mean():.4f} (x{ratio:.2f}) | "
                f"white noise {err_noise.mean():.4f} (x{ratio_noise:.2f}) | {train_seconds:.0f}s -> {verdict}")

    fig_path = _canary_figure(cfg, model, x, target, losses, err_train, err_perm, err_noise, device)
    out = {"milestone": "H2_canary", "n_windows": 64, "epochs": epochs, "n_params": int(n_params),
           "embed_dim": cfg["model"]["embed_dim"], "topk": cfg["model"]["topk"], "context": context,
           "loss_first": losses[0], "loss_last": losses[-1],
           "err_train_mean": float(err_train.mean()), "err_train_std": float(err_train.std()),
           "err_permuted_mean": float(err_perm.mean()), "err_noise_mean": float(err_noise.mean()),
           "ratio_permuted_over_train": ratio, "ratio_noise_over_train": ratio_noise,
           "verdict": verdict, "train_seconds": round(train_seconds, 1),
           "device": str(device), "figure": str(fig_path)}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "gdn_h2_canary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"saved {tables / 'gdn_h2_canary.json'} + {fig_path}")
    return out


def _canary_figure(cfg: dict, model: GDN, x: torch.Tensor, target: torch.Tensor,
                   losses: list[float], err_train: np.ndarray, err_perm: np.ndarray,
                   err_noise: np.ndarray, device: torch.device) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        pred = model(x.to(device)).cpu().numpy()
    real = target.numpy()
    feats = np.argsort(real.std(axis=0))[-2:]  # los 2 sensores mas dinamicos

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    ax.semilogy(range(1, len(losses) + 1), losses)
    ax.set_title("GDN canary: training loss (64 windows, Farm B)")
    ax.set_xlabel("epoch")
    ax.set_ylabel(cfg["training"].get("loss", "mse"))
    for ax, f in zip([axes[0, 1], axes[1, 0]], feats[::-1]):
        ax.scatter(real[:, f], pred[:, f], s=14, alpha=0.7)
        lim = [min(real[:, f].min(), pred[:, f].min()), max(real[:, f].max(), pred[:, f].max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"sensor idx {f}: predicted vs real (next step)")
        ax.set_xlabel("real")
        ax.set_ylabel("pred")
    ax = axes[1, 1]
    ax.boxplot([err_train, err_perm, err_noise], tick_labels=["trained", "time-permuted", "white noise"])
    ax.set_ylabel("per-window mean |forecast error|")
    ax.set_title(f"structure check: permuted x{err_perm.mean() / err_train.mean():.1f}, "
                 f"noise x{err_noise.mean() / err_train.mean():.1f}")
    fig.tight_layout()
    fig_dir = Path(cfg["output"]["figures_dir"]) / "gdn"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "h2_canary_forecast.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_smoke_b(cfg: dict, device: torch.device, seed: int, epochs: int,
                max_train: int = 120000) -> dict:
    """Farm B curated end-to-end con el harness sin tocar, para un primer ROC-ds y CARE-CV.
    submuestrea el train por velocidad, asi que no es una cifra final; valida el contrato."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    windows, feats, _ = load_frozen_windows(cfg["data"]["processed_path"], "B")
    take = min(max_train, len(windows))
    sub = windows[rng.choice(len(windows), take, replace=False)]  # subconjunto barajado
    scaler = WindowScaler(per_turbine=False, method="standard")
    scaled = scaler.fit_transform(sub, None).astype(np.float32)
    n_val = int(round(take * cfg["training"]["val_ratio"]))
    # el corte va sobre el subconjunto ya barajado, asi que val y train son intercambiables
    val_norm, train_fit = scaled[:n_val], scaled[n_val:]  # particion normal
    test_seqs = load_test_sequences(cfg["data"]["processed_path"], "B")
    for s in test_seqs:
        s["windows"] = scaler.transform(s["windows"]).astype(np.float32)
    k, context, w = scaled.shape[2], cfg["windowing"]["context"], scaled.shape[1]
    logger.info(f"Smoke B [curated]: {len(feats)} sensors | fit {train_fit.shape} val {val_norm.shape} "
                f"| {len(test_seqs)} events | device {device}")

    cfg_smoke = _deep_merge(cfg, {"training": {"batch_size": 256}})
    model = build_model(cfg_smoke, n_nodes=k).to(device)
    logger.info(f"GDN {sum(p.numel() for p in model.parameters()):,} params | epochs={epochs} "
                f"topk={cfg['model']['topk']} agg={cfg['scoring']['aggregation']}")
    x, target = forecast_tensors(train_fit, context)
    t0 = time.time()
    losses = train_gdn(model, x, target, cfg_smoke, device, seed=seed, epochs=epochs)
    train_seconds = time.time() - t0

    sc = cfg["scoring"]
    scorer = GDNScorer(model, context=context, device=device, aggregation=sc["aggregation"],
                       topk_q=sc["topk_q"], batch_size=sc["batch_size"],
                       percentile=cfg["threshold"]["percentile"])
    scorer.fit_normalization(val_norm)
    adapter = OfficialCAREAdapter(raw_path=cfg["data"]["raw_path"], farm="B", window_size=w)
    ev = run_model.evaluate(scorer, val_norm, test_seqs, adapter, seed)
    best, cv = ev["best"], ev["cv"]
    logger.info(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} | train {train_seconds:.0f}s")
    logger.info(f"Smoke B: CARE-CV={cv['care']:.4f} (oracle={best['care']:.4f}, optim={cv['optimism']:+.4f}) "
                f"| ROC-ds={best['roc_ds']:.4f} ROC-win={ev['roc_window']:.4f} | Cov={best['cov']:.3f} "
                f"Earl={best['earl']:.3f} Rel={best['rel']:.3f} Acc={best['acc']:.3f}")
    out = {"milestone": "H3_smoke", "farm": "B", "features": "curated", "seed": seed, "epochs": epochs,
           "n_fit": len(train_fit), "n_val": len(val_norm), "aggregation": sc["aggregation"],
           "loss_first": losses[0], "loss_last": losses[-1], "train_seconds": round(train_seconds, 1),
           "care_cv": cv["care"], "care_oracle": best["care"], "optimism": cv["optimism"],
           "roc_ds": best["roc_ds"], "roc_window": ev["roc_window"], "coverage": best["cov"],
           "earliness": best["earl"], "reliability": best["rel"], "accuracy": best["acc"]}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "gdn_h3_smoke_b.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"saved {tables / 'gdn_h3_smoke_b.json'}")
    return out


def compute_exo_idx(feats: list[str], keywords: list[str]) -> list[int]:
    """indices de los sensores exogenos que anclan el regimen, wind_speed por prefijo y
    ambiente por coincidencia exacta, para que sensor_80 nunca case con sensor_8_avg. mismo
    emparejado que run_imdiffusion.compute_exo_idx."""

    out = []
    for i, f in enumerate(feats):
        if any((f == k) or (k.endswith("_") and f.startswith(k)) or f.startswith(k + "_")
               for k in keywords):
            out.append(i)
    return out


def _load_b(cfg: dict, arm: str) -> tuple[np.ndarray, list[dict], list[str]]:
    """arm de representacion de Farm B. ventanas de sensor escaladas con un StandardScaler y
    filtradas opcionalmente por la mascara de retencion del Isolation Forest del AE, por
    paridad. los nodos de GDN son los sensores, sin PCA. curated 30 feats, soft/allavg 54. las
    secuencias de test usan el mismo scaler."""

    pp = cfg["data"]["processed_path"] if _B_ARM_DIR[arm] is None else _B_ARM_DIR[arm]
    d = np.load(Path(pp) / "farm_b_train_windows.npz", allow_pickle=True)
    windows64 = d["windows"]  # float64, tal como lo vio el AE
    feats = [str(c) for c in d["feature_cols"]]
    if cfg.get("isolation_forest_mask", False):
        ifc = cfg["isolation_forest"]
        mask, scaler, _ = compute_if_retention_mask(
            windows64, variance_threshold=ifc["variance_threshold"],
            contamination=ifc["contamination"], n_estimators=ifc["n_estimators"])
        scaled = scaler.transform(windows64).astype(np.float32)[mask]
    else:
        scaler = WindowScaler(per_turbine=False, method="standard")
        scaled = scaler.fit_transform(windows64, None).astype(np.float32)
    test_seqs = load_test_sequences(pp, "B")
    # el test se transforma con el scaler ya ajustado en train, nunca con uno propio
    for s in test_seqs:
        s["windows"] = scaler.transform(s["windows"]).astype(np.float32)
    return scaled, test_seqs, feats


def _load_c(cfg: dict, features: str) -> tuple[np.ndarray, list[dict], list[str]]:
    """representacion de Farm C. reconstruye desde el crudo el mismo camino pca-first que usan
    el AE e ImDiffusion, limpieza minima, luego los filtros de status, potencia y transicion, y
    luego dedup, pero para antes del PCA y ventanea los sensores escalados, que son los nodos
    de GDN. top100 es el subconjunto de 100 del XAI del AE y allavg es todo `_avg`; sin IF, C
    nunca lo uso. stride_train adelgaza las ~1.12M ventanas de train, el test va a stride 1."""

    import pickle

    from sklearn.preprocessing import StandardScaler
    from src.preprocessing.config import load_config as load_preproc_config
    from src.preprocessing.power_curve_filter import PowerCurveFilter
    from src.preprocessing.status_filter import StatusFilter
    from src.preprocessing.transition_excluder import TransitionExcluder
    from src.preprocessing.windower import Windower

    # la representacion escalada se cachea (determinista, no depende de la semilla) para que
    # las ejecuciones multi-seed y de sweep se salten los ~22 min de reconstruir C del crudo;
    # vive en el directorio experiments, que esta en el gitignore
    stride_train = cfg["training"].get("stride_train", 1)
    cache = Path("data/processed/experiments") / f"gdn_c_{features}_st{stride_train}.npz"
    cache_test = cache.with_suffix(".test.pkl")
    if cache.exists() and cache_test.exists():
        z = np.load(cache, allow_pickle=True)
        tw, feats = z["train_windows"], [str(c) for c in z["feats"]]
        with open(cache_test, "rb") as f:
            test_seqs = pickle.load(f)
        logger.info(f"Farm C repr [{features}] from cache: train {tw.shape} | {len(test_seqs)} events")
        return tw, test_seqs, feats

    pcfg = _deep_merge(load_preproc_config(), load_preproc_config().get("per_farm", {}).get("C", {}))
    pp = pcfg.get("preproc", {})
    df, event_info, feats, reg = run_model.preprocess_rows(
        "C", pcfg, pp.get("outlier", False), pp.get("temp_norm", False), pp.get("smooth", False))
    if features == "top100":
        sub = set(Path("config/farm_c_top100_features.txt").read_text(encoding="utf-8").strip().splitlines())
        feats = [c for c in feats if c in sub]
    elif features != "allavg":
        raise ValueError(f"unknown C features arm: {features}")
    df_train = df[df["train_test"] == "train"].copy()
    df_train, _ = StatusFilter(pcfg).transform(df_train)
    df_train, _ = PowerCurveFilter(pcfg, reg).transform(df_train)
    df_train, _ = TransitionExcluder(pcfg).transform(df_train, event_info)
    if pcfg["data"].get("dedup_train", True):
        df_train = df_train.drop_duplicates(["asset_id", "time_stamp"]).reset_index(drop=True)
    df_pred = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)
    # medias de train, tambien para rellenar los huecos de prediction, para no filtrar test
    col_means = df_train[feats].mean()

    def arr(src):
        return src[feats].fillna(col_means).fillna(0.0).to_numpy(np.float32)

    scaler = StandardScaler().fit(arr(df_train))

    def to_wdf(src, scaled):
        meta = src[[c for c in run_model.META if c in src.columns]].reset_index(drop=True)
        return pd.concat([meta, pd.DataFrame(scaled, columns=feats)], axis=1)

    # train se adelgaza con stride_train por coste; el test va siempre a stride 1, porque la
    # cobertura y la anticipacion del harness se miden ventana a ventana
    wtrain = Windower({"windowing": {**pcfg["windowing"], "stride_samples": stride_train}})
    wtest = Windower({"windowing": {**pcfg["windowing"], "stride_samples": 1}})
    train_windows, _, _ = wtrain.create_train_windows(to_wdf(df_train, scaler.transform(arr(df_train))), feats)
    test_obj, _ = wtest.create_test_sequences(to_wdf(df_pred, scaler.transform(arr(df_pred))), event_info, feats)
    test_seqs = [{"event_id": s.event_id, "event_label": s.event_label, "asset_id": s.asset_id,
                  "windows": s.windows.astype(np.float32), "labels": np.asarray(s.labels, np.int8),
                  "window_timestamps": s.window_timestamps} for s in test_obj]
    train_windows = train_windows.astype(np.float32)
    logger.info(f"Farm C repr [{features}]: {len(feats)} sensors | train {train_windows.shape} "
                f"(stride_train={stride_train}) | {len(test_seqs)} events")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, train_windows=train_windows, feats=np.array(feats, dtype=object))
    with open(cache_test, "wb") as f:
        pickle.dump(test_seqs, f)
    logger.info(f"cached C repr -> {cache}")
    return train_windows, test_seqs, feats


def load_representation(farm: str, cfg: dict, features: str, seed: int):
    """despacha al arm de representacion que toque, devuelve (train escalado (M,W,K)
    float32, test_seqs, feats)."""

    if farm == "B" and features in _B_ARM_DIR:
        return _load_b(cfg, features)
    if farm == "C" and features in ("top100", "allavg"):
        return _load_c(cfg, features)
    raise NotImplementedError(f"representation arm farm={farm} features={features} lands later")


def run_full(farm: str, cfg: dict, features: str, regime: bool, seed: int, epochs: int,
             device: torch.device) -> dict:
    """ejecucion completa de GDN por granja con la evaluacion del harness, una fila de tabla
    reproducible por (farm, arm, regime, seed). la metrica de seleccion es ROC-ds y se reporta
    CARE-CV."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaled, test_seqs, feats = load_representation(farm, cfg, features, seed)
    k, w = scaled.shape[2], scaled.shape[1]
    horizon = cfg["windowing"].get("horizon", 1)
    context = min(cfg["windowing"]["context"], w - horizon)  # context+horizon <= W
    perm = rng.permutation(len(scaled))
    n_val = int(round(len(scaled) * cfg["training"]["val_ratio"]))
    val_norm, train_fit = scaled[perm[:n_val]], scaled[perm[n_val:]]
    exo_idx = None
    if regime:
        kw = cfg["regime"].get("exo_keywords") or _DEFAULT_EXO[farm]
        exo_idx = compute_exo_idx(feats, kw)
        logger.info(f"regime ON: {len(exo_idx)} exogenous nodes off-score -> {[feats[i] for i in exo_idx]}")
    logger.info(f"Farm {farm} [{features}] regime={regime} h={horizon} ctx={context} seed={seed}: "
                f"{k} sensors | fit {train_fit.shape} val {val_norm.shape} | {len(test_seqs)} events | {device}")

    model = build_model(cfg, n_nodes=k, context=context, horizon=horizon).to(device)
    logger.info(f"GDN {sum(p.numel() for p in model.parameters()):,} params | epochs={epochs} "
                f"topk={cfg['model']['topk']} agg={cfg['scoring']['aggregation']} horizon={horizon}")
    x, target = forecast_tensors(train_fit, context, horizon)
    t0 = time.time()
    losses = train_gdn(model, x, target, cfg, device, seed=seed, epochs=epochs)
    secs = time.time() - t0

    sc = cfg["scoring"]
    scorer = GDNScorer(model, context=context, device=device, aggregation=sc["aggregation"],
                       topk_q=sc["topk_q"], exo_idx=exo_idx, batch_size=sc["batch_size"],
                       percentile=cfg["threshold"]["percentile"], horizon=horizon)
    scorer.fit_normalization(val_norm)
    adapter = OfficialCAREAdapter(raw_path=cfg["data"]["raw_path"], farm=farm, window_size=w)
    ev = run_model.evaluate(scorer, val_norm, test_seqs, adapter, seed)
    best, cv = ev["best"], ev["cv"]
    tag = (f"gdn_farm_{farm.lower()}_{features}{'_regime' if regime else ''}"
           f"{f'_h{horizon}' if horizon > 1 else ''}_s{seed}")
    logger.info(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} | train {secs:.0f}s")
    logger.info(f"{tag}: CARE-CV={cv['care']:.4f} (oracle={best['care']:.4f}, optim={cv['optimism']:+.4f}) "
                f"| ROC-ds={best['roc_ds']:.4f} ROC-win={ev['roc_window']:.4f} | Cov={best['cov']:.3f} "
                f"Earl={best['earl']:.3f} Rel={best['rel']:.3f} Acc={best['acc']:.3f}")

    models = Path(cfg["output"]["models_dir"])
    models.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feats": feats, "n_nodes": k, "context": context,
                "farm": farm, "features": features, "regime": regime, "seed": seed}, models / f"{tag}.pt")
    out = {"milestone": "H4", "tag": tag, "farm": farm, "features": features, "regime": regime,
           "seed": seed, "epochs": epochs, "n_nodes": k, "topk": cfg["model"]["topk"],
           "horizon": horizon, "context": context,
           "aggregation": sc["aggregation"], "if_mask": bool(cfg.get("isolation_forest_mask", False)),
           "n_fit": len(train_fit), "n_val": len(val_norm), "train_seconds": round(secs, 1),
           "loss_first": losses[0], "loss_last": losses[-1],
           "care_cv": cv["care"], "care_oracle": best["care"], "optimism": cv["optimism"],
           "roc_ds": best["roc_ds"], "roc_window": ev["roc_window"], "coverage": best["cov"],
           "earliness": best["earl"], "reliability": best["rel"], "accuracy": best["acc"],
           "fold_h_scale": cv["fold_h_scale"], "sweep": ev["sweep"]}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"saved {tables / (tag + '.json')} + checkpoint {models / (tag + '.pt')}")
    return out


def run_sweep(farm: str, cfg: dict, device: torch.device, seeds: list[int], topks: list[int],
              aggs: list[str], regimes: list[bool], mas: list[int], epochs: int,
              out_tag: str = "sweep") -> dict:
    """barrido en rejilla de una granja. carga la representacion una vez, entrena una vez por
    (seed, topk) y cachea las matrices de error por sensor, y luego barre agg x regime x
    suavizado MA como agregaciones post-hoc baratas. metrica de seleccion = ROC-ds medio sobre
    las semillas, y se reporta CARE-CV."""

    from collections import defaultdict

    scaled, test_seqs, feats = load_representation(farm, cfg, cfg["features"], seeds[0])
    context, w, kk = cfg["windowing"]["context"], scaled.shape[1], scaled.shape[2]
    exo_idx = compute_exo_idx(feats, cfg["regime"].get("exo_keywords") or _DEFAULT_EXO[farm])
    adapter = OfficialCAREAdapter(raw_path=cfg["data"]["raw_path"], farm=farm, window_size=w)
    sc = cfg["scoring"]
    logger.info(f"Sweep {farm} [{cfg['features']}]: {kk} sensors, {len(test_seqs)} events | seeds={seeds} "
                f"topks={topks} aggs={aggs} regimes={regimes} mas={mas} | exo={[feats[i] for i in exo_idx]}")
    rows = []
    for seed in seeds:
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(scaled))
        n_val = int(round(len(scaled) * cfg["training"]["val_ratio"]))
        val_norm, train_fit = scaled[perm[:n_val]], scaled[perm[n_val:]]
        x, target = forecast_tensors(train_fit, context)
        for topk in topks:
            cfg_tk = _deep_merge(cfg, {"model": {"topk": topk}})
            torch.manual_seed(seed)
            model = build_model(cfg_tk, n_nodes=kk).to(device)
            t0 = time.time()
            train_gdn(model, x, target, cfg, device, seed=seed, epochs=epochs)
            scorer = GDNScorer(model, context=context, device=device, batch_size=sc["batch_size"])
            val_err = scorer._sensor_errors(val_norm)
            ev_err = [scorer._sensor_errors(s["windows"]) for s in test_seqs]
            # normalizacion robusta por sensor, mediana e IQR de validacion, para que un
            # sensor ruidoso no domine la agregacion; el suelo 1e-6 evita dividir por cero
            # en un sensor plano
            med = np.median(val_err, axis=0)
            q25, q75 = np.percentile(val_err, [25, 75], axis=0)
            iqr = np.maximum(q75 - q25, 1e-6)
            logger.info(f"seed {seed} topk {topk}: trained {time.time() - t0:.0f}s -> sweep agg x regime x MA")
            for agg in aggs:
                for regime in regimes:
                    keep = [i for i in range(kk) if i not in exo_idx] if regime else list(range(kk))

                    def _score(err, keep=keep, agg=agg):
                        return aggregate_sensor_errors(((err - med) / iqr)[:, keep], agg,
                                                       sc["topk_q"]).astype(np.float64)

                    # el orden importa, _ReplayScorer los sirve tal cual, validacion primero
                    # y luego un evento por posicion, que es como llama evaluate
                    base = [_score(val_err)] + [_score(e) for e in ev_err]
                    for ma in mas:
                        scores = [_smooth(s, ma) for s in base]
                        ev = run_model.evaluate(_ReplayScorer(scores, cfg["threshold"]["percentile"]),
                                                val_norm, test_seqs, adapter, seed)
                        b = ev["best"]
                        rows.append({"seed": seed, "topk": topk, "agg": agg, "regime": regime, "ma": ma,
                                     "roc_ds": b["roc_ds"], "care_cv": ev["cv"]["care"],
                                     "optimism": ev["cv"]["optimism"], "roc_win": ev["roc_window"],
                                     "cov": b["cov"], "earl": b["earl"], "rel": b["rel"], "acc": b["acc"]})

    # se agrupa por configuracion y se promedia sobre semillas, nunca al reves
    grp = defaultdict(list)
    for r in rows:
        grp[(r["topk"], r["agg"], r["regime"], r["ma"])].append(r)
    summ = [{"topk": t, "agg": a, "regime": rg, "ma": m, "n_seeds": len(rs),
             "roc_ds_mean": float(np.mean([r["roc_ds"] for r in rs])),
             "roc_ds_std": float(np.std([r["roc_ds"] for r in rs])),
             "care_cv_mean": float(np.mean([r["care_cv"] for r in rs])),
             "care_cv_std": float(np.std([r["care_cv"] for r in rs])),
             "earl_mean": float(np.mean([r["earl"] for r in rs])),
             "cov_mean": float(np.mean([r["cov"] for r in rs]))}
            for (t, a, rg, m), rs in grp.items()]
    summ.sort(key=lambda r: r["roc_ds_mean"], reverse=True)
    out = {"milestone": "H5c", "farm": farm, "features": cfg["features"], "seeds": seeds,
           "topks": topks, "aggs": aggs, "regimes": regimes, "mas": mas, "epochs": epochs,
           "rows": rows, "summary": summ}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    out_path = tables / f"gdn_{out_tag}_{farm.lower()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"=== Farm {farm} {out_tag} — ranked by mean ROC-ds (multi-seed); CARE-CV in col ===")
    for r in sorted(summ, key=lambda r: r["care_cv_mean"], reverse=True):
        logger.info(f"topk={r['topk']:>2} agg={r['agg']:<5} reg={str(r['regime'])[0]} ma={r['ma']:>2} | "
                    f"ROC-ds={r['roc_ds_mean']:.4f}+-{r['roc_ds_std']:.3f} | "
                    f"CARE-CV={r['care_cv_mean']:.4f}+-{r['care_cv_std']:.3f} | "
                    f"Cov={r['cov_mean']:.3f} Earl={r['earl_mean']:.3f}")
    logger.info(f"saved {out_path}")
    return out


def run_fusion_b(cfg: dict, device: torch.device, seeds: list[int], alphas: list[float]) -> dict:
    """fusiona el score de reconstruccion del AE (results/models/ae_scores_farm_b.npz) con el
    score de desviacion de pronostico de GDN, sacado de checkpoints recargados, cada uno
    z-normalizado con sus stats de train y combinados como alpha*z_AE + (1-alpha)*z_GDN y como
    un max, y evaluados despues con el harness. alpha=1 reproduce el AE y alpha=0 es GDN solo.
    la correlacion GDN-AE en B es baja, ~0.3-0.44, asi que los dos son complementarios; se
    ordena por CARE-CV medio, cuyo objetivo es batir el 0.721 de AE-B. el volcado del AE es de
    una sola semilla, asi que la varianza de la fusion es solo de la semilla de GDN."""

    from collections import defaultdict

    aez = np.load("results/models/ae_scores_farm_b.npz")
    ae_train = aez["train_mae"].astype(np.float64)
    sc = cfg["scoring"]
    adapter = None
    rows = []
    for seed in seeds:
        scaled, test_seqs, feats = load_representation("B", cfg, cfg["features"], seed)
        if adapter is None:
            adapter = OfficialCAREAdapter(raw_path=cfg["data"]["raw_path"], farm="B",
                                          window_size=scaled.shape[1])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(scaled))
        n_val = int(round(len(scaled) * cfg["training"]["val_ratio"]))
        val_norm = scaled[perm[:n_val]]
        ck = torch.load(Path(cfg["output"]["models_dir"]) / f"gdn_farm_b_curated_s{seed}.pt",
                        map_location=device)
        model = build_model(cfg, n_nodes=scaled.shape[2]).to(device)
        model.load_state_dict(ck["state_dict"])
        scorer = GDNScorer(model, context=cfg["windowing"]["context"], device=device,
                           aggregation=sc["aggregation"], topk_q=sc["topk_q"], batch_size=sc["batch_size"])
        scorer.fit_normalization(val_norm)
        gdn_train = scorer.score_windows(scaled).astype(np.float64)
        # si las longitudes no casan es que la mascara IF de este run no es la del volcado
        # del AE, y entonces las dos series no estan alineadas ventana a ventana
        if len(gdn_train) != len(ae_train):
            raise RuntimeError(f"AE/GDN train length mismatch: {len(ae_train)} vs {len(gdn_train)} "
                               "(IF mask must match the AE dump)")
        gdn_ev = [scorer.score_windows(s["windows"]).astype(np.float64) for s in test_seqs]
        ae_ev = [aez[f"ev{s['event_id']}_mae"].astype(np.float64) for s in test_seqs]

        def _z(s, mu, sd):
            return (s - mu) / sd

        # las stats de z salen solo de train, para que la mezcla no vea los eventos
        amu, asd, gmu, gsd = ae_train.mean(), ae_train.std(), gdn_train.mean(), gdn_train.std()
        zat, zgt = _z(ae_train, amu, asd), _z(gdn_train, gmu, gsd)
        zae = [_z(e, amu, asd) for e in ae_ev]
        zge = [_z(e, gmu, gsd) for e in gdn_ev]
        for combo in [*alphas, "max"]:
            if combo == "max":
                ft, fe, label = np.maximum(zat, zgt), [np.maximum(a, g) for a, g in zip(zae, zge)], "max"
            else:
                a = float(combo)
                ft = a * zat + (1 - a) * zgt
                fe = [a * za + (1 - a) * zg for za, zg in zip(zae, zge)]
                label = f"a={a:.2f}"
            ev = run_model.evaluate(_ReplayScorer([ft] + fe, cfg["threshold"]["percentile"]),
                                    scaled, test_seqs, adapter, seed)
            b = ev["best"]
            rows.append({"seed": seed, "combo": label, "roc_ds": b["roc_ds"], "care_cv": ev["cv"]["care"],
                         "cov": b["cov"], "earl": b["earl"], "rel": b["rel"], "acc": b["acc"]})
        logger.info(f"seed {seed}: fused {len(alphas) + 1} combos")

    grp = defaultdict(list)
    for r in rows:
        grp[r["combo"]].append(r)
    summ = [{"combo": c, "n": len(rs),
             "care_cv_mean": float(np.mean([r["care_cv"] for r in rs])),
             "care_cv_std": float(np.std([r["care_cv"] for r in rs])),
             "roc_ds_mean": float(np.mean([r["roc_ds"] for r in rs])),
             "cov_mean": float(np.mean([r["cov"] for r in rs])),
             "earl_mean": float(np.mean([r["earl"] for r in rs])),
             "acc_mean": float(np.mean([r["acc"] for r in rs]))} for c, rs in grp.items()]
    summ.sort(key=lambda r: r["care_cv_mean"], reverse=True)
    out = {"milestone": "Tier2_fusion", "farm": "B", "seeds": seeds, "alphas": alphas,
           "rows": rows, "summary": summ}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "gdn_fusion_b.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("=== AE+GDN fusion B — ranked by mean CARE-CV (alpha=1 is AE-alone) ===")
    for r in summ:
        logger.info(f"{r['combo']:<7} | CARE-CV={r['care_cv_mean']:.4f}+-{r['care_cv_std']:.3f} | "
                    f"ROC-ds={r['roc_ds_mean']:.4f} | Cov={r['cov_mean']:.3f} Earl={r['earl_mean']:.3f} "
                    f"Acc={r['acc_mean']:.3f}")
    logger.info(f"saved {tables / 'gdn_fusion_b.json'}")
    return out


def run_horizon_b(cfg: dict, device: torch.device, seeds: list[int], horizons: list[int],
                  epochs: int) -> dict:
    """barrido del horizonte de pronostico en B, h pasos adelante. con h>1 los fallos
    sostenidos se vuelven persistentemente dificiles de pronosticar y el error se sostiene; el
    context se fija solo a W-h y h=1 es la baseline."""

    from collections import defaultdict

    rows = []
    for h in horizons:
        cfg_h = _deep_merge(cfg, {"windowing": {"horizon": h}})
        for seed in seeds:
            r = run_full("B", cfg_h, features=cfg["features"], regime=False, seed=seed,
                         epochs=epochs, device=device)
            rows.append({"horizon": h, "seed": seed, "context": r["context"], "roc_ds": r["roc_ds"],
                         "care_cv": r["care_cv"], "cov": r["coverage"], "earl": r["earliness"],
                         "rel": r["reliability"], "acc": r["accuracy"]})
    grp = defaultdict(list)
    for r in rows:
        grp[r["horizon"]].append(r)
    summ = [{"horizon": h, "context": rs[0]["context"], "n": len(rs),
             "care_cv_mean": float(np.mean([r["care_cv"] for r in rs])),
             "care_cv_std": float(np.std([r["care_cv"] for r in rs])),
             "roc_ds_mean": float(np.mean([r["roc_ds"] for r in rs])),
             "cov_mean": float(np.mean([r["cov"] for r in rs])),
             "earl_mean": float(np.mean([r["earl"] for r in rs])),
             "acc_mean": float(np.mean([r["acc"] for r in rs]))} for h, rs in grp.items()]
    summ.sort(key=lambda r: r["care_cv_mean"], reverse=True)
    out = {"milestone": "Tier3_horizon", "farm": "B", "seeds": seeds, "horizons": horizons,
           "rows": rows, "summary": summ}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "gdn_horizon_b.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("=== Farm B horizon sweep — ranked by mean CARE-CV (h=1 is the baseline) ===")
    for r in summ:
        logger.info(f"h={r['horizon']:>2} ctx={r['context']:>2} | CARE-CV={r['care_cv_mean']:.4f}+-{r['care_cv_std']:.3f} "
                    f"| ROC-ds={r['roc_ds_mean']:.4f} | Cov={r['cov_mean']:.3f} Earl={r['earl_mean']:.3f} Acc={r['acc_mean']:.3f}")
    logger.info(f"saved {tables / 'gdn_horizon_b.json'}")
    return out


def run_arms_b(cfg: dict, device: torch.device, arms: list[str], seeds: list[int],
               epochs: int) -> dict:
    """comparacion de arms de representacion de Farm B con el nucleo fijo, con topk, agg y
    regime tomados de la config. corre run_full por (arm, seed) y ordena los arms por ROC-ds
    medio."""

    from collections import defaultdict

    rows = []
    for arm in arms:
        for seed in seeds:
            r = run_full("B", cfg, features=arm, regime=False, seed=seed, epochs=epochs, device=device)
            rows.append({"arm": arm, "seed": seed, "roc_ds": r["roc_ds"], "care_cv": r["care_cv"],
                         "earl": r["earliness"], "cov": r["coverage"], "rel": r["reliability"],
                         "acc": r["accuracy"], "n_nodes": r["n_nodes"]})
    grp = defaultdict(list)
    for r in rows:
        grp[r["arm"]].append(r)
    summ = [{"arm": a, "n_nodes": rs[0]["n_nodes"], "n_seeds": len(rs),
             "roc_ds_mean": float(np.mean([r["roc_ds"] for r in rs])),
             "roc_ds_std": float(np.std([r["roc_ds"] for r in rs])),
             "care_cv_mean": float(np.mean([r["care_cv"] for r in rs])),
             "care_cv_std": float(np.std([r["care_cv"] for r in rs])),
             "earl_mean": float(np.mean([r["earl"] for r in rs]))}
            for a, rs in grp.items()]
    summ.sort(key=lambda r: r["roc_ds_mean"], reverse=True)
    out = {"milestone": "H4c", "farm": "B", "arms": arms, "seeds": seeds, "epochs": epochs,
           "topk": cfg["model"]["topk"], "aggregation": cfg["scoring"]["aggregation"],
           "rows": rows, "summary": summ}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "gdn_arms_b.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("=== Farm B representation arms — ranked by mean ROC-ds (multi-seed) ===")
    for r in summ:
        logger.info(f"arm={r['arm']:<8} nodes={r['n_nodes']:>2} | "
                    f"ROC-ds={r['roc_ds_mean']:.4f}+-{r['roc_ds_std']:.3f} | "
                    f"CARE-CV={r['care_cv_mean']:.4f}+-{r['care_cv_std']:.3f} | Earl={r['earl_mean']:.3f}")
    logger.info(f"saved {tables / 'gdn_arms_b.json'}")
    return out


def run_loss_ensemble(farm: str, cfg: dict, device: torch.device, seeds: list[int], losses: list[str],
                      epochs: int, split_seed: int = 42) -> dict:
    """pasada de ajuste sobre una granja, perdida de entrenamiento {mse, huber} x {una semilla,
    deep ensemble}. una perdida robusta (Huber) ataca los falsos positivos de pico normal que
    el MSE persigue, y el deep ensemble promedia los scores de desviacion por ventana entre
    semillas sobre una particion val/train fija donde solo cambian la inicializacion del modelo
    y el barajado del entrenamiento, o sea reduccion de varianza y no espiar el test. reusa la
    representacion cacheada; seleccion = ROC-ds medio y se reporta CARE-CV. la particion se
    ancla a `split_seed`, la de run_full ya establecida, asi que la ejecucion mse cuya
    semilla de inicializacion == split_seed reproduce exacta la baseline publicada
    (C s42 0.7297/0.8459, media 0.7228, AE-C 0.7432; B s42 0.6445/0.7593, AE-B 0.7211),
    un control de coherencia incorporado."""

    scaled, test_seqs, feats = load_representation(farm, cfg, cfg["features"], seeds[0])
    w, kk = scaled.shape[1], scaled.shape[2]
    horizon = cfg["windowing"].get("horizon", 1)
    context = min(cfg["windowing"]["context"], w - horizon)
    adapter = OfficialCAREAdapter(raw_path=cfg["data"]["raw_path"], farm=farm, window_size=w)
    sc, pct = cfg["scoring"], cfg["threshold"]["percentile"]

    # una sola particion fija para todas las semillas, asi el deep ensemble promedia sobre las
    # mismas ventanas de val y de evento; solo cambian la inicializacion y el barajado
    perm = np.random.default_rng(split_seed).permutation(len(scaled))
    n_val = int(round(len(scaled) * cfg["training"]["val_ratio"]))
    val_norm, train_fit = scaled[perm[:n_val]], scaled[perm[n_val:]]
    x, target = forecast_tensors(train_fit, context, horizon)
    logger.info(f"Ensemble {farm} [{cfg['features']}]: {kk} sensors | fit {train_fit.shape} "
                f"val {val_norm.shape} | {len(test_seqs)} events | losses={losses} seeds={seeds} | {device}")

    rows = []
    for loss in losses:
        cfg_l = _deep_merge(cfg, {"training": {"loss": loss}})
        per_seed = []  # uno por semilla, [val_score] + [event_scores]
        for seed in seeds:
            torch.manual_seed(seed)  # varia la inicializacion, train resiembra el barajado
            model = build_model(cfg_l, n_nodes=kk, context=context, horizon=horizon).to(device)
            t0 = time.time()
            train_gdn(model, x, target, cfg_l, device, seed=seed, epochs=epochs)
            scorer = GDNScorer(model, context=context, device=device, aggregation=sc["aggregation"],
                               topk_q=sc["topk_q"], batch_size=sc["batch_size"], percentile=pct,
                               horizon=horizon)
            scorer.fit_normalization(val_norm)
            scores = [scorer.score_windows(val_norm)] + [scorer.score_windows(s["windows"]) for s in test_seqs]
            per_seed.append(scores)
            ev = run_model.evaluate(_ReplayScorer(scores, pct), val_norm, test_seqs, adapter, seed)
            b = ev["best"]
            rows.append({"loss": loss, "mode": f"seed{seed}", "roc_ds": b["roc_ds"], "care_cv": ev["cv"]["care"],
                         "cov": b["cov"], "earl": b["earl"], "rel": b["rel"], "acc": b["acc"]})
            logger.info(f"{loss} seed {seed}: {time.time()-t0:.0f}s | CARE-CV={ev['cv']['care']:.4f} "
                        f"ROC-ds={b['roc_ds']:.4f} | Cov={b['cov']:.3f} Earl={b['earl']:.3f} Acc={b['acc']:.3f}")
        # se promedia posicion a posicion, no serie a serie, porque con el split fijo la
        # posicion i es la misma ventana en todas las semillas
        ens = [np.mean([ps[i] for ps in per_seed], axis=0) for i in range(len(per_seed[0]))]
        ev = run_model.evaluate(_ReplayScorer(ens, pct), val_norm, test_seqs, adapter, 0)
        b = ev["best"]
        rows.append({"loss": loss, "mode": f"ens{len(seeds)}", "roc_ds": b["roc_ds"], "care_cv": ev["cv"]["care"],
                     "cov": b["cov"], "earl": b["earl"], "rel": b["rel"], "acc": b["acc"]})
        logger.info(f"{loss} ENSEMBLE({len(seeds)}): CARE-CV={ev['cv']['care']:.4f} ROC-ds={b['roc_ds']:.4f} "
                    f"| Cov={b['cov']:.3f} Earl={b['earl']:.3f} Acc={b['acc']:.3f}")

    # por perdida, la media de los miembros sueltos frente a la unica fila del ensemble
    summ = []
    for loss in losses:
        single = [r for r in rows if r["loss"] == loss and r["mode"].startswith("seed")]
        e = next(r for r in rows if r["loss"] == loss and r["mode"].startswith("ens"))
        summ.append({"loss": loss, "single_care_mean": float(np.mean([r["care_cv"] for r in single])),
                     "single_care_std": float(np.std([r["care_cv"] for r in single])),
                     "single_rocds_mean": float(np.mean([r["roc_ds"] for r in single])),
                     "single_rocds_std": float(np.std([r["roc_ds"] for r in single])),
                     "ens_care": e["care_cv"], "ens_rocds": e["roc_ds"], "ens_cov": e["cov"],
                     "ens_earl": e["earl"], "ens_acc": e["acc"]})
    out = {"milestone": "loss_ensemble", "farm": farm, "features": cfg["features"], "seeds": seeds,
           "losses": losses, "epochs": epochs, "topk": cfg["model"]["topk"], "rows": rows, "summary": summ}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / f"gdn_loss_ensemble_{farm.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"=== Farm {farm} loss x ensemble (single mean+-std vs deep ensemble) ===")
    for s in summ:
        logger.info(f"loss={s['loss']:<5} | single CARE {s['single_care_mean']:.4f}+-{s['single_care_std']:.3f} "
                    f"ROC {s['single_rocds_mean']:.4f}+-{s['single_rocds_std']:.3f} | ENS CARE {s['ens_care']:.4f} "
                    f"ROC {s['ens_rocds']:.4f} | Cov {s['ens_cov']:.3f} Earl {s['ens_earl']:.3f} Acc {s['ens_acc']:.3f}")
    logger.info(f"saved {tables / f'gdn_loss_ensemble_{farm.lower()}.json'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="runner de GDN (Fase 2)")
    ap.add_argument("--farm", choices=["B", "C"], default="B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", default=None, help="arm de representacion (curated/soft/allavg/top100)")
    ap.add_argument("--window", type=int, default=None, help="sobrescribe windowing.window_size")
    ap.add_argument("--regime", action="store_true", help="ancla el viento y el ambiente exogenos como contexto del grafo")
    ap.add_argument("--score-only", action="store_true", help="recarga el checkpoint y vuelve a puntuar")
    ap.add_argument("--selftest", action="store_true", help="pasada adelante sobre datos sinteticos (Hito 1)")
    ap.add_argument("--canary", action="store_true", help="H2, control de sobreajuste sobre 64 ventanas de Farm B")
    ap.add_argument("--canary-epochs", type=int, default=400)
    ap.add_argument("--smoke", action="store_true", help="H3, Farm B curated end-to-end (ROC-ds + CARE-CV)")
    ap.add_argument("--sweep", action="store_true", help="rejilla (topk x agg x regime, multi-seed)")
    ap.add_argument("--smooth", action="store_true", help="barrido SMA x agg con topk fijo (coverage/earliness)")
    ap.add_argument("--arms", action="store_true", help="H4c, arms de representacion de Farm B (curated/soft/allavg)")
    ap.add_argument("--fuse", action="store_true", help="Tier 2, fusion AE+GDN en B (barrido de alpha + max)")
    ap.add_argument("--horizon-sweep", action="store_true", help="Tier 3, barrido del horizonte de pronostico en B")
    ap.add_argument("--loss-ensemble", action="store_true", help="loss {mse,huber} x deep-seed-ensemble (la granja va en --farm)")
    ap.add_argument("--max-train", type=int, default=120000, help="submuestra de train del smoke H3")
    ap.add_argument("--epochs", type=int, default=None, help="sobrescribe training.epochs")
    args = ap.parse_args()

    # el selftest sale antes de mirar el dispositivo, que es lo que lo hace correr sin GPU
    if args.selftest:
        selftest(args.farm, args.seed)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.canary:
        run_canary(load_farm_config("B"), device, epochs=args.canary_epochs, seed=args.seed)
        return
    if args.smoke:
        cfg = load_farm_config("B")
        run_smoke_b(cfg, device, seed=args.seed, epochs=args.epochs or cfg["training"]["epochs"],
                    max_train=args.max_train)
        return
    if args.sweep:
        cfg = load_farm_config(args.farm)
        if args.farm == "C":
            # barrido enfocado, topk x regime a max (la evaluacion de C va a ~110s por celda)
            topks, aggs = [12, 18, 25], ["max"]
        else:
            topks, aggs = [8, 12, 16], ["max", "mean", "topk"]
        run_sweep(args.farm, cfg, device, seeds=[42, 7], topks=topks, aggs=aggs,
                  regimes=[False, True], mas=[1], epochs=args.epochs or cfg["training"]["epochs"])
        return
    if args.smooth:
        cfg = load_farm_config(args.farm)
        run_sweep(args.farm, cfg, device, seeds=[42, 7], topks=[cfg["model"]["topk"]],
                  aggs=["max", "mean", "topk"], regimes=[False], mas=[1, 3, 5, 7, 9],
                  epochs=args.epochs or cfg["training"]["epochs"], out_tag="smooth")
        return
    if args.arms:
        cfg = load_farm_config("B")
        run_arms_b(cfg, device, arms=["curated", "soft", "allavg"], seeds=[42, 7],
                   epochs=args.epochs or cfg["training"]["epochs"])
        return
    if args.fuse:
        cfg = load_farm_config("B")
        run_fusion_b(cfg, device, seeds=[42, 7],
                     alphas=[0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0])
        return
    if args.horizon_sweep:
        cfg = load_farm_config("B")
        run_horizon_b(cfg, device, seeds=[42, 7], horizons=[1, 2, 3, 6],
                      epochs=args.epochs or cfg["training"]["epochs"])
        return
    if args.loss_ensemble:
        cfg = load_farm_config(args.farm)
        seeds = ([42, 7, 123, 2024, 1, 99, 11, 33, 55, 77, 88, 2025, 314, 271, 1618]
                 if args.farm == "C" else [42, 7, 123, 2024, 1, 99])
        run_loss_ensemble(args.farm, cfg, device, seeds=seeds, losses=["mse", "huber"],
                          epochs=args.epochs or cfg["training"]["epochs"])
        return
    # sin ningun modo, un run completo con el arm por defecto de la granja
    cfg = load_farm_config(args.farm)
    run_full(args.farm, cfg, features=args.features or cfg["features"], regime=args.regime,
             seed=args.seed, epochs=args.epochs or cfg["training"]["epochs"], device=device)


if __name__ == "__main__":
    main()
