"""CRGN runner, el nucleo de Yang et al. (Energies 2025), con train, score y evaluacion.

    python scripts/run_crgn.py --selftest --farm B

el selftest construye el modelo por parque sobre datos sinteticos y corre un forward mas un
InfoNCE, sin datos y sin GPU, para probar el cableado. los loaders de representacion se
reutilizan verbatim de las arms congeladas de GDN, en read-only, y la evaluacion pasa por el
harness intacto, run_model.evaluate via CRGNScorer.
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
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (imports hermanos)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz del repo (from src.X)

from src.crgn.data import stack_timesteps, to_node_major
from src.crgn.graph import build_adjacency
from src.crgn.model import CRGN
from src.crgn.scorer import CRGNScorer
# el loader de ventanas congeladas de GDN y su control de permutacion temporal, en read-only;
# src/gdn no se edita
from src.gdn.data import load_frozen_windows, time_permuted
from src.model_prep.model_scaler import WindowScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/crgn_config.yaml")
# nodos nominales para --selftest (curated B / top100 C); las ejecuciones reales leen feats
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


def build_model(cfg: dict, n_nodes: int, in_dim: int) -> CRGN:
    m = cfg["model"]
    return CRGN(n_nodes=n_nodes, in_dim=in_dim, embed_dim=m["embed_dim"], n_layers=m["gcn_layers"],
                slope=m["leaky_relu_slope"], readout=m["readout"], proj_dim=m.get("proj_dim"),
                temperature=cfg["contrastive"]["temperature"], noise_std=cfg["augmentation"]["noise_std"],
                mask_ratio=cfg["augmentation"]["mask_ratio"])


def train_crgn(model: CRGN, x: torch.Tensor, adj: torch.Tensor, cfg: dict, device: torch.device,
               seed: int, epochs: int | None = None) -> list[float]:
    """bucle de entrenamiento contrastivo auto-supervisado (InfoNCE sobre vistas aumentadas).
    x viene en node-major (N, K, W) y adj es el grafo Spearman fijo ya normalizado. devuelve
    la perdida media por epoca."""

    # se resiembra aqui para que el barajado del DataLoader dependa solo de esta semilla
    torch.manual_seed(seed)
    t = cfg["training"]
    epochs = epochs or t["epochs"]
    bs = min(t["batch_size"], x.shape[0])
    # drop_last solo si hay mas de un lote, porque con un lote unico tirarlo dejaria la epoca
    # vacia, y el InfoNCE necesita el lote entero para tener negativos
    loader = DataLoader(TensorDataset(x), batch_size=bs, shuffle=True,
                        num_workers=t.get("num_workers", 0), drop_last=x.shape[0] > bs)
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    adj = adj.to(device)
    model.to(device).train()
    losses: list[float] = []
    for _ in range(epochs):
        tot, nb = 0.0, 0
        for (xb,) in loader:
            opt.zero_grad()
            loss = model.contrastive_loss(xb.to(device), adj)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        losses.append(tot / max(nb, 1))
    return losses


def selftest(farm: str, seed: int) -> None:
    """construye el CRGN de la granja sobre ventanas sinteticas, probando que la construccion
    del grafo, el forward y el InfoNCE encajan entre si."""

    cfg = load_farm_config(farm)
    set_seed(seed)
    n_nodes = _NOMINAL_NODES[farm]
    w = cfg["windowing"]["window_size"]
    topk = cfg["graph"]["topk"]

    windows = np.random.randn(64, w, n_nodes).astype(np.float32)  # (N, W, K)
    adj = build_adjacency(stack_timesteps(windows), k=topk, weighted=cfg["graph"]["weighted"],
                          self_loops=cfg["graph"]["self_loops"])
    model = build_model(cfg, n_nodes, in_dim=w)
    n_params = sum(p.numel() for p in model.parameters())

    # el grafo se construye sobre (N, W, K) y el modelo consume node-major, de ahi la vuelta
    x = torch.from_numpy(to_node_major(windows))  # (N, K, W)
    a = torch.from_numpy(adj).float()
    emb = model.encode(x, a)
    loss = model.contrastive_loss(x, a)
    deg = (adj > 0).sum(axis=1)

    logger.info(f"[selftest farm {farm}] nodes={n_nodes} W={w} topk={topk} "
                f"embed_dim={cfg['model']['embed_dim']} layers={cfg['model']['gcn_layers']} "
                f"params={n_params:,}")
    logger.info(f"  graph: adjacency {adj.shape} | neighbours/node mean={deg.mean():.1f} "
                f"(min={deg.min()} max={deg.max()}) | symmetric={np.allclose(adj, adj.T)}")
    logger.info(f"  forward {tuple(x.shape)} -> embedding {tuple(emb.shape)} | "
                f"emb finite={bool(torch.isfinite(emb).all())} | "
                f"InfoNCE loss={loss.item():.4f} finite={bool(torch.isfinite(loss).all())}")


def run_canary(cfg: dict, device: torch.device, epochs: int, n_windows: int = 256,
               seed: int = 42) -> dict:
    """sobreajusta un punado de ventanas normales de Farm B con el objetivo contrastivo, y
    luego comprueba que el score de desviacion (el CRGNScorer de verdad) queda muy por encima
    en las copias permutadas en el tiempo y en las de ruido blanco. cifras mas figura.
    n_windows>>embed_dim mantiene bien condicionada la covarianza de Mahalanobis, y no hay
    auto-coincidencia porque el score es una distancia a la media."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    windows, _, _ = load_frozen_windows(cfg["data"]["processed_path"], "B")
    rng = np.random.default_rng(seed)
    sel = windows[rng.choice(len(windows), n_windows, replace=False)]
    scaled = WindowScaler(per_turbine=False, method="standard").fit_transform(sel, None).astype(np.float32)
    k_nodes, w = scaled.shape[2], scaled.shape[1]
    adj = build_adjacency(stack_timesteps(scaled), k=cfg["graph"]["topk"],
                          weighted=cfg["graph"]["weighted"], self_loops=cfg["graph"]["self_loops"])
    logger.info(f"Canary: {n_windows}/{len(windows):,} Farm B windows {scaled.shape[1:]} "
                f"(K={k_nodes}) | topk={cfg['graph']['topk']} | device {device}")

    model = build_model(cfg, n_nodes=k_nodes, in_dim=w).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} params | embed_dim={cfg['model']['embed_dim']} "
                f"layers={cfg['model']['gcn_layers']} | metric={cfg['scoring']['metric']}")
    x = torch.from_numpy(to_node_major(scaled))
    a = torch.from_numpy(adj).float()
    t0 = time.time()
    losses = train_crgn(model, x, a, cfg, device, seed=seed, epochs=epochs)
    train_seconds = time.time() - t0

    # dos controles, una permutacion temporal por ventana (mismas marginales, sin dinamica) y
    # ruido blanco
    perm = time_permuted(scaled, np.random.default_rng(123))
    noise = np.random.default_rng(123).normal(size=scaled.shape).astype(np.float32)
    sc = cfg["scoring"]
    scorer = CRGNScorer(model, adj, device, metric=sc["metric"], knn_k=sc["knn_k"],
                        shrinkage=sc["shrinkage"], batch_size=sc["batch_size"])
    scorer.fit_reference(scaled)  # la referencia normal son las ventanas entrenadas
    s_train, s_perm, s_noise = (scorer.score_windows(z) for z in (scaled, perm, noise))
    ratio = float(s_perm.mean() / s_train.mean())
    ratio_noise = float(s_noise.mean() / s_train.mean())
    # umbrales flojos a proposito, al canario solo se le pide ensenar que aprendio algo
    # de dinamica (>1.2 sobre lo permutado) y mucho de estructura (>2 sobre el ruido)
    verdict = "PASS" if ratio > 1.2 and ratio_noise > 2.0 else "CHECK"
    logger.info(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} | score train {s_train.mean():.3f}±"
                f"{s_train.std():.3f} | permuted {s_perm.mean():.3f} (x{ratio:.2f}) | "
                f"white noise {s_noise.mean():.3f} (x{ratio_noise:.2f}) | {train_seconds:.0f}s -> {verdict}")

    fig_path = _canary_figure(cfg, scorer, scaled, perm, noise, losses, s_train, s_perm, s_noise)
    out = {"milestone": "H1_canary", "n_windows": n_windows, "epochs": epochs, "n_params": int(n_params),
           "embed_dim": cfg["model"]["embed_dim"], "gcn_layers": cfg["model"]["gcn_layers"],
           "topk": cfg["graph"]["topk"], "metric": sc["metric"], "loss_first": losses[0],
           "loss_last": losses[-1], "score_train_mean": float(s_train.mean()),
           "score_train_std": float(s_train.std()), "score_permuted_mean": float(s_perm.mean()),
           "score_noise_mean": float(s_noise.mean()), "ratio_permuted_over_train": ratio,
           "ratio_noise_over_train": ratio_noise, "verdict": verdict,
           "train_seconds": round(train_seconds, 1), "device": str(device), "figure": str(fig_path)}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "crgn_h1_canary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"saved {tables / 'crgn_h1_canary.json'} + {fig_path}")
    return out


def _canary_figure(cfg: dict, scorer: CRGNScorer, scaled: np.ndarray, perm: np.ndarray,
                   noise: np.ndarray, losses: list[float], s_train: np.ndarray, s_perm: np.ndarray,
                   s_noise: np.ndarray) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb = np.concatenate([scorer._embed(z) for z in (scaled, perm, noise)])
    emb2 = emb - emb.mean(0)
    _, _, vt = np.linalg.svd(emb2, full_matrices=False)  # PCA 2D de los embeddings
    proj = emb2 @ vt[:2].T
    n = len(scaled)
    groups = [("trained normal", proj[:n]), ("time-permuted", proj[n:2 * n]), ("white noise", proj[2 * n:])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.semilogy(range(1, len(losses) + 1), losses)
    ax.set_title("CRGN canary: contrastive (InfoNCE) loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NT-Xent")
    ax = axes[1]
    for label, p in groups:
        ax.scatter(p[:, 0], p[:, 1], s=12, alpha=0.6, label=label)
    ax.set_title("window embeddings (2D PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax = axes[2]
    ax.boxplot([s_train, s_perm, s_noise], tick_labels=["trained", "time-permuted", "white noise"])
    ax.set_ylabel(f"deviation score ({cfg['scoring']['metric']})")
    ax.set_title(f"separation: permuted x{s_perm.mean() / s_train.mean():.1f}, "
                 f"noise x{s_noise.mean() / s_train.mean():.1f}")
    fig.tight_layout()
    fig_dir = Path(cfg["output"]["figures_dir"]) / "crgn"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "h1_canary.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _add_deriv(windows: np.ndarray) -> np.ndarray:
    """anade canales de primera diferencia (tasa temporal) a la feature de ventana de cada
    nodo, llevando (N, W, K) a (N, 2W, K), los valores crudos apilados con su diff en el
    tiempo (diff[0]=0). el grafo se construye antes sobre los valores crudos, asi que la
    conectividad no cambia y solo se enriquece la feature por nodo, in_dim de W a 2W. las
    tasas SCADA son senal temprana."""

    d = np.diff(windows, axis=1, prepend=windows[:, :1, :])
    return np.concatenate([windows, d], axis=1).astype(np.float32)


def run_full(cfg: dict, device: torch.device, farm: str, features: str, seed: int, epochs: int,
             max_train: int | None = None, milestone: str = "H3", tag: str | None = None,
             subset: str | None = None, derivatives: bool = False, repr_loader=None,
             dump_scores: bool = False) -> dict:
    """ejecucion completa de CRGN por granja con la evaluacion del harness, una fila
    reproducible por (farm, arm, seed). entrena el nucleo contrastivo con ventanas normales,
    ajusta la referencia de desviacion sobre la cola normal de val y puntua los eventos de test
    congelados con el run_model.evaluate intacto (CUSUM, h-sweep, CV y CARE oficial), igual
    que el AE y GDN. seleccion = ROC-ds y se reporta CARE-CV. max_train=None usa el train
    entero, que es la cifra real; un int lo acota para el smoke. subset es la clave de un
    subconjunto pcafirst ordenado por SHAP sobre las features cargadas. reusa en read-only la
    representacion congelada de GDN. un arm = un proceso acota la huella en GPU, porque varios
    arms en el mismo proceso acumulan y revientan el allocator de CUDA."""

    import run_model
    from run_gdn import load_representation
    from src.care_official.adapter import OfficialCAREAdapter

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaled, test_seqs, feats = (repr_loader(seed) if repr_loader is not None
                                else load_representation(farm, cfg, features, seed))
    if subset:
        with open(f"results/tables/farm_{farm.lower()}_pcafirst_feature_subsets.json", encoding="utf-8") as f:
            names = json.load(f)["subsets"][subset]
        # el mismo idx recorta train y test, que es lo que mantiene alineado el eje de nodos
        idx = [feats.index(n) for n in names if n in feats]
        scaled, feats = scaled[:, :, idx], [feats[i] for i in idx]
        test_seqs = [{**s, "windows": s["windows"][:, :, idx]} for s in test_seqs]
    k, w = scaled.shape[2], scaled.shape[1]
    perm = rng.permutation(len(scaled))
    n_val = int(round(len(scaled) * cfg["training"]["val_ratio"]))
    val_norm, train_all = scaled[perm[:n_val]], scaled[perm[n_val:]]
    train_fit = (train_all[rng.choice(len(train_all), max_train, replace=False)]
                 if max_train and len(train_all) > max_train else train_all)
    tag = tag or (f"crgn_farm_{farm.lower()}_{features}{('_' + subset) if subset else ''}"
                  f"{'_deriv' if derivatives else ''}_s{seed}")
    logger.info(f"{tag} [{features}{('/' + subset) if subset else ''}]: {len(feats)} sensors | "
                f"fit {train_fit.shape} val {val_norm.shape} | {len(test_seqs)} events | device {device}")

    g = cfg["graph"]
    # k-1 es el tope real de vecinos por nodo, con un subset corto el topk del config se pasa
    topk = min(g["topk"], k - 1)
    adj = build_adjacency(stack_timesteps(train_fit, max_rows=g.get("max_rows"), seed=seed),
                          k=topk, weighted=g["weighted"], self_loops=g["self_loops"])
    in_dim = w  # node feature length (W, or 2W con derivadas)
    if derivatives:  # el grafo ya esta construido sobre el crudo, asi que queda intacto
        train_fit, val_norm = _add_deriv(train_fit), _add_deriv(val_norm)
        test_seqs = [{**s, "windows": _add_deriv(s["windows"])} for s in test_seqs]
        in_dim = train_fit.shape[1]
    model = build_model(cfg, n_nodes=k, in_dim=in_dim).to(device)
    logger.info(f"CRGN {sum(p.numel() for p in model.parameters()):,} params | epochs={epochs} "
                f"topk={topk} in_dim={in_dim} deriv={derivatives} metric={cfg['scoring']['metric']}")
    x = torch.from_numpy(to_node_major(train_fit))
    a = torch.from_numpy(adj).float()
    t0 = time.time()
    losses = train_crgn(model, x, a, cfg, device, seed=seed, epochs=epochs)
    train_seconds = time.time() - t0

    sc = cfg["scoring"]
    scorer = CRGNScorer(model, adj, device, metric=sc["metric"], knn_k=sc["knn_k"],
                        shrinkage=sc["shrinkage"], batch_size=sc["batch_size"],
                        percentile=cfg["threshold"]["percentile"])
    scorer.fit_reference(val_norm)
    adapter = OfficialCAREAdapter(raw_path=cfg["data"]["raw_path"], farm=farm, window_size=w)
    ev = run_model.evaluate(scorer, val_norm, test_seqs, adapter, seed)
    best, cv = ev["best"], ev["cv"]
    logger.info(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} | train {train_seconds:.0f}s")
    logger.info(f"{tag}: CARE-CV={cv['care']:.4f} (oracle={best['care']:.4f}, optim={cv['optimism']:+.4f}) "
                f"| ROC-ds={best['roc_ds']:.4f} ROC-win={ev['roc_window']:.4f} | Cov={best['cov']:.3f} "
                f"Earl={best['earl']:.3f} Rel={best['rel']:.3f} Acc={best['acc']:.3f}")

    out = {"milestone": milestone, "tag": tag, "farm": farm, "features": features, "subset": subset or "",
           "derivatives": derivatives, "seed": seed, "epochs": epochs, "n_nodes": k, "topk": topk,
           "metric": sc["metric"],
           "if_mask": bool(cfg.get("isolation_forest_mask", False)), "n_fit": len(train_fit),
           "n_val": len(val_norm), "loss_first": losses[0], "loss_last": losses[-1],
           "train_seconds": round(train_seconds, 1), "care_cv": cv["care"], "care_oracle": best["care"],
           "optimism": cv["optimism"], "roc_ds": best["roc_ds"], "roc_window": ev["roc_window"],
           "coverage": best["cov"], "earliness": best["earl"], "reliability": best["rel"],
           "accuracy": best["acc"], "fold_h_scale": cv["fold_h_scale"], "sweep": ev["sweep"]}
    models = Path(cfg["output"]["models_dir"])
    models.mkdir(parents=True, exist_ok=True)
    # crgn_farm_c_top100_s42.pt es un artefacto congelado que ningun nivel del gate vigila, asi
    # que sobrescribirlo exige --overwrite-frozen explicito
    ckpt_path = models / f"{tag}.pt"
    if ckpt_path.exists() and not cfg["output"].get("overwrite_frozen", False):
        logger.warning(f"{ckpt_path} ya existe: se CONSERVA. Es un artefacto congelado; "
                       f"pasa --overwrite-frozen si de verdad quieres sobrescribirlo.")
        ckpt_note = f"checkpoint CONSERVADO (no reescrito): {ckpt_path}"
    else:
        torch.save({"state_dict": model.state_dict(), "adj": adj, "feats": feats, "n_nodes": k,
                    "in_dim": in_dim, "derivatives": derivatives, "farm": farm, "features": features,
                    "seed": seed}, ckpt_path)
        ckpt_note = f"checkpoint {ckpt_path}"
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"saved {tables / (tag + '.json')} + {ckpt_note}")
    if dump_scores:  # scores de desviacion por ventana para la fusion AE+CRGN
        fl = farm.lower()
        # la 42 va sin sufijo porque es la desplegada y sus npz ya estan citados asi
        sfx = "" if seed == 42 else f"_s{seed}"
        arrays = {"train_dev": np.asarray(scorer.score_windows(val_norm), np.float64)}
        meta = {}
        for s in test_seqs:
            eid = s["event_id"]
            arrays[f"ev{eid}_dev"] = np.asarray(scorer.score_windows(s["windows"]), np.float64)
            meta[str(eid)] = {"event_label": s["event_label"], "asset_id": int(s["asset_id"]),
                              "labels": np.asarray(s["labels"], int).tolist(),
                              "window_timestamps": [str(t) for t in np.asarray(s["window_timestamps"]).ravel()]}
        out_npz = models / f"crgn_scores_farm_{fl}{sfx}.npz"
        np.savez_compressed(out_npz, **arrays)
        with open(models / f"crgn_scores_farm_{fl}{sfx}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        logger.info(f"[dump-scores] CRGN: {len(test_seqs)} events -> {out_npz}")
    return out


def run_smoke_b(cfg: dict, device: torch.device, seed: int, epochs: int,
                max_train: int = 120000) -> dict:
    """Farm B curated end-to-end sobre una submuestra de train, validacion del contrato y no
    una cifra final. envoltorio fino sobre run_full con el tope de 120k y su propia tag."""

    return run_full(cfg, device, "B", "curated", seed, epochs, max_train=max_train,
                    milestone="H2_smoke", tag="crgn_h2_smoke_b")


def run_multiseed(cfg: dict, device: torch.device, farm: str, features: str, seeds: list[int],
                  epochs: int, max_train: int | None = None, subset: str | None = None,
                  milestone: str | None = None, derivatives: bool = False) -> dict:
    """multi-seed completo de un arm (features[/subset]). corre run_full por semilla, un solo
    proceso para que la GPU siga acotada, y reporta el CARE-CV y el ROC-ds medios, con ROC-ds
    como metrica de seleccion. el CARE varia entre semillas, asi que el multi-seed es
    obligatorio."""

    milestone = milestone or ("H3" if farm == "B" else "H4")
    rows = [run_full(cfg, device, farm, features, s, epochs, max_train=max_train, milestone=milestone,
                     subset=subset, derivatives=derivatives) for s in seeds]
    care = np.array([r["care_cv"] for r in rows])
    roc = np.array([r["roc_ds"] for r in rows])
    arm = f"{features}{('_' + subset) if subset else ''}{'_deriv' if derivatives else ''}"
    summary = {"milestone": milestone, "farm": farm, "features": features, "subset": subset or "",
               "derivatives": derivatives,
               "seeds": seeds, "epochs": epochs, "metric": cfg["scoring"]["metric"],
               "n_nodes": rows[0]["n_nodes"], "topk": rows[0]["topk"],
               "care_cv_mean": float(care.mean()), "care_cv_std": float(care.std()),
               "roc_ds_mean": float(roc.mean()), "roc_ds_std": float(roc.std()),
               "per_seed": [{"seed": r["seed"], "care_cv": r["care_cv"], "roc_ds": r["roc_ds"],
                             "coverage": r["coverage"], "earliness": r["earliness"],
                             "reliability": r["reliability"], "accuracy": r["accuracy"],
                             "optimism": r["optimism"]} for r in rows]}
    logger.info(f"=== CRGN {farm} [{arm}] multi-seed {seeds}: CARE-CV {care.mean():.4f}±{care.std():.4f} | "
                f"ROC-ds {roc.mean():.4f}±{roc.std():.4f} ===")
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    # el arm por defecto escribe en el nombre pelado; cualquier otro lleva su etiqueta, para
    # que un barrido no pise el fichero del baseline
    is_default = features == cfg["features"] and not subset and not derivatives
    out_path = tables / (f"crgn_{farm.lower()}_multiseed.json" if is_default
                         else f"crgn_{farm.lower()}_{arm}_multiseed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"saved {out_path}")
    return summary


def _fit_ref(val_emb: np.ndarray, metric: str, shrinkage: float, knn_k: int):
    """ajusta una referencia de embeddings normales para una metrica de desviacion, post-hoc y
    sobre los embeddings de val cacheados. la cuenta de Mahalanobis es identica byte a byte a
    la de CRGNScorer, asi que el punto baseline reproduce exacta la ejecucion publicada."""

    if metric == "mahalanobis":
        mean = val_emb.mean(axis=0)
        # el shrinkage sobre la diagonal es lo que deja invertible una covarianza casi singular
        cov = np.cov(val_emb, rowvar=False) + shrinkage * np.eye(val_emb.shape[1])
        return ("mahalanobis", mean, np.linalg.inv(cov))
    if metric == "knn":
        from sklearn.neighbors import NearestNeighbors
        return ("knn", NearestNeighbors(n_neighbors=knn_k).fit(val_emb))
    if metric == "cosine":
        mean = val_emb.mean(axis=0)
        return ("cosine", mean / (np.linalg.norm(mean) + 1e-12))
    if metric == "lof":
        from sklearn.neighbors import LocalOutlierFactor
        return ("lof", LocalOutlierFactor(n_neighbors=knn_k, novelty=True).fit(val_emb))
    if metric == "gmm":
        from sklearn.mixture import GaussianMixture
        # como maximo 8 componentes y al menos 50 ventanas por componente, para no sobreajustar
        n = min(8, max(1, len(val_emb) // 50))
        return ("gmm", GaussianMixture(n_components=n, covariance_type="diag",
                                       reg_covar=max(shrinkage, 1e-6), random_state=0).fit(val_emb))
    raise ValueError(f"unknown deviation metric {metric}")


def _score_emb(emb: np.ndarray, ref) -> np.ndarray:
    """desviacion por ventana de un lote de embeddings contra una referencia ya ajustada,
    donde mas alto es mas anomalo."""

    kind = ref[0]
    if kind == "mahalanobis":
        d = emb - ref[1]
        return np.einsum("ni,ij,nj->n", d, ref[2], d).astype(np.float64)
    if kind == "knn":
        return ref[1].kneighbors(emb)[0].mean(axis=1).astype(np.float64)
    if kind == "cosine":
        e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return (1.0 - e @ ref[1]).astype(np.float64)
    if kind == "lof":
        return (-ref[1].score_samples(emb)).astype(np.float64)  # higher = more outlier
    if kind == "gmm":
        return (-ref[1].score_samples(emb)).astype(np.float64)  # negative log-likelihood
    raise ValueError(f"unknown reference kind {kind}")


def run_sweep_scoretime(cfg: dict, device: torch.device, farm: str, features: str, seeds: list[int],
                        metrics: list[str] | None = None, shrinkages: list[float] | None = None,
                        knn_ks: list[int] | None = None, smas: list[int] | None = None) -> dict:
    """barrido post-hoc en tiempo de score sobre los checkpoints congelados, sin reentrenar.
    por semilla recarga el modelo entrenado, embebe val y eventos una sola vez, y luego barre
    metrica de desviacion x shrinkage x knn_k x SMA, que sale barato. seleccion = ROC-ds medio
    y se reporta CARE-CV. que el punto baseline (mahalanobis, shrinkage de config, sma=1)
    reproduzca exacta la ejecucion publicada es el control de la maquinaria."""

    import run_model
    from collections import defaultdict

    from run_gdn import _ReplayScorer, _smooth, load_representation
    from src.care_official.adapter import OfficialCAREAdapter

    # knn y lof quedan fuera de la lista por defecto, piden una referencia submuestreada
    metrics = metrics or ["mahalanobis", "cosine", "gmm"]
    shrinkages = shrinkages or [1e-4, 1e-3, 1e-2, 1e-1]
    knn_ks = knn_ks or [5, 10, 20, 50]
    smas = smas or [1, 3, 6, 12]
    models = Path(cfg["output"]["models_dir"])
    pct = cfg["threshold"]["percentile"]
    base_shrink = cfg["scoring"]["shrinkage"]
    rows: list[dict] = []

    for seed in seeds:
        scaled, test_seqs, feats = load_representation(farm, cfg, features, seed)
        w = scaled.shape[1]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(scaled))
        n_val = int(round(len(scaled) * cfg["training"]["val_ratio"]))
        val = scaled[perm[:n_val]]
        tag = f"crgn_farm_{farm.lower()}_{features}_s{seed}"
        # weights_only=False porque el ckpt es nuestro y guarda tambien adj, feats y n_nodes
        ckpt = torch.load(models / f"{tag}.pt", map_location=device, weights_only=False)
        model = build_model(cfg, n_nodes=ckpt["n_nodes"], in_dim=ckpt["in_dim"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        embedder = CRGNScorer(model, ckpt["adj"], device, batch_size=cfg["scoring"]["batch_size"])
        val_emb = embedder._embed(val)
        ev_embs = [embedder._embed(s["windows"]) for s in test_seqs]
        adapter = OfficialCAREAdapter(raw_path=cfg["data"]["raw_path"], farm=farm, window_size=w)
        logger.info(f"{tag}: embedded val {val_emb.shape} + {len(ev_embs)} events | sweeping score-time")

        def sweep_one(metric: str, shrink: float, kk: int) -> None:
            # el orden importa, _ReplayScorer los sirve tal cual, validacion primero y luego un
            # evento por posicion, que es como llama evaluate
            ref = _fit_ref(val_emb, metric, shrink, kk)
            base = [_score_emb(val_emb, ref)] + [_score_emb(e, ref) for e in ev_embs]
            for sma in smas:
                scores = [_smooth(s, sma) for s in base]
                ev = run_model.evaluate(_ReplayScorer(scores, pct), val, test_seqs, adapter, seed)
                b = ev["best"]
                rows.append({"seed": seed, "metric": metric, "shrinkage": shrink, "knn_k": kk,
                             "sma": sma, "roc_ds": b["roc_ds"], "care_cv": ev["cv"]["care"],
                             "cov": b["cov"], "earl": b["earl"], "rel": b["rel"], "acc": b["acc"]})

        # cada metrica barre solo el parametro que la gobierna, shrinkage la Mahalanobis y
        # k los vecinos; la gmm se queda con el shrinkage del config
        for metric in metrics:
            if metric == "mahalanobis":
                for sh in shrinkages:
                    sweep_one(metric, sh, 0)
            elif metric in ("knn", "lof"):
                for kk in knn_ks:
                    sweep_one(metric, 0.0, kk)
            elif metric == "gmm":
                sweep_one(metric, base_shrink, 0)
            else:
                sweep_one(metric, 0.0, 0)

    agg: dict = defaultdict(list)
    for r in rows:
        agg[(r["metric"], r["shrinkage"], r["knn_k"], r["sma"])].append(r)
    grid = []
    for (metric, sh, kk, sma), rs in agg.items():
        grid.append({"metric": metric, "shrinkage": sh, "knn_k": kk, "sma": sma,
                     "roc_ds_mean": float(np.mean([r["roc_ds"] for r in rs])),
                     "roc_ds_std": float(np.std([r["roc_ds"] for r in rs])),
                     "care_cv_mean": float(np.mean([r["care_cv"] for r in rs])),
                     "care_cv_std": float(np.std([r["care_cv"] for r in rs])),
                     "cov": float(np.mean([r["cov"] for r in rs])), "earl": float(np.mean([r["earl"] for r in rs])),
                     "rel": float(np.mean([r["rel"] for r in rs])), "acc": float(np.mean([r["acc"] for r in rs]))})
    grid.sort(key=lambda r: (r["roc_ds_mean"], r["care_cv_mean"]), reverse=True)
    best = grid[0]
    # el punto de control, la celda que coincide con la configuracion publicada
    repro = next((r for r in grid if r["metric"] == "mahalanobis"
                  and abs(r["shrinkage"] - base_shrink) < 1e-12 and r["sma"] == 1), None)
    logger.info(f"=== CRGN {farm} score-time sweep: {len(grid)} configs x {len(seeds)} seeds ===")
    if repro:
        logger.info(f"baseline repro (maha sh={base_shrink} sma=1): CARE-CV {repro['care_cv_mean']:.4f} | "
                    f"ROC-ds {repro['roc_ds_mean']:.4f} (debe = H3/H4)")
    logger.info(f"BEST by ROC-ds: {best['metric']} sh={best['shrinkage']} k={best['knn_k']} sma={best['sma']} "
                f"-> ROC-ds {best['roc_ds_mean']:.4f}±{best['roc_ds_std']:.4f} | "
                f"CARE-CV {best['care_cv_mean']:.4f}±{best['care_cv_std']:.4f} "
                f"(Cov {best['cov']:.3f} Earl {best['earl']:.3f} Rel {best['rel']:.3f} Acc {best['acc']:.3f})")
    out = {"milestone": "H5_ola1_scoretime", "farm": farm, "features": features, "seeds": seeds,
           "baseline_repro": repro, "best_by_roc_ds": best, "grid": grid, "rows": rows}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    out_path = tables / f"crgn_sweep_scoretime_{farm.lower()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"saved {out_path}")
    return out


def _ola2_arms(farm: str) -> list[tuple[str, dict]]:
    """rejilla de un eje cada vez, con el override fusionado en profundidad sobre la cfg de la
    granja y reentrenando cada arm. el baseline va primero, porque tiene que reproducir la
    ejecucion publicada. las palancas son epochs, embed_dim, gcn_layers, la temperatura
    contrastiva, el ruido y la mascara de augmentation, y el topk del grafo."""

    topks = {"B": [8, 16], "C": [12, 25]}[farm]
    arms: list[tuple[str, dict]] = [("baseline", {})]
    arms += [(f"ep{e}", {"training": {"epochs": e}}) for e in (50, 100)]
    arms += [(f"dim{d}", {"model": {"embed_dim": d}}) for d in (32, 128, 256)]
    arms += [(f"layers{n}", {"model": {"gcn_layers": n}}) for n in (1, 3)]
    arms += [(f"tau{t}", {"contrastive": {"temperature": t}}) for t in (0.1, 0.2, 1.0)]
    arms += [(f"noise{n}", {"augmentation": {"noise_std": n}}) for n in (0.05, 0.2)]
    arms += [(f"mask{m}", {"augmentation": {"mask_ratio": m}}) for m in (0.1, 0.3)]
    arms += [(f"topk{k}", {"graph": {"topk": k}}) for k in topks]
    return arms


def _ola2b_arms(farm: str) -> list[tuple[str, dict]]:
    """los ejes que no estan en _ola2_arms mas los combos CARE-positivos, todos como overrides
    por arm dentro de run_sweep_train, reusando las ventanas congeladas con una sola carga por
    semilla. quedan fuera pearson y el readout mean+max, que piden tocar codigo de grafo o de
    modelo, y stride_train, que obligaria a recargar las ventanas cacheadas. el baseline va
    primero, porque tiene que reproducir exacta la ejecucion publicada."""

    arms: list[tuple[str, dict]] = [("baseline", {})]
    arms += [("readout_max", {"model": {"readout": "max"}})]
    arms += [(f"proj{d}", {"model": {"proj_dim": d}}) for d in (32, 128)]
    arms += [("unweighted", {"graph": {"weighted": False}})]
    arms += [("noselfloop", {"graph": {"self_loops": False}})]
    arms += [("maxrows500k", {"graph": {"max_rows": 500000}})]
    arms += [(f"lr{lr:g}", {"training": {"learning_rate": lr}}) for lr in (3.0e-4, 3.0e-3)]
    arms += [("wd0", {"training": {"weight_decay": 0.0}})]
    if farm == "B":  # B's winners (noise0.2/layers3/ep50) combined, plus a harder noise
        arms += [("combo", {"augmentation": {"noise_std": 0.2}, "model": {"gcn_layers": 3},
                            "training": {"epochs": 50}})]
        arms += [("noise0.3", {"augmentation": {"noise_std": 0.3}})]
    else:  # C's CARE-positive knobs (topk25/noise0.2/mask0.3) combined
        arms += [("combo", {"graph": {"topk": 25}, "augmentation": {"noise_std": 0.2, "mask_ratio": 0.3}})]
    return arms


def _ola4_arms(farm: str) -> list[tuple[str, dict]]:
    """confirmacion multi-seed (>=5) de los candidatos CARE-positivos. solo el baseline, que
    debe reproducir la ejecucion publicada, y el candidato por granja (C proj32 / B noise0.3),
    para fijar el numero CRGN final por parque sin re-barrer la rejilla entera. reusa
    run_sweep_train con la misma maquinaria, 1 carga por semilla, override por arm, seleccion
    ROC-ds y CARE-CV titular."""

    arms: list[tuple[str, dict]] = [("baseline", {})]
    arms += [("proj32", {"model": {"proj_dim": 32}})] if farm == "C" else \
            [("noise0.3", {"augmentation": {"noise_std": 0.3}})]
    return arms


def run_sweep_train(cfg: dict, device: torch.device, farm: str, features: str, seeds: list[int],
                    arms: list[tuple[str, dict]] | None = None, out_tag: str = "") -> dict:
    """barrido con reentreno. carga la representacion una vez por semilla, y luego por arm
    fusiona en profundidad el override, reconstruye grafo y modelo, reentrena, puntua con la
    maha baseline y evalua con el harness intacto. seleccion = ROC-ds medio y se reporta
    CARE-CV. el arm 'baseline' tiene que reproducir la ejecucion publicada, con el RNG
    reiniciado por arm, el grafo fijado por sus parametros y la semilla, la inicializacion del
    modelo por la semilla de torch y el entrenamiento resembrandose solo."""

    import run_model
    from collections import defaultdict

    from run_gdn import load_representation
    from src.care_official.adapter import OfficialCAREAdapter

    arms = arms or _ola2_arms(farm)
    pct = cfg["threshold"]["percentile"]
    keys = ("epochs", "embed_dim", "gcn_layers", "temperature", "noise", "mask", "topk")
    rows: list[dict] = []

    for seed in seeds:
        scaled, test_seqs, feats = load_representation(farm, cfg, features, seed)
        w, k = scaled.shape[1], scaled.shape[2]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(scaled))
        n_val = int(round(len(scaled) * cfg["training"]["val_ratio"]))
        val_norm, train_fit = scaled[perm[:n_val]], scaled[perm[n_val:]]
        adapter = OfficialCAREAdapter(raw_path=cfg["data"]["raw_path"], farm=farm, window_size=w)
        x_nm = to_node_major(train_fit)  # (N,K,W) reused across arms
        logger.info(f"Sweep-train {farm} seed {seed}: {len(arms)} arms | fit {train_fit.shape} "
                    f"val {val_norm.shape} | {len(test_seqs)} events")
        for label, ov in arms:
            ci = _deep_merge(cfg, ov)
            # el RNG vuelve al mismo punto antes de cada arm, para que la unica diferencia
            # entre dos arms sea su override y no el estado que dejo el anterior
            np.random.seed(seed)
            torch.manual_seed(seed)
            g = ci["graph"]
            adj = build_adjacency(stack_timesteps(train_fit, max_rows=g.get("max_rows"), seed=seed),
                                  k=g["topk"], weighted=g["weighted"], self_loops=g["self_loops"])
            model = build_model(ci, n_nodes=k, in_dim=w).to(device)
            a = torch.from_numpy(adj).float()
            t0 = time.time()
            train_crgn(model, torch.from_numpy(x_nm), a, ci, device, seed=seed,
                       epochs=ci["training"]["epochs"])
            sc = ci["scoring"]
            scorer = CRGNScorer(model, adj, device, metric=sc["metric"], knn_k=sc["knn_k"],
                                shrinkage=sc["shrinkage"], batch_size=sc["batch_size"], percentile=pct)
            scorer.fit_reference(val_norm)
            ev = run_model.evaluate(scorer, val_norm, test_seqs, adapter, seed)
            b = ev["best"]
            rows.append({"seed": seed, "arm": label, "epochs": ci["training"]["epochs"],
                         "embed_dim": ci["model"]["embed_dim"], "gcn_layers": ci["model"]["gcn_layers"],
                         "temperature": ci["contrastive"]["temperature"], "noise": ci["augmentation"]["noise_std"],
                         "mask": ci["augmentation"]["mask_ratio"], "topk": g["topk"], "roc_ds": b["roc_ds"],
                         "care_cv": ev["cv"]["care"], "cov": b["cov"], "earl": b["earl"], "rel": b["rel"],
                         "acc": b["acc"], "secs": round(time.time() - t0, 1)})
            logger.info(f"  [{label}] {time.time()-t0:.0f}s | CARE-CV={ev['cv']['care']:.4f} "
                        f"ROC-ds={b['roc_ds']:.4f} Earl={b['earl']:.3f} Acc={b['acc']:.3f}")
            # sin soltar el modelo de cada arm la memoria de la GPU se acumula arm tras arm
            del model, scorer, a
            if device.type == "cuda":
                torch.cuda.empty_cache()

    agg: dict = defaultdict(list)
    for r in rows:
        agg[r["arm"]].append(r)
    grid = []
    for label, rs in agg.items():
        grid.append({"arm": label, **{kk: rs[0][kk] for kk in keys},
                     "roc_ds_mean": float(np.mean([r["roc_ds"] for r in rs])),
                     "roc_ds_std": float(np.std([r["roc_ds"] for r in rs])),
                     "care_cv_mean": float(np.mean([r["care_cv"] for r in rs])),
                     "care_cv_std": float(np.std([r["care_cv"] for r in rs])),
                     "cov": float(np.mean([r["cov"] for r in rs])), "earl": float(np.mean([r["earl"] for r in rs])),
                     "rel": float(np.mean([r["rel"] for r in rs])), "acc": float(np.mean([r["acc"] for r in rs]))})
    grid.sort(key=lambda r: (r["roc_ds_mean"], r["care_cv_mean"]), reverse=True)
    best, base = grid[0], next((r for r in grid if r["arm"] == "baseline"), None)
    logger.info(f"=== CRGN {farm} sweep-train: {len(grid)} arms x {len(seeds)} seeds ===")
    if base:
        logger.info(f"baseline arm: CARE-CV {base['care_cv_mean']:.4f} ROC-ds {base['roc_ds_mean']:.4f} (~H3/H4)")
    logger.info(f"BEST by ROC-ds: [{best['arm']}] ROC-ds {best['roc_ds_mean']:.4f}±{best['roc_ds_std']:.4f} | "
                f"CARE-CV {best['care_cv_mean']:.4f}±{best['care_cv_std']:.4f} (Cov {best['cov']:.3f} "
                f"Earl {best['earl']:.3f} Rel {best['rel']:.3f} Acc {best['acc']:.3f})")
    out = {"milestone": f"H5_ola2_train{('_' + out_tag) if out_tag else ''}", "farm": farm,
           "features": features, "seeds": seeds,
           "baseline_arm": base, "best_by_roc_ds": best, "grid": grid, "rows": rows}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    out_path = tables / f"crgn_sweep_train_{farm.lower()}{('_' + out_tag) if out_tag else ''}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"saved {out_path}")
    return out


def _tierc_arms(farm: str) -> list[tuple[str, str, str | None]]:
    """arms de representacion como (label, base_arm, subset_key|None). las claves de subset
    indexan los subconjuntos pcafirst ordenados por SHAP
    (farm_{b,c}_pcafirst_feature_subsets.json) sobre el conjunto allavg."""

    if farm == "B":
        return [("curated", "curated", None), ("soft", "soft", None), ("allavg", "allavg", None),
                ("shap_top30", "allavg", "S_top30"), ("shap_top20", "allavg", "S_top20"),
                ("shap_decluster", "allavg", "S_decluster"), ("shap_keep", "allavg", "S_keep")]
    return [("top100", "top100", None), ("allavg", "allavg", None),
            ("shap_top150", "allavg", "S_top150"), ("shap_top100", "allavg", "S_top100"),
            ("shap_top50", "allavg", "S_top50"), ("shap_top30", "allavg", "S_top30")]


def run_sweep_repr(cfg: dict, device: torch.device, farm: str, seeds: list[int],
                   arms: list[tuple[str, str, str | None]] | None = None) -> dict:
    """barrido de representacion y de features. por (base_arm, seed) carga la repr una vez, y
    luego por arm, que puede ser un subconjunto de esa base ordenado por SHAP, recorta las
    features, reentrena y evalua con el harness intacto. seleccion = ROC-ds medio y se reporta
    CARE-CV. prueba por primera vez sobre el modelo de grafos la palanca de subconjuntos SHAP
    que gano en el AE."""

    import run_model
    from collections import defaultdict

    from run_gdn import load_representation
    from src.care_official.adapter import OfficialCAREAdapter

    arms = arms or _tierc_arms(farm)
    with open(f"results/tables/farm_{farm.lower()}_pcafirst_feature_subsets.json", encoding="utf-8") as f:
        subdef = json.load(f)["subsets"]
    pct = cfg["threshold"]["percentile"]
    g, sc = cfg["graph"], cfg["scoring"]
    rows: list[dict] = []

    # se agrupa por arm base para cargar la representacion una sola vez por (base, semilla)
    for base_arm in sorted({a[1] for a in arms}):
        for seed in seeds:
            scaled0, test0, feats0 = load_representation(farm, cfg, base_arm, seed)
            w = scaled0.shape[1]
            rng = np.random.default_rng(seed)
            # la permutacion se calcula sobre la base y se reusa en todos sus subsets, porque
            # recortar columnas no cambia el numero de ventanas ni su orden
            perm = rng.permutation(len(scaled0))
            n_val = int(round(len(scaled0) * cfg["training"]["val_ratio"]))
            adapter = OfficialCAREAdapter(raw_path=cfg["data"]["raw_path"], farm=farm, window_size=w)
            for label, _, subkey in [a for a in arms if a[1] == base_arm]:
                if subkey:
                    idx = [feats0.index(n) for n in subdef[subkey] if n in feats0]
                    scaled, feats = scaled0[:, :, idx], [feats0[i] for i in idx]
                    test_seqs = [{**s, "windows": s["windows"][:, :, idx]} for s in test0]
                else:
                    scaled, feats, test_seqs = scaled0, feats0, test0
                k = scaled.shape[2]
                topk = min(g["topk"], k - 1)
                val_norm, train_fit = scaled[perm[:n_val]], scaled[perm[n_val:]]
                np.random.seed(seed)
                torch.manual_seed(seed)
                adj = build_adjacency(stack_timesteps(train_fit, max_rows=g.get("max_rows"), seed=seed),
                                      k=topk, weighted=g["weighted"], self_loops=g["self_loops"])
                model = build_model(cfg, n_nodes=k, in_dim=w).to(device)
                a = torch.from_numpy(adj).float()
                t0 = time.time()
                train_crgn(model, torch.from_numpy(to_node_major(train_fit)), a, cfg, device,
                           seed=seed, epochs=cfg["training"]["epochs"])
                scorer = CRGNScorer(model, adj, device, metric=sc["metric"], knn_k=sc["knn_k"],
                                    shrinkage=sc["shrinkage"], batch_size=sc["batch_size"], percentile=pct)
                scorer.fit_reference(val_norm)
                ev = run_model.evaluate(scorer, val_norm, test_seqs, adapter, seed)
                b = ev["best"]
                rows.append({"seed": seed, "arm": label, "base": base_arm, "subset": subkey or "",
                             "n_nodes": k, "topk": topk, "roc_ds": b["roc_ds"], "care_cv": ev["cv"]["care"],
                             "cov": b["cov"], "earl": b["earl"], "rel": b["rel"], "acc": b["acc"],
                             "secs": round(time.time() - t0, 1)})
                logger.info(f"  [{label}] base={base_arm} K={k} {time.time()-t0:.0f}s | "
                            f"CARE-CV={ev['cv']['care']:.4f} ROC-ds={b['roc_ds']:.4f} Earl={b['earl']:.3f}")
                # sin soltar el modelo de cada arm la memoria de la GPU se acumula
                del model, scorer, a
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    agg: dict = defaultdict(list)
    for r in rows:
        agg[r["arm"]].append(r)
    grid = []
    for label, rs in agg.items():
        grid.append({"arm": label, "base": rs[0]["base"], "subset": rs[0]["subset"], "n_nodes": rs[0]["n_nodes"],
                     "roc_ds_mean": float(np.mean([r["roc_ds"] for r in rs])),
                     "roc_ds_std": float(np.std([r["roc_ds"] for r in rs])),
                     "care_cv_mean": float(np.mean([r["care_cv"] for r in rs])),
                     "care_cv_std": float(np.std([r["care_cv"] for r in rs])),
                     "cov": float(np.mean([r["cov"] for r in rs])), "earl": float(np.mean([r["earl"] for r in rs])),
                     "rel": float(np.mean([r["rel"] for r in rs])), "acc": float(np.mean([r["acc"] for r in rs]))})
    grid.sort(key=lambda r: (r["roc_ds_mean"], r["care_cv_mean"]), reverse=True)
    best = grid[0]
    logger.info(f"=== CRGN {farm} repr sweep: {len(grid)} arms x {len(seeds)} seeds ===")
    logger.info(f"BEST by ROC-ds: [{best['arm']}] K={best['n_nodes']} ROC-ds {best['roc_ds_mean']:.4f}"
                f"±{best['roc_ds_std']:.4f} | CARE-CV {best['care_cv_mean']:.4f}±{best['care_cv_std']:.4f} "
                f"(Cov {best['cov']:.3f} Earl {best['earl']:.3f} Rel {best['rel']:.3f} Acc {best['acc']:.3f})")
    out = {"milestone": "H5_ola3_repr", "farm": farm, "seeds": seeds, "best_by_roc_ds": best,
           "grid": grid, "rows": rows}
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    out_path = tables / f"crgn_sweep_repr_{farm.lower()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"saved {out_path}")
    return out


def _load_c_preproc(features: str, stride: int, *, outlier: bool, temp_norm: bool, smooth: bool,
                    trans_before: int, trans_after: int, window_size: int):
    """representacion configurable de Farm C para el barrido de preproc y W. replica del camino
    pca-first de run_gdn._load_c, que se usa en read-only, filas y luego los filtros de status,
    potencia y transicion, dedup, StandardScaler y ventaneo, pero con las banderas de preproc,
    las horas de transicion y el tamano de ventana como parametros, y con una clave de cache
    que conoce el preproc, porque la de run_gdn solo guarda features+stride y aqui se quedaria
    obsoleta. queda fuera del ancla cuando window_size != 27. no se edita run_gdn y
    src.preprocessing se reusa en read-only."""

    import pickle

    import pandas as pd
    import run_model
    from sklearn.preprocessing import StandardScaler
    from src.preprocessing.config import load_config as load_preproc_config
    from src.preprocessing.power_curve_filter import PowerCurveFilter
    from src.preprocessing.status_filter import StatusFilter
    from src.preprocessing.transition_excluder import TransitionExcluder
    from src.preprocessing.windower import Windower

    pcfg = _deep_merge(load_preproc_config(), load_preproc_config().get("per_farm", {}).get("C", {}))
    pcfg["transition_exclusion"] = {**pcfg.get("transition_exclusion", {}),
                                    "hours_before_fault": trans_before, "hours_after_fault": trans_after}
    pcfg["windowing"] = {**pcfg["windowing"], "window_size_samples": window_size}
    # Smoother mira config.enabled (false por defecto), no el sitio de llamada, asi que la
    # bandera hay que escribirla en el config para que se honre
    pcfg["smoothing"] = {**pcfg.get("smoothing", {}), "enabled": smooth}
    # la clave de cache lleva los flags de preproc, la transicion y la ventana, porque la de
    # run_gdn solo lleva features y stride y aqui se quedaria rancia
    key = f"o{int(outlier)}t{int(temp_norm)}s{int(smooth)}_tb{trans_before}ta{trans_after}_w{window_size}"
    cache = Path("data/processed/experiments") / f"crgn_c_{features}_{key}_st{stride}.npz"
    cache_test = cache.with_suffix(".test.pkl")
    if cache.exists() and cache_test.exists():
        z = np.load(cache, allow_pickle=True)
        tw, feats = z["train_windows"], [str(c) for c in z["feats"]]
        with open(cache_test, "rb") as f:
            test_seqs = pickle.load(f)
        logger.info(f"Farm C repr [{features}/{key}] from cache: train {tw.shape} | {len(test_seqs)} events")
        return tw, test_seqs, feats

    df, event_info, feats, reg = run_model.preprocess_rows("C", pcfg, outlier, temp_norm, smooth)
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
    col_means = df_train[feats].mean()

    def arr(src):
        return src[feats].fillna(col_means).fillna(0.0).to_numpy(np.float32)

    scaler = StandardScaler().fit(arr(df_train))

    def to_wdf(src, sc):
        meta = src[[c for c in run_model.META if c in src.columns]].reset_index(drop=True)
        return pd.concat([meta, pd.DataFrame(sc, columns=feats)], axis=1)

    wtrain = Windower({"windowing": {**pcfg["windowing"], "stride_samples": stride}})
    wtest = Windower({"windowing": {**pcfg["windowing"], "stride_samples": 1}})
    train_windows, _, _ = wtrain.create_train_windows(to_wdf(df_train, scaler.transform(arr(df_train))), feats)
    test_obj, _ = wtest.create_test_sequences(to_wdf(df_pred, scaler.transform(arr(df_pred))), event_info, feats)
    test_seqs = [{"event_id": s.event_id, "event_label": s.event_label, "asset_id": s.asset_id,
                  "windows": s.windows.astype(np.float32), "labels": np.asarray(s.labels, np.int8),
                  "window_timestamps": s.window_timestamps} for s in test_obj]
    train_windows = train_windows.astype(np.float32)
    logger.info(f"Farm C repr [{features}/{key}]: {len(feats)} sensors | train {train_windows.shape} "
                f"| {len(test_seqs)} events")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, train_windows=train_windows, feats=np.array(feats, dtype=object))
    with open(cache_test, "wb") as f:
        pickle.dump(test_seqs, f)
    logger.info(f"cached C repr -> {cache}")
    return train_windows, test_seqs, feats


def _load_b_preproc(cfg: dict, features: str, stride: int, *, outlier: bool, temp_norm: bool, smooth: bool,
                    trans_before: int, trans_after: int, window_size: int):
    """representacion configurable de Farm B para el barrido de preproc y W. replica del camino
    window-first de run_gdn._load_b, que se usa en read-only, filas y luego limpieza, despues
    los filtros de status, potencia y transicion, dedup, ventaneo de los sensores crudos y por
    ultimo la mascara de retencion del IF mas WindowScaler sobre las ventanas, por paridad con
    el AE y GDN, pero con las banderas de preproc, las horas de transicion y el tamano de
    ventana como parametros, y con una clave de cache que conoce el preproc. conserva el
    conjunto de features curated y la mascara del IF. queda fuera del ancla cuando
    window_size != 27. no se edita run_gdn y src.preprocessing se reusa en read-only. la
    variante 'repro' tiene que reproducir la baseline publicada."""

    import pickle

    import pandas as pd
    import run_model
    from src.imdiffusion.data import compute_if_retention_mask
    from src.preprocessing.config import load_config as load_preproc_config
    from src.preprocessing.power_curve_filter import PowerCurveFilter
    from src.preprocessing.status_filter import StatusFilter
    from src.preprocessing.transition_excluder import TransitionExcluder
    from src.preprocessing.windower import Windower

    if features != "curated":
        raise ValueError(f"B preproc sweep only supports the curated arm (got {features})")
    pcfg = _deep_merge(load_preproc_config(), load_preproc_config().get("per_farm", {}).get("B", {}))
    pcfg["transition_exclusion"] = {**pcfg.get("transition_exclusion", {}),
                                    "hours_before_fault": trans_before, "hours_after_fault": trans_after}
    pcfg["windowing"] = {**pcfg["windowing"], "window_size_samples": window_size}
    # Smoother mira config.enabled, no el sitio de llamada
    pcfg["smoothing"] = {**pcfg.get("smoothing", {}), "enabled": smooth}
    key = f"o{int(outlier)}t{int(temp_norm)}s{int(smooth)}_tb{trans_before}ta{trans_after}_w{window_size}"
    cache = Path("data/processed/experiments") / f"crgn_b_{features}_{key}_st{stride}.npz"
    cache_test = cache.with_suffix(".test.pkl")
    if cache.exists() and cache_test.exists():
        z = np.load(cache, allow_pickle=True)
        tw, feats = z["train_windows"], [str(c) for c in z["feats"]]
        with open(cache_test, "rb") as f:
            test_seqs = pickle.load(f)
        logger.info(f"Farm B repr [{features}/{key}] from cache: train {tw.shape} | {len(test_seqs)} events")
        return tw, test_seqs, feats

    df, event_info, feats, reg = run_model.preprocess_rows("B", pcfg, outlier, temp_norm, smooth)
    df_train = df[df["train_test"] == "train"].copy()
    df_train, _ = StatusFilter(pcfg).transform(df_train)
    df_train, _ = PowerCurveFilter(pcfg, reg).transform(df_train)
    df_train, _ = TransitionExcluder(pcfg).transform(df_train, event_info)
    if pcfg["data"].get("dedup_train", True):
        df_train = df_train.drop_duplicates(["asset_id", "time_stamp"]).reset_index(drop=True)
    df_pred = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)
    col_means = df_train[feats].mean()

    # float64, el dtype con el que la mascara IF vio los datos en la ejecucion del AE
    def arr(src):
        return src[feats].fillna(col_means).fillna(0.0).to_numpy(np.float64)

    def to_wdf(src):
        meta = src[[c for c in run_model.META if c in src.columns]].reset_index(drop=True)
        return pd.concat([meta, pd.DataFrame(arr(src), columns=feats)], axis=1)

    wtrain = Windower({"windowing": {**pcfg["windowing"], "stride_samples": stride}})
    wtest = Windower({"windowing": {**pcfg["windowing"], "stride_samples": 1}})
    train_w64, _, _ = wtrain.create_train_windows(to_wdf(df_train), feats)
    test_obj, _ = wtest.create_test_sequences(to_wdf(df_pred), event_info, feats)
    train_w64 = np.asarray(train_w64, np.float64)

    # mascara IF y scaler sobre las ventanas crudas, el orden con paridad de _load_b curated
    ifc = cfg["isolation_forest"]
    mask, scaler, _ = compute_if_retention_mask(
        train_w64, variance_threshold=ifc["variance_threshold"],
        contamination=ifc["contamination"], n_estimators=ifc["n_estimators"])
    train_windows = scaler.transform(train_w64).astype(np.float32)[mask]
    test_seqs = [{"event_id": s.event_id, "event_label": s.event_label, "asset_id": s.asset_id,
                  "windows": scaler.transform(np.asarray(s.windows, np.float64)).astype(np.float32),
                  "labels": np.asarray(s.labels, np.int8), "window_timestamps": s.window_timestamps}
                 for s in test_obj]
    logger.info(f"Farm B repr [{features}/{key}]: {len(feats)} sensors | train {train_windows.shape} "
                f"| {len(test_seqs)} events")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, train_windows=train_windows, feats=np.array(feats, dtype=object))
    with open(cache_test, "wb") as f:
        pickle.dump(test_seqs, f)
    logger.info(f"cached B repr -> {cache}")
    return train_windows, test_seqs, feats


def _preproc_variants(farm: str) -> list[tuple[str, dict]]:
    """variantes de preproc y W, de la etiqueta a los kwargs del loader. 'repro' lleva los
    parametros baseline y tiene que reproducir la cifra publicada, lo que comprueba que la
    replica configurable es igual que run_gdn._load_c. w18 y w36 quedan fuera del ancla."""

    if farm == "C":
        base = dict(outlier=False, temp_norm=False, smooth=False, trans_before=168, trans_after=72,
                    window_size=27)
        return [("repro", {**base}),
                ("pp_outlier", {**base, "outlier": True}),
                ("pp_tempnorm", {**base, "temp_norm": True}),
                ("pp_smooth", {**base, "smooth": True}),
                ("pp_full", {**base, "outlier": True, "temp_norm": True, "smooth": True}),
                ("trans72", {**base, "trans_before": 72}),
                ("w18", {**base, "window_size": 18}),
                ("w36", {**base, "window_size": 36})]
    if farm == "B":
        # baseline de B, el arm curated con mascara IF, outlier y temp-norm ON, smooth OFF.
        # mismos ejes que C en direccion inversa, C parte de todo OFF y anade, B parte de la
        # limpieza ON y quita. en B la limpieza es esencial, al reves que C, porque minimal y
        # no_outlier se desploman; por eso las variantes de ventana y transicion se anclan
        # sobre no_tempnorm, la mejor base por ROC-ds. 'repro' reproduce el baseline curated
        # (~0.6445), con el offset de ir desde crudo frente al npz pre-horneado
        base = dict(outlier=True, temp_norm=True, smooth=False, trans_before=168, trans_after=72,
                    window_size=27)
        nt = {**base, "temp_norm": False}  # mejor base por ROC-ds, outlier ON y tempnorm OFF
        return [("repro", {**base}),
                ("no_outlier", {**base, "outlier": False}),
                ("no_tempnorm", {**nt}),
                ("minimal", {**base, "outlier": False, "temp_norm": False}),
                ("pp_smooth", {**base, "smooth": True}),
                ("nt_w18", {**nt, "window_size": 18}),
                ("nt_w36", {**nt, "window_size": 36}),
                ("nt_trans72", {**nt, "trans_before": 72})]
    raise NotImplementedError(f"preproc variants for farm {farm} unknown")


def run_sweep_preproc(cfg: dict, device: torch.device, farm: str, seeds: list[int], label: str,
                      noise: float | None = None) -> dict:
    """una variante de preproc y W, multi-seed, por el loader de representacion configurable.
    una variante = un proceso, robusto y guardando el progreso parcial. reusa el train, la
    evaluacion y el harness de run_full. la variante 'repro' tiene que reproducir la baseline,
    que es el control de la maquinaria, antes de leer las demas. noise es un override opcional
    de augmentation.noise_std para combinar el preproc con una palanca de entrenamiento, p.ej.
    no_tempnorm x noise0.3, y se etiqueta `_n<noise>` para no pisar la variante preproc
    pelada."""

    variants = dict(_preproc_variants(farm))
    if label not in variants:
        raise SystemExit(f"unknown variant '{label}'; options: {list(variants)}")
    params = variants[label]
    ci = _deep_merge(cfg, {"augmentation": {"noise_std": noise}}) if noise is not None else cfg
    sfx = f"_n{noise}" if noise is not None else ""
    stride = ci["training"].get("stride_train", 1)
    epochs = ci["training"]["epochs"]
    if farm == "C":
        loader = lambda s: _load_c_preproc(ci["features"], stride, **params)  # noqa: E731
    elif farm == "B":
        loader = lambda s: _load_b_preproc(ci, ci["features"], stride, **params)  # noqa: E731
    else:
        raise NotImplementedError(f"preproc loader for farm {farm} not wired")
    rows = [run_full(ci, device, farm, ci["features"], seed, epochs, milestone="H5_ola3c_preproc",
                     tag=f"crgn_farm_{farm.lower()}_{ci['features']}_{label}{sfx}_s{seed}", repr_loader=loader)
            for seed in seeds]
    care = np.array([r["care_cv"] for r in rows])
    roc = np.array([r["roc_ds"] for r in rows])
    summary = {"milestone": "H5_ola3c_preproc", "farm": farm, "features": ci["features"],
               "variant": label, "params": params, "noise_std": ci["augmentation"]["noise_std"],
               "seeds": seeds,
               "care_cv_mean": float(care.mean()), "care_cv_std": float(care.std()),
               "roc_ds_mean": float(roc.mean()), "roc_ds_std": float(roc.std()),
               "per_seed": [{"seed": r["seed"], "care_cv": r["care_cv"], "roc_ds": r["roc_ds"],
                             "coverage": r["coverage"], "earliness": r["earliness"],
                             "reliability": r["reliability"], "accuracy": r["accuracy"]} for r in rows]}
    logger.info(f"=== CRGN {farm} preproc[{label}{sfx}] {params} noise={ci['augmentation']['noise_std']}: "
                f"CARE-CV {care.mean():.4f}±{care.std():.4f} | ROC-ds {roc.mean():.4f}±{roc.std():.4f} ===")
    tables = Path(ci["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    out_path = tables / f"crgn_{farm.lower()}_{label}{sfx}_multiseed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"saved {out_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="runner de CRGN (nucleo Energies 2025)")
    ap.add_argument("--farm", choices=["B", "C"], default="B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true", help="forward + InfoNCE en sintetico (H0)")
    ap.add_argument("--canary", action="store_true", help="H1, sobreajuste en ventanas normales de B")
    ap.add_argument("--canary-epochs", type=int, default=400)
    ap.add_argument("--canary-windows", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="H2 curated Farm B end-to-end (ROC-ds + CARE-CV)")
    ap.add_argument("--full", action="store_true", help="H3/H4 full multi-seed por parque (numero real)")
    ap.add_argument("--sweep-score", action="store_true",
                    help="H5 Ola 1: sweep score-time post-hoc sobre checkpoints (sin reentrenar)")
    ap.add_argument("--sweep-train", action="store_true",
                    help="H5 Ola 2: sweep con reentreno (epochs/dim/capas/temp/aug/topk)")
    ap.add_argument("--sweep-train2", action="store_true",
                    help="H5 Ola 2b: combos + ejes Tier B sin tocar (readout/proj/weighted/self_loops/max_rows/lr/wd)")
    ap.add_argument("--sweep-repr", action="store_true",
                    help="H5 Ola 3 (Tier C): sweep de representacion (arms base + subsets SHAP)")
    ap.add_argument("--max-train", type=int, default=120000, help="H2, submuestra de train del smoke")
    ap.add_argument("--epochs", type=int, default=None, help="sobrescribe training.epochs")
    ap.add_argument("--features", default=None, help="sobrescribe el arm de representacion (Tier C: 1 arm por proceso)")
    ap.add_argument("--subset", default=None, help="subset SHAP pcafirst (S_top100, S_top30, ...) sobre --features")
    ap.add_argument("--derivatives", action="store_true",
                    help="H5 Ola 3b: anade canales de primeras diferencias (in_dim W a 2W; grafo intacto)")
    ap.add_argument("--sweep-preproc", action="store_true",
                    help="H5 Ola 3c: una variante preproc/W (--variant) multi-seed, loader CRGN configurable")
    ap.add_argument("--variant", default=None, help="etiqueta variante preproc (repro/pp_full/trans72/w18/w36/...)")
    ap.add_argument("--noise", type=float, default=None,
                    help="Ola 6 combo: sobrescribe augmentation.noise_std sobre --variant (p.ej. no_tempnorm --noise 0.3)")
    ap.add_argument("--ola4", action="store_true",
                    help="H5 Ola 4: confirmacion multi-seed de los candidatos Ola2b (C proj32 / B noise0.3)")
    ap.add_argument("--seeds", default=None,
                    help="sobrescribe reproducibility.seeds, p.ej. 42,7,123,1234,12345 (por defecto, config)")
    ap.add_argument("--dump-scores", action="store_true",
                    help="H5 Ola 5: dump scores CRGN por ventana (train_dev/ev{ID}_dev) para fusion AE+CRGN")
    ap.add_argument("--overwrite-frozen", action="store_true",
                    help="permite sobrescribir los artefactos congelados de results/models/")
    args = ap.parse_args()

    # el selftest sale antes de mirar el dispositivo, que es lo que lo hace correr sin GPU
    if args.selftest:
        selftest(args.farm, args.seed)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.canary:
        run_canary(load_farm_config("B"), device, epochs=args.canary_epochs,
                   n_windows=args.canary_windows, seed=args.seed)
        return
    if args.smoke:
        cfg = load_farm_config("B")
        run_smoke_b(cfg, device, seed=args.seed, epochs=args.epochs or cfg["training"]["epochs"],
                    max_train=args.max_train)
        return
    if args.full:
        cfg = load_farm_config(args.farm)
        # la etiqueta de hito sale de las banderas, no del config, para que un mismo arm no
        # escriba dos veces bajo nombres distintos
        ms = ("H5_ola3b_deriv" if args.derivatives else
              ("H5_ola3_repr" if (args.features or args.subset) else None))
        run_multiseed(cfg, device, args.farm, args.features or cfg["features"],
                      cfg["reproducibility"]["seeds"], epochs=args.epochs or cfg["training"]["epochs"],
                      subset=args.subset, derivatives=args.derivatives, milestone=ms)
        return
    if args.sweep_score:
        cfg = load_farm_config(args.farm)
        run_sweep_scoretime(cfg, device, args.farm, cfg["features"], cfg["reproducibility"]["seeds"])
        return
    if args.sweep_train:
        cfg = load_farm_config(args.farm)
        run_sweep_train(cfg, device, args.farm, cfg["features"], cfg["reproducibility"]["seeds"])
        return
    if args.sweep_train2:
        cfg = load_farm_config(args.farm)
        run_sweep_train(cfg, device, args.farm, cfg["features"], cfg["reproducibility"]["seeds"],
                        arms=_ola2b_arms(args.farm), out_tag="ola2b")
        return
    if args.ola4:
        cfg = load_farm_config(args.farm)
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["reproducibility"]["seeds"]
        run_sweep_train(cfg, device, args.farm, cfg["features"], seeds,
                        arms=_ola4_arms(args.farm), out_tag="ola4")
        return
    if args.dump_scores:
        cfg = load_farm_config(args.farm)
        # la bandera viaja por el cfg porque quien decide conservar el ckpt es run_full
        cfg["output"]["overwrite_frozen"] = args.overwrite_frozen
        run_full(cfg, device, args.farm, cfg["features"], args.seed,
                 epochs=args.epochs or cfg["training"]["epochs"], dump_scores=True)
        return
    if args.sweep_repr:
        cfg = load_farm_config(args.farm)
        run_sweep_repr(cfg, device, args.farm, cfg["reproducibility"]["seeds"])
        return
    if args.sweep_preproc:
        cfg = load_farm_config(args.farm)
        run_sweep_preproc(cfg, device, args.farm, cfg["reproducibility"]["seeds"], args.variant,
                          noise=args.noise)
        return
    raise SystemExit("solo --selftest/--canary/--smoke/--full/--sweep-score/--sweep-train/"
                     "--sweep-train2/--sweep-repr/--sweep-preproc cableados")


if __name__ == "__main__":
    main()
