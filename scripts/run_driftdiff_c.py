"""D3R para Farm C bajo el protocolo oficial, sobre la representacion ganadora de C.

esa representacion son los componentes pca-first del top-100, K componentes con W=27, la misma
que usa el BiLSTM-AE (CARE-CV 0.7432 desplegado, 0.7206 honesto). reutiliza run_model en solo
lectura, preprocess_rows mas RowReducer mas Windower mas evaluate, y evalua con ReplayScorer y
run_model.evaluate, o sea CUSUM, h-sweep y CARE-CV sobre eventos, de modo que sale comparable
al AE y fusable, con los scores alineados por timestamp con ae y crgn. espeja
scripts/run_timeaddm_c.py.

ablaciones

    --no-decompose : difusor sin descomposicion, al estilo de TimeADDM-raw
    --t            : paso de ruido de la difusion
    --period       : ventana de la mediana movil del target stable
    --p            : pico de perturbacion

con --seeds 42,7,123,1234,12345 va multi-seed, hace el prep una sola vez y entrena y puntua
por semilla.

    .venv\\Scripts\\python.exe scripts/run_driftdiff_c.py --seeds 42 [--smoke]
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
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (imports hermanos)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz del repo (from src.X)

import run_model
from run_timeaddm_c import get_components
from sweep_timeaddm_scoring import ReplayScorer
from src.care_official.adapter import OfficialCAREAdapter
from src.driftdiff.model import build_dddr
from src.driftdiff.data import time_features, compute_stable_target
from src.driftdiff import scorer as ddscorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPR_SEED = 42  # la representacion PCA se fija a seed 42, como en TimeADDM
# la cache de filas es la parte cara del prep y no depende del period, por eso va aparte
ROWS_CACHE = Path("data/processed/farm_c_driftdiff_rows.pkl")
RAW_CACHE = Path("data/processed/farm_c_driftdiff_rows_raw.pkl")  # features crudas, sin PCA
SMOOTH = [1, 3, 5, 7]  # medias moviles sobre el score por ventana, gratis, como en TimeADDM
VARIANTS = ["recon", "trend", "recon+trend"]


def _cfgs():
    base = yaml.safe_load(open("config/model_config.yaml", encoding="utf-8"))
    cfg = run_model._deep_merge(base, base.get("per_farm", {}).get("C", {}))
    pcfg_base = run_model.load_preproc_config()
    pcfg = run_model._deep_merge(pcfg_base, pcfg_base.get("per_farm", {}).get("C", {}))
    return cfg, pcfg


def _prep_rows():
    """filas con meta y componentes PCA, para train y prediction. cacheado, porque es la parte
    cara del prep, unos 20 minutos."""
    if ROWS_CACHE.exists():
        logger.info(f"Loading row-level cache {ROWS_CACHE}")
        d = pickle.load(open(ROWS_CACHE, "rb"))
        return d["df_train"], d["df_pred"], d["event_info"], d["K"]
    logger.info("Generating row-level components from raw (preprocess_rows + RowReducer) — ~20 min ...")
    cfg, pcfg = _cfgs()
    pp = pcfg.get("preproc", {})
    df, event_info, feats, reg = run_model.preprocess_rows(
        "C", pcfg, pp.get("outlier", False), pp.get("temp_norm", False), pp.get("smooth", False))
    feat_file = cfg.get("features", {}).get("subset_file")
    if feat_file and Path(feat_file).exists():
        sub = set(Path(feat_file).read_text(encoding="utf-8").strip().splitlines())
        feats = [c for c in feats if c in sub]
        logger.info(f"  feature subset: {len(feats)} features")
    df_train = df[df["train_test"] == "train"].copy()
    df_train, _ = run_model.StatusFilter(pcfg).transform(df_train)
    df_train, _ = run_model.PowerCurveFilter(pcfg, reg).transform(df_train)
    df_train, _ = run_model.TransitionExcluder(pcfg).transform(df_train, event_info)
    if pcfg["data"].get("dedup_train", True):
        df_train = df_train.drop_duplicates(["asset_id", "time_stamp"]).reset_index(drop=True)
    df_pred = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)
    col_means = df_train[feats].mean()

    def arr(src):
        return src[feats].fillna(col_means).fillna(0.0).to_numpy(np.float32)

    reducer = run_model.RowReducer(variance_threshold=cfg["pca"]["variance_threshold"], random_state=REPR_SEED)
    comps_tr = reducer.fit_transform(arr(df_train))
    comps_te = reducer.transform(arr(df_pred))
    K = reducer.n_components_
    cols = [f"pc_{i}" for i in range(K)]
    df_train = df_train[[c for c in run_model.META if c in df_train.columns]].reset_index(drop=True)
    df_train = df_train.join(pd.DataFrame(comps_tr, columns=cols))
    df_pred = df_pred[[c for c in run_model.META if c in df_pred.columns]].reset_index(drop=True)
    df_pred = df_pred.join(pd.DataFrame(comps_te, columns=cols))
    ROWS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"df_train": df_train, "df_pred": df_pred, "event_info": event_info, "K": int(K)},
                open(ROWS_CACHE, "wb"))
    logger.info(f"  cached -> {ROWS_CACHE}  (train {len(df_train):,} rows, pred {len(df_pred):,} rows, K={K})")
    return df_train, df_pred, event_info, int(K)


def _prep_rows_raw():
    """filas con las features crudas del subset XAI top-100, sin PCA. cacheado."""
    if RAW_CACHE.exists():
        logger.info(f"Loading raw row-level cache {RAW_CACHE}")
        d = pickle.load(open(RAW_CACHE, "rb"))
        return d["df_train"], d["df_pred"], d["event_info"], d["feats"]
    logger.info("Generating RAW row-level features from raw (preprocess_rows, sin PCA) — ~20 min ...")
    cfg, pcfg = _cfgs()
    pp = pcfg.get("preproc", {})
    df, event_info, feats, reg = run_model.preprocess_rows(
        "C", pcfg, pp.get("outlier", False), pp.get("temp_norm", False), pp.get("smooth", False))
    feat_file = cfg.get("features", {}).get("subset_file")
    if feat_file and Path(feat_file).exists():
        sub = set(Path(feat_file).read_text(encoding="utf-8").strip().splitlines())
        feats = [c for c in feats if c in sub]
        logger.info(f"  feature subset: {len(feats)} features")
    df_train = df[df["train_test"] == "train"].copy()
    df_train, _ = run_model.StatusFilter(pcfg).transform(df_train)
    df_train, _ = run_model.PowerCurveFilter(pcfg, reg).transform(df_train)
    df_train, _ = run_model.TransitionExcluder(pcfg).transform(df_train, event_info)
    if pcfg["data"].get("dedup_train", True):
        df_train = df_train.drop_duplicates(["asset_id", "time_stamp"]).reset_index(drop=True)
    df_pred = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)
    col_means = df_train[feats].mean()
    keep = [c for c in run_model.META if c in df_train.columns] + feats
    df_train = df_train[keep].copy()
    df_train[feats] = df_train[feats].fillna(col_means).fillna(0.0)
    df_pred = df_pred[[c for c in run_model.META if c in df_pred.columns] + feats].copy()
    df_pred[feats] = df_pred[feats].fillna(col_means).fillna(0.0)
    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"df_train": df_train, "df_pred": df_pred, "event_info": event_info, "feats": feats},
                open(RAW_CACHE, "wb"))
    logger.info(f"  cached -> {RAW_CACHE}  (train {len(df_train):,} rows, {len(feats)} feats)")
    return df_train, df_pred, event_info, feats


def prep_c(period: int, raw: bool = False):
    """ensambla las ventanas de data, los componentes PCA estandarizados o las features crudas
    si raw=True, mas stable, el target detrend, mas time, el calendario."""

    # el test sale del Windower oficial, que es lo que lo deja alineado con la meta y fusable
    _cfg, pcfg = _cfgs()
    if raw:
        df_train, df_pred, event_info, cols = _prep_rows_raw()
        K = len(cols)
    else:
        df_train, df_pred, event_info, K = _prep_rows()
        cols = [f"pc_{i}" for i in range(K)]
    st_cols = [f"st_{i}" for i in range(K)]
    tf_cols = [f"tf_{i}" for i in range(5)]
    wcfg = {"windowing": pcfg["windowing"]}
    W = pcfg["windowing"]["window_size_samples"]
    max_gap = pcfg["windowing"]["max_gap_minutes"]
    meta_tr = [c for c in run_model.META if c in df_train.columns]
    meta_te = [c for c in run_model.META if c in df_pred.columns]

    # los componentes se estandarizan por columna, en unidades de std, para que p y t queden en
    # una escala consistente entre ellos
    scaler = StandardScaler().fit(df_train[cols].to_numpy(np.float32))
    comps_tr = scaler.transform(df_train[cols].to_numpy(np.float32)).astype(np.float32)
    comps_te = scaler.transform(df_pred[cols].to_numpy(np.float32)).astype(np.float32)

    stable = compute_stable_target(comps_tr, df_train[["asset_id", "time_stamp"]].reset_index(drop=True),
                                   period=period, max_gap_minutes=max_gap)
    tf_tr = time_features(df_train["time_stamp"].values)
    tf_te = time_features(df_pred["time_stamp"].values)

    windower = run_model.Windower(wcfg)
    # el wdf se ensambla de una vez, que insertar columna a columna fragmenta el DataFrame
    wdf_tr = pd.concat([df_train[meta_tr].reset_index(drop=True),
                        pd.DataFrame(comps_tr, columns=cols),
                        pd.DataFrame(stable, columns=st_cols),
                        pd.DataFrame(tf_tr, columns=tf_cols)], axis=1)
    train_all, train_tids, _ = windower.create_train_windows(wdf_tr, cols + st_cols + tf_cols)
    # el Windower devuelve las tres familias pegadas en el eje de canal, y este es el orden en
    # que se pidieron, primero los K componentes, luego los K stable y luego las 5 de fecha
    train_data = np.ascontiguousarray(train_all[:, :, :K])
    train_stable = np.ascontiguousarray(train_all[:, :, K:2 * K])
    train_time = np.ascontiguousarray(train_all[:, :, 2 * K:2 * K + 5])
    del train_all

    wdf_te = pd.concat([df_pred[meta_te].reset_index(drop=True),
                        pd.DataFrame(comps_te, columns=cols),
                        pd.DataFrame(tf_te, columns=tf_cols)], axis=1)
    test_obj, _ = windower.create_test_sequences(wdf_te, event_info, cols + tf_cols)
    test_seqs = []
    for s in test_obj:
        win = s.windows.astype(np.float32)
        test_seqs.append({"event_id": int(s.event_id), "event_label": s.event_label,
                          "asset_id": int(s.asset_id), "windows": win[:, :, :K],
                          "data": np.ascontiguousarray(win[:, :, :K]),
                          "time": np.ascontiguousarray(win[:, :, K:K + 5]),
                          "labels": np.asarray(s.labels, int), "window_timestamps": s.window_timestamps})

    # sanity, mismo numero y orden de ventanas que get_components, que va por la misma
    # representacion y la misma seed 42; solo aplica al camino de componentes
    if not raw:
        try:
            gc_train, _gc_test, gc_K = get_components()
            ok = gc_K == K and gc_train.shape == train_data.shape
            logger.info(f"  sanity get_components: {'OK' if ok else 'MISMATCH'} "
                        f"{train_data.shape} vs {gc_train.shape} (K {K} vs {gc_K})")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  sanity get_components salto: {e}")

    logger.info(f"prep C | K={K} W={W} | train {train_data.shape} | {len(test_seqs)} eventos | period={period}")
    return {"train_data": train_data, "train_stable": train_stable, "train_time": train_time,
            "train_tids": np.asarray(train_tids), "test_seqs": test_seqs, "K": K, "W": W}


def train_dddr(model, data, stable, time_, cfg, device, seed):
    """entrena D3R con loss = 0.5*MSE(stable, target) + 0.5*MSE(recon, data) y early stopping
    en val."""

    # train_cap submuestrea solo el bucle de entrenamiento; el scoring y la calibracion siguen
    # usando el train completo
    rng = np.random.default_rng(seed)
    n = len(data)
    cap = cfg.get("train_cap", 0) or n
    # se pide un 10% de mas sobre el cap, y ese 10% es justo el que se aparta para val
    work = rng.permutation(n)[:min(cap + int(0.1 * cap) + 1, n)]
    nval = max(1, int(0.1 * len(work)))
    val_idx, tr_idx = work[:nval], work[nval:]
    bs = cfg["batch"]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    crit = torch.nn.MSELoss()
    decompose, p = cfg["decompose"], cfg["p"]
    best, best_state, ctr = float("inf"), None, 0
    logger.info(f"  train windows {len(tr_idx):,} (cap {cap:,}) | val {len(val_idx):,} | batch {bs}")

    def batch_loss(bidx, train):
        x = torch.as_tensor(data[bidx], dtype=torch.float32, device=device)
        tf = torch.as_tensor(time_[bidx], dtype=torch.float32, device=device)
        stable_pred, _trend, recon = model(x, tf, p=(p if train else 0.0))
        loss = crit(recon, x)
        # sin descomposicion el modelo es un difusor raw y solo se le pide la reconstruccion
        if decompose:
            st = torch.as_tensor(stable[bidx], dtype=torch.float32, device=device)
            loss = 0.5 * crit(stable_pred, st) + 0.5 * loss
        return loss

    for e in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        tr_perm = rng.permutation(len(tr_idx))
        tl = []
        for s in range(0, len(tr_idx), bs):
            b = tr_idx[tr_perm[s:s + bs]]
            opt.zero_grad()
            loss = batch_loss(b, train=True)
            loss.backward()
            opt.step()
            tl.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = [batch_loss(val_idx[s:s + bs], train=False).item() for s in range(0, len(val_idx), bs)]
        tr_l, val_l = float(np.mean(tl)), float(np.mean(vl))
        logger.info(f"  epoch {e:>2} | train {tr_l:.5f} | val {val_l:.5f} | {time.time()-t0:.0f}s")
        if val_l < best - cfg.get("min_delta", 1e-4):
            best, best_state, ctr = val_l, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            ctr += 1
            if ctr >= cfg["patience"]:
                logger.info(f"  early stop @ epoch {e} (best val {best:.5f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _smooth(s, w):
    return s if w <= 1 else np.convolve(s, np.ones(w) / w, mode="same")


def run_one_seed(prep, adapter, args, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    decompose = not args.no_decompose
    K, W = prep["K"], prep["W"]
    tag = f"_s{seed}" + (f"_{args.exp_tag}" if args.exp_tag else "")
    # d es el corte de frecuencia del bloque de descomposicion, media ventana
    mcfg = {"model_dim": args.model_dim, "ff_dim": args.ff_dim, "atten_dim": args.atten_dim,
            "block_num": args.block_num, "head_num": args.head_num, "dropout": args.dropout,
            "d": max(1, W // 2), "t": args.t, "decompose": decompose}
    model = build_dddr(K, W, 5, mcfg, DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    logger.info(f"[seed {seed}] DDDR ({nparams:,} params) decompose={decompose} t={args.t} "
                f"period={args.period} p={args.p}")

    tcfg = {"batch": args.batch, "lr": args.lr, "epochs": args.epochs, "patience": args.patience,
            "decompose": decompose, "p": args.p, "train_cap": args.train_cap, "min_delta": args.min_delta}
    t0 = time.time()
    train_dddr(model, prep["train_data"], prep["train_stable"], prep["train_time"], tcfg, DEVICE, seed)
    logger.info(f"  train done | {time.time()-t0:.0f}s")

    t0 = time.time()
    train_out = ddscorer.run_model_scores(model, prep["train_data"], prep["train_time"], DEVICE, draws=args.draws)
    event_outs = [ddscorer.run_model_scores(model, s["data"], s["time"], DEVICE, draws=args.draws)
                  for s in prep["test_seqs"]]
    logger.info(f"  scoring done | {time.time()-t0:.0f}s")

    # sin descomposicion no hay canal trend, asi que la unica variante de score es recon
    variants = ["recon"] if not decompose else VARIANTS
    rows = []
    for variant in variants:
        tr_s, ev_s = ddscorer.assemble_scores(train_out, event_outs, variant)
        for sw in SMOOTH:
            ev_sm = [_smooth(e, sw) for e in ev_s]
            # percentil 75 y no 99, el score de D3R es menos apuntado que el MAE del AE
            scorer = ReplayScorer([tr_s] + ev_sm, percentile=75.0)
            ev = run_model.evaluate(scorer, prep["train_data"], prep["test_seqs"], adapter, 42)  # cv-seed 42 fijo
            b, cv = ev["best"], ev["cv"]
            rows.append({"variant": variant, "smooth": sw, "care_cv": cv["care"], "oracle": b["care"],
                         "optim": cv["optimism"], "roc_ds": b["roc_ds"], "roc_win": ev["roc_window"],
                         "f1": b["f1"], "cov": b["cov"], "earl": b["earl"], "rel": b["rel"], "acc": b["acc"],
                         "earl_cond": b["earl_conditional"], "h_scale": b["h_scale"]})
            logger.info(f"  {variant:<12} MA{sw}  CARE-CV={cv['care']:.4f}  optim={cv['optimism']:+.4f}  "
                        f"ROC-ds={b['roc_ds']:.3f}  Earl={b['earl']:.3f}  Acc={b['acc']:.3f}")
    rows.sort(key=lambda r: -r["care_cv"])
    best = rows[0]
    logger.info(f"[seed {seed}] BEST {best['variant']} MA{best['smooth']} -> CARE-CV={best['care_cv']:.4f} "
                f"(ROC-ds {best['roc_ds']:.3f}, Earl {best['earl']:.3f})")

    out = {"farm": "C", "model": "driftdiff_d3r", "representation": "pca_first_top100_components_stdscaled",
           "K": K, "window_size": int(W), "seed": seed, "decompose": decompose, "t": args.t,
           "period": args.period, "p": args.p, "draws": args.draws, "train_cap": args.train_cap,
           "exp_tag": args.exp_tag, "model_dim": args.model_dim, "block_num": args.block_num,
           "n_params": int(nparams), "smoke": args.smoke, "sweep": rows}
    p_out = Path(f"results/tables/ownA_farm_c_sweep{tag}.json")
    p_out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p_out, "w"), indent=2, default=run_model._np_default)
    logger.info(f"  saved -> {p_out}")

    if args.dump_scores:
        _dump_scores(train_out, event_outs, prep["test_seqs"], seed, decompose)
    return best


def _dump_scores(train_out, event_outs, test_seqs, seed, decompose):
    """vuelca los scores por ventana, train y ev{ID}, alineados con la meta del AE para que
    fuse_multi los pueda fusionar."""

    # escribe recon y, si hay descomposicion, trend; crudos, que fuse_multi ya z-normaliza
    meta = json.load(open("results/models/ae_scores_farm_c_meta.json", encoding="utf-8"))
    arrays = {}
    variants = [("recon", "recon")] + ([("trend", "trend")] if decompose else [])
    for variant, key in variants:
        tr_s, ev_s = ddscorer.assemble_scores(train_out, event_outs, variant)
        arrays[f"train_{key}"] = np.asarray(tr_s, np.float64)
        for s, e in zip(test_seqs, ev_s):
            eid = int(s["event_id"])
            if str(eid) not in meta:
                raise RuntimeError(f"evento {eid} ausente en la meta de C")
            if len(e) != len(meta[str(eid)]["labels"]):
                raise RuntimeError(f"desalineado ev{eid}: {len(e)} vs meta {len(meta[str(eid)]['labels'])}")
            arrays[f"ev{eid}_{key}"] = np.asarray(e, np.float64)
    # la clave es ev{ID}_{variante}, asi que el id sale entre el 'ev' y el primer guion bajo
    covered = {int(k[2:].split("_")[0]) for k in arrays if k.startswith("ev")}
    missing = set(int(k) for k in meta) - covered
    if missing:
        raise RuntimeError(f"faltan eventos en el volcado: {sorted(missing)}")
    out = Path(f"results/models/ownA_scores_farm_c_s{seed}.npz")
    np.savez_compressed(out, **arrays)
    logger.info(f"  [dump] ownA ({[v for v, _ in variants]}): {len(arrays)} arrays -> {out}")


def _keep_awake():
    """evita que Windows suspenda el sistema durante el run."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        logger.info("keep-awake ON (SetThreadExecutionState)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"keep-awake no disponible: {e}")


def main():
    _keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42", help="lista de semillas, ej 42,7,123,1234,12345")
    ap.add_argument("--exp-tag", type=str, default="")
    ap.add_argument("--period", type=int, default=144, help="ventana mediana movil del target stable (144=1 dia)")
    ap.add_argument("--t", type=int, default=500, help="paso de ruido fijo de la difusion")
    ap.add_argument("--p", type=float, default=1.0, help="pico de perturbacion (std, sobre comps estandarizados)")
    ap.add_argument("--no-decompose", action="store_true", help="ablacion: sin descomposicion (difusor raw)")
    ap.add_argument("--model-dim", type=int, default=128)
    ap.add_argument("--ff-dim", type=int, default=256)
    ap.add_argument("--block-num", type=int, default=2)
    ap.add_argument("--head-num", type=int, default=4)
    ap.add_argument("--atten-dim", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-delta", type=float, default=1e-4, help="mejora minima de val para no contar como estancado")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--train-cap", type=int, default=300000, help="max ventanas de entrenamiento (0=todas)")
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="train_cap 80k + 3 epocas (validar pipeline)")
    ap.add_argument("--dump-scores", action="store_true", help="volcar scores para fusion (results/models/ownA_*)")
    ap.add_argument("--raw", action="store_true", help="Prueba 3: features crudas (100) en vez de componentes PCA (off-anchor)")
    args = ap.parse_args()

    # el smoke solo valida el pipeline, con poco train y tres epocas
    if args.smoke:
        args.train_cap = 80000
        args.epochs = min(args.epochs, 3)
    seeds = [int(s) for s in args.seeds.split(",")]
    logger.info(f"Device: {DEVICE} | Farm C | D3R | seeds {seeds} | train_cap {args.train_cap}")

    # el prep es comun a todas las semillas, se hace una vez y se reutiliza
    prep = prep_c(args.period, raw=args.raw)
    adapter = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm="C", window_size=prep["W"])
    bests = []
    for seed in seeds:
        best = run_one_seed(prep, adapter, args, seed)
        best = {**best, "seed": seed}
        bests.append(best)

    cares = np.array([b["care_cv"] for b in bests])
    suffix = f"_{args.exp_tag}" if args.exp_tag else ""
    summary = {"model": "driftdiff_d3r", "farm": "C", "seeds": seeds,
               "decompose": not args.no_decompose, "t": args.t, "period": args.period, "p": args.p,
               "train_cap": args.train_cap, "batch": args.batch,
               "mean": float(cares.mean()), "std": float(cares.std()),
               "wins_vs_ae_honest_0.7206": int((cares > 0.7206).sum()),
               "per_seed": bests}
    json.dump(summary, open(f"results/tables/ownA_multiseed_summary{suffix}.json", "w"),
              indent=2, default=run_model._np_default)
    logger.info("=" * 70)
    logger.info(f"MULTI-SEED D3R C: CARE-CV {cares.mean():.4f} +- {cares.std():.4f} "
                f"(por-semilla {[(b['seed'], round(b['care_cv'], 4), b['variant']) for b in bests]}) | "
                f"wins vs AE-honesto 0.7206: {int((cares > 0.7206).sum())}/{len(seeds)} "
                f"| AE deployed 0.7432 / AE+CRGN 0.7452")
    logger.info(f"summary -> results/tables/ownA_multiseed_summary{suffix}.json")


if __name__ == "__main__":
    main()
