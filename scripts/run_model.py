"""pipeline de deteccion de anomalias con el BiLSTM-autoencoder, para Farm B y Farm C.

hay un pipeline por granja porque la configuracion optima no es la misma en las dos y no
son unificables. Farm B va window-first: el PreprocessingPipeline entero corre desde el
crudo y en el mismo proceso, con las features curated, la limpieza pesada de outlier y
temp-norm, dedup, sin suavizado y con imputacion por interpolacion, y tras el ventaneo
entran WindowScaler, WindowPCA por paso de tiempo e Isolation Forest. Farm C va pca-first:
las filas se reprocesan con limpieza minima, de todos los `_avg` al subset de 100 de XAI, y
luego RowReducer, o sea StandardScaler y PCA a nivel de fila, y se ventanean los
componentes, sin Isolation Forest. las dos granjas se reproducen de punta a punta desde el
crudo con una sola orden, sin un paso de run_preprocessing aparte.

`order` y el resto de decisiones por granja viven en config/model_config.yaml bajo
`per_farm:`, y run_farm las funde sobre la base. la deteccion usa el CUSUM estadistico con
la `h` elegida por CV sobre eventos, sin fuga del test, y la puntuacion es la del evaluador
CARE oficial. las metricas que reporta son el CARE de CV con su optimismo, Coverage Fβ, la
Earliness WS oficial y ademas la earliness condicional, que solo mira los detectados,
Reliability EFβ, accuracy, F1 con precision y recall, y AUC-ROC a nivel de dataset, con la
de ventana como secundaria.

    python run_model.py --farm B              # window-first, se preprocesa el solo del crudo
    python run_model.py --farm C              # pca-first, reprocesa las filas el solo
    python run_model.py --farm C --dump-artifacts   # y persiste scaler/pca/seqs para XAI
"""
from __future__ import annotations

import argparse
import copy
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
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/, imports hermanos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz, para from src.X

from src.preprocessing.config import load_config as load_preproc_config
from src.preprocessing.data_loader import CareDataLoader
from src.preprocessing.zero_handler import ZeroHandler
from src.preprocessing.feature_selector import FeatureSelector
from src.preprocessing.outlier_handler import OutlierHandler
from src.preprocessing.normalizer import TemperatureNormalizer
from src.preprocessing.smoother import Smoother
from src.preprocessing.status_filter import StatusFilter
from src.preprocessing.power_curve_filter import PowerCurveFilter
from src.preprocessing.transition_excluder import TransitionExcluder
from src.preprocessing.windower import Windower
from src.preprocessing.pipeline import PreprocessingPipeline
from src.model_prep.model_scaler import WindowScaler
from src.model_prep.pca_reducer import WindowPCA
from src.model_prep.row_reducer import RowReducer
from src.model_prep.isolation_forest_cleaner import IFCleaner
from src.models.bilstm_autoencoder import BiLSTMAutoencoder
from src.models.trainer import Trainer
from src.models.anomaly_detector import AnomalyDetector, run_cusum, fit_cusum_params, H_SCALES
from src.care_official.adapter import OfficialCAREAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# columnas que no son sensores: viajan con el dataframe pero nunca entran al modelo
META = ["time_stamp", "asset_id", "id", "train_test", "status_type_id",
        "event_id", "event_label", "event_start", "event_end", "event_description"]


def _np_default(o):
    return float(o) if isinstance(o, np.floating) else int(o)


def _deep_merge(base: dict, override: dict) -> dict:
    # fusion recursiva, un dict se funde clave a clave y cualquier otra cosa sustituye entera
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def preprocess_rows(farm: str, pcfg: dict, outlier: bool, temp_norm: bool, smooth: bool):
    """carga y limpia las filas de las features seleccionadas, y para antes del ventaneo.

    es la mitad a nivel de fila del camino pca-first; el ventaneo viene despues, sobre los
    componentes reducidos y no sobre los sensores."""

    loader = CareDataLoader(pcfg)
    df, event_info, reg = loader.load_farm(farm)
    df, _ = ZeroHandler(pcfg, reg).transform(df)
    # la seleccion de features y el fit del outlier miran solo filas de operacion normal,
    # para que el propio fallo no defina lo que se considera normal
    normal = pcfg["status_filtering"]["normal_statuses"]
    dfn = df[df["status_type_id"].isin(normal)]
    feats, _ = FeatureSelector(pcfg, reg).select(dfn)
    logger.info(f"  features selected: {len(feats)}")
    if outlier:
        oh = OutlierHandler(pcfg, reg)
        oh.fit(df[df["status_type_id"].isin(normal)], feats)
        df, _ = oh.transform(df, feats)
        logger.info("  outlier-IQR applied")
    if temp_norm:
        df, _ = TemperatureNormalizer(pcfg, reg).transform(df, feats)
        logger.info("  temp-norm applied")
    if smooth:
        df, _ = Smoother(pcfg, reg).transform(df, feats)
        logger.info("  smoothing applied")
    keep = [c for c in META if c in df.columns] + feats
    # el viento se arrastra aunque no sea feature: lo necesita el filtro de curva de potencia
    keep += [c for c in df.columns if c.endswith("_avg") and "wind_speed" in c and c not in keep]
    return df[keep].copy(), event_info, feats, reg


def dump_ae_scores(detector: AnomalyDetector, train_clean: np.ndarray, test_seqs: list[dict],
                   farm: str, out_dir: Path, seed: int = 42) -> None:
    """persiste los scores de reconstruccion por ventana del AE, train y cada evento, mas la
    metadata de evento, para poder repetirlos en CPU sin reentrenar.

    la convencion de claves es la del npz de scores de ImDiffusion, train_mae y ev{ID}_mae,
    para que quien los relea no dependa del modelo que los produjo. hoy el consumidor es la
    fusion: la entrada 'ae' de fuse_multi.DETECTORS lee exactamente
    results/models/ae_scores_farm_{f}{sfx(s)}.npz, y el gate de Nivel B repite ese corpus.
    los ficheros llevan sufijo de semilla; la 42 conserva el nombre sin sufijo."""

    out_dir.mkdir(parents=True, exist_ok=True)
    fl = farm.lower()
    suffix = "" if seed == 42 else f"_s{seed}"
    arrays = {"train_mae": np.asarray(detector.score_windows(train_clean), np.float64)}
    meta = {}
    for s in test_seqs:
        eid = s["event_id"]
        arrays[f"ev{eid}_mae"] = np.asarray(detector.score_windows(s["windows"]), np.float64)
        meta[str(eid)] = {"event_label": s["event_label"], "asset_id": int(s["asset_id"]),
                          "labels": np.asarray(s["labels"], int).tolist(),
                          "window_timestamps": [str(t) for t in np.asarray(s["window_timestamps"]).ravel()]}
    np.savez_compressed(out_dir / f"ae_scores_farm_{fl}{suffix}.npz", **arrays)
    with open(out_dir / f"ae_scores_farm_{fl}{suffix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    logger.info(f"[dump-scores] AE: {len(test_seqs)} events -> {out_dir}/ae_scores_farm_{fl}{suffix}.npz")


def evaluate(detector: AnomalyDetector, train_clean: np.ndarray, test_seqs: list[dict],
             adapter: OfficialCAREAdapter, seed: int) -> dict:
    """CUSUM estadistico, barrido de h, CV de h sin fuga de test y CARE oficial."""

    # mu, k y h_base se calibran solo con el train: el umbral no puede mirar los eventos
    train_err = detector.score_windows(train_clean)
    mu, k, h_base = fit_cusum_params(train_err, slack=1.0, percentile=99.0)

    # se puntua cada evento una vez y se guarda su CUSUM acumulado, para barrer h sin repuntuar
    ev_meta, ev_S, gt, sc = [], [], [], []
    for s in test_seqs:
        e = detector.score_windows(s["windows"])
        ev_S.append(run_cusum(e, mu, k))
        ev_meta.append({"event_id": s["event_id"], "asset_id": s["asset_id"], "event_label": s["event_label"],
                        "window_timestamps": s["window_timestamps"]})
        gt.append(np.asarray(s["labels"], int))
        sc.append(e)
    gt = np.concatenate(gt)
    sc = np.concatenate(sc)
    # AUC de ventana, secundaria: pide las dos clases presentes o no esta definida
    roc_win = float(roc_auc_score(gt, sc)) if gt.sum() and (gt == 0).sum() else float("nan")
    ev_lab = np.array([1 if m["event_label"] == "anomaly" else 0 for m in ev_meta])

    def _preds(h):
        return [{**ev_meta[i], "window_predictions": (ev_S[i] > h).astype(np.int8)} for i in range(len(ev_meta))]

    # barrido de h: el mejor CARE de aqui es el oracle, optimista porque elige mirando el test
    sweep = []
    for hs in H_SCALES:
        h = h_base * hs
        res, diag = adapter.evaluate(_preds(h))
        # AUC a nivel dataset: un evento puntua por su fraccion de filas marcadas
        yt = [1 if d["event_label"] == "anomaly" else 0 for d in diag]
        ys = [d["n_flagged_rows"] / d["n_pred_rows"] if d["n_pred_rows"] else 0.0 for d in diag]
        roc_ds = float(roc_auc_score(yt, ys)) if (0 in yt and 1 in yt) else float("nan")
        f1 = float(f1_score(gt, np.concatenate([(ev_S[i] > h).astype(np.int8) for i in range(len(ev_meta))]),
                            zero_division=0))
        cond = adapter.conditional_earliness(_preds(h))
        sweep.append({"h_scale": hs, "care": res.care, "f1": f1, "roc_ds": roc_ds, "cov": res.coverage,
                      "earl": res.earliness, "rel": res.reliability, "acc": res.accuracy,
                      "earl_conditional": cond["conditional_earliness"]})
    best = max(sweep, key=lambda r: r["care"])

    # el CARE honesto: h se elige en los eventos de train de cada fold y se aplica a los de
    # test, asi que ninguna prediccion se hizo con un umbral que hubiera visto ese evento
    n_anom = int(ev_lab.sum())
    n_splits = max(2, min(5, n_anom // 2))  # al menos 2 anomalias por fold, B da 3 y C da 5
    h_grid = [h_base * r["h_scale"] for r in sweep]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cvp = [None] * len(ev_meta)
    fold_hs = []
    for tr, te in skf.split(np.arange(len(ev_meta)), ev_lab):
        hstar = max(h_grid, key=lambda h: adapter.evaluate([_preds(h)[i] for i in tr])[0].care)
        fold_hs.append(round(hstar / h_base, 4))
        for i in te:
            cvp[i] = (ev_S[i] > hstar).astype(np.int8)
    cv_care = float(adapter.evaluate([{**ev_meta[i], "window_predictions": cvp[i]}
                                      for i in range(len(ev_meta))])[0].care)

    return {"best": best, "roc_window": roc_win,
            "cv": {"folds": n_splits, "care": cv_care, "fold_h_scale": fold_hs,
                   "oracle": best["care"], "optimism": best["care"] - cv_care},
            "sweep": sweep,
            "threshold": float(np.percentile(train_err, detector.percentile))}


def _prep_window_first(farm: str, cfg: dict, pcfg_base: dict, seed: int):
    """desde el crudo, simetrico con pca-first: PreprocessingPipeline entero en el proceso, de
    la carga al ventaneo, y luego reduccion por paso de tiempo con WindowScaler y WindowPCA.

    el pipeline ademas persiste sus ventanas en processed_path, asi que ese .npz es una salida
    regenerada del crudo y nunca una entrada de esta ejecucion."""

    pcfg = _deep_merge(pcfg_base, pcfg_base.get("per_farm", {}).get(farm, {}))
    result = PreprocessingPipeline(pcfg).run(farm)
    feats = list(result.feature_cols)
    train_raw = result.train_windows
    test_dicts = [{"event_id": s.event_id, "event_label": s.event_label, "asset_id": s.asset_id,
                   "windows": np.asarray(s.windows, np.float32), "labels": np.asarray(s.labels, int),
                   "window_timestamps": s.window_timestamps} for s in result.test_sequences]
    # subset opcional de features; el orden lo fija el fichero, y train y test se cortan con
    # los mismos indices para que la columna k siga siendo la misma feature en los dos
    feat_file = cfg.get("features", {}).get("subset_file")
    if feat_file and Path(feat_file).exists():
        sub = Path(feat_file).read_text(encoding="utf-8").strip().splitlines()
        idx = [feats.index(f) for f in sub if f in feats]
        feats = [feats[i] for i in idx]
        train_raw = train_raw[:, :, idx]
        for s in test_dicts:
            s["windows"] = s["windows"][:, :, idx]
    scaler = WindowScaler(feature_range=tuple(cfg["scaling"]["feature_range"]), per_turbine=False,
                          method=cfg["scaling"].get("method", "standard"))
    train_scaled = scaler.fit_transform(train_raw, None)
    pca = WindowPCA(variance_threshold=cfg["pca"]["variance_threshold"], random_state=seed)
    train_windows = pca.fit_transform(train_scaled)

    def _seq(s, win):
        return {"event_id": s["event_id"], "event_label": s["event_label"], "asset_id": s["asset_id"],
                "windows": win.astype(np.float32), "labels": np.asarray(s["labels"], int),
                "window_timestamps": s["window_timestamps"]}

    # dos vistas del test: las ventanas crudas, que son las que entiende XAI, y las reducidas,
    # que son las que ve el modelo. scaler y pca van ajustados solo con train
    raw_test = [_seq(s, s["windows"]) for s in test_dicts]
    test_seqs = [_seq(s, pca.transform(scaler.transform(s["windows"]))) for s in test_dicts]
    return train_windows, test_seqs, raw_test, feats, pca.n_components_, scaler, pca


def _prep_pca_first(farm: str, cfg: dict, pcfg_base: dict, seed: int):
    """camino pca-first: reprocesa las filas con limpieza minima, RowReducer y ventaneo de los
    componentes."""

    pcfg = _deep_merge(pcfg_base, pcfg_base.get("per_farm", {}).get(farm, {}))
    pp = pcfg.get("preproc", {})
    df, event_info, feats, reg = preprocess_rows(farm, pcfg, pp.get("outlier", False),
                                                 pp.get("temp_norm", False), pp.get("smooth", False))
    feat_file = cfg.get("features", {}).get("subset_file")
    if feat_file and Path(feat_file).exists():
        sub = set(Path(feat_file).read_text(encoding="utf-8").strip().splitlines())
        feats = [c for c in feats if c in sub]
        logger.info(f"  feature subset: {len(feats)} features")
    # la limpieza fuerte va solo sobre train: en prediction hay que puntuar todas las filas,
    # incluidas las anomalas, o se estaria tirando justo lo que se quiere detectar
    df_train = df[df["train_test"] == "train"].copy()
    df_train, _ = StatusFilter(pcfg).transform(df_train)
    df_train, _ = PowerCurveFilter(pcfg, reg).transform(df_train)
    df_train, _ = TransitionExcluder(pcfg).transform(df_train, event_info)
    if pcfg["data"].get("dedup_train", True):
        df_train = df_train.drop_duplicates(["asset_id", "time_stamp"]).reset_index(drop=True)
    df_pred = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)
    # las medias de imputacion salen de train y se reusan en prediction, para no filtrar
    col_means = df_train[feats].mean()
    logger.info(f"  train rows {len(df_train):,}")
    windower = Windower({"windowing": pcfg["windowing"]})

    def to_wdf(src, data_, names):
        out = src[[c for c in META if c in src.columns]].copy().reset_index(drop=True)
        out[names] = data_
        return out

    def arr(src):
        return src[feats].fillna(col_means).fillna(0.0).to_numpy(np.float32)

    # se reduce a nivel de fila y se ventanean los componentes; al reves que en window-first
    reducer = RowReducer(variance_threshold=cfg["pca"]["variance_threshold"], random_state=seed)
    comps = reducer.fit_transform(arr(df_train))
    K = reducer.n_components_
    cols = [f"pc_{i}" for i in range(K)]
    train_windows, _, _ = windower.create_train_windows(to_wdf(df_train, comps, cols), cols)
    test_obj, _ = windower.create_test_sequences(to_wdf(df_pred, reducer.transform(arr(df_pred)), cols),
                                                 event_info, cols)
    # mismo ventaneo sobre las features crudas: da las secuencias que necesita XAI
    raw_obj, _ = windower.create_test_sequences(to_wdf(df_pred, arr(df_pred), feats), event_info, feats)

    def _seq(s):
        return {"event_id": s.event_id, "event_label": s.event_label, "asset_id": s.asset_id,
                "windows": s.windows.astype(np.float32), "labels": np.asarray(s.labels, int),
                "window_timestamps": s.window_timestamps}

    return train_windows, [_seq(s) for s in test_obj], [_seq(s) for s in raw_obj], feats, K, reducer.scaler, reducer.pca


def run_farm(farm: str, base_cfg: dict, pcfg_base: dict, device: str, raw_path: str,
             dump_artifacts: bool = False, dump_scores: bool = False) -> dict:
    # la config efectiva es la base con el bloque per_farm de esta granja fundido encima
    cfg = _deep_merge(base_cfg, base_cfg.get("per_farm", {}).get(farm, {}))
    order = cfg.get("order", "window_first")
    seed = cfg["reproducibility"]["random_seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    fl = farm.lower()
    logger.info(f"Farm {farm} | order={order}")

    t0 = time.time()
    logger.info("[1-3/6] Preprocessing + scale + PCA ...")
    if order == "window_first":
        train_windows, test_seqs, raw_test, feats, K, scaler_obj, pca_obj = _prep_window_first(farm, cfg, pcfg_base, seed)
    else:
        train_windows, test_seqs, raw_test, feats, K, scaler_obj, pca_obj = _prep_pca_first(farm, cfg, pcfg_base, seed)
    W = train_windows.shape[1]
    logger.info(f"  order={order}: {len(feats)} feats -> K={K} | train {train_windows.shape} | {time.time()-t0:.0f}s")

    if cfg["isolation_forest"].get("enabled", True):
        ifc = cfg["isolation_forest"]
        train_clean = IFCleaner(contamination=ifc["contamination"], n_estimators=ifc["n_estimators"],
                                random_state=ifc["random_state"], n_jobs=ifc["n_jobs"]).fit_transform(train_windows)
        logger.info(f"[4/6] IF cleaned: {len(train_windows):,} -> {len(train_clean):,}")
    else:
        train_clean = train_windows
        logger.info(f"[4/6] (IF disabled) train windows {len(train_clean):,}")

    ae_cfg = cfg["autoencoder"]
    # la salida tiene que poder ser negativa si el escalado es standard; con minmax basta relu
    out_act = "linear" if cfg["scaling"].get("method", "standard") == "standard" else "relu"
    model = BiLSTMAutoencoder(input_size=K, encoder_units=ae_cfg["encoder_units"],
                              decoder_units=ae_cfg["decoder_units"], dropout=ae_cfg["dropout"], out_activation=out_act)
    logger.info(f"[5/6] Training BiLSTM ({sum(p.numel() for p in model.parameters()):,} params) ...")
    cfg["score_mode"] = cfg.get("score_mode", "mae_window")
    Trainer(model=model, config=cfg, device=device).fit(train_clean, farm=farm)

    t0 = time.time()
    detector = AnomalyDetector(model=model, config=cfg, device=device)
    adapter = OfficialCAREAdapter(raw_path=raw_path, farm=farm, window_size=W)
    if dump_scores:
        dump_ae_scores(detector, train_clean, test_seqs, farm, Path(cfg["output"]["models_dir"]), seed)
    ev = evaluate(detector, train_clean, test_seqs, adapter, seed)
    best, cv = ev["best"], ev["cv"]
    logger.info(f"[6/6] Farm {farm}: K={K} | CARE-CV={cv['care']:.4f} (oracle={best['care']:.4f}, "
                f"optimismo={cv['optimism']:+.4f}, folds={cv['fold_h_scale']}) | {time.time()-t0:.0f}s")
    logger.info(f"  CARE={best['care']:.4f} F1={best['f1']:.4f} ROC-ds={best['roc_ds']:.4f} ROC-win={ev['roc_window']:.4f} "
                f"| Cov={best['cov']:.3f} Earl={best['earl']:.3f} EarlCond={best['earl_conditional']:.3f} "
                f"Rel={best['rel']:.3f} Acc={best['acc']:.3f}")

    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    models = Path(cfg["output"]["models_dir"])
    models.mkdir(parents=True, exist_ok=True)
    summary = {"farm": farm, "order": order, "n_features": len(feats), "K": int(K), "window_size": int(W),
               "threshold": ev["threshold"], "roc_window": ev["roc_window"],
               "care_cv": cv["care"], "care_oracle": best["care"], "optimism": cv["optimism"],
               "coverage_fbeta": best["cov"], "earliness_ws": best["earl"],
               "earliness_conditional": best["earl_conditional"], "reliability_efbeta": best["rel"],
               "accuracy": best["acc"], "f1": best["f1"], "auc_roc_dataset": best["roc_ds"],
               "auc_roc_window": ev["roc_window"], "best": best, "cv": cv, "sweep": ev["sweep"]}
    with open(tables / f"farm_{fl}_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=_np_default)

    # artefactos de explicabilidad: los transformadores ajustados y el test sin reducir, que
    # es el unico espacio donde una atribucion se puede leer como un sensor con nombre
    if dump_artifacts:
        with open(models / f"farm_{fl}_scaler.pkl", "wb") as f:
            pickle.dump(scaler_obj, f)
        with open(models / f"farm_{fl}_pca.pkl", "wb") as f:
            pickle.dump(pca_obj, f)
        (models / f"farm_{fl}_model_feature_cols.txt").write_text("\n".join(feats), encoding="utf-8")
        npz = {str(s["event_id"]): s["windows"] for s in raw_test}
        meta = {str(s["event_id"]): {"event_label": s["event_label"], "asset_id": int(s["asset_id"]),
                                     "labels": np.asarray(s["labels"], int).tolist(),
                                     "window_timestamps": [str(t) for t in np.asarray(s["window_timestamps"]).ravel()]}
                for s in raw_test}
        dproc = Path(cfg["data"]["processed_path"])
        dproc.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dproc / f"farm_{fl}_xai_test_sequences.npz", **npz)
        with open(dproc / f"farm_{fl}_xai_test_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"  [dump] scaler/pca/feature_cols + {len(npz)} raw test seqs saved")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", choices=["B", "C"], default=None)
    ap.add_argument("--raw-path", default="data/raw/CARE_To_Compare")
    ap.add_argument("--dump-artifacts", action="store_true")
    ap.add_argument("--dump-scores", action="store_true",
                    help="persiste los scores por ventana del AE y la meta de evento para la fusion")
    ap.add_argument("--seed", type=int, default=None,
                    help="sobrescribe reproducibility.random_seed, para las ejecuciones multi-semilla")
    ap.add_argument("--overwrite-frozen", action="store_true",
                    help="permite sobrescribir los artefactos congelados de results/models/")
    args = ap.parse_args()

    with open("config/model_config.yaml", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    if args.seed is not None:
        base_cfg["reproducibility"]["random_seed"] = args.seed
    base_cfg["output"]["overwrite_frozen"] = args.overwrite_frozen
    pcfg_base = load_preproc_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PyTorch {torch.__version__} | Device: {device}")

    farms = [args.farm] if args.farm else base_cfg["data"]["farms"]
    results = {}
    for farm in farms:
        results[farm] = run_farm(farm, base_cfg, pcfg_base, device, args.raw_path,
                                 args.dump_artifacts, args.dump_scores)

    logger.info(f"{'Farm':<6}{'order':<13}{'CARE-CV':>9}{'optim':>8}{'Cov':>7}{'Earl':>7}{'EarlC':>7}{'Rel':>7}{'Acc':>7}{'ROC-ds':>8}")
    for farm, r in results.items():
        logger.info(f"{farm:<6}{r['order']:<13}{r['care_cv']:>9.4f}{r['optimism']:>+8.4f}{r['coverage_fbeta']:>7.3f}"
                    f"{r['earliness_ws']:>7.3f}{r['earliness_conditional']:>7.3f}{r['reliability_efbeta']:>7.3f}"
                    f"{r['accuracy']:>7.3f}{r['auc_roc_dataset']:>8.4f}")
