"""atlas de senales de dominio model-free para Farm C, seed-42, en CPU.

mapea el espacio no-avg (238 sensores x {std,max,min}, virgen para ae, crgn y fisica y
verificado en la cabecera del CSV) mas extensiones de coherencia, y mide antes de invertir:
el CARE-CV de seed-42 en solitario y la correlacion al nivel del score CUSUM contra ae, crgn
y phys95maxrate. solo pasan al dump de 5 semillas y a la fusion las senales con perfil de
campeon, correlacion baja contra los tres y fuerza >=~0.65.

familias, todas con la receta del campeon, z leakage-free del train mas agg por ventana mas agg
entre canales:

    F1a std__*    : burst de varianza intra-10min del canal _std. la varianza sube antes que la
                    media, precursor clasico de gearbox/NBM, candidata a Earliness ortogonal
    F1b rng__*    : rango intra-10min (_max - _min), transitorios y extremos
    F2  asym__*   : coherencia intra-sensor (_max - _avg), rotura de la relacion entre la media
                    y el extremo
    F3  sdp__*    : coherencia de pares redundantes (>=0.95) sobre el canal _std; el gemelo
                    cuya varianza se separa del par es fallo local, extension de phys95
    F4  grp__*    : residuo a la mediana de grupos redundantes de >=3 miembros (avg,
                    pares>=0.90)
    F5  pav rate2 : segunda derivada del residuo de pares avg, mas alla de maxrate
    sanity pav    : reproduce phys95max y phys95maxrate desde los canales pav, valida el prep y
                    la alineacion contra results/models/phys_scores_farm_c_s42.npz

el prep es espejo de run_gdn._load_c, con los mismos filtros, orden, scaler y Windower, para
que la alineacion coincida con la meta del AE, y con el keep ampliado a las 4 stats de los
sensores top100. los canales se computan sobre filas escaladas y lo que se ventanea son los
canales, no las columnas crudas. los pares se descubren data-driven sobre el Spearman del
train normal (subsample de 200k, rng 42) a nivel de fila, y vale porque Spearman es rank,
invariante a escala y equivalente a descubrirlos sobre ventanas. cache propia, gitignored,
en data/processed/experiments/atlas_c_st8.*

    .venv/Scripts/python.exe scripts/signal_atlas.py [--rebuild] [--top-n 20]

escribe results/tables/signal_atlas_c.json, el centinela, ademas de la consola.
"""
from __future__ import annotations

import argparse
import ctypes
import io
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_model
from sweep_timeaddm_scoring import ReplayScorer
from src.care_official.adapter import OfficialCAREAdapter
from src.crgn.graph import spearman_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT = Path("results/tables/signal_atlas_c.json")
STRIDE_TRAIN = 8  # mismo que la via CRGN/GDN/phys de C, crgn_config per_farm C
VAL_RATIO = 0.2  # idem; leakage-free, stats del train-split y calibracion sobre el val-split
PAIR_ROWS = 200000  # subsample de filas de train para el Spearman, = graph.max_rows de CRGN


def cache_paths(tag: str = "") -> tuple[Path, Path]:
    """cache por variante, '' es el atlas original de 153 canales y '3' la v3, que
    anade los canales val__ y el panel ampliado."""
    base = Path(f"data/processed/experiments/atlas{tag}_c_st8")
    return base.with_suffix(".npz"), Path(str(base) + ".test.pkl")


CACHE, CACHE_TEST = cache_paths("")  # compat con quien llama a load_cache() sin tag


# evita que Windows suspenda la maquina durante el build de la cache, que dura ~25 min
def keep_awake() -> None:
    if sys.platform == "win32":
        # CONTINUOUS | SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)


def _components(pairs: list[tuple[str, str, float]]) -> list[list[str]]:
    """componentes conexas del grafo de pares, con union-find plano."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[str, list[str]] = {}
    for n in parent:
        groups.setdefault(find(n), []).append(n)
    return [sorted(g) for g in groups.values()]


def build_cache(top_n: int, with_val: bool = False, tag: str = "", save_rows: bool = False) -> None:
    """espejo de run_gdn._load_c con el keep ampliado a las 4 stats; lo que se ventanea son
    los canales, no las columnas crudas."""
    cache, cache_test = cache_paths(tag)
    from sklearn.preprocessing import StandardScaler
    from src.preprocessing.config import load_config as load_preproc_config
    from src.preprocessing.data_loader import CareDataLoader
    from src.preprocessing.zero_handler import ZeroHandler
    from src.preprocessing.feature_selector import FeatureSelector
    from src.preprocessing.status_filter import StatusFilter
    from src.preprocessing.power_curve_filter import PowerCurveFilter
    from src.preprocessing.transition_excluder import TransitionExcluder
    from src.preprocessing.windower import Windower

    # la config de C se superpone sobre la de la raiz, igual que hace run_gdn._load_c
    pcfg = run_model._deep_merge(load_preproc_config(), load_preproc_config().get("per_farm", {}).get("C", {}))
    loader = CareDataLoader(pcfg)
    df, event_info, reg = loader.load_farm("C")
    df, _ = ZeroHandler(pcfg, reg).transform(df)
    normal = pcfg["status_filtering"]["normal_statuses"]
    dfn = df[df["status_type_id"].isin(normal)]
    feats_all, _ = FeatureSelector(pcfg, reg).select(dfn)
    del dfn
    sub = set(Path("config/farm_c_top100_features.txt").read_text(encoding="utf-8").strip().splitlines())
    feats = [c for c in feats_all if c in sub]  # orden del selector, exacto a _load_c
    logger.info(f"top100 en orden selector: {len(feats)}")
    # keep = META + top100_avg + las 4 stats de esos sensores (espacio no-avg) + anclas viento
    bases = [f[: -len("_avg")] for f in feats]
    extra = [f"{b}_{s}" for b in bases for s in ("std", "max", "min") if f"{b}_{s}" in df.columns]
    keep = [c for c in run_model.META if c in df.columns] + feats + extra
    keep += [c for c in df.columns if c.endswith("_avg") and "wind_speed" in c and c not in keep]
    df = df[keep].copy()
    logger.info(f"filas {len(df)} | columnas kept {len(keep)} (top100 + {len(extra)} stats no-avg)")

    df_train = df[df["train_test"] == "train"].copy()
    df_train, _ = StatusFilter(pcfg).transform(df_train)
    df_train, _ = PowerCurveFilter(pcfg, reg).transform(df_train)
    df_train, _ = TransitionExcluder(pcfg).transform(df_train, event_info)
    if pcfg["data"].get("dedup_train", True):
        df_train = df_train.drop_duplicates(["asset_id", "time_stamp"]).reset_index(drop=True)
    df_pred = df[df["train_test"] == "prediction"].copy().reset_index(drop=True)
    del df

    cols = feats + extra  # columnas numericas escaladas
    col_means = df_train[cols].mean()

    def arr(src: pd.DataFrame) -> np.ndarray:
        return src[cols].fillna(col_means).fillna(0.0).to_numpy(np.float32)

    # el scaler se ajusta solo sobre train y luego se aplica a pred, para no filtrar el futuro
    a_train = arr(df_train)
    scaler = StandardScaler().fit(a_train)
    s_train = scaler.transform(a_train).astype(np.float32)
    del a_train
    s_pred = scaler.transform(arr(df_pred)).astype(np.float32)
    cidx = {c: i for i, c in enumerate(cols)}

    # pares redundantes data-driven sobre filas de train escaladas
    # Spearman es rank, asi que la escala de cada columna da igual
    rng = np.random.default_rng(42)
    idx = rng.choice(len(s_train), size=min(PAIR_ROWS, len(s_train)), replace=False)
    avg_idx = [cidx[f] for f in feats]
    sp = np.abs(spearman_matrix(s_train[idx][:, avg_idx]))
    np.fill_diagonal(sp, 0.0)
    iu = np.triu_indices(len(feats), 1)
    pairs90 = sorted([(bases[i], bases[j], float(sp[i, j]))
                      for i, j in zip(*iu) if sp[i, j] >= 0.90], key=lambda p: -p[2])
    pairs95 = [p for p in pairs90 if p[2] >= 0.95]
    logger.info(f"pares |Spearman|>=0.90: {len(pairs90)} | >=0.95: {len(pairs95)}")
    for a, b, w in pairs95:
        logger.info(f"  par95 {a} ~ {b}  |r|={w:.3f}")

    # canales, de nombre a vector por fila
    top_bases = bases[:top_n]
    pair_bases = sorted({b for a, b2, _ in pairs90 for b in (a, b2)})
    sel = list(dict.fromkeys(top_bases + pair_bases))  # union ordenada, sin duplicados
    logger.info(f"sensores para canales: {len(sel)} (top-{top_n} ∪ {len(pair_bases)} de pares90)")

    def col(name: str, src: np.ndarray) -> np.ndarray:
        return src[:, cidx[name]]

    # with_val (v3) suma los canales val__{sensor}, el avg escalado de todos los top100, que
    # mete el mundo del AE dentro del mismo score
    def build_channels(src: np.ndarray) -> tuple[np.ndarray, list[str]]:
        chs: list[np.ndarray] = []
        names: list[str] = []

        def add(name: str, v: np.ndarray) -> None:
            names.append(name)
            chs.append(v.astype(np.float32))

        for b in sel:
            if f"{b}_std" in cidx:
                add(f"std__{b}", col(f"{b}_std", src))
            if f"{b}_max" in cidx and f"{b}_min" in cidx:
                add(f"rng__{b}", col(f"{b}_max", src) - col(f"{b}_min", src))
            if f"{b}_max" in cidx:
                add(f"asym__{b}", col(f"{b}_max", src) - col(f"{b}_avg", src))
        for a, b, _w in pairs95:
            if f"{a}_std" in cidx and f"{b}_std" in cidx:
                add(f"sdp__{a}__{b}", col(f"{a}_std", src) - col(f"{b}_std", src))
            add(f"pav__{a}__{b}", col(f"{a}_avg", src) - col(f"{b}_avg", src))
        for gi, grp in enumerate(g for g in _components(pairs90) if len(g) >= 3):
            med = np.median(np.stack([col(f"{m}_avg", src) for m in grp], axis=1), axis=1)
            for m in grp:
                add(f"grp__{gi}__{m}", col(f"{m}_avg", src) - med)
        if with_val:  # v3, el mundo del AE dentro del panel
            for b in bases:
                add(f"val__{b}", col(f"{b}_avg", src))
        return np.stack(chs, axis=1), names

    ch_train, ch_names = build_channels(s_train)
    ch_pred, _ = build_channels(s_pred)
    del s_train, s_pred
    logger.info(f"canales: {len(ch_names)}")

    # save_rows (v4, horizonte largo) vuelca ademas las filas de canales, el train con asset y
    # time para poder segmentarlo y el pred por evento; secuencias mas largas que W
    if save_rows:
        rows_path = Path(f"data/processed/experiments/atlas{tag}_c_rows.npz")
        np.savez_compressed(
            rows_path,
            ch_names=np.array(ch_names, dtype=object),
            train_rows=ch_train.astype(np.float32),
            train_asset=df_train["asset_id"].to_numpy(np.int64),
            train_time=df_train["time_stamp"].to_numpy("datetime64[ns]").astype(np.int64),
            pred_rows=ch_pred.astype(np.float32),
            pred_event=df_pred["event_id"].to_numpy(np.int64),
            pred_asset=df_pred["asset_id"].to_numpy(np.int64),
            pred_time=df_pred["time_stamp"].to_numpy("datetime64[ns]").astype(np.int64))
        logger.info(f"filas de canales -> {rows_path}")

    def to_wdf(meta_src: pd.DataFrame, ch: np.ndarray) -> pd.DataFrame:
        meta = meta_src[[c for c in run_model.META if c in meta_src.columns]].reset_index(drop=True)
        return pd.concat([meta, pd.DataFrame(ch, columns=ch_names)], axis=1)

    # train con stride 8 para abaratar; test con stride 1, que el CARE puntua ventana a ventana
    wtrain = Windower({"windowing": {**pcfg["windowing"], "stride_samples": STRIDE_TRAIN}})
    wtest = Windower({"windowing": {**pcfg["windowing"], "stride_samples": 1}})
    train_windows, _, _ = wtrain.create_train_windows(to_wdf(df_train, ch_train), ch_names)
    test_obj, _ = wtest.create_test_sequences(to_wdf(df_pred, ch_pred), event_info, ch_names)
    test_seqs = [{"event_id": s.event_id, "event_label": s.event_label, "asset_id": s.asset_id,
                  "windows": s.windows.astype(np.float32), "labels": np.asarray(s.labels, np.int8),
                  "window_timestamps": s.window_timestamps} for s in test_obj]
    train_windows = train_windows.astype(np.float32)
    logger.info(f"train windows {train_windows.shape} (stride {STRIDE_TRAIN}) | {len(test_seqs)} eventos")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, train_windows=train_windows, ch_names=np.array(ch_names, dtype=object),
                        pairs90=np.array([f"{a}|{b}|{w}" for a, b, w in pairs90], dtype=object))
    with open(cache_test, "wb") as f:
        pickle.dump(test_seqs, f)
    logger.info(f"cache -> {cache} + {cache_test}")


def load_cache(tag: str = ""):
    cache, cache_test = cache_paths(tag)
    z = np.load(cache, allow_pickle=True)
    tw = z["train_windows"]
    ch_names = [str(c) for c in z["ch_names"]]
    pairs90 = []
    for s in z["pairs90"]:
        a, b, w = str(s).split("|")
        pairs90.append((a, b, float(w)))
    with open(cache_test, "rb") as f:
        test_seqs = pickle.load(f)
    return tw, ch_names, pairs90, test_seqs


# scoring, z leakage-free por canal, agg de ventana {level,rate,rate2}x{max} y agg de canales

def window_dev(w: np.ndarray, mu: np.ndarray, sd: np.ndarray, mode: str) -> np.ndarray:
    """de w (N,27,C) a dev (N,C), el max sobre la ventana de |z| del canal en modo level
    o de su derivada primera o segunda en rate y rate2."""
    if mode == "level":
        x = w
    elif mode == "rate":
        x = np.diff(w, axis=1)
    elif mode == "rate2":
        x = np.diff(np.diff(w, axis=1), axis=1)
    else:
        raise ValueError(mode)
    return (np.abs(x - mu) / sd).max(axis=1)


def rate_stats(tw: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """mu y sd por canal del train-split, en el mismo espacio que window_dev, level, rate
    o rate2."""
    if mode == "level":
        x = tw
    elif mode == "rate":
        x = np.diff(tw, axis=1)
    else:
        x = np.diff(np.diff(tw, axis=1), axis=1)
    flat = x.reshape(-1, x.shape[2])
    return flat.mean(axis=0), flat.std(axis=0) + 1e-8


def agg_channels(dev: np.ndarray, how: str, w: np.ndarray | None = None) -> np.ndarray:
    """de dev (N,C) a score (N,), agregando entre canales con mean, top3 o wmean de |corr|."""
    if how == "mean":
        return dev.mean(axis=1)
    if how == "top3":
        k = min(3, dev.shape[1])
        return np.sort(dev, axis=1)[:, -k:].mean(axis=1)
    if how == "wmean":
        return (dev * w).sum(axis=1) / max(float(w.sum()), 1e-8)
    raise ValueError(how)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-tag", default="", help="'' = atlas original; '3' = v3 (val + panel ampliado)")
    ap.add_argument("--with-val", action="store_true", help="v3: canales val__ (avg escalado top100)")
    ap.add_argument("--save-rows", action="store_true", help="v4: volcar tambien FILAS de canales")
    ap.add_argument("--build-only", action="store_true", help="solo construir la cache y salir")
    args = ap.parse_args()
    keep_awake()

    cache, cache_test = cache_paths(args.cache_tag)
    if args.rebuild or not (cache.exists() and cache_test.exists()):
        logger.info(f"construyendo cache del atlas{args.cache_tag or ''} (~25 min, una vez) ...")
        build_cache(args.top_n, with_val=args.with_val, tag=args.cache_tag, save_rows=args.save_rows)
    if args.build_only:
        logger.info("build-only: hecho")
        return
    tw, ch_names, pairs90, test_seqs = load_cache(args.cache_tag)
    pairs95 = [p for p in pairs90 if p[2] >= 0.95]
    logger.info(f"cache: train {tw.shape} | {len(ch_names)} canales | {len(test_seqs)} eventos | "
                f"pares95 {len(pairs95)}")

    # split leakage-free identico a dump_phys, stats del train-split y calibracion en val
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tw))
    n_val = int(round(len(tw) * VAL_RATIO))
    tw_val, tw_fit = tw[perm[:n_val]], tw[perm[n_val:]]
    del tw

    # cada familia agrupa sus canales por el prefijo del nombre, nunca por posicion
    fam_idx = {pref: [i for i, c in enumerate(ch_names) if c.startswith(pref)]
               for pref in ("std__", "rng__", "asym__", "sdp__", "pav__", "grp__")}
    for pref, idxs in fam_idx.items():
        logger.info(f"  {pref:<7} {len(idxs)} canales")
    # los pesos |corr| se leen del nombre del canal, sdp__{a}__{b} y pav__{a}__{b}, no del
    # orden de las columnas
    wdict = {(a, b): w for a, b, w in pairs95}

    def fam_weights(pref: str) -> np.ndarray:
        names = [ch_names[i] for i in fam_idx[pref]]
        return np.array([wdict[tuple(n.split("__")[1:3])] for n in names], dtype=np.float64)

    pw_sdp = fam_weights("sdp__") if fam_idx["sdp__"] else None
    pw_pav = fam_weights("pav__") if fam_idx["pav__"] else None

    signals: list[tuple[str, str, str, str, np.ndarray | None]] = [
        # (nombre, prefijo, agg canales, modo ventana, pesos)
        ("std_max_mean",    "std__",  "mean",  "level", None),
        ("std_max_top3",    "std__",  "top3",  "level", None),
        ("std_rate_mean",   "std__",  "mean",  "rate",  None),
        ("std_rate_top3",   "std__",  "top3",  "rate",  None),
        ("rng_max_mean",    "rng__",  "mean",  "level", None),
        ("rng_rate_mean",   "rng__",  "mean",  "rate",  None),
        ("asym_max_mean",   "asym__", "mean",  "level", None),
        ("asym_rate_mean",  "asym__", "mean",  "rate",  None),
        ("sdp_max_wmean",   "sdp__",  "wmean", "level", pw_sdp),
        ("sdp_rate_wmean",  "sdp__",  "wmean", "rate",  pw_sdp),
        ("pav_rate2_wmean", "pav__",  "wmean", "rate2", pw_pav),
        ("sanity_phys95max",     "pav__", "wmean", "level", pw_pav),
        ("sanity_phys95maxrate", "pav__", "wmean", "rate",  pw_pav),
    ]
    if fam_idx["grp__"]:
        signals += [("grp_max_mean", "grp__", "mean", "level", None),
                    ("grp_rate_mean", "grp__", "mean", "rate", None)]

    # referencias para la correlacion al nivel CUSUM, seed-42; se concatenan en el mismo orden
    # de eventos que los scores de cada senal, que es lo que hace comparables los dos vectores
    meta = json.load(open("results/models/ae_scores_farm_c_meta.json", encoding="utf-8"))
    eids = [int(s["event_id"]) for s in test_seqs]
    refs = {}
    for name, path, key in (("ae", "results/models/ae_scores_farm_c.npz", "mae"),
                            ("crgn", "results/models/crgn_scores_farm_c.npz", "dev"),
                            ("phys", "results/models/phys_scores_farm_c_s42.npz", "phys95maxrate")):
        z = np.load(path)
        refs[name] = np.concatenate([np.asarray(z[f"ev{e}_{key}"], float) for e in eids])

    adapter = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm="C", window_size=27)
    # el evaluador solo mira labels y timestamps, las ventanas van vacias y no se copian
    slim = [{**{k: s[k] for k in ("event_id", "event_label", "asset_id", "labels", "window_timestamps")},
             "windows": np.empty(0)} for s in test_seqs]
    logger.info("refs: AE-honesto 0.7206 | phys95maxrate solo ~0.674 (Earl 0.72, corr 0.034/0.040) | "
                "campeon fusion 0.7694")

    rows = []
    for name, pref, how, mode, w in signals:
        idxs = fam_idx[pref]
        if not idxs:
            continue
        mu, sd = rate_stats(tw_fit[:, :, idxs], mode)
        tr_sc = agg_channels(window_dev(tw_val[:, :, idxs], mu, sd, mode), how, w)
        ev_sc = []
        for s in test_seqs:
            sc = agg_channels(window_dev(s["windows"][:, :, idxs], mu, sd, mode), how, w)
            if len(sc) != len(meta[str(s["event_id"])]["labels"]):
                raise RuntimeError(f"desalineado ev{s['event_id']}: {len(sc)} vs meta")
            ev_sc.append(sc)
        # el ReplayScorer sirve scores ya calculados, primero el de train y luego cada evento
        rs = ReplayScorer([tr_sc] + ev_sc)
        ev = run_model.evaluate(rs, np.empty(0), slim, adapter, 42)
        b, cv = ev["best"], ev["cv"]
        cat = np.concatenate(ev_sc)
        corr = {k: float(np.corrcoef(cat, v)[0, 1]) for k, v in refs.items()}
        row = {"signal": name, "family": pref.rstrip("_"), "n_ch": len(idxs), "care_cv": cv["care"],
               "oracle": b["care"], "roc_ds": b["roc_ds"], "cov": b["cov"], "earl": b["earl"],
               "rel": b["rel"], "acc": b["acc"], "corr_ae": corr["ae"], "corr_crgn": corr["crgn"],
               "corr_phys": corr["phys"]}
        rows.append(row)
        logger.info(f"  {name:<22} CARE-CV={cv['care']:.4f} ROC={b['roc_ds']:.3f} Earl={b['earl']:.3f} "
                    f"Acc={b['acc']:.3f} | corr ae={corr['ae']:+.3f} crgn={corr['crgn']:+.3f} "
                    f"phys={corr['phys']:+.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"farm": "C", "seed": args.seed, "stride_train": STRIDE_TRAIN,
               "pairs95": [f"{a}~{b}:{w:.3f}" for a, b, w in pairs95],
               "n_pairs90": len(pairs90), "rows": rows}, open(OUT, "w"), indent=2)
    logger.info(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
