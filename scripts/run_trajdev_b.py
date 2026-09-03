"""TrajDev sobre el mundo de valores de Farm B, con la receta que gano en Farm C.

la receta es identica a la de C y no se tunea nada en B: forecaster GRU de 3 miembros, de
ctx18 a H9, sobre los canales de valores escalados; el score es la departure mas z por
canal, con stats de train para que sea leakage-free, y max sobre los canales. de ahi salen
`dep_zmax_val`, su co-configuracion `dep_zpersist_val` y el diagnostico `dep_zslope_val`.

el mundo de valores de B son las 30 features curated crudas de `farm_b_train_windows.npz`
con escalado global ajustado en train y sin PCA, espejo de lo que en C es val__, el top-100
avg escalado. la razon de no meter PCA es que el espacio PCA del AE re-deriva al propio AE.
la higiene es la oficial de B: IF-clean de las ventanas de train, con los mismos parametros
que load_farm_b y random_state fijo a 42, antes de entrenar y de calibrar, porque en B el
modelo entrena sobre train_clean.

el scaler propio se ajusta sobre ese mismo train congelado, sin usar farm_b_scaler.pkl ni
farm_b_pca.pkl. la comparacion con el AE-B honesto es a nivel de CARE-CV, contra
ae_identical_b_s*.json y pareada por semilla sobre las comunes 42, 7 y 123.

    .venv/Scripts/python.exe scripts/run_trajdev_b.py --seeds 42,7,123,1234,12345 --dump-scores

escribe results/tables/trajdev_farm_b_gate_s{seed}[_tag].json, un centinela por semilla, y
con --dump-scores tambien results/models/trajdevb_scores_farm_b_s{seed}.npz.
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
import pandas as pd
import torch

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_model
from run_driftdiff_c import _keep_awake
from run_trajdev_c import train_forecaster
from sweep_timeaddm_scoring import ReplayScorer
from src.care_official.adapter import OfficialCAREAdapter
from src.model_prep.isolation_forest_cleaner import IFCleaner
from src.trajdev.model import predict_windows, dep_variants

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# baseline del AE-B por semilla, ya coherente con cv-seed 42; sale de los single_cares de
# fuse_multi sobre los dumps ident, no de los ae_identical_b_s*.json, que evaluaban con la
# cv-seed igual a la semilla de train y por tanto con splits distintos en cada una
AE_B_HONESTO = {42: 0.7211, 7: 0.6597, 123: 0.6410, 1234: 0.6430, 12345: 0.6627}


def prep_b(proc_dir: str = "data/processed") -> dict:
    """ventanas crudas de B, escalado global ajustado en train, sin PCA, y despues IF-clean.

    con un proc_dir distinto del de por defecto, que es la ablation de representacion, train,
    test y meta salen todos del mismo directorio y el conjunto es autoconsistente. ahi el
    ventaneo puede no coincidir con el curated, asi que la alineacion se comprueba contra las
    labels propias y no contra la meta del AE, y run_one_seed omite las corr por ventana."""

    z = np.load(f"{proc_dir}/farm_b_train_windows.npz", allow_pickle=True)
    w = z["windows"].astype(np.float32)  # (411249, 27, 30)
    feats = [str(c) for c in z["feature_cols"]]
    k = w.shape[2]
    # media y desviacion sobre todas las filas de todas las ventanas: un solo escalado global,
    # no uno por paso de tiempo. en float64 porque son 11 millones de filas por columna
    flat = w.reshape(-1, k)
    mu = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = (flat.std(axis=0, dtype=np.float64) + 1e-8).astype(np.float32)
    w -= mu
    w /= sd
    n0 = len(w)
    train = IFCleaner(contamination=0.01, n_estimators=100, random_state=42, n_jobs=-1).fit_transform(w)
    logger.info(f"prep B: {n0:,} ventanas -> IF-clean {len(train):,} | K={k} | scaler propio fit-en-train")

    # el test se escala con mu y sd de train, nunca con los suyos
    meta = json.load(open(f"{proc_dir}/farm_b_test_metadata.json", encoding="utf-8"))
    zt = np.load(f"{proc_dir}/farm_b_test_sequences.npz", allow_pickle=True)
    seqs = []
    for eid, m in meta.items():
        d = zt[eid].astype(np.float32)
        d -= mu
        d /= sd
        seqs.append({"event_id": int(eid), "asset_id": m["asset_id"], "event_label": m["event_label"],
                     "labels": np.asarray(m["labels"], int),
                     "window_timestamps": pd.to_datetime(m["window_timestamps"]),
                     "data": d})
    return {"train_data": train, "K": k, "ch_names": [f"val__{c}" for c in feats], "test_seqs": seqs,
            "mu": mu, "sd": sd}


def corr_refs(eids: list[int]) -> dict[str, np.ndarray]:
    """referencias para la correlacion al nivel del score por ventana.

    en B solo hay dumps de la semilla 42, la desplegada, asi que la referencia es la misma
    para todas las semillas. es informativa y no decide nada."""

    out = {}
    for name, path, key in (("ae", "results/models/ae_scores_farm_b.npz", "mae"),
                            ("crgn", "results/models/crgn_scores_farm_b.npz", "dev")):
        try:
            z = np.load(path)
            out[name] = np.concatenate([np.asarray(z[f"ev{e}_{key}"], float) for e in eids])
        except (FileNotFoundError, KeyError) as exc:
            # una ref que falta solo quita esa correlacion, no invalida la ejecucion
            logger.warning(f"ref {name} no disponible ({exc})")
    return out


def run_one_seed(prep, adapter, args, seed: int) -> dict:
    np.random.seed(seed)
    data = prep["train_data"]
    C, H = args.context, args.horizon
    W = data.shape[1]
    if C + H > W:
        raise ValueError(f"context+horizon {C}+{H} > W={W}")
    # sub-horizontes de las variantes de departure, capados al horizonte real
    sub_h = sorted({min(h, H) for h in (int(x) for x in args.sub_h.split(","))})

    t0 = time.time()
    models = [train_forecaster(data, args, seed, mi) for mi in range(args.n_models)]
    n_par = sum(p.numel() for p in models[0].parameters())
    logger.info(f"  train done ({args.n_models} modelo(s), {n_par:,} params) | {time.time() - t0:.0f}s")

    t0 = time.time()
    err_tr, dis_tr = predict_windows(models, data, C, H, DEVICE)
    ev_out = [predict_windows(models, s["data"], C, H, DEVICE) for s in prep["test_seqs"]]
    logger.info(f"  scoring done | {time.time() - t0:.0f}s")

    # familia z por canal con stats de train, leakage-free; en B la mascara val son todos los
    # canales. la mediana sobre el horizonte es la alternativa robusta a picos de la media
    agg_h = (lambda x: np.median(x, axis=1)) if args.agg_h == "median" else (lambda x: x.mean(axis=1))
    tr_var = dep_variants(err_tr, sub_h)
    a_mean_tr = agg_h(np.abs(err_tr))  # (N, K)
    mu_c = a_mean_tr.mean(axis=0)
    sd_c = a_mean_tr.std(axis=0) + 1e-8
    # slope del error en el horizonte, |err| final menos |err| inicial: una trayectoria que
    # diverge puntua antes de que el error llegue a ser grande
    a_slope_tr = np.abs(err_tr)[:, -1, :] - np.abs(err_tr)[:, 0, :]
    mu_s = a_slope_tr.mean(axis=0)
    sd_s = a_slope_tr.std(axis=0) + 1e-8
    # persistencia direccional, |media con signo del err|: la deriva pequena pero consistente
    # despega pronto en fallo, y en normal el signo alterna y se cancela
    a_pers_tr = np.abs(err_tr.mean(axis=1))
    mu_p = a_pers_tr.mean(axis=0)
    sd_p = a_pers_tr.std(axis=0) + 1e-8

    # pesos para los cuadernos de XAI: ensemble, stats leakage-free y el scaler de prep_b,
    # que aqui si hay que guardar porque no vive en ningun pkl
    if getattr(args, "save_model", False):
        outm = Path(f"results/models/trajdevb_model_farm_b_s{seed}.pt")
        torch.save({"state_dicts": [m.state_dict() for m in models],
                    "arch": {"k": int(data.shape[2]), "context": C, "horizon": H,
                             "hidden": args.hidden, "layers": args.layers, "n_models": args.n_models},
                    "z_stats": {"mu_c": mu_c, "sd_c": sd_c, "mu_s": mu_s, "sd_s": sd_s,
                                "mu_p": mu_p, "sd_p": sd_p},
                    "scaler": {"mu": prep.get("mu"), "sd": prep.get("sd")},
                    "sub_h": sub_h, "ch_names": prep.get("ch_names"),
                    "ema": args.ema, "agg_h": args.agg_h, "seed": seed}, outm)
        logger.info(f"  modelo -> {outm}")

    def z_family(a_mean: np.ndarray, a_slope: np.ndarray, a_pers: np.ndarray) -> dict[str, np.ndarray]:
        # z por canal con mu y sd de train, para comparar canales de escalas distintas
        z = (a_mean - mu_c) / sd_c
        zs = (a_slope - mu_s) / sd_s
        zp = (a_pers - mu_p) / sd_p
        # ordenados una vez: de aqui salen los agregadores de meseta top-k, de k=2 a k=5
        zsort = np.sort(z, axis=1)
        return {"dep_z": z.mean(axis=1),
                "dep_ztop2_val": zsort[:, -2:].mean(axis=1),
                "dep_z_top3": zsort[:, -3:].mean(axis=1),
                "dep_ztop4_val": zsort[:, -4:].mean(axis=1),
                "dep_ztop5_val": zsort[:, -5:].mean(axis=1),
                "dep_zmax_val": z.max(axis=1),
                "dep_zslope_val": zs.max(axis=1),
                "dep_zpersist_val": zp.max(axis=1)}

    tr_var.update(z_family(a_mean_tr, a_slope_tr, a_pers_tr))
    if args.n_models > 1:
        tr_var["dis"] = dis_tr
    ev_var: dict[str, list[np.ndarray]] = {k: [] for k in tr_var}
    for err_e, dis_e in ev_out:
        v = dep_variants(err_e, sub_h)
        v.update(z_family(agg_h(np.abs(err_e)),
                          np.abs(err_e)[:, -1, :] - np.abs(err_e)[:, 0, :],
                          np.abs(err_e.mean(axis=1))))
        if args.n_models > 1:
            v["dis"] = dis_e
        for k in ev_var:
            ev_var[k].append(v[k])

    eids = [int(s["event_id"]) for s in prep["test_seqs"]]
    first = next(iter(ev_var))
    default_dir = (args.proc_dir == "data/processed")
    if default_dir:  # curated, comprobacion fuerte contra el ventaneo del AE, apples-to-apples
        meta = json.load(open("results/models/ae_scores_farm_b_meta.json", encoding="utf-8"))
        ref_len = {int(e): len(meta[str(e)]["labels"]) for e in eids}
    else:  # ablation, el ventaneo difiere del AE: autoconsistencia con las labels propias
        ref_len = {int(s["event_id"]): len(s["labels"]) for s in prep["test_seqs"]}
    for i, eid in enumerate(eids):
        if len(ev_var[first][i]) != ref_len[eid]:
            raise RuntimeError(f"desalineado ev{eid}: {len(ev_var[first][i])} vs {ref_len[eid]}")
    refs = corr_refs(eids) if default_dir else {}  # fuera de curated no alinean por ventana
    # el evaluador solo necesita labels y timestamps: las ventanas van vacias porque el
    # ReplayScorer ya trae los scores calculados
    slim = [{**{k: s[k] for k in ("event_id", "event_label", "asset_id", "labels", "window_timestamps")},
             "windows": np.empty(0)} for s in prep["test_seqs"]]

    ae_ref = AE_B_HONESTO.get(seed)
    rows = []
    for name, tr_sc in tr_var.items():
        evs = ev_var[name]
        rs = ReplayScorer([tr_sc] + evs)
        ev = run_model.evaluate(rs, np.empty(0), slim, adapter, 42)
        b, cv = ev["best"], ev["cv"]
        cat = np.concatenate(evs)
        corr = {k: float(np.corrcoef(cat, v)[0, 1]) for k, v in refs.items()}
        # comparacion pareada por semilla contra el AE-B honesto de esa misma semilla
        beat = (cv["care"] > ae_ref) if ae_ref is not None else None
        rows.append({"variant": name, "care_cv": cv["care"], "oracle": b["care"], "roc_ds": b["roc_ds"],
                     "cov": b["cov"], "earl": b["earl"], "rel": b["rel"], "acc": b["acc"],
                     **{f"corr_{k}": v for k, v in corr.items()},
                     "ae_b_honesto_seed": ae_ref, "beats_ae_seed": beat})
        logger.info(f"  {name:<16} CARE-CV={cv['care']:.4f} ROC={b['roc_ds']:.3f} Earl={b['earl']:.3f} "
                    f"Acc={b['acc']:.3f} | corr ae={corr.get('ae', float('nan')):+.3f} "
                    f"crgn={corr.get('crgn', float('nan')):+.3f}"
                    + (f" | vs AE-B s{seed} {ae_ref}: {'GANA' if beat else 'pierde'}" if ae_ref else ""))

    if args.dump_scores:
        arrays = {}
        for name in tr_var:
            arrays[f"train_{name}"] = np.asarray(tr_var[name], np.float64)
            for eid, sc in zip(eids, ev_var[name]):
                arrays[f"ev{eid}_{name}"] = np.asarray(sc, np.float64)
        out = Path(f"results/models/trajdevb_scores_farm_b_s{seed}.npz")
        np.savez_compressed(out, **arrays)
        logger.info(f"  dump -> {out} ({len(arrays)} arrays)")

    return {"seed": seed, "context": C, "horizon": H, "n_models": args.n_models,
            "hidden": args.hidden, "params": int(n_par), "ema": args.ema, "agg_h": args.agg_h,
            "rows": rows}


def main() -> None:
    _keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--context", type=int, default=18)
    ap.add_argument("--horizon", type=int, default=9)
    ap.add_argument("--sub-h", default="3,6,9")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=3e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--train-cap", type=int, default=250000)
    ap.add_argument("--n-models", type=int, default=3)
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dump-scores", action="store_true")
    ap.add_argument("--save-model", action="store_true",
                    help="guarda ensemble+stats+scaler en results/models/trajdevb_model_farm_b_s{seed}.pt")
    ap.add_argument("--proc-dir", default="data/processed",
                    help="directorio processed; !=default = ablation de representacion (p.ej. all-avg)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="palanca L2: EWMA causal in-window sobre los valores (0 = off)")
    ap.add_argument("--agg-h", default="mean", choices=["mean", "median"],
                    help="palanca L3: agregador de |err| sobre el horizonte para la familia z")
    args = ap.parse_args()
    if args.smoke:
        args.train_cap, args.epochs = 80000, 3

    prep = prep_b(args.proc_dir)
    # suavizado opcional: EWMA causal a lo largo de la ventana, aplicada igual a train y test
    if args.ema > 0:
        def _ema_inplace(x: np.ndarray, a: float) -> None:
            for t in range(1, x.shape[1]):
                x[:, t] = a * x[:, t] + (1 - a) * x[:, t - 1]
        _ema_inplace(prep["train_data"], args.ema)
        for s in prep["test_seqs"]:
            _ema_inplace(s["data"], args.ema)
        logger.info(f"palanca L2: EWMA in-window alpha={args.ema} aplicada a train+test")
    adapter = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm="B", window_size=27)
    logger.info(f"TrajDev-B | K={prep['K']} W={prep['train_data'].shape[1]} | ctx={args.context} "
                f"H={args.horizon} | train {prep['train_data'].shape} | {len(prep['test_seqs'])} eventos "
                f"| {DEVICE}")
    logger.info("refs: AE-B honesto cv42 0.6655±0.029 (s42 .7211 / s7 .6597 / s123 .6410 / s1234 .6430 "
                "/ s12345 .6627) | deployed 0.7211 | CRGN-B 0.6514")

    tag = f"_{args.exp_tag}" if args.exp_tag else ""
    for seed in [int(s) for s in args.seeds.split(",")]:
        logger.info(f"=== seed {seed} ===")
        res = run_one_seed(prep, adapter, args, seed)
        out = Path(f"results/tables/trajdev_farm_b_gate_s{seed}{tag}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out, "w"), indent=2, default=run_model._np_default)
        logger.info(f"saved -> {out}")


if __name__ == "__main__":
    main()
