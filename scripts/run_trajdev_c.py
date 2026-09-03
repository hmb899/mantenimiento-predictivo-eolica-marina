"""earliness-first por desviacion de trayectoria en Farm C, camino pca-first W=27 K=55.

forecaster GRU sobre los componentes PCA estandarizados, con el prep de
run_driftdiff_c.prep_c y su cache de filas reutilizada: contexto de C pasos y pronostico
directo de los H siguientes dentro de la misma ventana, con C+H<=27. el score por ventana
es la desviacion de la trayectoria respecto al pronostico, en las variantes de
src/trajdev/model.py. la evaluacion es identica a la del resto del proyecto, ReplayScorer
mas run_model.evaluate con cv-seed 42, y ademas se mide la correlacion al nivel CUSUM
contra ae, crgn y phys95maxrate, que es lo que dice si el score aporta algo a la fusion.

    .venv/Scripts/python.exe scripts/run_trajdev_c.py --seeds 42 [--context 18 --horizon 9]
    [--n-models 3]     anade la variante `dis`, desacuerdo de ensemble al estilo FATE
    [--dump-scores]    results/models/trajdev_scores_farm_c_s{seed}.npz, para la fusion

escribe results/tables/trajdev_farm_c_gate_s{seed}[_tag].json, un centinela por semilla.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_model
from run_driftdiff_c import prep_c, _keep_awake
from sweep_timeaddm_scoring import ReplayScorer
from src.care_official.adapter import OfficialCAREAdapter
from src.trajdev.model import TrajForecaster, predict_windows, dep_variants

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_forecaster(data: np.ndarray, args, seed: int, model_idx: int) -> TrajForecaster:
    """entrena un forecaster de ctx a H con early stopping en val, el 10% del subset capado."""

    # semilla propia de cada miembro del ensemble, para que los modelos no salgan iguales
    torch.manual_seed(seed * 1000 + model_idx)
    rng = np.random.default_rng(seed * 1000 + model_idx)
    n = len(data)
    cap = args.train_cap or n
    # el subset se capa a train_cap y se pide un 10% de mas, que es el que se aparta para val
    work = rng.permutation(n)[:min(cap + int(0.1 * cap) + 1, n)]
    nval = max(1, int(0.1 * len(work)))
    val_idx, tr_idx = work[:nval], work[nval:]
    C, H = args.context, args.horizon
    model = TrajForecaster(data.shape[2], C, H, hidden=args.hidden, layers=args.layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    crit = torch.nn.MSELoss()
    bs = args.batch
    xv = torch.from_numpy(np.ascontiguousarray(data[val_idx][:, :C, :], dtype=np.float32)).to(DEVICE)
    yv = torch.from_numpy(np.ascontiguousarray(data[val_idx][:, C:C + H, :], dtype=np.float32)).to(DEVICE)
    # val entra entero en GPU una sola vez; el train va por lotes
    best, best_state, ctr = float("inf"), None, 0
    for e in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(tr_idx)
        tot, nb = 0.0, 0
        for i in range(0, len(tr_idx), bs):
            idx = tr_idx[i:i + bs]
            xb = torch.from_numpy(np.ascontiguousarray(data[idx][:, :C, :], dtype=np.float32)).to(DEVICE)
            yb = torch.from_numpy(np.ascontiguousarray(data[idx][:, C:C + H, :], dtype=np.float32)).to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        model.eval()
        with torch.no_grad():
            vl = float(crit(model(xv), yv))
        logger.info(f"  m{model_idx} epoch {e}: train {tot / max(nb, 1):.5f} | val {vl:.5f}")
        # la mejora tiene que superar min_delta para contar; si no, suma paciencia
        if vl < best - args.min_delta:
            best, ctr = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            ctr += 1
            if ctr >= args.patience:
                logger.info(f"  m{model_idx} early stop @ {e} (best val {best:.5f})")
                break
    # se devuelve el mejor estado por val, no el ultimo
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def corr_refs(seed: int, eids: list[int]) -> dict[str, np.ndarray]:
    # el dump de la semilla 42 es el desplegado y va sin sufijo; el resto lo llevan
    sfx = "" if seed == 42 else f"_s{seed}"
    out = {}
    for name, path, key in (("ae", f"results/models/ae_scores_farm_c{sfx}.npz", "mae"),
                            ("crgn", f"results/models/crgn_scores_farm_c{sfx}.npz", "dev"),
                            ("phys", f"results/models/phys_scores_farm_c_s{seed}.npz", "phys95maxrate")):
        try:
            z = np.load(path)
            out[name] = np.concatenate([np.asarray(z[f"ev{e}_{key}"], float) for e in eids])
        except FileNotFoundError:
            # una ref que falta solo quita esa correlacion, no invalida la ejecucion
            logger.warning(f"ref {name} no disponible ({path})")
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

    # variantes por ventana; dep_z usa stats del |error| de train, leakage-free
    tr_var = dep_variants(err_tr, sub_h)
    a_mean_tr = np.abs(err_tr).mean(axis=1)  # (N, K), una sola pasada sobre el horizonte
    mu_c = a_mean_tr.mean(axis=0)
    sd_c = a_mean_tr.std(axis=0) + 1e-8

    # con la repr atlas los canales son heterogeneos, asi que el score va z por canal y se
    # agrega por familia; dep_z_pav es la trayectoria de la coherencia pura, el hibrido fisico
    ch_names = prep.get("ch_names")
    masks: dict[str, np.ndarray] = {}
    if ch_names:
        masks["pav"] = np.array([c.startswith(("pav__", "sdp__")) for c in ch_names])
        masks["sens"] = np.array([c.startswith(("std__", "rng__", "asym__")) for c in ch_names])
        if any(c.startswith("val__") for c in ch_names):  # el mundo del AE dentro del panel
            masks["val"] = np.array([c.startswith("val__") for c in ch_names])
            masks["all"] = np.ones(len(ch_names), dtype=bool)  # dep_zmax_all, todos

    # slope del error dentro del horizonte, |err| final menos |err| inicial por canal: una
    # trayectoria que diverge puntua antes de que el error llegue a ser grande
    a_slope_tr = np.abs(err_tr)[:, -1, :] - np.abs(err_tr)[:, 0, :]  # (N, K)
    mu_s = a_slope_tr.mean(axis=0)
    sd_s = a_slope_tr.std(axis=0) + 1e-8
    # persistencia direccional, |media con signo del err| por canal: la deriva pequena pero
    # consistente despega pronto en fallo, mientras que en normal el signo alterna y se
    # cancela, lo que la convierte en un supresor natural de falsas alarmas
    a_pers_tr = np.abs(err_tr.mean(axis=1))  # (N, K), err con signo promediado sobre H
    mu_p = a_pers_tr.mean(axis=0)
    sd_p = a_pers_tr.std(axis=0) + 1e-8

    # pesos para los cuadernos de XAI: ensemble mas las stats de train, que son leakage-free;
    # la representacion atlas no se guarda porque se rehace del cache
    if getattr(args, "save_model", False):
        stem = args.npz_stem or ("trajdev2" if getattr(args, "repr", "pca") == "atlas" else "trajdev")
        outm = Path(f"results/models/{stem}_model_farm_c_s{seed}.pt")
        torch.save({"state_dicts": [m.state_dict() for m in models],
                    "arch": {"k": int(data.shape[2]), "context": C, "horizon": H,
                             "hidden": args.hidden, "layers": args.layers, "n_models": args.n_models},
                    "z_stats": {"mu_c": mu_c, "sd_c": sd_c, "mu_s": mu_s, "sd_s": sd_s,
                                "mu_p": mu_p, "sd_p": sd_p},
                    "sub_h": sub_h, "ch_names": prep.get("ch_names"),
                    "repr": getattr(args, "repr", "pca"), "atlas_tag": getattr(args, "atlas_tag", ""),
                    "seed": seed}, outm)
        logger.info(f"  modelo -> {outm}")

    def z_family(a_mean: np.ndarray, a_slope: np.ndarray, a_pers: np.ndarray) -> dict[str, np.ndarray]:
        # z por canal con mu y sd de train, para comparar canales de escalas distintas
        z = (a_mean - mu_c) / sd_c  # (N, C)
        zs = (a_slope - mu_s) / sd_s
        zp = (a_pers - mu_p) / sd_p
        out = {"dep_z": z.mean(axis=1),
               "dep_z_top3": np.sort(z, axis=1)[:, -3:].mean(axis=1)}
        for mname, m in masks.items():
            if m.any():
                out[f"dep_z_{mname}"] = z[:, m].mean(axis=1)
                out[f"dep_zmax_{mname}"] = z[:, m].max(axis=1)
                out[f"dep_zslope_{mname}"] = zs[:, m].max(axis=1)
                out[f"dep_zpersist_{mname}"] = zp[:, m].max(axis=1)
        return out

    tr_var.update(z_family(a_mean_tr, a_slope_tr, a_pers_tr))
    if args.n_models > 1:
        tr_var["dis"] = dis_tr
    ev_var: dict[str, list[np.ndarray]] = {k: [] for k in tr_var}
    for err_e, dis_e in ev_out:
        v = dep_variants(err_e, sub_h)
        v.update(z_family(np.abs(err_e).mean(axis=1),
                          np.abs(err_e)[:, -1, :] - np.abs(err_e)[:, 0, :],
                          np.abs(err_e.mean(axis=1))))
        if args.n_models > 1:
            v["dis"] = dis_e
        for k in ev_var:
            ev_var[k].append(v[k])

    # comprobacion fuerte de alineacion, el ventaneo tiene que coincidir evento a evento
    # con el del AE, o la comparacion CARE deja de ser apples-to-apples
    meta = json.load(open("results/models/ae_scores_farm_c_meta.json", encoding="utf-8"))
    eids = [int(s["event_id"]) for s in prep["test_seqs"]]
    first = next(iter(ev_var))
    for i, eid in enumerate(eids):
        if len(ev_var[first][i]) != len(meta[str(eid)]["labels"]):
            raise RuntimeError(f"desalineado ev{eid} vs meta: {len(ev_var[first][i])}")
    refs = corr_refs(seed, eids)
    # el evaluador solo necesita labels y timestamps: las ventanas van vacias porque el
    # ReplayScorer ya trae los scores calculados
    slim = [{**{k: s[k] for k in ("event_id", "event_label", "asset_id", "labels", "window_timestamps")},
             "windows": np.empty(0)} for s in prep["test_seqs"]]

    rows = []
    for name, tr_sc in tr_var.items():
        evs = ev_var[name]
        rs = ReplayScorer([tr_sc] + evs)
        ev = run_model.evaluate(rs, np.empty(0), slim, adapter, 42)
        b, cv = ev["best"], ev["cv"]
        cat = np.concatenate(evs)
        corr = {k: float(np.corrcoef(cat, v)[0, 1]) for k, v in refs.items()}
        # la variante pasa si es poco redundante con el AE y ademas llega sola al umbral
        gate = corr.get("ae", 1.0) < 0.5 and cv["care"] >= 0.68
        rows.append({"variant": name, "care_cv": cv["care"], "oracle": b["care"], "roc_ds": b["roc_ds"],
                     "cov": b["cov"], "earl": b["earl"], "rel": b["rel"], "acc": b["acc"],
                     **{f"corr_{k}": v for k, v in corr.items()}, "gate_pass": bool(gate)})
        logger.info(f"  {name:<8} CARE-CV={cv['care']:.4f} ROC={b['roc_ds']:.3f} Earl={b['earl']:.3f} "
                    f"Acc={b['acc']:.3f} | corr ae={corr.get('ae', float('nan')):+.3f} "
                    f"crgn={corr.get('crgn', float('nan')):+.3f} phys={corr.get('phys', float('nan')):+.3f} "
                    f"| GATE {'PASS' if gate else 'fail'}")

    if args.dump_scores:
        arrays = {}
        for name in tr_var:
            arrays[f"train_{name}"] = np.asarray(tr_var[name], np.float64)
            for eid, sc in zip(eids, ev_var[name]):
                arrays[f"ev{eid}_{name}"] = np.asarray(sc, np.float64)
        stem = args.npz_stem or ("trajdev2" if getattr(args, "repr", "pca") == "atlas" else "trajdev")
        out = Path(f"results/models/{stem}_scores_farm_c_s{seed}.npz")
        np.savez_compressed(out, **arrays)
        logger.info(f"  dump -> {out} ({len(arrays)} arrays)")

    return {"seed": seed, "context": C, "horizon": H, "n_models": args.n_models,
            "hidden": args.hidden, "params": int(n_par), "rows": rows}


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
    ap.add_argument("--n-models", type=int, default=1)
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dump-scores", action="store_true")
    ap.add_argument("--save-model", action="store_true",
                    help="guarda ensemble+stats en results/models/{stem}_model_farm_c_s{seed}.pt")
    ap.add_argument("--repr", default="pca", choices=["pca", "atlas"],
                    help="pca = componentes K=55 (camino AE); atlas = canales fisicos no-avg (v2)")
    ap.add_argument("--atlas-tag", default="", help="'' = atlas 153ch; '3' = v3 (val + panel ampliado)")
    ap.add_argument("--npz-stem", default="", help="stem del dump (def: trajdev2 atlas / trajdev pca)")
    args = ap.parse_args()
    if args.smoke:
        args.train_cap, args.epochs = 80000, 3

    # dos representaciones: el atlas viene ya ventaneado del cache, el camino pca lo construye
    if args.repr == "atlas":
        from signal_atlas import load_cache
        tw, ch_names, _pairs, seqs = load_cache(args.atlas_tag)
        prep = {"train_data": tw, "K": tw.shape[2], "ch_names": ch_names,
                "test_seqs": [{**s, "data": s["windows"]} for s in seqs]}
    else:
        prep = prep_c(144)
        prep["ch_names"] = None
    adapter = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm="C", window_size=27)
    logger.info(f"C1 trajdev | K={prep['K']} W={prep['train_data'].shape[1]} | ctx={args.context} "
                f"H={args.horizon} | train {prep['train_data'].shape} | {len(prep['test_seqs'])} eventos "
                f"| {DEVICE}")
    logger.info("refs: AE-honesto 0.7206 | GATE: corr-AE<0.5 Y CARE-CV>=0.68 | campeon fusion 0.7694")

    tag = f"_{args.exp_tag}" if args.exp_tag else ""
    for seed in [int(s) for s in args.seeds.split(",")]:
        logger.info(f"=== seed {seed} ===")
        res = run_one_seed(prep, adapter, args, seed)
        out = Path(f"results/tables/trajdev_farm_c_gate_s{seed}{tag}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out, "w"), indent=2, default=run_model._np_default)
        logger.info(f"saved -> {out}")


if __name__ == "__main__":
    main()
