"""GraphDiff para Farm C bajo el protocolo oficial, sobre la representacion de CRGN.

esa representacion son los top-100 XAI como nodos del grafo, con W=27, y no el camino PCA,
porque un grafo Spearman sobre componentes PCA saldria casi diagonal y la ortogonalidad de CRGN
nace del grafo de sensores crudos. reutiliza en solo lectura run_gdn.load_representation y
src/crgn, o sea el grafo, el encoder y el train contrastivo, y evalua con run_model.evaluate, o
sea CUSUM, h-sweep y CARE-CV sobre eventos, comparable al AE y fusable, con los eventos
alineados 1:1 con la meta del AE igual que CRGN. espeja scripts/run_driftdiff_c.py.

modos

    frozen : entrena el CRGN contrastivo, lo congela y entrena gather, difusion y decoder
    joint  : todo a la vez, al estilo de NGLS-Diff

la puerta temprana imprime corr(eps, AE) y corr(eps, CRGN) al nivel del CUSUM, que es lo que
dice si el eps latente es a la vez fuerte y ortogonal a los dos.

    .venv\\Scripts\\python.exe scripts/run_graphdiff_c.py --seeds 42 --mode frozen [--smoke]
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

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (imports hermanos)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz del repo (from src.X)

import run_model
from run_crgn import load_farm_config, train_crgn
from run_gdn import load_representation
from sweep_timeaddm_scoring import ReplayScorer
from src.care_official.adapter import OfficialCAREAdapter
from src.crgn.data import stack_timesteps, to_node_major
from src.crgn.graph import build_adjacency
from src.graphdiff.model import build_graphdiff
from src.graphdiff import scorer as gdscorer
from src.graphdiff import physics as ph

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTH = [1, 3, 5, 7]
VARIANTS = ["eps", "recon", "eps+recon"]
# referencias del proyecto contra las que se lee el multi-seed, todas en CARE-CV de C
AE_HONEST, AE_DEPLOYED, CHAMPION = 0.7206, 0.7432, 0.7452


# evita que Windows suspenda el sistema durante el run
def _keep_awake():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        logger.info("keep-awake ON (SetThreadExecutionState)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"keep-awake no disponible: {e}")


def prep_c(seed: int, phys_thresh: float = 0.95):
    """representacion CRGN de C, top-100 en node-major, con el split val/train de
    run_crgn.run_full y el grafo Spearman top-k."""

    # la representacion es invariante a la semilla y esta cacheada; el split y el grafo si
    # dependen de ella, igual que en CRGN multi-seed. los eventos van alineados 1:1 con la
    # meta del AE
    cfg = load_farm_config("C")
    scaled, test_seqs, feats = load_representation("C", cfg, "top100", seed)
    k, w = scaled.shape[2], scaled.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(scaled))
    n_val = int(round(len(scaled) * cfg["training"]["val_ratio"]))
    val_norm, train_fit = scaled[perm[:n_val]], scaled[perm[n_val:]]
    g = cfg["graph"]
    # el topk se capa a K-1, que un nodo no puede tener mas vecinos que el resto del grafo
    topk = min(g["topk"], k - 1)
    adj = build_adjacency(stack_timesteps(train_fit, max_rows=g.get("max_rows"), seed=seed),
                          k=topk, weighted=g["weighted"], self_loops=g["self_loops"])
    # los pares redundantes solo se usan si w_phys > 0, pero salen aqui para que el grafo y los
    # pares vengan del mismo subsample de filas de train
    pairs = ph.find_redundant_pairs(stack_timesteps(train_fit, max_rows=g.get("max_rows"), seed=seed),
                                    thresh=phys_thresh)
    logger.info(f"prep C | K={k} W={w} topk={topk} | fit {train_fit.shape} val {val_norm.shape} | "
                f"{len(test_seqs)} eventos | pares fisicos(>= {phys_thresh})={len(pairs)}")
    return {"train_fit": train_fit, "val_norm": val_norm, "test_seqs": test_seqs, "adj": adj,
            "K": k, "W": w, "feats": feats, "cfg": cfg, "pairs": pairs}


def train_model(model, prep, args, seed):
    """en modo frozen entrena el CRGN contrastivo, lo congela y luego entrena gather, difusion
    y decoder; en modo joint entrena todo a la vez."""

    x_nm = to_node_major(prep["train_fit"])  # (N, K, W)
    adj_np = prep["adj"].astype(np.float32)
    a_t = torch.from_numpy(adj_np).to(DEVICE)
    cfg = prep["cfg"]

    if args.mode == "frozen":
        t0 = time.time()
        logger.info(f"  [frozen] fase 1: CRGN contrastivo ({args.enc_epochs} ep) ...")
        train_crgn(model.backbone, torch.from_numpy(x_nm), torch.from_numpy(adj_np), cfg, DEVICE,
                   seed=seed, epochs=args.enc_epochs)
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        params = (list(model.gather.parameters()) + list(model.diffusion.parameters())
                  + list(model.decoder.parameters()))
        with_contrast = False
        logger.info(f"  [frozen] backbone congelado | {time.time()-t0:.0f}s")
    else:
        params = list(model.parameters())
        # con w_contrast=0 queda NGLS-Diff puro, con el latente libre
        with_contrast = args.w_contrast > 0

    # coherencia fisica, y w_phys=0 la apaga
    pairs = prep.get("pairs") if args.w_phys > 0 else None
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    n, bs = len(x_nm), args.batch
    # seed+1 para que el barajado de la fase 2 no repita el de la fase 1 contrastiva
    rng = np.random.default_rng(seed + 1)
    logger.info(f"  fase 2: gather+difusion+decoder ({args.epochs} ep, {n:,} ventanas, batch {bs}) "
                f"| w_gather={args.w_gather} w_recon={args.w_recon} "
                f"w_contrast={args.w_contrast if with_contrast else 0} "
                f"w_phys={args.w_phys}{f' ({len(pairs)} pares)' if pairs else ''}")
    for e in range(args.epochs):
        t0 = time.time()
        model.train()
        perm = rng.permutation(n)
        agg = {"total": [], "diff": [], "kl": [], "recon": [], "contrast": [], "phys": []}
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = torch.as_tensor(x_nm[idx], dtype=torch.float32, device=DEVICE)
            opt.zero_grad()
            losses = model.training_losses(xb, a_t, args.w_gather, args.w_recon, args.w_contrast,
                                           with_contrast, pairs=pairs, w_phys=args.w_phys)
            losses["total"].backward()
            opt.step()
            for kk in agg:
                agg[kk].append(float(losses[kk].detach()))
        logger.info(f"  epoch {e:>2} | total {np.mean(agg['total']):.4f} | diff {np.mean(agg['diff']):.4f} "
                    f"| kl {np.mean(agg['kl']):.3f} | recon {np.mean(agg['recon']):.4f} "
                    f"| contr {np.mean(agg['contrast']):.4f} | phys {np.mean(agg['phys']):.4f} | {time.time()-t0:.0f}s")
    return model


def _smooth(s, w):
    return s if w <= 1 else np.convolve(s, np.ones(w) / w, mode="same")


def _pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def gate_diagnostic(event_raws, test_seqs, levels, meta):
    """puerta temprana, la correlacion del eps por nivel y la del recon contra AE y CRGN, al
    nivel del score que entra al CUSUM y con los eventos concatenados."""

    # lo que se busca es un eps decorrelado de los dos a la vez y fuerte. el nivel de ruido
    # mueve el trade-off, bajo sale ortogonal pero debil y alto fuerte pero redundante, asi
    # que se reportan todos los niveles
    order = [int(x) for x in meta.keys()]
    # los scores se concatenan en el orden de eventos de la meta del AE, que es el unico orden
    # en el que las referencias y el eps son comparables punto a punto
    by_id = {int(s["event_id"]): e for s, e in zip(test_seqs, event_raws)}
    rec_cat = np.concatenate([by_id[i]["recon"] for i in order])
    refs = {}
    for name, npz, keyf in [("ae", "results/models/ae_scores_farm_c.npz", lambda e: f"ev{e}_mae"),
                            ("crgn", "results/models/crgn_scores_farm_c.npz", lambda e: f"ev{e}_dev")]:
        try:
            z = np.load(npz)
            refs[name] = np.concatenate([np.asarray(z[keyf(i)], float) for i in order])
        except Exception as ex:  # noqa: BLE001
            logger.warning(f"  [gate] no pude cargar {name}: {ex}")
    out = {}
    for L in levels:
        eps_cat = np.concatenate([by_id[i][f"eps{L}"] for i in order])
        out[f"L{L}"] = {f"eps_vs_{n}": _pearson(eps_cat, r) for n, r in refs.items() if len(r) == len(eps_cat)}
    out["recon"] = {f"recon_vs_{n}": _pearson(rec_cat, r) for n, r in refs.items() if len(r) == len(rec_cat)}
    return out


def run_one_seed(prep, adapter, args, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    K, W = prep["K"], prep["W"]
    mcfg = {"embed_dim": args.embed_dim, "gcn_layers": args.gcn_layers, "d_lat": args.d_lat,
            "proj_dim": args.proj_dim, "temperature": prep["cfg"]["contrastive"]["temperature"],
            "noise_std": prep["cfg"]["augmentation"]["noise_std"],
            "mask_ratio": prep["cfg"]["augmentation"]["mask_ratio"], "diff_num_res": args.diff_num_res,
            "diff_attn": not args.no_attn, "n_steps": args.n_steps}
    model = build_graphdiff(K, W, mcfg, DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    logger.info(f"[seed {seed}] GraphDiff ({nparams:,} params) mode={args.mode} d_lat={args.d_lat} "
                f"n_steps={args.n_steps} levels={args.levels}")

    train_model(model, prep, args, seed)

    levels = [int(x) for x in args.levels.split(",")]
    t0 = time.time()
    train_raw = gdscorer.run_scores(model, to_node_major(prep["val_norm"]), prep["adj"], DEVICE,
                                    levels, draws=args.draws)
    event_raws = [gdscorer.run_scores(model, to_node_major(s["windows"]), prep["adj"], DEVICE,
                                      levels, draws=args.draws) for s in prep["test_seqs"]]
    logger.info(f"  scoring done | {time.time()-t0:.0f}s")

    rows = []
    for variant in VARIANTS:
        for L in levels:
            if variant == "recon" and L != levels[0]:  # recon no depende del nivel
                continue
            tr_s, ev_s = gdscorer.assemble(train_raw, event_raws, variant, L)
            for sw in SMOOTH:
                ev_sm = [_smooth(e, sw) for e in ev_s]
                # percentil 75 y no 99, igual que en D3R, el score no es tan apuntado
                sc = ReplayScorer([tr_s] + ev_sm, percentile=75.0)
                ev = run_model.evaluate(sc, prep["val_norm"], prep["test_seqs"], adapter, 42)  # cv-seed 42 fijo
                b, cv = ev["best"], ev["cv"]
                rows.append({"variant": variant, "level": L, "smooth": sw, "care_cv": cv["care"],
                             "oracle": b["care"], "optim": cv["optimism"], "roc_ds": b["roc_ds"],
                             "roc_win": ev["roc_window"], "f1": b["f1"], "cov": b["cov"], "earl": b["earl"],
                             "rel": b["rel"], "acc": b["acc"], "h_scale": b["h_scale"]})
            best_sw = max((r for r in rows if r["variant"] == variant and r["level"] == L),
                          key=lambda r: r["care_cv"])
            logger.info(f"  {variant:<10} L{L:<4} best MA{best_sw['smooth']}  CARE-CV={best_sw['care_cv']:.4f}  "
                        f"ROC-ds={best_sw['roc_ds']:.3f}  Earl={best_sw['earl']:.3f}  Acc={best_sw['acc']:.3f}")
    rows.sort(key=lambda r: -r["care_cv"])
    best = rows[0]

    meta = json.load(open("results/models/ae_scores_farm_c_meta.json", encoding="utf-8"))
    gate = gate_diagnostic(event_raws, prep["test_seqs"], levels, meta)
    # el eps solo se reporta aparte del mejor global, porque es el que decide la puerta
    best_eps = max((r for r in rows if r["variant"] == "eps"), key=lambda r: r["care_cv"])
    logger.info(f"[seed {seed}] BEST {best['variant']} L{best['level']} MA{best['smooth']} -> "
                f"CARE-CV={best['care_cv']:.4f} (ROC-ds {best['roc_ds']:.3f}, Earl {best['earl']:.3f})")
    logger.info(f"[seed {seed}] EPS-solo mejor: CARE-CV={best_eps['care_cv']:.4f} (L{best_eps['level']} "
                f"MA{best_eps['smooth']}, Earl {best_eps['earl']:.3f}, Acc {best_eps['acc']:.3f})")
    for L in levels:
        g = gate.get(f"L{L}", {})
        logger.info(f"[seed {seed}] PUERTA L{L}: eps_vs_AE={g.get('eps_vs_ae'):.3f} "
                    f"eps_vs_CRGN={g.get('eps_vs_crgn'):.3f}" if g else f"[seed {seed}] PUERTA L{L}: (sin refs)")
    logger.info(f"[seed {seed}] PUERTA recon: {gate.get('recon', {})}")

    out = {"farm": "C", "model": "graphdiff", "representation": "crgn_top100_nodemajor", "K": K,
           "window_size": int(W), "seed": seed, "mode": args.mode, "d_lat": args.d_lat,
           "n_steps": args.n_steps, "levels": levels, "draws": args.draws, "enc_epochs": args.enc_epochs,
           "epochs": args.epochs, "w_gather": args.w_gather, "w_recon": args.w_recon,
           "w_contrast": args.w_contrast, "w_phys": args.w_phys, "phys_thresh": args.phys_thresh,
           "diff_attn": not args.no_attn, "n_params": int(nparams),
           "smoke": args.smoke, "eps_best": best_eps, "gate": gate, "best": best, "sweep": rows}
    tag = f"_s{seed}" + (f"_{args.exp_tag}" if args.exp_tag else "")
    p_out = Path(f"results/tables/ownB_farm_c_sweep{tag}.json")
    p_out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p_out, "w"), indent=2, default=run_model._np_default)
    logger.info(f"  saved -> {p_out}")

    if args.dump_scores:
        _dump_scores(train_raw, event_raws, prep["test_seqs"], seed, levels, meta)
    if args.save_ckpt:
        ck = Path(f"results/models/graphdiff_farm_c{tag}.pt")
        torch.save({"state_dict": model.state_dict(), "adj": prep["adj"], "K": K, "W": W,
                    "mcfg": mcfg, "mode": args.mode, "seed": seed, "levels": levels}, ck)
        logger.info(f"  [ckpt] -> {ck}")
    return {**best, "eps_care": best_eps["care_cv"], "gate": gate, "seed": seed}


def _dump_scores(train_raw, event_raws, test_seqs, seed, levels, meta):
    """vuelca el eps de todos los niveles, train_eps{L} y ev{e}_eps{L}, mas el recon, alineados
    con la meta del AE."""

    # con todos los niveles volcados, fuse_multi elige el suyo sin reentrenar, ownB{L}
    by_id = {int(s["event_id"]): e for s, e in zip(test_seqs, event_raws)}
    arrays = {"train_recon": np.asarray(train_raw["recon"], np.float64)}
    for L in levels:
        arrays[f"train_eps{L}"] = np.asarray(train_raw[f"eps{L}"], np.float64)
    for eid_s, m in meta.items():
        eid = int(eid_s)
        if eid not in by_id:
            raise RuntimeError(f"evento {eid} ausente en el scoring de graphdiff")
        e = by_id[eid]
        if len(e["recon"]) != len(m["labels"]):
            raise RuntimeError(f"desalineado ev{eid}: {len(e['recon'])} vs meta {len(m['labels'])}")
        arrays[f"ev{eid}_recon"] = np.asarray(e["recon"], np.float64)
        for L in levels:
            arrays[f"ev{eid}_eps{L}"] = np.asarray(e[f"eps{L}"], np.float64)
    out = Path(f"results/models/ownB_scores_farm_c_s{seed}.npz")
    np.savez_compressed(out, **arrays)
    logger.info(f"  [dump] ownB (eps {levels} + recon): {len(arrays)} arrays -> {out}")


def main():
    _keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--mode", choices=["frozen", "joint"], default="frozen")
    ap.add_argument("--exp-tag", type=str, default="")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--gcn-layers", type=int, default=2)
    ap.add_argument("--d-lat", type=int, default=64, help="dim del latente (multiplo de 32)")
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--diff-num-res", type=int, default=2)
    ap.add_argument("--no-attn", action="store_true", help="ablacion: sin self-attention entre nodos")
    ap.add_argument("--n-steps", type=int, default=500, help="pasos de la difusion (T)")
    ap.add_argument("--levels", type=str, default="50,100,250", help="niveles de ruido para el eps")
    ap.add_argument("--enc-epochs", type=int, default=20, help="epocas del CRGN contrastivo (modo frozen)")
    ap.add_argument("--epochs", type=int, default=20, help="epocas de gather+difusion+decoder")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--w-gather", type=float, default=1.0, help="peso KL (normal-gathering, beta-VAE)")
    ap.add_argument("--w-recon", type=float, default=1.0, help="peso recon del decoder")
    ap.add_argument("--w-contrast", type=float, default=1.0, help="peso NT-Xent (solo modo joint)")
    ap.add_argument("--w-phys", type=float, default=0.0, help="peso coherencia fisica (§6; 0=off)")
    ap.add_argument("--phys-thresh", type=float, default=0.95, help="|Spearman| minimo para par redundante")
    ap.add_argument("--draws", type=int, default=4, help="realizaciones de ruido promediadas por ventana")
    ap.add_argument("--smoke", action="store_true", help="pocas epocas (validar pipeline + puerta)")
    ap.add_argument("--dump-scores", action="store_true", help="volcar scores para fusion (ownB_*)")
    ap.add_argument("--save-ckpt", action="store_true", help="guardar checkpoint (re-score sin reentrenar)")
    args = ap.parse_args()

    # el smoke solo valida el pipeline y la puerta, con pocas epocas
    if args.smoke:
        args.enc_epochs = min(args.enc_epochs, 5)
        args.epochs = min(args.epochs, 5)
    seeds = [int(s) for s in args.seeds.split(",")]
    logger.info(f"Device: {DEVICE} | Farm C | GraphDiff | seeds {seeds} | mode {args.mode}")

    # el prep se rehace por semilla, al contrario que en D3R, porque aqui el split y el grafo
    # dependen de la semilla y no solo el entrenamiento
    bests = []
    for seed in seeds:
        prep = prep_c(seed, args.phys_thresh)
        adapter = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm="C", window_size=prep["W"])
        bests.append(run_one_seed(prep, adapter, args, seed))

    cares = np.array([b["care_cv"] for b in bests])
    eps_cares = np.array([b["eps_care"] for b in bests])
    suffix = f"_{args.exp_tag}" if args.exp_tag else ""
    summary = {"model": "graphdiff", "farm": "C", "seeds": seeds, "mode": args.mode, "d_lat": args.d_lat,
               "n_steps": args.n_steps, "epochs": args.epochs, "mean": float(cares.mean()),
               "std": float(cares.std()), "eps_mean": float(eps_cares.mean()), "eps_std": float(eps_cares.std()),
               "wins_vs_ae_honest_0.7206": int((cares > AE_HONEST).sum()),
               "gate_per_seed": [{"seed": b["seed"], **(b["gate"] or {})} for b in bests],
               "per_seed": [{k: v for k, v in b.items() if k != "gate"} for b in bests]}
    json.dump(summary, open(f"results/tables/ownB_multiseed_summary{suffix}.json", "w"),
              indent=2, default=run_model._np_default)
    logger.info("=" * 74)
    logger.info(f"MULTI-SEED GraphDiff C: mejor-variante CARE-CV {cares.mean():.4f} +- {cares.std():.4f} | "
                f"EPS-solo {eps_cares.mean():.4f} +- {eps_cares.std():.4f} | "
                f"wins vs AE-honesto {AE_HONEST}: {int((cares > AE_HONEST).sum())}/{len(seeds)} "
                f"| AE deployed {AE_DEPLOYED} / campeon {CHAMPION}")
    logger.info(f"summary -> results/tables/ownB_multiseed_summary{suffix}.json")


if __name__ == "__main__":
    main()
