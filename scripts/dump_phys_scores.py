"""vuelca el score model-free de coherencia fisica alineado a la meta del AE, por semilla.

descubre los pares redundantes, |Spearman| >= thr, sobre el split train-normal de cada semilla
y puntua la violacion de coherencia por ventana. escribe
results/models/phys_scores_farm_c_s{seed}.npz con las claves train_phys{90,95} y
ev{e}_phys{90,95}, que es lo que consume la fusion multi-seed. corre en CPU y no carga ningun
modelo.

    python scripts/dump_phys_scores.py --seeds 42,7,123,1234,12345
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_crgn import load_farm_config
from run_gdn import load_representation
from src.crgn.data import stack_timesteps, to_node_major
from src.graphdiff import physics as ph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,123,1234,12345")
    ap.add_argument("--farm", default="C", choices=["B", "C"])
    ap.add_argument("--thresholds", default="", help="ej '0.85,0.90' (por defecto C=0.90,0.95 / B=0.85,0.90)")
    args = ap.parse_args()
    farm, fl = args.farm, args.farm.lower()
    cfg = load_farm_config(farm)
    features = cfg["features"]  # C=top100, B=curated
    thrs = ([float(x) for x in args.thresholds.split(",")] if args.thresholds
            else ([0.90, 0.95] if farm == "C" else [0.85, 0.90]))
    # las tres variantes son (agg, rate); max sin rate es el campeon phys95max, y max con rate
    # es su derivada, que sale aun mas ortogonal al resto de detectores
    variants = [("mean", False), ("max", False), ("max", True)]
    # el tag es el nombre con el que fuse_multi registra el detector, phys95maxrate y compania
    thresholds = [(t, agg, rate, f"phys{int(round(t*100))}{'' if agg == 'mean' else agg}{'rate' if rate else ''}")
                  for t in thrs for agg, rate in variants]
    meta = json.load(open(f"results/models/ae_scores_farm_{fl}_meta.json", encoding="utf-8"))
    for seed in [int(s) for s in args.seeds.split(",")]:
        scaled, test_seqs, _feats = load_representation(farm, cfg, features, seed)
        # split leakage-free, los pares y las stats salen del train y la calibracion del val
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(scaled))
        n_val = int(round(len(scaled) * cfg["training"]["val_ratio"]))
        val_nm = to_node_major(scaled[perm[:n_val]])
        train_nm = to_node_major(scaled[perm[n_val:]])
        # los pares se descubren a nivel de fila, no de ventana, con el subsample de CRGN
        rows = stack_timesteps(scaled[perm[n_val:]], max_rows=cfg["graph"].get("max_rows"), seed=seed)
        by_id = {int(s["event_id"]): s for s in test_seqs}
        arrays = {}
        for thr, agg, rate, tag in thresholds:
            pairs = ph.find_redundant_pairs(rows, thresh=thr)
            stats = ph.coherence_stats(train_nm, pairs, rate=rate)
            arrays[f"train_{tag}"] = ph.coherence_score(val_nm, pairs, stats, agg=agg, rate=rate).astype(np.float64)
            for eid_s, m in meta.items():
                eid = int(eid_s)
                sc = ph.coherence_score(to_node_major(by_id[eid]["windows"]), pairs, stats, agg=agg, rate=rate).astype(np.float64)
                # el dump solo vale si cada evento casa ventana a ventana con la meta del AE
                if len(sc) != len(m["labels"]):
                    raise RuntimeError(f"desalineado s{seed} ev{eid}: {len(sc)} vs {len(m['labels'])}")
                arrays[f"ev{eid}_{tag}"] = sc
        out = Path(f"results/models/phys_scores_farm_{fl}_s{seed}.npz")
        np.savez_compressed(out, **arrays)
        print(f"[{farm}] seed {seed}: {len(arrays)} arrays ({[t for *_, t in thresholds]}) -> {out}")


if __name__ == "__main__":
    main()
