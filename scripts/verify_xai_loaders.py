"""gate de la fase XAI. comprueba que los cargadores de src/xai reproducen exactamente los
numeros committeados, antes de construir ningun notebook encima. se invoca con
python scripts/verify_xai_loaders.py [--skip-c-ae].

[B-1] cargador de TrajDev-B, dep_zmax_val y dep_z_top3 recomputados por evento contra el
      dump trajdevb_scores_farm_b_s42.npz, con max|diff| < 1e-4
[B-2] transform del AE-B, loader.transform(raw test) contra las ventanas K de load_farm_b,
      exacto
[B-3] AE-B de punta a punta, evaluate(ident_s42 via loader) da CARE-CV 0.721343
      (ae_identical_b_s42.json). el dump deployed no sirve de arbitro, es del modelo perdido
[C-1] cargador de TrajDev-C, dep_zmax_val por evento contra trajdev3_scores_farm_c_s42.npz
[C-2] AE-C de punta a punta, transform(farm_c_xai raw) da la K del cache oficial y
      evaluate(ident_s42 via loader) da CARE-CV 0.743199 (ae_identical_c_s42.json). el dump
      deployed no es arbitro, es de la otra linea
[X-1] ida y vuelta de la PCA, transform(inverse(k)) == k, o sea la exactitud de la
      retro-proyeccion, en las dos granjas
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xai import (load_ae, load_ae_events, load_meta, load_scores, load_trajdev,
                     load_trajdev_events, trajdev_channel_z)
from src.xai.loaders import ae_scores, trajdev_forecast_errors, trajdev_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
TOL = 1e-4
FAILS: list[str] = []


# un check no aborta, apunta el fallo y sigue, para que una ejecucion liste todo lo roto
def check(name: str, ok: bool, detail: str) -> None:
    logger.info(f"[{name}] {'PASS' if ok else 'FAIL'} — {detail}")
    if not ok:
        FAILS.append(name)


# recomputa el score de TrajDev desde el cargador y lo compara contra el dump de la fusion.
# en B se comprueban los dos agregadores registrados y en C solo el campeon
def verify_trajdev(farm: str) -> None:
    variants = {"dep_zmax_val": ("ownb3_zmax" if farm == "B" else "own3_val")}
    if farm == "B":
        variants["dep_z_top3"] = "ownb3_ztop3"
    bundle = load_trajdev(farm)
    events = load_trajdev_events(farm)
    ref = {v: load_scores(farm, det)[1] for v, det in variants.items()}
    worst = {v: 0.0 for v in variants}
    for ev in events:
        # el error de pronostico se pasa a z por canal y de ahi a un escalar por ventana
        err, _dis = trajdev_forecast_errors(bundle, ev["data"])
        z = trajdev_channel_z(bundle, err)["z"]
        for v in variants:
            s = trajdev_score(bundle, z, v)
            r = ref[v][ev["event_id"]]
            # otra longitud ya no es cuestion de tolerancia, son otras ventanas
            if len(s) != len(r):
                check(f"{farm}-trajdev-{v}", False, f"ev{ev['event_id']} len {len(s)} vs dump {len(r)}")
                return
            worst[v] = max(worst[v], float(np.max(np.abs(s - r))))
    for v in variants:
        check(f"{farm}-trajdev-{v}", worst[v] < TOL, f"max|diff| {worst[v]:.2e} sobre {len(events)} eventos")


# dos comprobaciones encadenadas en B, primero que la representacion coincide y
# despues que el AE sobre ella reproduce su CARE, para no confundir un fallo con otro
def verify_ae_b() -> None:
    from compute_lead_times import load_farm_b
    import run_model
    from sweep_timeaddm_scoring import ReplayScorer
    from src.care_official.adapter import OfficialCAREAdapter

    ae = load_ae("B")
    train_k, seqs, _ = load_farm_b()
    evs = load_ae_events("B")
    by_id = {s["event_id"]: s for s in seqs}
    worst = 0.0
    for ev in evs:
        k_ours = ae["transform"](ev["raw"])
        k_ref = by_id[ev["event_id"]]["windows"]
        worst = max(worst, float(np.max(np.abs(k_ours - k_ref))))
    check("B-ae-transform", worst < 1e-5, f"max|diff| vs load_farm_b {worst:.2e}")

    tr_scores = ae_scores(ae, train_k.astype(np.float32, copy=False))
    ev_scores = [ae_scores(ae, ae["transform"](ev["raw"])) for ev in evs]
    adapter = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm="B", window_size=27)
    # el harness pide las secuencias completas, pero con un replay las ventanas sobran
    slim = [{"event_id": ev["event_id"], "event_label": ev["event_label"], "asset_id": ev["asset_id"],
             "labels": ev["labels"], "window_timestamps": ev["window_timestamps"],
             "windows": np.empty(0)} for ev in evs]
    res = run_model.evaluate(ReplayScorer([tr_scores] + ev_scores), np.empty(0), slim, adapter, 42)
    cv = res["cv"]["care"]
    check("B-ae-care", abs(cv - 0.721343) < 5e-4, f"CARE-CV {cv:.6f} vs ident_s42 0.721343")


def verify_ae_c() -> None:
    """cadena C reproducible por la linea-cache, transform(raw) da la K del cache oficial y
    el AE ident_s42 sobre esa representacion reproduce su CARE exacto, 0.743199, que es el
    nivel deployed. el dump deployed no es arbitro aqui, porque es de la otra linea, la del
    fit de scaler y pca que se perdio."""

    import pickle

    import run_model
    from sweep_timeaddm_scoring import ReplayScorer
    from src.care_official.adapter import OfficialCAREAdapter

    ae = load_ae("C")
    evs = load_ae_events("C")
    with open("data/processed/farm_c_pcafirst_timeaddm.pkl", "rb") as f:
        cache = pickle.load(f)
    k_by = {int(s["event_id"]): np.asarray(s["windows"], np.float32) for s in cache["test"]}
    train_k = np.asarray(cache["train"], np.float32)
    del cache  # el cache de C pesa y ya solo hacen falta las vistas de arriba

    worst = 0.0
    for ev in evs:
        worst = max(worst, float(np.max(np.abs(ae["transform"](ev["raw"]) - k_by[ev["event_id"]]))))
    check("C-ae-transform", worst < 1e-3, f"transform(raw) vs cache oficial: max|diff| {worst:.2e}")

    tr_scores = ae_scores(ae, train_k)
    ev_scores = [ae_scores(ae, k_by[ev["event_id"]]) for ev in evs]
    adapter = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm="C", window_size=27)
    slim = [{"event_id": ev["event_id"], "event_label": ev["event_label"], "asset_id": ev["asset_id"],
             "labels": ev["labels"], "window_timestamps": ev["window_timestamps"],
             "windows": np.empty(0)} for ev in evs]
    res = run_model.evaluate(ReplayScorer([tr_scores] + ev_scores), np.empty(0), slim, adapter, 42)
    cv = res["cv"]["care"]
    check("C-ae-care", abs(cv - 0.743199) < 5e-4, f"CARE-CV {cv:.6f} vs ident_c_s42 0.743199")


# la retro-proyeccion pierde lo que quedo fuera de las componentes, asi que la ida y
# vuelta solo tiene que cerrar en el espacio K. bastan 64 ventanas del primer evento
def verify_roundtrip(farm: str) -> None:
    ae = load_ae(farm)
    evs = load_ae_events(farm)
    k = ae["transform"](evs[0]["raw"][:64])
    k2 = ae["transform"](ae["inverse"](k))
    d = float(np.max(np.abs(k - k2)))
    check(f"{farm}-pca-roundtrip", d < 1e-3, f"transform(inverse(k)) max|diff| {d:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-c-ae", action="store_true", help="si rebuild_farm_c aun no ha verificado")
    args = ap.parse_args()

    # --skip-c-ae deja fuera los dos checks de C que dependen del cache reconstruido
    verify_trajdev("B")
    verify_ae_b()
    verify_roundtrip("B")
    verify_trajdev("C")
    if not args.skip_c_ae:
        verify_ae_c()
        verify_roundtrip("C")
    logger.info("=" * 70)
    if FAILS:
        logger.error(f"GATE FALLIDO: {FAILS}")
        sys.exit(1)
    logger.info("GATE XAI: TODOS LOS CHECKS PASS — los cargadores reproducen lo committeado.")


if __name__ == "__main__":
    main()
