"""fusion a nivel de score, generica y multi-detector.

carga los scores por ventana ya guardados de N detectores (AE, CRGN, difusion-eps, GDN...),
los alinea por evento contra la meta canonica, cuyos labels y timestamps son invariantes a la
semilla, y evalua con el harness CARE-CV oficial (run_model.evaluate via ReplayScorer) tanto
cada detector solo, que es el baseline honesto, como cada fusion (normalizador x combinador).
agrega sobre varias semillas de modelo (media+-std, wins pareados vs AE). todo en CPU, sin
recargar modelos.

escribe siempre a ficheros nuevos, results/tables/fuse_<tag>_<farm>.json; no sobrescribe nada.

ejemplo, los mejores combinadores y la normalizacion sobre AE+CRGN en Farm C con 5 semillas:

    python scripts/fuse_multi.py --farm C --detectors ae,crgn --seeds 42,7,123,1234,12345 \
        --normalizers zscore,gaussian_tail --tag a1_ae_crgn
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz del repo

import run_model
from run_model import _np_default
from src.care_official.adapter import OfficialCAREAdapter
from src.fusion import combiners as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# los volcados de la semilla 42 no llevan sufijo, el resto va con _s{seed}
def sfx(seed: int) -> str:
    return "" if seed == 42 else f"_s{seed}"


# registro de detectores, como localizar sus scores npz por (farm, seed)
DETECTORS = {
    "ae":   {"npz": lambda f, s: f"results/models/ae_scores_farm_{f}{sfx(s)}.npz",
             "train": "train_mae", "ev": lambda e: f"ev{e}_mae"},
    "crgn": {"npz": lambda f, s: f"results/models/crgn_scores_farm_{f}{sfx(s)}.npz",
             "train": "train_dev", "ev": lambda e: f"ev{e}_dev"},
    # difusion-eps, siempre con sufijo _s{seed} incluida la 42. solo 3 semillas, porque el
    # backbone AE va fijo en seed-42 y lo unico que varia con la semilla es el difusor
    "diff": {"npz": lambda f, s: f"results/models/diff_scores_farm_{f}_s{s}.npz",
             "train": "train_eps", "ev": lambda e: f"ev{e}_eps"},
    # D3R, volcado por scripts/run_driftdiff_c.py --dump-scores, siempre con sufijo _s{seed} y
    # alineado con la meta. dos scores por ventana, recon (residuo) y trend (drift anomalo)
    "ownA":       {"npz": lambda f, s: f"results/models/ownA_scores_farm_{f}_s{s}.npz",
                   "train": "train_recon", "ev": lambda e: f"ev{e}_recon"},
    "ownA_trend": {"npz": lambda f, s: f"results/models/ownA_scores_farm_{f}_s{s}.npz",
                   "train": "train_trend", "ev": lambda e: f"ev{e}_trend"},
    # pendiente del trend, o sea la dinamica del nivel, ortogonal al AE (~0.25). es model-free,
    # asi que las semillas comparten contenido
    "slope18": {"npz": lambda f, s: f"results/models/slope_scores_farm_{f}_s{s}.npz",
                "train": "train_slope18", "ev": lambda e: f"ev{e}_slope18"},
    "slope36": {"npz": lambda f, s: f"results/models/slope_scores_farm_{f}_s{s}.npz",
                "train": "train_slope36", "ev": lambda e: f"ev{e}_slope36"},
    # GraphDiff, volcado por scripts/run_graphdiff_c.py --dump-scores, siempre _s{seed} y
    # alineado 1:1 con la meta AE. eps de la difusion latente nativa a varios niveles de ruido
    # (ownB{L}), donde el nivel modula ortogonalidad contra fuerza, mas el recon del grafo-VAE
    "ownB25":     {"npz": lambda f, s: f"results/models/ownB_scores_farm_{f}_s{s}.npz",
                   "train": "train_eps25", "ev": lambda e: f"ev{e}_eps25"},
    "ownB50":     {"npz": lambda f, s: f"results/models/ownB_scores_farm_{f}_s{s}.npz",
                   "train": "train_eps50", "ev": lambda e: f"ev{e}_eps50"},
    "ownB100":    {"npz": lambda f, s: f"results/models/ownB_scores_farm_{f}_s{s}.npz",
                   "train": "train_eps100", "ev": lambda e: f"ev{e}_eps100"},
    "ownB250":    {"npz": lambda f, s: f"results/models/ownB_scores_farm_{f}_s{s}.npz",
                   "train": "train_eps250", "ev": lambda e: f"ev{e}_eps250"},
    "ownB_recon": {"npz": lambda f, s: f"results/models/ownB_scores_farm_{f}_s{s}.npz",
                   "train": "train_recon", "ev": lambda e: f"ev{e}_recon"},
    # score fisico model-free de violacion de coherencia entre sensores redundantes
    # (|Spearman| >= thr), ortogonal por construccion al no ser ni recon ni difusion, y de
    # dominio. phys90, 13 pares, es el mas fuerte; phys95, 9 pares, el mas ortogonal.
    # siempre _s{seed} y alineado a la meta AE
    "phys90": {"npz": lambda f, s: f"results/models/phys_scores_farm_{f}_s{s}.npz",
               "train": "train_phys90", "ev": lambda e: f"ev{e}_phys90"},
    "phys95": {"npz": lambda f, s: f"results/models/phys_scores_farm_{f}_s{s}.npz",
               "train": "train_phys95", "ev": lambda e: f"ev{e}_phys95"},
    # agg=max, rotura transitoria de la coherencia, ultra-ortogonal ~0.07-0.11 y Earliness alta
    "phys90max": {"npz": lambda f, s: f"results/models/phys_scores_farm_{f}_s{s}.npz",
                  "train": "train_phys90max", "ev": lambda e: f"ev{e}_phys90max"},
    "phys95max": {"npz": lambda f, s: f"results/models/phys_scores_farm_{f}_s{s}.npz",
                  "train": "train_phys95max", "ev": lambda e: f"ev{e}_phys95max"},
    # rate=max sobre la derivada del residuo, o sea la rotura de la tasa de coherencia, y el
    # mas ortogonal de todos (~0.035)
    "phys90maxrate": {"npz": lambda f, s: f"results/models/phys_scores_farm_{f}_s{s}.npz",
                      "train": "train_phys90maxrate", "ev": lambda e: f"ev{e}_phys90maxrate"},
    "phys95maxrate": {"npz": lambda f, s: f"results/models/phys_scores_farm_{f}_s{s}.npz",
                      "train": "train_phys95maxrate", "ev": lambda e: f"ev{e}_phys95maxrate"},
    # umbrales flojos para Farm B, cuyos pares redundantes son menos perfectos
    "phys85": {"npz": lambda f, s: f"results/models/phys_scores_farm_{f}_s{s}.npz",
               "train": "train_phys85", "ev": lambda e: f"ev{e}_phys85"},
    "phys85max": {"npz": lambda f, s: f"results/models/phys_scores_farm_{f}_s{s}.npz",
                  "train": "train_phys85max", "ev": lambda e: f"ev{e}_phys85max"},
    # ImDiffusion, 2 variantes escalares por ventana. 'mean' es redundante con el AE en raw
    # (Pearson 0.83) pero decorrelable por calibracion (Spearman 0.56); 'median' es
    # rank-ortogonal a todo (~0.2). la ruta depende de la granja y solo hay 2 semillas, 42 y 7.
    # en B las ventanas no alinean 1:1 con la meta (split3/regime) y el harness aborta al verlo
    "imdiff":     {"npz": lambda f, s: _imdiff_npz(f, s),
                   "train": "train_mean", "ev": lambda e: f"ev{e}_mean"},
    "imdiff_med": {"npz": lambda f, s: _imdiff_npz(f, s),
                   "train": "train_median", "ev": lambda e: f"ev{e}_median"},
    # senales no-avg del atlas, salidas de la criba seed-42 (signal_atlas_c.json). rng es el
    # rango max-min, ultra-ortogonal (0.021/0.027), Earl 0.76. stdr es el burst de varianza
    # rate-top3 (0.116/0.107, Earl 0.72). ambos ~0.6 con phys
    "atlas_rng":  {"npz": lambda f, s: f"results/models/atlas_scores_farm_{f}_s{s}.npz",
                   "train": "train_rng_max_mean", "ev": lambda e: f"ev{e}_rng_max_mean"},
    "atlas_stdr": {"npz": lambda f, s: f"results/models/atlas_scores_farm_{f}_s{s}.npz",
                   "train": "train_std_rate_top3", "ev": lambda e: f"ev{e}_std_rate_top3"},
    # canal de disponibilidad, la fraccion de status distinto de 0 por ventana sobre el flujo
    # sin filtrar. determinista, las 5 semillas comparten contenido. ve 4 eventos oscuros
    # (35, 70, 81, 90) que ningun detector de valores ve, con Earl baja (0.31)
    "avail":      {"npz": lambda f, s: f"results/models/avail_scores_farm_{f}_s{s}.npz",
                   "train": "train_avail_status", "ev": lambda e: f"ev{e}_avail_status"},
    # TrajDev v2, forecast-departure sobre canales fisicos no-avg (run_trajdev_c.py --repr
    # atlas --n-models 3 --dump-scores). own2_zmax = dep_zmax_sens (0.7314±0.001 solo, corr ae
    # 0.017, crgn 0.015, phys 0.62); own2_ztop3 = dep_z_top3 (0.7209±0.003; 0.034/0.031/0.61);
    # own2_end = dep_end, la mas ortogonal a phys (0.26)
    "own2_zmax":  {"npz": lambda f, s: f"results/models/trajdev2_scores_farm_{f}_s{s}.npz",
                   "train": "train_dep_zmax_sens", "ev": lambda e: f"ev{e}_dep_zmax_sens"},
    "own2_ztop3": {"npz": lambda f, s: f"results/models/trajdev2_scores_farm_{f}_s{s}.npz",
                   "train": "train_dep_z_top3", "ev": lambda e: f"ev{e}_dep_z_top3"},
    "own2_end":   {"npz": lambda f, s: f"results/models/trajdev2_scores_farm_{f}_s{s}.npz",
                   "train": "train_dep_end", "ev": lambda e: f"ev{e}_dep_end"},
    # TrajDev v3, cabeza de valores (atlas3, val__ top100 mas panel top-40). dep_zmax_val es el
    # forecast-departure max-z sobre el mundo del AE, o sea un AE mejor (0.7399+-0.010, Acc
    # 0.906, corr ae 0.88); dep_zmax_all es la union interna plana
    "own3_val": {"npz": lambda f, s: f"results/models/trajdev3_scores_farm_{f}_s{s}.npz",
                 "train": "train_dep_zmax_val", "ev": lambda e: f"ev{e}_dep_zmax_val"},
    # tasa del error de pronostico sobre valores (zslope_val), fuerte (0.733) y a la vez
    # ortogonal a los tres miembros del campeon (ae 0.09, crgn 0.23, phys 0.07)
    "own3_slope": {"npz": lambda f, s: f"results/models/trajdev3s_scores_farm_{f}_s{s}.npz",
                   "train": "train_dep_zslope_val", "ev": lambda e: f"ev{e}_dep_zslope_val"},
    "own3_all": {"npz": lambda f, s: f"results/models/trajdev3_scores_farm_{f}_s{s}.npz",
                 "train": "train_dep_zmax_all", "ev": lambda e: f"ev{e}_dep_zmax_all"},
    # dep_z_pav, media-z de los canales de coherencia pav__ y sdp__ del panel atlas3. es el
    # detector v3 mas temprano (Earl 0.630, frente a 0.536 de own3_all y 0.466 de own3_val) y
    # el de mayor cobertura (0.844), pero poco fiable (Rel 0.628), con muchas falsas alarmas
    "own3_pav": {"npz": lambda f, s: f"results/models/trajdev3_scores_farm_{f}_s{s}.npz",
                 "train": "train_dep_z_pav", "ev": lambda e: f"ev{e}_dep_z_pav"},
    # dep_z_val, media-z sobre los canales val en lugar del max de own3_val. es el segundo mas
    # temprano (Earl 0.590) y se basa en valores, asi que su evidencia es coherente con el
    # score, sin el decoupling entre pav y val
    "own3_valmean": {"npz": lambda f, s: f"results/models/trajdev3_scores_farm_{f}_s{s}.npz",
                     "train": "train_dep_z_val", "ev": lambda e: f"ev{e}_dep_z_val"},
    # TrajDev-B, la replica de v3 en Farm B (run_trajdev_b.py --dump-scores, curated stride 1),
    # alinea 1:1 con la meta AE y siempre lleva _s{seed}. ownb3_zmax = dep_zmax_val, la
    # ganadora de C, que en B da 0.6864+-0.008 con corr_ae ~0.50; ownb3_ztop3 = dep_z_top3, el
    # mejor agregador curated de B, 0.7021 con corr_ae ~0.62
    "ownb3_zmax":  {"npz": lambda f, s: f"results/models/trajdevb_scores_farm_{f}_s{s}.npz",
                    "train": "train_dep_zmax_val", "ev": lambda e: f"ev{e}_dep_zmax_val"},
    "ownb3_ztop3": {"npz": lambda f, s: f"results/models/trajdevb_scores_farm_{f}_s{s}.npz",
                    "train": "train_dep_z_top3", "ev": lambda e: f"ev{e}_dep_z_top3"},
    # gdn se anade cuando se vuelquen sus npz alineados
}


# la ruta del volcado de ImDiffusion cambia de stem segun la granja
def _imdiff_npz(f: str, s: int) -> str:
    f = f.lower()
    stem = "top100_offset_regime_split3" if f == "c" else "soft_offset_regime_split3"
    return f"results/models/imdiffusion_farm_{f}_{stem}_s{s}_scores.npz"


def parse_calib(spec: str, dets: list[str]) -> dict:
    """mapea cada detector a su normalizador; los que no cita la spec se quedan con zscore."""

    # la spec son pares detector:normalizador, 'ae:zscore,crgn:zscore,imdiff:gmm_tail'
    m = {}
    for part in spec.split(","):
        d, nm = part.split(":")
        if nm.strip() not in C.NORMALIZERS:
            raise ValueError(f"normalizador desconocido: {nm}")
        m[d.strip()] = nm.strip()
    return {d: m.get(d, "zscore") for d in dets}


class ReplayScorer:
    """devuelve scores precomputados en orden, primero el train y luego cada evento."""

    def __init__(self, scores):
        self._scores, self._i = scores, 0


    # el harness llama una vez por bloque y en orden, basta con avanzar el cursor
    def score_windows(self, windows):  # noqa: ARG002
        s = self._scores[self._i]
        self._i += 1
        return s

    percentile = 99.0  # solo afecta al campo 'threshold' reportado, no al CARE-CV


# localiza el npz y devuelve el train y los eventos, en el orden de eids
def load_detector(farm: str, name: str, seed: int, eids: list[str]):
    d = DETECTORS[name]
    z = np.load(d["npz"](farm, seed))
    train = np.asarray(z[d["train"]], dtype=float)
    evs = [np.asarray(z[d["ev"](e)], dtype=float) for e in eids]
    return train, evs


# secuencias de test para el harness. la meta aporta labels y timestamps, y las
# ventanas van vacias porque el scorer es un replay y no vuelve a puntuar nada
def build_test_seqs(meta: dict) -> list[dict]:
    return [{"event_id": int(e), "event_label": m["event_label"], "asset_id": int(m["asset_id"]),
             "windows": np.empty(0), "labels": np.array(m["labels"], np.int8),
             "window_timestamps": np.array(m["window_timestamps"], dtype="datetime64[ns]")}
            for e, m in meta.items()]


# corre el harness oficial sobre un score ya fusionado. care_cv es la cifra que
# se compara, el resto son sub-metricas del mejor h del sweep, informativas
def eval_scores(train_fused, ev_fused, test_seqs, adapter, cv_seed=42) -> dict:
    rs = ReplayScorer([train_fused] + list(ev_fused))
    ev = run_model.evaluate(rs, np.empty(0), test_seqs, adapter, cv_seed)
    b, cv = ev["best"], ev["cv"]
    return {"care_cv": cv["care"], "oracle": b["care"], "cov": b["cov"], "earl": b["earl"],
            "rel": b["rel"], "acc": b["acc"], "roc_ds": b["roc_ds"]}


# media, desviacion y la lista por semilla, que es lo que acaba en el JSON
def agg(cares: list[float]) -> dict:
    a = np.array(cares, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "seeds": [round(x, 4) for x in a]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", required=True, choices=["B", "C"])
    ap.add_argument("--detectors", required=True, help="lista separada por comas, ej: ae,crgn")
    ap.add_argument("--seeds", default="42,7,123,1234,12345")
    ap.add_argument("--normalizers", default="zscore,gaussian_tail",
                    help="modo UNIFORME: mismo normalizador a todos los detectores (barre cada uno). "
                         "Opciones: zscore,gaussian_tail,ecdf,gmm_tail")
    ap.add_argument("--calib", default="", help="modo SELECTIVO por-detector (anula --normalizers), "
                    "ej: 'ae:zscore,crgn:zscore,imdiff:gmm_tail'. Detectores no citados usan zscore")
    ap.add_argument("--cv-seed", type=int, default=42, help="semilla del split de CV (fijo, como las campanas)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    farm = args.farm
    fl = farm.lower()
    dets = args.detectors.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    norms = args.normalizers.split(",")
    n = len(dets)

    # la meta es canonica, invariante a la semilla, y fija el orden de eventos
    meta = json.load(open(f"results/models/ae_scores_farm_{fl}_meta.json", encoding="utf-8"))
    eids = list(meta.keys())
    test_seqs = build_test_seqs(meta)
    adapter = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm=farm, window_size=27)
    mode = f"calib[{args.calib}]" if args.calib else f"norm{norms}"
    logger.info(f"Farm {farm} | detectores {dets} | seeds {seeds} | {mode} | {len(eids)} eventos")

    # dos modos de calibracion, uniforme (un normalizador para todos, que se barre) o selectivo
    # con --calib. cada variante es (nombre, mapa de detector a normalizador), y el mapa se
    # guarda por referencia para precalibrar una sola vez por semilla y variante en el bucle de
    # abajo, que es lo que abarata el gmm_tail
    if args.calib:
        calib_variants = [("calib", parse_calib(args.calib, dets))]
    else:
        calib_variants = [(nm, {d: nm for d in dets}) for nm in norms]

    # cada config es (etiqueta, calibracion, combinador) y no depende de la semilla
    configs = []
    for cname, cmap in calib_variants:
        for w in C.weight_grid(n):
            wtag = "/".join(f"{x:.2f}" for x in w)
            configs.append((f"{cname}|avg[{wtag}]", cmap, lambda zs, w=w: C.combine_average(zs, w)))
        configs.append((f"{cname}|max", cmap, lambda zs: C.combine_maximum(zs)))
        # aom y moa agregan por grupos, y con menos de tres detectores son degenerados
        if n >= 3:
            configs.append((f"{cname}|aom", cmap, lambda zs: C.combine_aom(zs, n_buckets=2)))
            configs.append((f"{cname}|moa", cmap, lambda zs: C.combine_moa(zs, n_buckets=2)))

    # acumuladores. single_cares es el baseline de cada detector solo, y fused_last
    # guarda las sub-metricas de la ultima semilla, que no deciden nada
    single_cares = {d: [] for d in dets}
    fused_cares = {lbl: [] for lbl, _, _ in configs}
    fused_last = {lbl: None for lbl, _, _ in configs}
    corr_acc = np.zeros((n, n))

    for seed in seeds:
        raw = {}  # por detector, su train y la lista de scores por evento
        for d in dets:
            tr, evs = load_detector(fl, d, seed, eids)
            # tantos scores como labels, o el detector no esta alineado
            for e, ev, m in zip(eids, evs, test_seqs):
                if len(ev) != len(m["labels"]):
                    raise RuntimeError(f"desalineado {d} seed{seed} ev{e}: {len(ev)} vs meta {len(m['labels'])}")
            raw[d] = (tr, evs)

        # los trains no estan alineados 1:1 (AE stride 1 frente a CRGN stride 8) y solo sirven
        # para calibrar el CUSUM del score fusionado. h_base = percentile(CUSUM(train)) corre
        # secuencialmente, de modo que la calibracion depende del orden y de la autocorrelacion
        # y el remuestreo tiene que preservar los dos. de ahi el repeat, un ZOH x8. las
        # ventanas adyacentes solapan 26 de 27, asi que aproxima la secuencia stride-1 real. el
        # detector mas largo (n_big) queda intacto, y con peso 0 al resto los endpoints salen
        # exactos. los eventos si alinean 1:1
        n_big = max(len(raw[d][0]) for d in dets)
        train_rs = {d: (raw[d][0] if len(raw[d][0]) >= n_big
                        else np.repeat(raw[d][0], -(-n_big // len(raw[d][0])))[:n_big]) for d in dets}

        # correlacion entre detectores sobre test alineado (nivel del score que entra al CUSUM)
        cat = {d: np.concatenate(raw[d][1]) for d in dets}
        for i in range(n):
            for j in range(n):
                corr_acc[i, j] += float(np.corrcoef(cat[dets[i]], cat[dets[j]])[0, 1])

        # baseline, cada detector con su raw sin normalizar, o sea su CARE real
        for d in dets:
            tr, evs = raw[d]
            single_cares[d].append(eval_scores(tr, evs, test_seqs, adapter, args.cv_seed)["care_cv"])

        # por variante de calibracion se calibra cada detector una sola vez y se le aplican los
        # combinadores de esa variante, asi gmm_tail ajusta un GMM por detector y semilla, no
        # uno por config. endpoint exacto, con calib zscore y peso 0 reproduce el AE
        for cname, cmap in calib_variants:
            # las stats salen del train completo y se aplican al train ya remuestreado
            st = {d: C.NORMALIZERS[cmap[d]][0](raw[d][0]) for d in dets}
            zt = {d: C.NORMALIZERS[cmap[d]][1](train_rs[d], st[d]) for d in dets}
            ze = {d: [C.NORMALIZERS[cmap[d]][1](ev, st[d]) for ev in raw[d][1]] for d in dets}
            for lbl, cm, fn in configs:
                if cm is not cmap:  # identidad, no igualdad, solo los de esta variante
                    continue
                train_fused = fn([zt[d] for d in dets])
                ev_fused = [fn([ze[d][i] for d in dets]) for i in range(len(eids))]
                r = eval_scores(train_fused, ev_fused, test_seqs, adapter, args.cv_seed)
                fused_cares[lbl].append(r["care_cv"])
                fused_last[lbl] = r
        logger.info(f"  seed {seed} listo | AE-solo {single_cares[dets[0]][-1]:.4f}")

    corr = (corr_acc / len(seeds)).tolist()
    ae_seed_cares = single_cares[dets[0]]  # el primero de la lista es la referencia

    # ranking de fusiones por CARE-CV medio, con wins pareados vs AE
    rows = []
    for lbl, _, _ in configs:
        a = agg(fused_cares[lbl])
        wins = int(sum(1 for f, ae in zip(fused_cares[lbl], ae_seed_cares) if f > ae))
        delta = float(np.mean(np.array(fused_cares[lbl]) - np.array(ae_seed_cares)))
        rows.append({"config": lbl, **a, "wins_vs_ae": f"{wins}/{len(seeds)}", "delta_vs_ae": round(delta, 4),
                     "last_seed": {k: round(v, 4) for k, v in fused_last[lbl].items()}})
    rows.sort(key=lambda r: -r["mean"])

    singles = {d: agg(single_cares[d]) for d in dets}
    logger.info("=" * 78)
    logger.info(f"BASELINES (raw, {len(seeds)} seeds):")
    for d in dets:
        logger.info(f"  {d:<6} CARE-CV = {singles[d]['mean']:.4f} +- {singles[d]['std']:.4f}  {singles[d]['seeds']}")
    logger.info(f"CORR (test, medio): {[[round(c,3) for c in r] for r in corr]}  ({dets})")
    logger.info(f"TOP-5 fusiones:")
    for r in rows[:5]:
        logger.info(f"  {r['config']:<26} {r['mean']:.4f} +- {r['std']:.4f}  wins {r['wins_vs_ae']}  d{r['delta_vs_ae']:+.4f}")

    # la configuracion entera viaja con los resultados, y siempre a fichero nuevo
    out = {"farm": farm, "detectors": dets, "seeds": seeds, "normalizers": norms, "calib": args.calib,
           "cv_seed": args.cv_seed, "corr_test_mean": corr, "baselines": singles, "fusions": rows}
    p = Path(f"results/tables/fuse_{args.tag}_{fl}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=2, default=_np_default)
    logger.info(f"guardado -> {p}")


if __name__ == "__main__":
    main()
