"""corre el pipeline de preprocesado de Farm B y Farm C.

    python run_preprocessing.py
    python run_preprocessing.py --farm B
    python run_preprocessing.py --patch config/experiments/l6_light.yaml --out mi_corpus \\
        --farm B
    python run_preprocessing.py --patch config/experiments/canonico_C.yaml --out mi_corpus \\
        --farm C
"""
import sys
import io
import argparse
import logging
from pathlib import Path

import yaml

# la terminal de Windows puede no traer UTF-8 por defecto
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

from src.preprocessing import load_config, PreprocessingPipeline
from src.preprocessing.utils import col_to_sensor_name


def _deep_merge(base: dict, patch: dict) -> dict:
    # fusion recursiva, el patch pisa hoja a hoja y los dicts anidados se mezclan en vez de
    # sustituirse enteros
    result = base.copy()
    for k, v in patch.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def print_feature_table(feature_cols, sensor_registry, farm_label):
    # el registro describe cada sensor por su nombre, y la columna lleva el stat pegado detras
    desc_map = dict(zip(sensor_registry["sensor_name"], sensor_registry["description"]))
    print(f"\n{'='*70}")
    print(f"  Farm {farm_label} — {len(feature_cols)} features selected")
    print(f"{'='*70}")
    print(f"  {'#':<4} {'Column':<35} {'Description'}")
    print(f"  {'-'*4} {'-'*35} {'-'*30}")
    for i, col in enumerate(feature_cols, 1):
        base = col_to_sensor_name(col)
        desc = desc_map.get(base, "—")
        print(f"  {i:<4} {col:<35} {desc}")


def compare_features(result_b, result_c):
    # las dos granjas vienen anonimizadas por separado, asi que las columnas no se pueden
    # emparejar por nombre y se emparejan por el texto de la descripcion
    desc_b = dict(zip(result_b.sensor_registry["sensor_name"],
                      result_b.sensor_registry["description"].str.lower()))
    desc_c = dict(zip(result_c.sensor_registry["sensor_name"],
                      result_c.sensor_registry["description"].str.lower()))

    feats_b = {col_to_sensor_name(c): c for c in result_b.feature_cols}
    feats_c = {col_to_sensor_name(c): c for c in result_c.feature_cols}

    matched = []
    unmatched_b = []
    unmatched_c = list(feats_c.keys())

    # a cada columna de B se le busca la de C con mas palabras en comun, en Jaccard sobre
    # las palabras de las dos descripciones
    for sn_b, col_b in feats_b.items():
        kws_b = set(desc_b.get(sn_b, "").split())
        best_match_sn = None
        best_score = 0.0
        for sn_c in unmatched_c:
            kws_c = set(desc_c.get(sn_c, "").split())
            denom = max(len(kws_b | kws_c), 1)
            score = len(kws_b & kws_c) / denom
            if score > best_score:
                best_score = score
                best_match_sn = sn_c

        # 0.4 de solape es el corte; por debajo, la columna de B se declara sin pareja
        if best_match_sn and best_score >= 0.4:
            matched.append((col_b, feats_c[best_match_sn], best_score,
                            desc_b.get(sn_b, ""), desc_c.get(best_match_sn, "")))
            # el emparejamiento es uno a uno, el candidato elegido sale de la lista
            unmatched_c.remove(best_match_sn)
        else:
            unmatched_b.append((col_b, desc_b.get(sn_b, "")))

    print(f"  FEATURE OVERLAP ANALYSIS — Farm B ({len(result_b.feature_cols)}) vs Farm C ({len(result_c.feature_cols)})")
    print(f"  Matched pairs: {len(matched)}  |  Only in B: {len(unmatched_b)}  |  Only in C: {len(unmatched_c)}")
    print(f"\n  {'Farm B column':<35} {'Farm C column':<35} {'Sim':>5}  Description (B)")
    print(f"  {'-'*35} {'-'*35} {'-'*5}  {'-'*35}")
    for col_b, col_c, sim, desc, _ in sorted(matched, key=lambda x: -x[2]):
        print(f"  {col_b:<35} {col_c:<35} {sim:>5.2f}  {desc[:50]}")

    if unmatched_b:
        print(f"\n  Features only in Farm B:")
        for col, desc in unmatched_b:
            print(f"    {col:<35} {desc}")

    if unmatched_c:
        print(f"\n  Features only in Farm C (no B match):")
        for sn in unmatched_c:
            print(f"    {feats_c[sn]:<35} {desc_c.get(sn, '')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm", choices=["B", "C"], default=None)
    parser.add_argument("--patch", default=None, metavar="YAML",
                        help="YAML de parche para sobrescribir claves concretas de la config")
    parser.add_argument("--out", default=None, metavar="SUFFIX",
                        help="sufijo del directorio de salida, guarda en data/processed_{SUFFIX}/")
    args = parser.parse_args()

    config = load_config()

    # el patch YAML se superpone sobre la config canonica, no la sustituye entera
    if args.patch:
        patch_path = Path(args.patch)
        if not patch_path.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_path}")
        with open(patch_path, encoding="utf-8") as f:
            patch = yaml.safe_load(f)
        config = _deep_merge(config, patch)
        print(f"Patch applied: {patch_path.name} -> {patch}")

    # con --out el corpus va a un directorio aparte, para no pisar el procesado oficial
    if args.out:
        config["data"]["processed_path"] = f"data/processed_{args.out}"
        print(f"Output directory: {config['data']['processed_path']}")

    pipeline = PreprocessingPipeline(config)
    farms = [args.farm] if args.farm else ["B", "C"]
    results = {}

    for farm in farms:
        print(f"\n>>> Running Farm {farm} …")
        result = pipeline.run(farm)
        print_feature_table(result.feature_cols, result.sensor_registry, farm)
        results[farm] = result

    # la comparacion de features solo tiene sentido si se han corrido las dos granjas
    if "B" in results and "C" in results:
        compare_features(results["B"], results["C"])

    out_dir = config["data"]["processed_path"]
    print("\n SUMMARY")
    for farm, r in results.items():
        tw = r.train_windows
        print(f"  Farm {farm}: train={tw.shape}, test_seqs={len(r.test_sequences)}, "
              f"features={len(r.feature_cols)}")
    print(f"Done. Outputs in {out_dir}/")
