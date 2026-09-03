"""antelacion de deteccion por evento, en horas, de los BiLSTM-AE congelados.

reconstruye las decisiones del protocolo oficial en su punto de operacion publicado, con el
CUSUM estadistico de mu, k y h_base al percentil 99 del train y h = h_base * h_scale elegido
por CV, que sale constante en todos los folds, B=1.0 y C=0.002. de cada evento registra el
primer cruce S > h.

    t_alarm            timestamp de la ventana donde se dispara la alarma
    lead_hours_to_end  horas de antelacion respecto a event_end, o sea al fallo
    offset_start_h     horas desde event_start hasta la alarma, negativo si la alarma
                       llega antes que el evento

lleva un sanity check dentro. con esos mismos scores reproduce el CARE-CV por
run_model.evaluate y lo compara con el congelado, B 0.7211 y C 0.7432; si no casa, los
tiempos no son fiables y el script lo avisa.

no reentrena nada, usa los checkpoints y escaladores congelados y los datos ya procesados.

    Farm B  data/processed/farm_b_{train_windows,test_sequences}.npz, mas
            farm_b_test_metadata.json y results/models/farm_b_{ident_s42_best.pt,scaler.pkl,pca.pkl};
            es la ruta window-first, de 30 feats a K=18
    Farm C  data/processed/farm_c_pcafirst_timeaddm.pkl, la representacion ganadora de K=55
            componentes, mas results/models/farm_c_best.pt y event_info.csv para los limites
            de cada evento

    python compute_lead_times.py [--farm B|C]

sin --farm corre las dos granjas.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (imports hermanos)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz del repo (from src.X)

from src.models.bilstm_autoencoder import BiLSTMAutoencoder
from src.models.anomaly_detector import fit_cusum_params, run_cusum
from src.model_prep.isolation_forest_cleaner import IFCleaner
from src.care_official.adapter import OfficialCAREAdapter
from sweep_timeaddm_scoring import ReplayScorer
import run_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
# el ckpt desplegado de B, farm_b_best.pt, se perdio, y la copia que queda en el historico de
# git es K=14 y no 18, asi que su load_state_dict revienta por size mismatch. en su lugar va el
# reentrenamiento identico de la semilla 42, que replica el camino oficial del 0.7211 y
# reproduce el punto de operacion publicado, CARE-CV 0.7213 y la misma tabla de antelaciones
FROZEN = {"B": {"care_cv": 0.7211, "h_scale": 1.0, "K": 18,
                "ckpt": "results/models/farm_b_ident_s42_best.pt"},
          "C": {"care_cv": 0.7432, "h_scale": 0.002, "K": 55, "ckpt": "results/models/farm_c_best.pt"}}


@torch.no_grad()
def mae_scores(model: torch.nn.Module, windows: np.ndarray, batch: int = 1024) -> np.ndarray:
    """error de reconstruccion MAE por ventana, la misma definicion que
    AnomalyDetector.mae_window."""
    out = []
    for s in range(0, len(windows), batch):
        x = torch.tensor(windows[s:s + batch], dtype=torch.float32, device=DEVICE)
        out.append((x - model(x)).abs().mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def load_farm_b():
    """artefactos window-first congelados, las ventanas crudas de 30 feats pasadas por el
    scaler y luego por el PCA hasta K=18 componentes."""
    scaler = pickle.load(open("results/models/farm_b_scaler.pkl", "rb"))
    pca = pickle.load(open("results/models/farm_b_pca.pkl", "rb"))
    z = np.load("data/processed/farm_b_train_windows.npz", allow_pickle=True)
    train = pca.transform(scaler.transform(z["windows"].astype(np.float32)))
    # la ejecucion oficial de B limpia con IsolationForest antes de calibrar, encendido en la
    # config base
    train = IFCleaner(contamination=0.01, n_estimators=100, random_state=42, n_jobs=-1).fit_transform(train)
    meta = json.load(open("data/processed/farm_b_test_metadata.json", encoding="utf-8"))
    zt = np.load("data/processed/farm_b_test_sequences.npz", allow_pickle=True)
    seqs, bounds = [], {}
    for eid, m in meta.items():
        comps = pca.transform(scaler.transform(zt[eid].astype(np.float32)))
        seqs.append({"event_id": int(eid), "asset_id": m["asset_id"], "event_label": m["event_label"],
                     "windows": comps, "labels": np.asarray(m["labels"], int),
                     "window_timestamps": pd.to_datetime(m["window_timestamps"])})
        bounds[int(eid)] = (pd.Timestamp(m["event_start"]), pd.Timestamp(m["event_end"]))
    return train, seqs, bounds


def load_farm_c():
    """cache pca-first congelada, la representacion ganadora de K=55 componentes. aqui no hay
    IF, per_farm C lo desactiva."""
    d = pickle.load(open("data/processed/farm_c_pcafirst_timeaddm.pkl", "rb"))
    info = pd.read_csv("data/raw/CARE_To_Compare/Wind Farm C/event_info.csv", sep=";", encoding="latin-1")
    bounds = {int(r.event_id): (pd.Timestamp(r.event_start), pd.Timestamp(r.event_end))
              for r in info.itertuples()}
    seqs = []
    for s in d["test"]:
        s = dict(s)
        s["window_timestamps"] = pd.to_datetime(np.asarray(s["window_timestamps"]))
        seqs.append(s)
    return d["train"].astype(np.float32, copy=False), seqs, bounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", choices=["B", "C"], default=None, help="por defecto, las dos")
    args = ap.parse_args()
    farms = [args.farm] if args.farm else ["B", "C"]

    for farm in farms:
        fz = FROZEN[farm]
        logger.info(f"=== Farm {farm} | frozen CARE-CV {fz['care_cv']} | h_scale {fz['h_scale']} ===")
        train, seqs, bounds = load_farm_b() if farm == "B" else load_farm_c()
        # el evaluador oficial necesita el tamano de ventana, y sale de la representacion
        W = train.shape[1]
        model = BiLSTMAutoencoder(input_size=fz["K"], encoder_units=[64, 16], decoder_units=[16, 64],
                                  dropout=0.1, out_activation="linear")
        # strict=False tolera claves que sobren o falten, pero no un tamano distinto
        model.load_state_dict(torch.load(fz["ckpt"], map_location=DEVICE), strict=False)
        model.to(DEVICE).eval()

        logger.info(f"scoring train {train.shape} + {len(seqs)} events ...")
        train_err = mae_scores(model, train)
        ev_err = [mae_scores(model, s["windows"]) for s in seqs]

        # sanity, esos mismos scores por el evaluador oficial dan el CARE-CV congelado
        adapter = OfficialCAREAdapter(raw_path="data/raw/CARE_To_Compare", farm=farm, window_size=W)
        ev = run_model.evaluate(ReplayScorer([train_err] + ev_err), train, seqs, adapter, SEED)
        cv = ev["cv"]["care"]
        delta = cv - fz["care_cv"]
        # 2e-3 de tolerancia; por encima, el punto de operacion no es el publicado
        status = "OK" if abs(delta) < 2e-3 else "MISMATCH — tiempos NO fiables"
        logger.info(f"[Sanity] CARE-CV recomputado = {cv:.4f} vs congelado {fz['care_cv']:.4f} "
                    f"(delta {delta:+.4f}) -> {status}")

        # tiempos de alarma en el punto de operacion publicado, con el h_base del percentil 99
        # del train escalado por el h_scale de la granja
        mu, k, h_base = fit_cusum_params(train_err, slack=1.0, percentile=99.0)
        h = h_base * fz["h_scale"]
        rows = []
        for s, e in zip(seqs, ev_err):
            S = run_cusum(e, mu, k)
            # la alarma es el primer cruce del umbral; si no hay ninguno, el evento no alarma
            above = np.flatnonzero(S > h)
            idx = int(above[0]) if len(above) else None
            ts = s["window_timestamps"]
            t_alarm = ts[idx] if idx is not None else None
            ev_start, ev_end = bounds.get(int(s["event_id"]), (None, None))
            is_anom = s["event_label"] == "anomaly"
            # la antelacion solo va en anomalias; un cruce en un normal es falsa alarma
            lead_end = (ev_end - t_alarm).total_seconds() / 3600 if (is_anom and t_alarm is not None) else np.nan
            off_start = (t_alarm - ev_start).total_seconds() / 3600 if (is_anom and t_alarm is not None) else np.nan
            rows.append({"event_id": int(s["event_id"]), "asset_id": s["asset_id"],
                         "event_label": s["event_label"],
                         "event_start": ev_start, "event_end": ev_end,
                         "alarm_raised": idx is not None,
                         "t_alarm": t_alarm,
                         "lead_hours_to_end": round(lead_end, 1) if np.isfinite(lead_end) else None,
                         "offset_start_h": round(off_start, 1) if np.isfinite(off_start) else None,
                         "false_alarm": (idx is not None) and not is_anom})
        df = pd.DataFrame(rows).sort_values(["event_label", "event_id"])
        out = Path(f"results/tables/farm_{farm.lower()}_lead_times.csv")
        df.to_csv(out, index=False)

        an = df[df.event_label == "anomaly"]
        det = an[an.alarm_raised]
        fa = df[(df.event_label == "normal") & df.false_alarm]
        logger.info(f"[{farm}] anomalias detectadas: {len(det)}/{len(an)} | falsas alarmas: {len(fa)}/{(df.event_label=='normal').sum()}")
        if len(det):
            logger.info(f"[{farm}] antelacion (h hasta event_end): media {det.lead_hours_to_end.mean():.1f} | "
                        f"mediana {det.lead_hours_to_end.median():.1f} | min {det.lead_hours_to_end.min():.1f} | "
                        f"max {det.lead_hours_to_end.max():.1f}")
        logger.info(f"saved -> {out}")


if __name__ == "__main__":
    main()
