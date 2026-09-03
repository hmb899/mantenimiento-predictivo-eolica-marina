"""
CUSUM oficial para los timelines de los notebooks

usa las mismas primitivas que el harness, src.models.anomaly_detector.fit_cusum_params y
run_cusum, y replica el h-sweep de run_model.

official_cusum
    calibra (mu, k, h_base) sobre los scores de train
cusum_series, first_alarm
    traza acumulada de un evento y primer cruce del umbral
h_sweep_best
    barre los multiplicadores de h y se queda con el punto de operacion de mejor CARE
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# los notebooks importan esto sin instalar el paquete, de ahi el sys.path a mano
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.anomaly_detector import fit_cusum_params, run_cusum  # noqa: E402

# mismos multiplicadores que run_model.H_SCALES (el sweep oficial)
H_SCALES = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def official_cusum(train_scores: np.ndarray) -> tuple[float, float, float]:
    """(mu, k, h_base) con la receta oficial, slack=1.0 y percentil 99 sobre el train
    secuencial."""

    return fit_cusum_params(np.asarray(train_scores, float), slack=1.0, percentile=99.0)


def cusum_series(scores: np.ndarray, mu: float, k: float) -> np.ndarray:
    # traza CUSUM de un evento con los parametros ya calibrados en train
    return run_cusum(np.asarray(scores, float), mu, k)


def first_alarm(S: np.ndarray, h: float) -> int | None:
    # indice del primer cruce del umbral, None si la traza no lo cruza nunca
    above = np.flatnonzero(S > h)
    return int(above[0]) if len(above) else None


def h_sweep_best(train_scores: np.ndarray, ev_scores: dict[int, np.ndarray], meta: dict[int, dict],
                 adapter) -> dict:
    """replica el h-sweep oficial y devuelve el punto de operacion de mejor CARE (oracle del
    sweep, para timelines ilustrativos; los numeros de decision siguen siendo los CARE-CV
    committeados)."""

    mu, k, h_base = official_cusum(train_scores)
    ev_S = {e: cusum_series(s, mu, k) for e, s in ev_scores.items()}
    best = None
    # cada escala multiplica el h calibrado y se evalua sobre todos los eventos a la vez
    for hs in H_SCALES:
        h = h_base * hs
        preds = [{"event_id": e, "asset_id": meta[e]["asset_id"], "event_label": meta[e]["event_label"],
                  "window_timestamps": meta[e]["window_timestamps"],
                  "window_predictions": (ev_S[e] > h).astype(np.int8)} for e in ev_scores]
        res, _diag = adapter.evaluate(preds)
        if best is None or res.care > best["care"]:
            best = {"care": float(res.care), "h": float(h), "h_scale": hs, "cov": float(res.coverage),
                    "earl": float(res.earliness), "rel": float(res.reliability), "acc": float(res.accuracy)}
    # las trazas viajan con el resultado para que el notebook dibuje sin recalcularlas
    return {**best, "mu": mu, "k": k, "h_base": h_base, "ev_S": ev_S}
