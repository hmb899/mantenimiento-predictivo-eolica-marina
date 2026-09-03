"""
evaluador del CARE Score y metricas estandar de ML, cuatro sub-scores por evento a
partir del contador de criticidad (Guck et al., 2024)

coverage (Fβ)
    precision y recall a nivel de ventana, solo sobre los eventos de anomalia
    β=0.5 pesa mas la precision, lo que penaliza las falsas alarmas
accuracy (Ācc)
    accuracy medio por evento (TP_rate) en los eventos normales, donde toda ventana
    deberia ser normal
reliability (EFβ)
    Fβ a nivel de evento sobre todos los eventos, un evento da alarma cuando
    max(criticality) >= tc
    verdadero positivo en un evento de anomalia, falso positivo en uno normal
earliness (WS)
    score pesado de deteccion temprana, solo en los eventos de anomalia con alarma
    WS = Σ w_i con w_i = (N - i) / N para la ventana i del periodo de anomalia que
    dispara la alarma, i = 0 es muy pronto y N-1 muy tarde

CARE Score final = (Fβ + WS + EFβ + 2·Ācc) / 5
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocessing import TestSequenceDict


def _fbeta(precision: float, recall: float, beta: float) -> float:
    """calcula el F-beta, devuelve 0 si el denominador es cero."""

    b2 = beta**2
    # con beta menor que 1 el denominador lo domina el recall y el peso cae en la precision
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


class CAREEvaluator:
    """calcula el CARE Score y sus metricas de apoyo."""

    def __init__(self, config: dict) -> None:
        """saca de config el beta y el umbral de criticidad."""

        ev_cfg = config["evaluation"]["care_score"]
        self.beta: float = ev_cfg["beta"]
        self.tc: int = config["cusum"]["alarm_threshold"]  # misma clave que el AnomalyDetector
        # el tamano de ventana solo se usa como suavizado por defecto en standard_metrics
        self.window_size: int = config.get("windowing", {}).get("window_size_samples", 1)


    def evaluate(self, test_sequences: list[TestSequenceDict], cusum_results: list[dict]) -> dict:
        """calcula los cuatro sub-scores de CARE y el CARE Score final."""

        # las dos listas van pareadas por posicion, un evento por indice
        assert len(test_sequences) == len(cusum_results)

        f_beta, prec_cov, rec_cov = self._coverage(test_sequences, cusum_results)
        acc_mean = self._accuracy(test_sequences, cusum_results)
        ef_beta, prec_rel, rec_rel = self._reliability(test_sequences, cusum_results)
        ws = self._earliness(test_sequences, cusum_results)

        # accuracy pesa doble y el total se divide entre cinco, no entre cuatro
        care_score = (f_beta + ws + ef_beta + 2 * acc_mean) / 5.0

        # resumen por evento para inspeccion, no entra en el score
        per_event = []
        for seq, res in zip(test_sequences, cusum_results):
            gt = seq["labels"]
            pred = res["window_predictions"]
            per_event.append(
                {
                    "event_id": seq["event_id"],
                    "event_label": seq["event_label"],
                    "n_windows": len(gt),
                    "n_anomaly_windows_gt": int(gt.sum()),
                    "n_predicted_anomaly": int(pred.sum()),
                    "alarm_raised": res["alarm_raised"],
                    "alarm_window_idx": res["alarm_window_idx"],
                    "max_criticality": res["max_criticality"],
                }
            )

        return {
            "coverage_precision": prec_cov,
            "coverage_recall": rec_cov,
            "coverage_fbeta": f_beta,
            "accuracy": acc_mean,
            "reliability_precision": prec_rel,
            "reliability_recall": rec_rel,
            "reliability_efbeta": ef_beta,
            "earliness_ws": ws,
            "care_score": care_score,
            "beta": self.beta,
            "tc": self.tc,
            "per_event": per_event,
        }


    def _coverage(self, test_sequences: list[TestSequenceDict], cusum_results: list[dict]) -> tuple[float, float, float]:
        """calcula el Fbeta a nivel de ventana en los eventos de anomalia, devuelve fbeta,
        precision y recall."""

        # los conteos se acumulan sobre todos los eventos de anomalia, es un micro promedio
        # y un evento largo pesa mas que uno corto
        tp = fp = fn = 0
        for seq, res in zip(test_sequences, cusum_results):
            # los eventos normales no cuentan aqui, van a accuracy
            if seq["event_label"] != "anomaly":
                continue
            gt = seq["labels"]
            pred = res["window_predictions"]
            tp += int(((pred == 1) & (gt == 1)).sum())
            fp += int(((pred == 1) & (gt == 0)).sum())
            fn += int(((pred == 0) & (gt == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return _fbeta(prec, rec, self.beta), prec, rec


    def _accuracy(self, test_sequences: list[TestSequenceDict], cusum_results: list[dict]) -> float:
        """calcula el accuracy medio por evento en los eventos normales."""

        # media de medias, cada evento normal pesa igual sin importar su longitud
        scores: list[float] = []
        for seq, res in zip(test_sequences, cusum_results):
            if seq["event_label"] != "normal":
                continue
            gt = seq["labels"]
            pred = res["window_predictions"]
            # un evento sin ventanas no aporta media, se salta
            if len(gt) == 0:
                continue
            scores.append(float((pred == gt).mean()))
        return float(np.mean(scores)) if scores else 0.0


    def _reliability(self, test_sequences: list[TestSequenceDict], cusum_results: list[dict]) -> tuple[float, float, float]:
        """calcula el Fbeta a nivel de evento sobre todos los eventos, devuelve efbeta,
        precision y recall."""

        # aqui la unidad es el evento entero, con alarma o sin ella, no la ventana
        tp = fp = fn = 0
        for seq, res in zip(test_sequences, cusum_results):
            alarm = res["alarm_raised"]
            if seq["event_label"] == "anomaly":
                if alarm:
                    tp += 1
                else:
                    fn += 1
            else:
                # un evento normal sin alarma no suma nada, el Fbeta no cuenta negativos
                if alarm:
                    fp += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return _fbeta(prec, rec, self.beta), prec, rec


    def _earliness(self, test_sequences: list[TestSequenceDict], cusum_results: list[dict]) -> float:
        """calcula el earliness pesado en los eventos de anomalia que dieron alarma."""

        scores: list[float] = []
        for seq, res in zip(test_sequences, cusum_results):
            # solo puntuan las anomalias con alarma, la no detectada ya penaliza en reliability
            if seq["event_label"] != "anomaly" or not res["alarm_raised"]:
                continue
            gt = seq["labels"]
            alarm_idx = res["alarm_window_idx"]
            anomaly_indices = np.where(gt == 1)[0]
            if len(anomaly_indices) == 0:
                continue
            # N son las ventanas anomalas del evento, no la longitud del evento
            N = len(anomaly_indices)
            # i = ventanas anomalas hasta la posicion de la alarma, ella incluida
            i = int((anomaly_indices <= alarm_idx).sum())
            # el peso es 1.0 con la alarma en la primera ventana anomala y 0.0 en la ultima
            w = (N - i) / N if N > 0 else 0.0
            scores.append(w)
        return float(np.mean(scores)) if scores else 0.0


    def standard_metrics(self, test_sequences: list[TestSequenceDict], cusum_results: list[dict], smooth_window: int | None = None) -> dict:
        """calcula precision, recall, F1 y AUC a nivel de ventana sobre todos los eventos."""

        # sin smooth_window explicito se suaviza con el tamano de ventana del config
        if smooth_window is None:
            smooth_window = self.window_size

        all_gt: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_scores_smooth: list[np.ndarray] = []

        for seq, res in zip(test_sequences, cusum_results):
            raw = res["window_errors"]
            all_gt.append(seq["labels"])
            all_pred.append(res["window_predictions"])
            all_scores.append(raw)
            # la media movil por evento evita artefactos de frontera entre eventos
            if smooth_window > 1 and len(raw) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(raw, kernel, mode="same")
            else:
                smoothed = raw.copy()
            all_scores_smooth.append(smoothed)

        # una vez suavizado por evento ya se puede concatenar, las metricas son globales
        gt = np.concatenate(all_gt)
        pred = np.concatenate(all_pred)
        scores = np.concatenate(all_scores)
        scores_smooth = np.concatenate(all_scores_smooth)

        # precision, recall y F1 salen de la prediccion binaria, las AUC del score continuo
        metrics = {
            "precision": float(precision_score(gt, pred, zero_division=0)),
            "recall": float(recall_score(gt, pred, zero_division=0)),
            "f1": float(f1_score(gt, pred, zero_division=0)),
        }

        # las AUC piden las dos clases presentes, si falta una salen NaN en vez de romper
        has_both = gt.sum() > 0 and (gt == 0).sum() > 0
        if has_both:
            metrics["auc_roc"] = float(roc_auc_score(gt, scores))
            metrics["auc_pr"] = float(average_precision_score(gt, scores))
            metrics["auc_roc_smooth"] = float(roc_auc_score(gt, scores_smooth))
            metrics["auc_pr_smooth"] = float(average_precision_score(gt, scores_smooth))
        else:
            metrics["auc_roc"] = float("nan")
            metrics["auc_pr"] = float("nan")
            metrics["auc_roc_smooth"] = float("nan")
            metrics["auc_pr_smooth"] = float("nan")

        return metrics
