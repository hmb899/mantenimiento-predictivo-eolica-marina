"""tests unitarios de la implementacion de la metrica CARE.

todos son deterministas y calculados a mano donde se puede. se ejecutan con
pytest tests/test_care_official.py -v
"""
import numpy as np
import pytest

from src.care_official.coverage import f_beta, coverage_score
from src.care_official.accuracy import accuracy_score, accuracy_score_single
from src.care_official.reliability import (
    reliability_score,
    ef_beta,
    criticality_alarm,
)
from src.care_official.earliness import weighted_score_single, earliness_score
from src.care_official.care import compute_care, weighted_average, CAREResult
from src.care_official.labels import compute_earliness_weights, NORMAL_STATUS


# auxiliares, los dicts minimos de dataset anomalo y normal

def _anomaly_ds(y_pred_normal_pred, y_pred_all_pred, statuses_all_pred=None, y_pred_event=None,
                weights_event=None):
    n = len(y_pred_event) if y_pred_event is not None else 0
    w = weights_event if weights_event is not None else compute_earliness_weights(n)
    evt = y_pred_event if y_pred_event is not None else np.array([], dtype=int)
    y_pred_all = np.asarray(y_pred_all_pred, dtype=int)
    # status por defecto todo normal (0), asi el Criticality Algorithm procesa las predicciones
    if statuses_all_pred is None:
        statuses = np.zeros(len(y_pred_all), dtype=int)
    else:
        statuses = np.asarray(statuses_all_pred, dtype=int)
    event_mask = np.zeros(len(y_pred_all), dtype=bool)
    return {
        "y_pred_normal_pred": np.asarray(y_pred_normal_pred, dtype=int),
        "y_true_normal_pred": np.ones(len(y_pred_normal_pred), dtype=int),
        "event_mask_in_pred": event_mask,
        "y_pred_all_pred": y_pred_all,
        "statuses_all_pred": statuses,
        "weights_event": np.asarray(w, dtype=float),
        "y_pred_event": np.asarray(evt, dtype=int),
    }


def _normal_ds(y_pred_all_pred, statuses_all_pred=None):
    y_pred_all = np.asarray(y_pred_all_pred, dtype=int)
    # status por defecto todo normal, asi el Criticality Algorithm procesa las predicciones
    if statuses_all_pred is None:
        statuses = np.zeros(len(y_pred_all), dtype=int)
    else:
        statuses = np.asarray(statuses_all_pred, dtype=int)
    normal_mask = np.isin(statuses, list(NORMAL_STATUS))
    return {
        "y_pred_all_pred": y_pred_all,
        "statuses_all_pred": statuses,
        "y_pred_normal_pred": y_pred_all[normal_mask],
    }


# F_beta, la pieza basica de Coverage

class TestFBeta:
    """f_beta, que el paper usa con beta=1/2."""

    def test_perfect_score(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        assert f_beta(y_true, y_pred, beta=0.5) == pytest.approx(1.0)


    def test_zero_tp(self):
        # el modelo no predice nada, fn=4
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        assert f_beta(y_true, y_pred, beta=0.5) == pytest.approx(0.0)


    def test_all_fp_no_positives(self):
        # la verdad de referencia es todo normal y el modelo predice todo anomalia
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        assert f_beta(y_true, y_pred, beta=0.5) == pytest.approx(0.0)


    def test_hand_computed_beta_half(self):
        # tp=2, fn=2, fp=1, beta=0.5
        # F_{1/2} = 1.25*2 / (1.25*2 + 0.25*2 + 1) = 2.5 / 4 = 0.625
        y_true = np.array([1, 1, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 1])
        expected = 2.5 / 4.0
        assert f_beta(y_true, y_pred, beta=0.5) == pytest.approx(expected, rel=1e-9)


    def test_denominator_zero_returns_zero(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        assert f_beta(y_true, y_pred, beta=0.5) == 0.0


    def test_beta_half_penalizes_fp_more(self):
        # con beta=0.5 un modelo de precision alta y cobertura baja gana al contrario,
        # que es lo opuesto a lo que pasaria con beta=2
        y_true = np.array([1, 1, 1, 1, 0, 0])
        # 1 tp, 3 fn, 0 fp da F_{1/2} = 1.25*1 / (1.25*1 + 0.25*3) = 1.25/2 = 0.625
        y_pred_high_prec = np.array([1, 0, 0, 0, 0, 0])
        # 3 tp, 1 fn, 2 fp da F_{1/2} = 1.25*3 / (1.25*3 + 0.25*1 + 2) = 3.75/6 = 0.625
        y_pred_high_rec  = np.array([1, 1, 1, 0, 1, 1])
        score_hp = f_beta(y_true, y_pred_high_prec, beta=0.5)
        score_hr = f_beta(y_true, y_pred_high_rec,  beta=0.5)
        assert score_hp >= score_hr


    def test_accepts_any_beta(self):
        # la formula tiene que valer tambien con beta=2, que algun test usa
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        assert f_beta(y_true, y_pred, beta=2.0) == pytest.approx(1.0)


class TestCoverageScore:
    def test_single_dataset_perfect(self):
        gt = np.array([1, 1, 1])
        pred = np.array([1, 1, 1])
        assert coverage_score([(gt, pred)]) == pytest.approx(1.0)


    def test_average_over_two_datasets(self):
        # dataset 1 con F_{1/2} = 1.0 y dataset 2 con 0.0, o sea media 0.5
        d1 = (np.array([1, 1]), np.array([1, 1]))
        d2 = (np.array([1, 1]), np.array([0, 0]))
        result = coverage_score([d1, d2])
        assert result == pytest.approx(0.5)


    def test_empty_list(self):
        assert coverage_score([]) == pytest.approx(0.0)


# Accuracy

class TestAccuracySingle:
    def test_all_correct(self):
        assert accuracy_score_single(np.array([0, 0, 0, 0])) == pytest.approx(1.0)


    def test_all_wrong(self):
        assert accuracy_score_single(np.array([1, 1, 1, 1])) == pytest.approx(0.0)


    def test_half_wrong(self):
        assert accuracy_score_single(np.array([0, 1, 0, 1])) == pytest.approx(0.5)


    def test_empty(self):
        assert accuracy_score_single(np.array([], dtype=int)) == pytest.approx(1.0)


class TestAccuracyScore:
    def test_multiple_datasets(self):
        d1 = np.array([0, 0, 0, 1])  # 0.75
        d2 = np.array([0, 0, 0])  # 1.0
        assert accuracy_score([d1, d2]) == pytest.approx(0.875)


    def test_empty_list(self):
        assert accuracy_score([]) == pytest.approx(1.0)


# Reliability, el Criticality Algorithm

class TestCriticalityAlarm:
    """el contador de criticidad, que es el Algorithm 1 del paper."""

    def test_all_anomaly_normal_status_reaches_threshold(self):
        # 100 predicciones de anomalia seguidas con status normal dan crit = 100 >= 72, alarma
        y_pred = np.ones(100, dtype=int)
        statuses = np.zeros(100, dtype=int)  # todo normal (status 0)
        assert criticality_alarm(y_pred, statuses, tc=72) is True


    def test_all_normal_prediction_no_alarm(self):
        # con todas las predicciones a 0, crit se queda en 0 y no hay alarma
        y_pred = np.zeros(100, dtype=int)
        statuses = np.zeros(100, dtype=int)
        assert criticality_alarm(y_pred, statuses, tc=72) is False


    def test_counter_resets_on_normal_predictions(self):
        # con 50 de anomalia y 50 normales, crit sube a 50 y baja, nunca llega a 72
        y_pred = np.array([1] * 50 + [0] * 50, dtype=int)
        statuses = np.zeros(100, dtype=int)
        assert criticality_alarm(y_pred, statuses, tc=72) is False


    def test_abnormal_status_ignored(self):
        # 200 predicciones de anomalia pero todas en status anormal, crit se queda en 0
        y_pred = np.ones(200, dtype=int)
        statuses = np.ones(200, dtype=int) * 4  # status 4 = parada (anormal)
        assert criticality_alarm(y_pred, statuses, tc=72) is False


    def test_mixed_status_only_normal_counts(self):
        # crit solo sube en los pasos de status normal y las predicciones en status anormal
        # no lo mueven; alternando quedan 50 pasos normales, 50 < 72, no hay alarma
        y_pred = np.ones(100, dtype=int)
        statuses = np.array([0, 4] * 50, dtype=int)  # normal y anormal alternados
        assert criticality_alarm(y_pred, statuses, tc=72) is False


    def test_normal_status_two_is_processed(self):
        # el status 2 tambien es "normal" y tiene que subir crit
        y_pred = np.ones(100, dtype=int)
        statuses = np.ones(100, dtype=int) * 2  # todo status 2 (al ralenti, normal)
        assert criticality_alarm(y_pred, statuses, tc=72) is True


    def test_threshold_exact(self):
        # con tc predicciones de anomalia hay alarma, y con tc-1 no
        tc = 10
        y_pred_exact = np.ones(tc, dtype=int)
        y_pred_short = np.ones(tc - 1, dtype=int)
        statuses = np.zeros(tc, dtype=int)
        assert criticality_alarm(y_pred_exact, statuses, tc=tc) is True
        assert criticality_alarm(y_pred_short[:tc-1], statuses[:tc-1], tc=tc) is False


class TestEFBeta:
    def test_all_detected_no_fp(self):
        # TP_e=3, FN_e=0, FP_e=0 da EF_{1/2} = 1.0
        assert ef_beta(3, 0, 0, beta=0.5) == pytest.approx(1.0)


    def test_none_detected_no_fp(self):
        # TP_e=0, FN_e=3, FP_e=0 da 0
        assert ef_beta(0, 3, 0, beta=0.5) == pytest.approx(0.0)


    def test_fp_reduces_score(self):
        assert ef_beta(2, 0, 0, beta=0.5) > ef_beta(2, 0, 5, beta=0.5)


    def test_hand_computed_beta_half(self):
        # TP_e=2, FN_e=1, FP_e=2, beta=0.5
        # EF_{1/2} = 1.25*2 / (1.25*2 + 0.25*1 + 2) = 2.5 / 4.75
        expected = 2.5 / 4.75
        assert ef_beta(2, 1, 2, beta=0.5) == pytest.approx(expected, rel=1e-9)


    def test_denom_zero(self):
        assert ef_beta(0, 0, 0, beta=0.5) == pytest.approx(0.0)


class TestReliabilityScore:
    """el sub-score con el Criticality Algorithm y tc=72."""

    def test_constant_anomaly_raises_alarm_on_normal_status(self):
        # 100 predicciones de anomalia seguidas sobre status normal dan alarma en los dos
        # datasets, o sea TP_e=1 en el anomalo y FP_e=1 en el normal
        anomaly = [(np.ones(100, dtype=int), np.zeros(100, dtype=int))]
        normal  = [(np.ones(100, dtype=int), np.zeros(100, dtype=int))]
        # TP_e=1, FN_e=0, FP_e=1 da EF_{1/2} = 1.25/(1.25+1) = 1.25/2.25
        expected = 1.25 / 2.25
        score = reliability_score(anomaly, normal, beta=0.5, tc=72)
        assert score == pytest.approx(expected, rel=1e-6)


    def test_constant_normal_no_alarms(self):
        # sin predicciones crit se queda en 0 y no hay alarmas, o sea TP_e=0, FN_e=1, FP_e=0
        anomaly = [(np.zeros(100, dtype=int), np.zeros(100, dtype=int))]
        normal  = [(np.zeros(100, dtype=int), np.zeros(100, dtype=int))]
        score = reliability_score(anomaly, normal, beta=0.5, tc=72)
        assert score == pytest.approx(0.0)


    def test_abnormal_status_prevents_alarm(self):
        # 200 predicciones de anomalia todas en status anormal, sin alarmas, TP_e=0 y score 0
        y_pred = np.ones(200, dtype=int)
        statuses = np.ones(200, dtype=int) * 4  # todo status 4 (parada)
        anomaly = [(y_pred, statuses)]
        normal  = [(y_pred, statuses)]
        score = reliability_score(anomaly, normal, beta=0.5, tc=72)
        assert score == pytest.approx(0.0)


    def test_good_model_no_fp(self):
        # el anomalo da alarma con 80 predicciones en status normal y el normal no da ninguna,
        # asi que TP_e=1, FN_e=0, FP_e=0 y EF_{1/2} = 1.0
        anomaly = [(np.ones(80, dtype=int), np.zeros(80, dtype=int))]
        normal  = [(np.zeros(80, dtype=int), np.zeros(80, dtype=int))]
        score = reliability_score(anomaly, normal, beta=0.5, tc=72)
        assert score == pytest.approx(1.0)


# pesos de Earliness

class TestEarlinessWeights:
    def test_empty(self):
        w = compute_earliness_weights(0)
        assert len(w) == 0


    def test_single_point(self):
        w = compute_earliness_weights(1)
        assert len(w) == 1
        assert w[0] == pytest.approx(1.0)


    def test_two_points(self):
        w = compute_earliness_weights(2)
        assert w[0] == pytest.approx(1.0)
        assert w[1] == pytest.approx(0.0)


    def test_four_points(self):
        w = compute_earliness_weights(4)
        assert w[0] == pytest.approx(1.0)
        assert w[1] == pytest.approx(1.0)
        assert w[2] == pytest.approx(1.0)
        assert w[3] == pytest.approx(0.0)


    def test_weights_are_non_increasing_in_second_half(self):
        w = compute_earliness_weights(10)
        half = 5
        second_half = w[half:]
        assert all(second_half[i] >= second_half[i + 1] for i in range(len(second_half) - 1))


    def test_last_weight_is_zero(self):
        for n in [3, 6, 10, 20]:
            w = compute_earliness_weights(n)
            assert w[-1] == pytest.approx(0.0), f"n={n}: last weight should be 0"


    def test_first_half_all_ones(self):
        n = 8
        w = compute_earliness_weights(n)
        half = n // 2
        assert all(w[i] == pytest.approx(1.0) for i in range(half))


class TestWeightedScoreSingle:
    def test_all_detected_all_weight_one(self):
        y_pred = np.array([1, 1, 1, 1])
        w = np.array([1.0, 1.0, 1.0, 1.0])
        assert weighted_score_single(y_pred, w) == pytest.approx(1.0)


    def test_none_detected(self):
        y_pred = np.array([0, 0, 0, 0])
        w = np.array([1.0, 1.0, 1.0, 1.0])
        assert weighted_score_single(y_pred, w) == pytest.approx(0.0)


    def test_early_detection_scores_higher(self):
        w = compute_earliness_weights(4)
        pred_early = np.array([1, 0, 0, 0])
        pred_late  = np.array([0, 0, 0, 1])
        assert weighted_score_single(pred_early, w) > weighted_score_single(pred_late, w)


    def test_empty(self):
        assert weighted_score_single(np.array([]), np.array([])) == pytest.approx(0.0)


class TestEarlinessScore:
    def test_single_perfect(self):
        pred = np.array([1, 1, 1, 1])
        w = np.ones(4)
        assert earliness_score([(pred, w)]) == pytest.approx(1.0)


    def test_average_over_two(self):
        d1 = (np.array([1, 1]), np.array([1.0, 1.0]))
        d2 = (np.array([0, 0]), np.array([1.0, 0.0]))
        result = earliness_score([d1, d2])
        assert result == pytest.approx(0.5)


# media ponderada

class TestWeightedAverage:
    def test_all_ones(self):
        wa = weighted_average(1.0, 1.0, 1.0, 1.0, 1, 1, 1, 2)
        assert wa == pytest.approx(1.0)


    def test_all_zeros(self):
        wa = weighted_average(0.0, 0.0, 0.0, 0.0, 1, 1, 1, 2)
        assert wa == pytest.approx(0.0)


    def test_hand_computed(self):
        # WA = (1*0.5 + 1*0.8 + 1*0.6 + 2*0.9) / 5 = 3.7/5 = 0.74
        wa = weighted_average(0.5, 0.8, 0.6, 0.9, 1, 1, 1, 2)
        assert wa == pytest.approx(0.74, rel=1e-9)


    def test_accuracy_double_weight(self):
        wa_high_acc = weighted_average(0.5, 0.5, 0.5, 1.0, 1, 1, 1, 2)
        wa_low_acc  = weighted_average(0.5, 0.5, 0.5, 0.0, 1, 1, 1, 2)
        wa_high_cov = weighted_average(1.0, 0.5, 0.5, 0.5, 1, 1, 1, 2)
        wa_low_cov  = weighted_average(0.0, 0.5, 0.5, 0.5, 1, 1, 1, 2)
        diff_acc = wa_high_acc - wa_low_acc
        diff_cov = wa_high_cov - wa_low_cov
        assert diff_acc > diff_cov


# la formula a trozos de CARE (eq. 5 del paper)

class TestCAREPiecewise:
    """las tres ramas de la formula de CARE."""

    def _make_never_detects(self, n_anomaly=3, n_normal=3, n_pts=80):
        """todas las predicciones a 0, la baseline de normal constante."""

        anomaly_ds = [
            _anomaly_ds(
                y_pred_normal_pred=np.zeros(n_pts, dtype=int),
                y_pred_all_pred=np.zeros(n_pts, dtype=int),
                y_pred_event=np.zeros(4, dtype=int),
                weights_event=compute_earliness_weights(4),
            )
            for _ in range(n_anomaly)
        ]
        normal_ds = [_normal_ds(np.zeros(n_pts, dtype=int)) for _ in range(n_normal)]
        return anomaly_ds, normal_ds


    def _make_always_anomaly(self, n_anomaly=3, n_normal=3, n_pts=80):
        """todas las predicciones a 1, la baseline de anomalia constante."""

        anomaly_ds = [
            _anomaly_ds(
                y_pred_normal_pred=np.ones(n_pts, dtype=int),
                y_pred_all_pred=np.ones(n_pts, dtype=int),
                y_pred_event=np.ones(4, dtype=int),
                weights_event=compute_earliness_weights(4),
            )
            for _ in range(n_anomaly)
        ]
        normal_ds = [_normal_ds(np.ones(n_pts, dtype=int)) for _ in range(n_normal)]
        return anomaly_ds, normal_ds


    def test_no_detections_care_is_zero(self):
        """con normal constante no se detecta ninguna anomalia y CARE = 0, la rama 1."""

        anomaly_ds, normal_ds = self._make_never_detects()
        result = compute_care(anomaly_ds, normal_ds)
        assert result.care == pytest.approx(0.0)


    def test_no_detections_accuracy_reported_correctly(self):
        """aunque salte la rama 1 y CARE valga 0, el sub-score de accuracy sigue bien."""

        anomaly_ds, normal_ds = self._make_never_detects(n_normal=3, n_pts=80)
        result = compute_care(anomaly_ds, normal_ds)
        # los datasets normales quedan todos bien clasificados, con las predicciones a 0
        assert result.accuracy == pytest.approx(1.0)


    def test_acc_below_half_care_equals_acc(self):
        """con anomalia constante Acc=0 < 0.5, o sea CARE = Acc = 0, la rama 2."""

        anomaly_ds, normal_ds = self._make_always_anomaly()
        result = compute_care(anomaly_ds, normal_ds)
        assert result.accuracy == pytest.approx(0.0)
        assert result.care == pytest.approx(result.accuracy)  # = 0


    def test_good_model_uses_weighted_average(self):
        """con Accuracy buena (>= 0.5) y alguna deteccion, CARE es la media ponderada."""

        # solo se predice anomalia en los puntos de la ventana de evento, y el dataset
        # normal no da ninguna alarma
        n_pts = 80
        # 72 predicciones de anomalia seguidas en status normal dan alarma en los anomalos
        y_pred_anomaly = np.zeros(n_pts, dtype=int)
        y_pred_anomaly[:72] = 1

        anomaly_ds = [
            _anomaly_ds(
                y_pred_normal_pred=y_pred_anomaly.copy(),
                y_pred_all_pred=y_pred_anomaly.copy(),
                y_pred_event=np.ones(4, dtype=int),
                weights_event=compute_earliness_weights(4),
            )
            for _ in range(3)
        ]
        # datasets normales con todas las predicciones a 0, sin alarma y con Acc = 1.0
        normal_ds = [_normal_ds(np.zeros(n_pts, dtype=int)) for _ in range(3)]

        result = compute_care(anomaly_ds, normal_ds)
        assert result.accuracy >= 0.5
        expected_wa = weighted_average(
            result.coverage, result.earliness, result.reliability, result.accuracy
        )
        assert result.care == pytest.approx(expected_wa, rel=1e-9)


    def test_predict_nothing_worse_than_some_detections(self):
        """la rama 1 (CARE=0) queda por debajo de un modelo que detecta algo."""

        anomaly_none, normal_none = self._make_never_detects()
        result_none = compute_care(anomaly_none, normal_none)

        # un modelo que predice anomalia en 80 puntos de prediccion con status normal,
        # suficientes para disparar la alarma, y todo normal en los datasets normales
        n_pts = 80
        y_pred_some = np.ones(n_pts, dtype=int)
        anomaly_some = [
            _anomaly_ds(
                y_pred_normal_pred=y_pred_some.copy(),
                y_pred_all_pred=y_pred_some.copy(),
                y_pred_event=np.ones(4, dtype=int),
                weights_event=compute_earliness_weights(4),
            )
            for _ in range(3)
        ]
        normal_some = [_normal_ds(np.zeros(n_pts, dtype=int)) for _ in range(3)]
        result_some = compute_care(anomaly_some, normal_some)

        assert result_some.care > result_none.care


# la constante NORMAL_STATUS

class TestNormalStatus:
    def test_contains_zero_and_two(self):
        assert 0 in NORMAL_STATUS
        assert 2 in NORMAL_STATUS


    def test_does_not_contain_abnormal(self):
        for s in [1, 3, 4, 5]:
            assert s not in NORMAL_STATUS


    def test_is_frozenset(self):
        assert isinstance(NORMAL_STATUS, frozenset)
