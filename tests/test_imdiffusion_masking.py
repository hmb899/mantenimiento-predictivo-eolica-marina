"""tests estaticos de las mascaras de rejilla, complementariedad, cobertura y anchura
exacta de banda, y del algebra de las mascaras cond/target. incluye la variante de
rejilla con desfase por feature, donde cada feature lleva su propia fase y la mitad
de los sensores queda observada en cada instante."""
import numpy as np
import pytest

from src.imdiffusion.masking import (build_cond_target, complementary_masks, grating_mask,
                                     grating_mask_2d, gratings_for_model)


@pytest.mark.parametrize("w,split", [(27, 9), (27, 3), (27, 5), (100, 10), (28, 9), (12, 4)])
def test_masks_complementary_and_cover_window(w, split):
    m0, m1 = complementary_masks(w, split)
    assert not (m0 & m1).any(), "intersection must be empty"
    assert (m0 | m1).all(), "union must cover the whole window"
    assert m0.shape == m1.shape == (w,)


def test_strip_width_exact_w27_split9():
    m0 = grating_mask(27, 9, 0)
    # la config por defecto es W=27 y split=9, o sea bandas de 3 muestras (30 min)
    boundaries = np.flatnonzero(np.diff(m0.astype(int)) != 0) + 1
    run_lengths = np.diff(np.r_[0, boundaries, 27])
    assert (run_lengths == 3).all()
    assert m0[:3].all() and not m0[3:6].any()  # la estrategia 0 empieza marcando en t=0


def test_strip_width_exact_w27_split3():
    m0 = grating_mask(27, 3, 0)
    boundaries = np.flatnonzero(np.diff(m0.astype(int)) != 0) + 1
    run_lengths = np.diff(np.r_[0, boundaries, 27])
    assert (run_lengths == 9).all()  # con split=3 las bandas son de 9 muestras (90 min)


def test_odd_split_asymmetry():
    # con un numero impar de bandas, la estrategia 0 (la de las pares) se lleva una mas
    m0, m1 = complementary_masks(27, 9)
    assert m0.sum() == 15 and m1.sum() == 12
    m0, m1 = complementary_masks(27, 3)
    assert m0.sum() == 18 and m1.sum() == 9


def test_validation_errors():
    with pytest.raises(ValueError):
        grating_mask(27, 1, 0)  # con split=1 la estrategia 1 se queda sin objetivos
    with pytest.raises(ValueError):
        grating_mask(27, 28, 0)  # split mayor que window_size
    with pytest.raises(ValueError):
        grating_mask(27, 9, 2)  # estrategia invalida


def test_build_cond_target_algebra():
    rng = np.random.default_rng(0)
    observed = rng.random((5, 27, 4)) > 0.1  # algunas posiciones sin observar
    g = grating_mask(27, 9, 0)
    cond, target = build_cond_target(observed, g)
    assert not (cond & target).any()
    assert ((cond | target) == observed).all()
    # una posicion sin observar no pertenece a ninguna de las dos mascaras
    assert not (cond & ~observed).any() and not (target & ~observed).any()
    # con observed todo a True, target es la rejilla difundida sobre las features
    cond_f, target_f = build_cond_target(np.ones((2, 27, 4), bool), g)
    assert (target_f == g[None, :, None]).all()


def test_build_cond_target_shape_validation():
    g = grating_mask(27, 9, 0)
    with pytest.raises(ValueError):
        build_cond_target(np.ones((27, 4), bool), g)  # falta la dimension de lote
    with pytest.raises(ValueError):
        build_cond_target(np.ones((2, 18, 4), bool), g)  # W no cuadra


@pytest.mark.parametrize("w,k,split", [(27, 30, 9), (27, 54, 9), (27, 54, 3), (12, 5, 4)])
def test_offset_masks_complementary_and_cover(w, k, split):
    m0 = grating_mask_2d(w, k, split, 0)
    m1 = grating_mask_2d(w, k, split, 1)
    assert m0.shape == m1.shape == (w, k)
    assert not (m0 & m1).any() and (m0 | m1).all()


def test_offset_masks_balance_per_timestep():
    # en cada instante queda observada mas o menos la mitad de las features, que es el sentido
    # de la variante, porque la imputacion se apoya en los otros sensores del mismo t
    m0 = grating_mask_2d(27, 54, 9, 0)
    per_t = m0.sum(axis=1)
    assert (per_t == 27).all()  # con K par, exactamente la mitad enmascarada
    m_odd = grating_mask_2d(27, 5, 4, 0)
    assert set(m_odd.sum(axis=1)) <= {2, 3}  # con K impar, suelo y techo de la mitad


def test_offset_masks_per_feature_strip_width():
    m0 = grating_mask_2d(27, 8, 9, 0)
    for k in range(8):
        col = m0[:, k]
        boundaries = np.flatnonzero(np.diff(col.astype(int)) != 0) + 1
        run_lengths = np.diff(np.r_[0, boundaries, 27])
        assert (run_lengths == 3).all()  # las bandas siguen siendo de 3 muestras por feature
    # las features contiguas van en fase opuesta
    assert (m0[:, 0] == ~m0[:, 1]).all()


def test_gratings_for_model_layouts():
    t = gratings_for_model(27, 6, 9, "time")
    assert t.shape == (2, 6, 27) and (t[0] == ~t[1]).all()
    assert (t[0] == t[0][0]).all()  # en modo time, las mismas bandas para cada feature
    o = gratings_for_model(27, 6, 9, "feature_offset")
    assert o.shape == (2, 6, 27) and (o[0] == ~o[1]).all()
    assert (o[0][0] == ~o[0][1]).all()  # en modo offset, la fase alterna
    np.testing.assert_array_equal(o[0], grating_mask_2d(27, 6, 9, 0).T)
    with pytest.raises(ValueError):
        gratings_for_model(27, 6, 9, "diagonal")


def test_regime_conditioning_anchors_exogenous():
    # las filas 0 y 3 son anclas exogenas y nunca son objetivo en ninguna de las estrategias
    exo = [0, 3]
    for mode in ("time", "feature_offset"):
        pair = gratings_for_model(27, 6, 9, mode, exo_idx=exo)
        assert pair.shape == (2, 6, 27)
        # las filas exogenas no llevan objetivo, siempre son condicionamiento
        assert not pair[:, exo, :].any()
        # las endogenas siguen cubriendo la ventana entera y de forma complementaria
        endo = [1, 2, 4, 5]
        assert (pair[0] | pair[1])[endo].all()
        assert not (pair[0][endo] & pair[1][endo]).any()


def test_regime_conditioning_empty_is_noop():
    base = gratings_for_model(27, 5, 5, "feature_offset")
    same = gratings_for_model(27, 5, 5, "feature_offset", exo_idx=[])
    np.testing.assert_array_equal(base, same)
