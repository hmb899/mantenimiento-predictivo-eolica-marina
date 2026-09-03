"""
grating masking, deterministic complementary imputation masks over the time axis

ImDiffusion turns anomaly detection into conditional imputation, each window is cut into
strips of width window_size // split, strategy 0 makes the even strips imputation targets
and strategy 1 the odd ones. the two strategies are complementary, so scoring a window
with both imputes every position exactly once
# ref: IMDiffusion dataset.py:get_mask (official defaults: W=100, split=10)

mask convention, all bool
  grating  True means the position is an imputation target, hidden from the model
  observed True means the position carries a real value. the frozen pipeline imputes NaN
           before windowing, so in practice observed is all-True, and it is kept explicit
           so a non-observed position can never enter the loss or the score
  cond   = observed & ~grating   (visible conditioning values)
  target = observed &  grating   (positions where loss and score are computed)
"""
from __future__ import annotations

import numpy as np


def grating_mask(window_size: int, split: int, strategy: int) -> np.ndarray:
    """return the (window_size,) bool grating mask for one strategy, the positions whose
    strip index has the same parity as strategy."""

    if not isinstance(split, int) or not 2 <= split <= window_size:
        raise ValueError(f"split must be an int in [2, window_size={window_size}], got {split}")
    if strategy not in (0, 1):
        raise ValueError(f"strategy must be 0 or 1, got {strategy}")
    # si split no divide exacto, el resto sobrante sigue alternando la paridad y la
    # complementariedad se mantiene igual
    width = window_size // split
    strip_index = np.arange(window_size) // width
    # con split impar las dos estrategias quedan asimetricas, la 0 se lleva una banda mas
    return (strip_index % 2) == strategy


def complementary_masks(window_size: int, split: int) -> tuple[np.ndarray, np.ndarray]:
    """return both strategy masks, together they tile the window exactly once."""

    m0 = grating_mask(window_size, split, 0)
    m1 = grating_mask(window_size, split, 1)
    # el assert es la garantia estructural, ni se solapan ni dejan hueco
    assert not (m0 & m1).any() and (m0 | m1).all(), "grating masks must be complementary"
    return m0, m1


def grating_mask_2d(window_size: int, n_features: int, split: int, strategy: int) -> np.ndarray:
    """(window_size, n_features) grating with the strip phase shifted per feature, so that
    feature k targets the positions with parity (t // width + k) % 2 == strategy."""

    # variante propia, la rejilla oficial enmascara todos los sensores a la vez y deja solo
    # contexto temporal, interpolable incluso en regimen de fallo (ROC-win 0.4978)
    if not isinstance(split, int) or not 2 <= split <= window_size:
        raise ValueError(f"split must be an int in [2, window_size={window_size}], got {split}")
    if strategy not in (0, 1):
        raise ValueError(f"strategy must be 0 or 1, got {strategy}")
    width = window_size // split
    strip_index = np.arange(window_size) // width
    k = np.arange(n_features)
    # el desfase por feature deja observada la mitad de los sensores en cada paso, asi imputar
    # uno enmascarado obliga a usar los demas, que es la relacion que rompe un fallo
    return ((strip_index[:, None] + k[None, :]) % 2) == strategy


def gratings_for_model(window_size: int, n_features: int, split: int,
                       mode: str = "time", exo_idx: list[int] | None = None) -> np.ndarray:
    """both complementary masks in model layout, (2, n_features, window_size) bool. mode
    "time" is the official grating and "feature_offset" the per-feature phase-shifted one."""

    if mode == "time":
        # la misma banda para todas las features, se difunde la mascara 1D a las K filas
        m0 = grating_mask(window_size, split, 0)
        g0 = np.broadcast_to(m0[None, :], (n_features, window_size)).copy()
    elif mode == "feature_offset":
        g0 = grating_mask_2d(window_size, n_features, split, 0).T.copy()
    else:
        raise ValueError(f"unknown masking mode: {mode}")
    # la segunda estrategia es la negacion de la primera, de ahi que sean complementarias
    pair = np.stack([g0, ~g0])
    endo = np.ones(n_features, dtype=bool)
    if exo_idx is not None and len(exo_idx) > 0:
        # las filas exogenas (viento, ambiente) hacen de ancla de regimen, nunca son objetivo
        # y siempre condicionan, asi lo endogeno se imputa dado el regimen limpio y desviarse
        # de esa normalidad condicional, que es lo que rompe un fallo, sube el score
        idx = list(exo_idx)
        pair[:, idx, :] = False  # exogenous rows, never target and always conditioning
        endo[idx] = False  # y por tanto no puntuan
    assert not (pair[0] & pair[1]).any(), "complementary strategies must not overlap"
    assert (pair[0] | pair[1])[endo].all(), "endogenous positions must be covered exactly once"
    assert not (pair[0] | pair[1])[~endo].any(), "exogenous positions must never be a target"
    return pair


def build_cond_target(observed: np.ndarray, grating: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """split an observed mask (N, W, K) into (cond, target) using a (W,) time-axis grating,
    with cond ∧ target = ∅ and cond ∨ target = observed asserted at runtime."""

    # observed True donde el valor es real, grating True donde la posicion es objetivo
    if observed.ndim != 3:
        raise ValueError(f"observed must be (N, W, K), got shape {observed.shape}")
    if grating.shape != (observed.shape[1],):
        raise ValueError(f"grating must be (W={observed.shape[1]},), got {grating.shape}")
    # la rejilla es (W,) y se difunde al lote y a las features con los ejes vacios
    g = grating[None, :, None]
    cond = observed & ~g
    target = observed & g
    assert not (cond & target).any(), "cond and target masks overlap"
    assert ((cond | target) == observed).all(), "cond ∪ target must equal observed"
    return cond, target
