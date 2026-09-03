"""
normalizadores y combinadores de scores para fusion a nivel de score

todas las funciones operan sobre arrays 1-D de scores de anomalia por ventana.

normalizadores
    se ajustan sobre el train de cada detector y se aplican a train y a eventos, para que
    detectores en escalas distintas sean comparables antes de combinar
    z-score, y el Gaussian-tail de Kriegel et al. 2011, "Interpreting and Unifying Outlier
    Scores"
combinadores
    reciben una lista de arrays ya normalizados, uno por detector y todos de la misma longitud,
    y devuelven un unico array fusionado (Aggarwal & Sathe 2017, Outlier Ensembles)

nota: con z-score, una fusion con alpha=0, o sea peso 1 al AE y 0 al resto, reproduce
exactamente el baseline del AE, porque el CUSUM de la evaluacion es invariante a las
transformaciones afines
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf
from scipy.stats import norm


# normalizadores
def fit_zscore(train: np.ndarray) -> tuple[float, float]:
    # el epsilon de la desviacion evita dividir por cero con un detector plano en train
    return float(np.mean(train)), float(np.std(train) + 1e-12)


def apply_zscore(x: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    mu, sd = stats
    return (np.asarray(x, dtype=float) - mu) / sd


def fit_gaussian_tail(train: np.ndarray) -> tuple[float, float]:
    return float(np.mean(train)), float(np.std(train) + 1e-12)


def apply_gaussian_tail(x: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    """Gaussian scaling de Kriegel, estandariza y mapea por la cola erf a [0,1). los valores
    por debajo de la media, que no son anomalos, se aplanan a 0, y la cola alta se comprime a
    [0,1), evitando que un detector de cola pesada domine la suma."""

    mu, sd = stats
    # el sqrt(2) es lo que convierte el z en el argumento que espera erf
    z = (np.asarray(x, dtype=float) - mu) / (sd * np.sqrt(2.0))
    return np.maximum(0.0, erf(z))


def fit_ecdf(train: np.ndarray) -> np.ndarray:
    """calibrador rank, distribution-free, que guarda el train ordenado."""

    return np.sort(np.asarray(train, dtype=float))


def apply_ecdf(x: np.ndarray, stats: np.ndarray) -> np.ndarray:
    """ECDF empirica, mapea cada score a su cuantil en el train, a [0,1] y monotona. es el
    decorrelador maximo, porque el Pearson de dos scores ECDF es casi su Spearman. satura a
    1.0 por encima del max del train y ahi pierde discriminacion, y por eso se acompana de
    gmm_tail, que no satura."""

    # searchsorted da la posicion en el train ordenado, y dividir por len la vuelve un cuantil
    s = stats
    return np.searchsorted(s, np.asarray(x, dtype=float), side="right") / len(s)


def fit_gmm_tail(train: np.ndarray, k: int = 3, subsample: int = 50000):
    """ajusta una GMM de k componentes al score normal del train, al estilo M2AD, y devuelve
    (weights, means, stds) para una cola de supervivencia. subsamplea por velocidad, y si el
    ajuste falla por colas patologicas cae a k=1, una sola gaussiana, de forma robusta."""

    from sklearn.mixture import GaussianMixture
    tr = np.asarray(train, dtype=float).reshape(-1, 1)
    # semilla fija en el subsampleo y en el fit, la calibracion tiene que ser reproducible
    if len(tr) > subsample:
        tr = tr[np.random.default_rng(0).choice(len(tr), subsample, replace=False)]
    try:
        gm = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-6,
                             random_state=0).fit(tr)
        return (gm.weights_.ravel(), gm.means_.ravel(), np.sqrt(gm.covariances_.ravel()) + 1e-12)
    except Exception:  # ajuste degenerado, se cae a una gaussiana unica
        return (np.array([1.0]), np.array([float(tr.mean())]), np.array([float(tr.std()) + 1e-12]))


def apply_gmm_tail(x: np.ndarray, stats) -> np.ndarray:
    """score de anomalia calibrado por la cola de supervivencia de la GMM normal, -log( sum_i
    w_i * SF_i(x) ). es monotona creciente, porque la supervivencia es monotona decreciente, y
    no satura: sigue creciendo por encima del max del train, al reves que ECDF, preservando la
    discriminacion entre las ventanas mas anomalas. receta M2AD (AISTATS 2025)."""

    w, mu, sd = stats
    x = np.asarray(x, dtype=float)
    # se suma la supervivencia de cada componente, ponderada por su peso en la mezcla
    sf = np.zeros_like(x)
    for wi, mi, si in zip(w, mu, sd):
        sf += wi * norm.sf(x, loc=mi, scale=si)
    # el clip por abajo evita el log(0) cuando la cola se agota numericamente
    return -np.log(np.clip(sf, 1e-300, 1.0))


NORMALIZERS = {
    "zscore": (fit_zscore, apply_zscore),
    "gaussian_tail": (fit_gaussian_tail, apply_gaussian_tail),
    "ecdf": (fit_ecdf, apply_ecdf),
    "gmm_tail": (fit_gmm_tail, apply_gmm_tail),
}


# combinadores
def _stack(norm_scores: list[np.ndarray]) -> np.ndarray:
    # (n_detectores, n_ventanas), el detector es la fila y la ventana la columna
    return np.vstack([np.asarray(s, dtype=float) for s in norm_scores])


def combine_average(norm_scores: list[np.ndarray], weights=None) -> np.ndarray:
    """media, ponderada si se le pasan pesos, que se normalizan a suma 1."""

    S = _stack(norm_scores)
    if weights is None:
        return S.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    return (w[:, None] * S).sum(axis=0)


def combine_maximum(norm_scores: list[np.ndarray]) -> np.ndarray:
    """maximo elemento a elemento, que favorece la cobertura porque basta con que dispare
    uno."""

    return _stack(norm_scores).max(axis=0)


def _buckets(n: int, n_buckets: int) -> list[np.ndarray]:
    # reparto lo mas igualado posible, y nunca mas buckets que detectores
    return [b for b in np.array_split(np.arange(n), min(max(1, n_buckets), n)) if len(b)]


def combine_aom(norm_scores: list[np.ndarray], n_buckets: int = 2) -> np.ndarray:
    """Average-of-Maximum (Aggarwal-Sathe), el max dentro de cada bucket y luego la media entre
    buckets. reduce varianza conservando algo del sesgo bajo del max, y necesita >=3 detectores
    para no ser degenerado."""

    S = _stack(norm_scores)
    return np.mean([S[b].max(axis=0) for b in _buckets(S.shape[0], n_buckets)], axis=0)


def combine_moa(norm_scores: list[np.ndarray], n_buckets: int = 2) -> np.ndarray:
    """Maximum-of-Average (Aggarwal-Sathe), la media dentro de cada bucket y luego el max entre
    buckets. necesita >=3 detectores para no ser degenerado."""

    S = _stack(norm_scores)
    return np.max([S[b].mean(axis=0) for b in _buckets(S.shape[0], n_buckets)], axis=0)


def weight_grid(n: int) -> list[list[float]]:
    """rejilla de pesos para la media ponderada. con n=2 es el barrido de alpha, el peso del
    segundo detector, en [0, .2, .35, .5, .65, .8, 1], compatible con las campanas; con n>=3 es
    un simplex grueso de paso 0.25, todas las combinaciones que suman 1."""

    if n == 2:
        return [[1 - a, a] for a in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)]
    step = 0.25
    ticks = int(round(1.0 / step))
    grid = []
    # con n=3 el tercer peso queda determinado por los otros dos, de ahi el doble bucle
    if n == 3:
        for i in range(ticks + 1):
            for j in range(ticks + 1 - i):
                k = ticks - i - j
                grid.append([i * step, j * step, k * step])
        return grid
    # con n>=4, baseline equipeso y cada detector dominante, que el barrido completo seria
    # enorme
    grid.append([1.0 / n] * n)
    for d in range(n):
        # 0.85 al dominante y el 0.15 restante repartido entre los demas
        w = [0.15 / (n - 1)] * n
        w[d] = 0.85
        grid.append(w)
    return grid


if __name__ == "__main__":  # self-test rapido
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=100), rng.normal(size=100)
    st = fit_zscore(a)
    assert np.allclose(apply_zscore(a, st).mean(), 0, atol=1e-6)
    gt = apply_gaussian_tail(a, fit_gaussian_tail(a))
    assert gt.min() >= 0.0 and gt.max() < 1.0
    # ecdf, monotona en [0,1], y satura a 1.0 por encima del max del train
    xs = np.linspace(a.min(), a.max(), 50)
    ec = apply_ecdf(xs, fit_ecdf(a))
    assert np.all(np.diff(ec) >= 0) and 0 <= ec.min() and ec.max() <= 1.0
    assert apply_ecdf(np.array([a.max() + 10.0]), fit_ecdf(a))[0] == 1.0     # satura
    # gmm_tail, monotona creciente y sin saturar, crece mas alla del max del train
    st = fit_gmm_tail(a)
    gm = apply_gmm_tail(xs, st)
    assert np.all(np.diff(gm) >= -1e-9), "gmm_tail no monotona"
    assert apply_gmm_tail(np.array([a.max() + 10.0]), st)[0] > apply_gmm_tail(np.array([a.max()]), st)[0]
    assert np.allclose(combine_average([a, b]), (a + b) / 2)
    assert np.allclose(combine_maximum([a, b]), np.maximum(a, b))
    assert len(weight_grid(2)) == 7 and all(abs(sum(w) - 1) < 1e-9 for w in weight_grid(3))
    print("combiners self-test OK")
