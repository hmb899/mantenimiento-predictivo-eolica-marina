"""tests estaticos del modelo, formas del denoiser, el schedule, la perdida restringida a
las posiciones objetivo, la ausencia de fuga desde los valores sin observar, el contrato
de imputacion y el contrato del scorer."""
import numpy as np
import pytest
import torch

from src.imdiffusion.diffusion import ImDiffusionModel, masked_noise_loss, quad_beta_schedule
from src.imdiffusion.masking import grating_mask
from src.imdiffusion.scorer import ImDiffusionScorer, StepEnsemble, aggregate_window_errors

TINY = dict(n_features=4, window_size=12, num_steps=8, channels=8, layers=2, nheads=2,
            diffusion_embedding_dim=16, feature_embedding_dim=4, time_embedding_dim=8)


def _tiny_model() -> ImDiffusionModel:
    torch.manual_seed(0)
    return ImDiffusionModel(**TINY)


def _full_masks(b: int = 3, strategy: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """mascaras cond/target para observed ≡ 1 con W=12 y split=4."""

    g = torch.from_numpy(grating_mask(12, 4, strategy))
    target = g[None, None, :].expand(b, 4, 12).float()
    return 1.0 - target, target


def test_quad_schedule_endpoints_and_monotonic():
    betas = quad_beta_schedule(50, 1e-4, 0.5)
    assert betas.shape == (50,)
    assert torch.isclose(betas[0], torch.tensor(1e-4))
    assert torch.isclose(betas[-1], torch.tensor(0.5))
    assert (betas.diff() > 0).all()


def test_calc_loss_scalar_finite_and_shapes():
    model = _tiny_model()
    cond, target = _full_masks()
    x = torch.randn(3, 4, 12)
    loss = model.calc_loss(x, cond, target, strategy=0)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_denoiser_output_shape():
    model = _tiny_model()
    cond, _ = _full_masks(b=2)
    x = torch.randn(2, 2, 4, 12)  # (B, 2, K, L)
    side = model._side_info(cond[:2])
    t = torch.zeros(2, dtype=torch.long)
    s = torch.zeros(2, dtype=torch.long)
    out = model.denoiser(x, side, t, s)
    assert out.shape == (2, 4, 12)


def test_loss_ignores_non_target_predictions():
    torch.manual_seed(1)
    noise = torch.randn(2, 3, 5)
    predicted = torch.randn(2, 3, 5)
    target = (torch.rand(2, 3, 5) > 0.5).float()
    base = masked_noise_loss(noise, predicted, target)
    perturbed = predicted + 100.0 * (1.0 - target)  # cambia solo las posiciones no objetivo
    assert torch.equal(masked_noise_loss(noise, perturbed, target), base)
    inside = predicted + 1.0 * target  # tocar las posiciones objetivo si tiene que notarse
    assert not torch.equal(masked_noise_loss(noise, inside, target), base)


def test_unobserved_values_never_reach_the_loss():
    model = _tiny_model()
    b = 2
    g = torch.from_numpy(grating_mask(12, 4, 0))
    observed = torch.ones(b, 4, 12)
    observed[:, 1, 2:5] = 0.0  # recorta unas posiciones sin observar
    target = observed * g[None, None, :]
    cond = observed * (~g)[None, None, :]
    t = torch.full((b,), 3, dtype=torch.long)
    x = torch.randn(b, 4, 12)
    torch.manual_seed(7)
    loss_a = model.calc_loss(x, cond, target, strategy=0, t=t)
    x2 = x.clone()
    x2[:, 1, 2:5] += 1000.0  # basura en las posiciones sin observar
    torch.manual_seed(7)
    loss_b = model.calc_loss(x2, cond, target, strategy=0, t=t)
    assert torch.equal(loss_a, loss_b)


def test_overlapping_masks_rejected():
    model = _tiny_model()
    x = torch.randn(2, 4, 12)
    ones = torch.ones(2, 4, 12)
    with pytest.raises(ValueError):
        model.calc_loss(x, ones, ones, strategy=0)


def test_impute_contract():
    model = _tiny_model().eval()
    cond, target = _full_masks(b=2)
    x = torch.randn(2, 4, 12)
    torch.manual_seed(3)
    x_hat, inter = model.impute(x, cond, strategy=0, ddim_steps=4, return_intermediates=True)
    assert x_hat.shape == x.shape and torch.isfinite(x_hat).all()
    assert torch.equal(x_hat * cond, x * cond)  # el condicionamiento pasa intacto
    assert len(inter) == 4 and all(i.shape == x.shape for i in inter)
    # lo que cae fuera de cond no llega al modelo, y con la misma semilla poner basura
    # ahi da exactamente la misma salida
    x2 = x + 50.0 * target
    torch.manual_seed(3)
    x_hat2 = model.impute(x2, cond, strategy=0, ddim_steps=4)
    assert torch.allclose(x_hat * target, x_hat2 * target)


def test_scorer_contract_and_determinism():
    model = _tiny_model()
    scorer = ImDiffusionScorer(model, split=4, ddim_steps=4, batch_size=3, device="cpu")
    assert hasattr(scorer, "percentile")  # atributo del contrato de arnes
    windows = np.random.default_rng(0).normal(size=(5, 12, 4)).astype(np.float32)
    torch.manual_seed(11)
    s1 = scorer.score_windows(windows)
    assert s1.shape == (5,) and s1.dtype == np.float64 and np.isfinite(s1).all()
    torch.manual_seed(11)
    s2 = scorer.score_windows(windows)
    np.testing.assert_array_equal(s1, s2)


def test_scorer_feature_offset_mode():
    model = _tiny_model()
    scorer = ImDiffusionScorer(model, split=4, ddim_steps=4, batch_size=3, device="cpu",
                               mask_mode="feature_offset")
    windows = np.random.default_rng(3).normal(size=(4, 12, 4)).astype(np.float32)
    torch.manual_seed(31)
    s1 = scorer.score_windows(windows)
    assert s1.shape == (4,) and np.isfinite(s1).all()
    torch.manual_seed(31)
    np.testing.assert_array_equal(scorer.score_windows(windows), s1)
    # los dos modos enmascaran conjuntos de posiciones distintos, asi que los scores difieren
    base = ImDiffusionScorer(model, split=4, ddim_steps=4, batch_size=3, device="cpu")
    torch.manual_seed(31)
    assert not np.array_equal(base.score_windows(windows), s1)


def test_scorer_regime_conditioning():
    model = _tiny_model()
    # las filas exogenas nunca se enmascaran, de modo que su error de imputacion es cero exacto
    scorer = ImDiffusionScorer(model, split=4, ddim_steps=4, batch_size=3, device="cpu",
                               mask_mode="feature_offset", exo_idx=[0, 2])
    windows = np.random.default_rng(5).normal(size=(4, 12, 4)).astype(np.float32)
    torch.manual_seed(41)
    s = scorer.score_windows(windows)
    assert s.shape == (4,) and np.isfinite(s).all()
    # en los scores por feature las exogenas (0, 2) no acumulan error, nunca son objetivo
    torch.manual_seed(41)
    full = scorer.score_windows_full(windows)
    assert np.allclose(full["perfeat"][:, [0, 2]], 0.0)
    assert (full["perfeat"][:, [1, 3]] > 0).any()  # la senal la llevan las endogenas


def test_scorer_rejects_bad_input():
    model = _tiny_model()
    scorer = ImDiffusionScorer(model, split=4, ddim_steps=4, device="cpu")
    with pytest.raises(ValueError):
        scorer.score_windows(np.zeros((3, 9, 4), np.float32))  # W equivocada
    bad = np.zeros((3, 12, 4), np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        scorer.score_windows(bad)  # no finito, se rechaza


def test_scorer_per_step_shape():
    model = _tiny_model()
    scorer = ImDiffusionScorer(model, split=4, ddim_steps=4, batch_size=2, device="cpu")
    windows = np.random.default_rng(1).normal(size=(3, 12, 4)).astype(np.float32)
    steps = scorer.score_windows_steps(windows)
    assert steps.shape == (3, 4) and np.isfinite(steps).all()


def test_score_windows_full_consistent_with_contract():
    model = _tiny_model()
    scorer = ImDiffusionScorer(model, split=4, ddim_steps=4, batch_size=2, device="cpu")
    windows = np.random.default_rng(2).normal(size=(5, 12, 4)).astype(np.float32)
    torch.manual_seed(21)
    full = scorer.score_windows_full(windows)
    torch.manual_seed(21)
    contract = scorer.score_windows(windows)
    np.testing.assert_array_equal(full["mean"], contract)  # mismas tiradas de RNG, identico
    assert full["steps"].shape == (5, 4)
    assert set(full) == {"mean", "median", "topk_q", "steps", "perfeat"}
    # el ultimo paso DDIM es la imputacion final, asi que su media es el score del contrato
    np.testing.assert_allclose(full["steps"][:, -1], contract, rtol=1e-6)
    # los scores por feature son (N, K) y su media sobre features da la media de la ventana
    assert full["perfeat"].shape == (5, 4)
    np.testing.assert_allclose(full["perfeat"].mean(axis=1), contract, rtol=1e-5)


def test_aggregations_exact_values():
    err = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert aggregate_window_errors(err, "mean").item() == pytest.approx(2.5)
    assert aggregate_window_errors(err, "median").item() == pytest.approx(2.0)  # mediana baja
    assert aggregate_window_errors(err, "topk_q", topk_q=0.5).item() == pytest.approx(3.5)
    with pytest.raises(ValueError):
        aggregate_window_errors(err, "max")


def test_step_ensemble_votes_in_unit_interval():
    rng = np.random.default_rng(0)
    train = np.abs(rng.normal(size=(200, 4)))
    ens = StepEnsemble(percentile=90.0).fit(train)
    votes = ens.score(np.abs(rng.normal(size=(10, 4))))
    assert votes.shape == (10,) and ((votes >= 0) & (votes <= 1)).all()
    # una ventana con scores por paso enormes tiene que votar anomala en todos los pasos
    assert ens.score(train.max(axis=0, keepdims=True) * 10).item() == 1.0
