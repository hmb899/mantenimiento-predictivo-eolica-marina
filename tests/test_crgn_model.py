"""tests estaticos del codificador GCN residual de CRGN y de su InfoNCE/NT-Xent. sintetico,
sin datos y sin GPU."""
import numpy as np
import pytest
import torch

from src.crgn.graph import build_adjacency
from src.crgn.model import CRGN, ResidualGCN, nt_xent_loss


def _adj(n=10, t=300, k=3, seed=0):
    x = np.random.default_rng(seed).standard_normal((t, n))
    return torch.from_numpy(build_adjacency(x, k=k)).float()


def test_residual_gcn_shapes():
    enc = ResidualGCN(in_dim=27, hidden_dim=64, n_layers=2)
    x = torch.randn(8, 10, 27)
    out = enc(x, _adj(10))
    assert out.shape == (8, 10, 64)
    assert torch.isfinite(out).all()


def test_residual_gcn_requires_one_layer():
    with pytest.raises(ValueError):
        ResidualGCN(in_dim=27, hidden_dim=64, n_layers=0)


def test_crgn_encode_and_readout():
    torch.manual_seed(0)
    model = CRGN(n_nodes=10, in_dim=27, embed_dim=64, n_layers=2)
    x = torch.randn(8, 10, 27)
    emb = model.encode(x, _adj(10))
    assert emb.shape == (8, 64)  # readout de media sobre los nodos
    assert torch.isfinite(emb).all()


def test_crgn_max_readout():
    model = CRGN(n_nodes=10, in_dim=27, embed_dim=32, n_layers=2, readout="max")
    emb = model.encode(torch.randn(4, 10, 27), _adj(10))
    assert emb.shape == (4, 32)


def test_crgn_rejects_bad_readout():
    with pytest.raises(ValueError):
        CRGN(n_nodes=10, in_dim=27, readout="sum")


def test_contrastive_loss_finite_and_nonnegative():
    torch.manual_seed(1)
    model = CRGN(n_nodes=10, in_dim=27, embed_dim=64, n_layers=2)
    loss = model.contrastive_loss(torch.randn(16, 10, 27), _adj(10))
    assert loss.item() >= 0.0
    assert torch.isfinite(loss).all()
    loss.backward()  # los gradientes fluyen, es seguro para autograd
    assert any(p.grad is not None for p in model.parameters())


def test_nt_xent_identical_views_low_loss():
    # dos vistas identicas y bien separadas, cada positivo domina y la perdida sale pequena
    z = torch.eye(4)
    loss_same = nt_xent_loss(z, z.clone(), temperature=0.1)
    rng = torch.Generator().manual_seed(0)
    z2 = torch.randn(4, 8, generator=rng)
    loss_rand = nt_xent_loss(z2, torch.randn(4, 8, generator=rng), temperature=0.5)
    assert loss_same < loss_rand  # positivos alineados, NT-Xent mas bajo


def test_crgn_deterministic_with_seed():
    def emb():
        torch.manual_seed(7)
        m = CRGN(n_nodes=10, in_dim=27, embed_dim=32, n_layers=2)
        with torch.no_grad():
            return m.encode(torch.ones(2, 10, 27), _adj(10))
    assert torch.allclose(emb(), emb())
