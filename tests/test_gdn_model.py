"""tests de la red de pronostico de GDN, formas del forward, la adyacencia inyectable, el flujo
de gradiente y el determinismo. todo sintetico."""
import pytest
import torch
import torch.nn.functional as F

from src.gdn.model import GDN


def _model(n_nodes: int = 6, context: int = 10, topk: int = 2, **kw) -> GDN:
    torch.manual_seed(0)
    return GDN(n_nodes=n_nodes, context=context, embed_dim=8, out_layer_inter_dim=16,
               topk=topk, **kw)


def test_forward_shape_and_finite():
    model = _model()
    out = model(torch.randn(4, 6, 10))
    assert out.shape == (4, 6)
    assert torch.isfinite(out).all()


def test_forward_input_validation():
    model = _model()
    with pytest.raises(ValueError):
        model(torch.randn(4, 5, 10))  # n_nodes equivocado
    with pytest.raises(ValueError):
        model(torch.randn(4, 6, 9))  # context equivocado


def test_deeper_out_mlp_shape():
    model = _model(out_layer_num=3)
    out = model(torch.randn(4, 6, 10))
    assert out.shape == (4, 6) and torch.isfinite(out).all()


def test_horizon_output_shape():
    model = _model(horizon=3)
    out = model(torch.randn(4, 6, 10))
    assert out.shape == (4, 6, 3) and torch.isfinite(out).all()  # (B, N, horizon) con h>1
    assert _model(horizon=1)(torch.randn(4, 6, 10)).shape == (4, 6)  # h=1 colapsa a (B, N)


def test_learned_adjacency_degree():
    model = _model(topk=2, include_self_loops=True)
    adj = model.adjacency()
    assert adj.shape == (6, 6)
    assert (adj.sum(dim=1) == 3).all()  # 2 vecinos mas el propio


def test_injected_adjacency_overrides_learned():
    model = _model()
    model.eval()  # congela BN para que lo unico que cambie sea el grafo
    x = torch.randn(4, 6, 10)
    out_learned = model(x)
    out_identity = model(x, adjacency=torch.eye(6))  # solo self-loops
    assert out_identity.shape == (4, 6)
    assert not torch.allclose(out_learned, out_identity)  # el grafo importa de verdad


def test_injected_adjacency_self_loops_only_is_finite():
    model = _model()
    model.eval()
    out = model(torch.randn(2, 6, 10), adjacency=torch.eye(6))
    assert torch.isfinite(out).all()


def test_gradients_flow_including_embedding():
    model = _model()
    x = torch.randn(8, 6, 10)
    target = torch.randn(8, 6)
    F.mse_loss(model(x), target).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)
    # el embedding de sensor gobierna el grafo y la fusion, asi que tiene que recibir gradiente
    assert model.embedding.weight.grad.abs().sum().item() > 0


def test_determinism_same_seed():
    x = torch.randn(4, 6, 10)
    torch.testing.assert_close(_model()(x), _model()(x))
