"""Dense visible-space oracles for the regularized Cholesky coordinates."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.test_local_optimizer import DEVICES, model, step
from torchcst import AdamWConfig, CSTLocalAdam, LocalAdamConfig
from torchcst.optim.atom_grad import AtomGradientObservation
from torchcst.optim.moments.transported_rms import (
    CholeskyParameterRMS,
    TransportedParameterRMS,
)


def context(j, old, point):
    def cross(a, b, *, local):
        assert local
        return j.transpose(-1, -2) @ (j if a is b else old)

    return SimpleNamespace(
        current_point=point, geometry=SimpleNamespace(cross=cross, visible_shape=(6, 1))
    )


@pytest.mark.parametrize("damping", [0.0, 0.01])
def test_transport_and_metric_match_visible_oracle(damping):
    torch.manual_seed(6)
    rms = CholeskyParameterRMS(0.9, eps=1e-6, damping=damping)
    old_j = torch.randn(2, 6, 3, dtype=torch.float64)
    old_q = old_c = None
    for t in range(1, 4):
        j = old_j + 0.1 * torch.randn_like(old_j)
        point = torch.full((2, 3), float(t), dtype=torch.float64)
        ctx = context(j, old_j, point)
        if t == 1:
            state = rms.initialize(ctx)
        g = torch.randn(6, dtype=torch.float64)
        observation = AtomGradientObservation(jg=(j.transpose(-1, -2) @ g))
        result = rms.expand(state, observation, ctx, next_step=t)
        saved = result.pending_state
        q = j @ saved.basis
        r = j.transpose(-1, -2) @ j
        eye = torch.eye(3, dtype=torch.float64)
        torch.testing.assert_close(
            saved.basis.transpose(-1, -2) @ (r + damping * eye) @ saved.basis,
            eye.expand(2, 3, 3),
        )
        observed = g[:, None] @ g[None, :]
        history = 0 if old_q is None else old_q @ old_c @ old_q.transpose(-1, -2)
        expected = q.transpose(-1, -2) @ (0.9 * history + 0.1 * observed) @ q
        torch.testing.assert_close(saved.value, expected)
        values, vectors = torch.linalg.eigh(expected / (1 - 0.9**t))
        root = (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.transpose(
            -1, -2
        ) + 1e-6 * eye
        visible_metric = q @ root @ q.transpose(-1, -2)
        torch.testing.assert_close(
            result.metric.blocks,
            j.transpose(-1, -2) @ visible_metric @ j,
            atol=1e-7,
            rtol=1e-6,
        )
        old_j, old_q, old_c, state = j, q, expected, saved


def test_undamped_cholesky_and_eigen_metrics_agree():
    torch.manual_seed(71)
    j = torch.randn(2, 6, 3, dtype=torch.float64)
    point = torch.zeros(2, 3, dtype=torch.float64)
    ctx = context(j, j, point)
    obs = AtomGradientObservation(jg=torch.randn_like(point))
    components = [
        CholeskyParameterRMS(0.9, eps=1e-6, damping=0),
        TransportedParameterRMS(0.9, eps=1e-6),
    ]
    results = [c.expand(c.initialize(ctx), obs, ctx, next_step=1) for c in components]
    torch.testing.assert_close(
        results[0].metric.blocks, results[1].metric.blocks, atol=1e-7, rtol=1e-6
    )


def test_singular_gram_is_smooth_and_does_not_use_eigh(monkeypatch):
    rms = CholeskyParameterRMS(0.9, eps=1e-8, damping=0.01)

    def forbidden(*args, **kwargs):
        raise AssertionError("whitening must not perform spectral selection")

    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    r = torch.diag_embed(torch.tensor([[0.0, 1e-6, 1.0]], dtype=torch.float64))
    b = rms.whitening_basis(r)
    perturbed = rms.whitening_basis(r + 1e-10 * torch.eye(3, dtype=r.dtype))
    assert torch.isfinite(b).all()
    assert (perturbed - b).abs().max() < 1e-6


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("deferred", [False, True])
def test_mixed_checkpoint_continuation(device, dtype, deferred):
    a = model(dtype, device)
    cfg = LocalAdamConfig(whitening="cholesky", device_execution=deferred)
    o = CSTLocalAdam(a, cst=cfg, dense=AdamWConfig())
    for _ in range(3):
        step(a, o)
    b = model(dtype, device)
    b.load_state_dict(a.state_dict())
    p = CSTLocalAdam(b, cst=cfg, dense=AdamWConfig())
    p.load_state_dict(o.state_dict())
    for _ in range(2):
        step(a, o)
        step(b, p)
    o.check_errors()
    p.check_errors()
    for actual, expected in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    eigen = CSTLocalAdam(
        model(dtype, device), cst=replace(cfg, whitening="eigen"), dense=AdamWConfig()
    )
    with pytest.raises(ValueError, match="contract"):
        eigen.load_state_dict(o.state_dict())


def test_invalid_mode_rejected():
    assert LocalAdamConfig(0.05).lr == 0.05
    with pytest.raises(ValueError, match="whitening"):
        LocalAdamConfig(whitening="unknown")
