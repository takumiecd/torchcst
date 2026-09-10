"""Public API parity, mixed ownership, and exact checkpoint continuation."""

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.test_training_smoke import amplitude_bandwidth
from torchcst import (
    AdamWConfig,
    Chart,
    CSTLinear,
    CSTLocalAdam,
    LocalAdamConfig,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def model(dtype=torch.float32, device="cpu"):
    torch.manual_seed(13)
    return nn.Sequential(
        CSTLinear(
            Chart.linspace(3),
            Chart.linspace(2),
            atoms=2,
            kernel=amplitude_bandwidth(),
            backend="factored",
            dtype=dtype,
        ),
        nn.Tanh(),
        nn.Linear(2, 2, dtype=dtype),
        CSTLinear(
            Chart.linspace(2),
            Chart.linspace(2),
            atoms=2,
            kernel=amplitude_bandwidth(),
            backend="factored",
            dtype=dtype,
        ),
    ).to(device)


def step(m, o):
    p = next(m.parameters())
    x = torch.tensor(
        [[1.0, -0.5, 0.3], [0.2, 0.7, -0.1]], device=p.device, dtype=p.dtype
    )
    o.zero_grad()
    m(x).square().mean().backward()
    o.step()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("deferred", [False, True])
def test_public_resumes_exactly(device, dtype, deferred):
    a = model(dtype, device)
    cfg = LocalAdamConfig(
        lr=0.05, betas=(0.9, 0.99), whitening="eigen", device_execution=deferred
    )
    new = CSTLocalAdam(a, cst=cfg, dense=AdamWConfig())
    for _ in range(3):
        step(a, new)
    new.check_errors()
    saved_model, saved_state = copy.deepcopy(a.state_dict()), new.state_dict()
    c = model(dtype, device)
    c.load_state_dict(saved_model)
    resumed = CSTLocalAdam(c, cst=cfg, dense=AdamWConfig())
    resumed.load_state_dict(saved_state)
    for site in resumed._sites:
        assert site.state.second.basis.dtype == torch.float64
        assert site.state.second.value.dtype == dtype
    for _ in range(2):
        step(a, new)
        step(c, resumed)
    for actual, expected in zip(a.parameters(), c.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert all(
        r.backend == "atom_block_direct" for r in new.last_step.compression_results
    )


def test_dense_subset_matches_torch_adamw():
    m = model(torch.float64)
    o = CSTLocalAdam(m, dense=AdamWConfig(lr=0.003))
    reference = [nn.Parameter(p.detach().clone()) for p in m[2].parameters()]
    adam = torch.optim.AdamW(reference, lr=0.003)
    for _ in range(3):
        o.zero_grad()
        m(torch.ones(2, 3, dtype=torch.float64)).square().mean().backward()
        for ref, p in zip(reference, m[2].parameters()):
            ref.grad = p.grad.clone()
        o.step()
        adam.step()
        for ref, p in zip(reference, m[2].parameters()):
            torch.testing.assert_close(p, ref, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("deferred", [False, True])
def test_invalid_update_cannot_partially_commit_mixed_model(deferred):
    m = model()
    o = CSTLocalAdam(m, device_execution=deferred, dense=AdamWConfig())
    step(m, o)
    before = [p.detach().clone() for p in m.parameters()]
    solve = o._solve

    def broken(*args):
        result = solve(*args)
        return replace(
            result, displacement=torch.full_like(result.displacement, float("nan"))
        )

    o._solve = broken
    if deferred:
        step(m, o)
        with pytest.raises(FloatingPointError):
            o.check_errors()
    else:
        with pytest.raises(FloatingPointError):
            step(m, o)
    for actual, expected in zip(m.parameters(), before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"first_moment_damping": 0},
        {"update_damping": 0},
        {"update_damping": float("nan")},
        {"solve_rtol": 1},
    ],
)
def test_config_rejects_invalid_regularization(kwargs):
    with pytest.raises(ValueError):
        LocalAdamConfig(**kwargs)


def test_no_trust_radius_option_and_checkpoint_contract():
    with pytest.raises(TypeError):
        LocalAdamConfig(trust_radius=0.25)
    m = model()
    o = CSTLocalAdam(m, dense=AdamWConfig())
    other = CSTLocalAdam(model(), update_damping=0.02, dense=AdamWConfig())
    with pytest.raises(ValueError, match="contract"):
        other.load_state_dict(o.state_dict())


def test_training_never_requests_a_full_gram(monkeypatch):
    from torchcst._derivatives.local_tangent import AtomLocalTangentGeometry

    original = AtomLocalTangentGeometry.cross

    def local_only(self, *args, **kwargs):
        assert kwargs.get("local") is True
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AtomLocalTangentGeometry, "cross", local_only)
    m = model()
    o = CSTLocalAdam(m, dense=AdamWConfig())
    for _ in range(3):
        step(m, o)
