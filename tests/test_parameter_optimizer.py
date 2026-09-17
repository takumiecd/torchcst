import copy
import math

import pytest
import torch
from torch import nn

from torchcst import (
    AdamWConfig,
    Amplitude,
    Chart,
    CSTLinear,
    CSTParameterAdam,
    Gaussian,
    ParameterAdamConfig,
    Separable,
)


def model(atoms=3, inputs=5, outputs=4):
    return CSTLinear(
        Chart.linspace(inputs, low=-1.0, high=1.0),
        Chart.linspace(outputs, low=-1.0, high=1.0),
        atoms=atoms,
        kernel=Amplitude(
            Separable(input_profile=Gaussian(0.4), output_profile=Gaussian(0.3))
        ),
        backend="factored",
        dtype=torch.float64,
    )


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_matches_scheduled_pytorch_adam_and_resumes(device):
    torch.manual_seed(7)
    site = model().to(device)
    oracle = copy.deepcopy(site)
    opt = CSTParameterAdam(site, cst=ParameterAdamConfig(decay_steps=6))
    ref = torch.optim.Adam(
        oracle.parameters(), lr=0.03, betas=(0.5, 0.99), foreach=False
    )
    for step in range(9):
        gradient = torch.randn_like(site.atoms.p)
        site.atoms.p.grad = gradient.clone()
        oracle.atoms.p.grad = gradient.clone()
        ref.param_groups[0]["lr"] = 0.03 * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / 5, 1)))
        )
        opt.step()
        ref.step()
        torch.testing.assert_close(site.atoms.p, oracle.atoms.p, rtol=1e-12, atol=1e-12)
        if step == 3:
            restored = CSTParameterAdam(site)
            restored.load_state_dict(copy.deepcopy(opt.state_dict()))
            opt = restored
    assert opt.param_groups[0]["schedule_step"] == 9
    assert opt.param_groups[0]["lr"] == 0.03


def test_mixed_model_matches_adamw_without_scheduling_dense_block():
    site = model()
    mixed = nn.Sequential(site, nn.Linear(4, 2, dtype=torch.float64))
    oracle = copy.deepcopy(mixed)
    opt = CSTParameterAdam(mixed, dense=AdamWConfig(lr=0.002, weight_decay=0.1))
    ref = torch.optim.AdamW(
        oracle[1].parameters(), lr=0.002, weight_decay=0.1, foreach=False
    )
    for _ in range(3):
        for p, q in zip(mixed.parameters(), oracle.parameters()):
            p.grad = torch.ones_like(p)
            q.grad = torch.ones_like(q)
        opt.step()
        ref.step()
    for p, q in zip(mixed[1].parameters(), oracle[1].parameters()):
        torch.testing.assert_close(p, q)


@pytest.mark.parametrize("atoms,inputs,outputs", [(3, 5, 4), (6, 101, 97)])
def test_state_scales_only_with_parameters_and_never_materializes(
    monkeypatch, atoms, inputs, outputs
):
    site = model(atoms, inputs, outputs)

    def forbidden(*args, **kwargs):
        raise AssertionError("dense representation or derivative was requested")

    for name in (
        "dense_weight",
        "materialized_atoms",
        "_materialize_atoms",
        "cst_derivatives",
    ):
        monkeypatch.setattr(site, name, forbidden)
    opt = CSTParameterAdam(site)
    for _ in range(2):
        opt.zero_grad()
        site(torch.randn(2, inputs, dtype=torch.float64)).square().mean().backward()
        opt.step()
    state = opt.state[site.atoms.p]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    assert state["exp_avg"].shape == site.atoms.p.shape
    assert state["exp_avg_sq"].shape == site.atoms.p.shape
    assert sum(t.numel() for t in state.values()) == 2 * site.atoms.p.numel() + 1
    assert site.atoms.grad is None


def test_no_grad_and_invalid_gradient_do_not_advance_or_partially_update():
    site = model()
    opt = CSTParameterAdam(site)
    opt.step()
    assert opt.param_groups[0]["schedule_step"] == 0
    assert not opt.state
    site.atoms.p.grad = torch.full_like(site.atoms.p, float("nan"))
    before = site.atoms.p.detach().clone()
    with pytest.raises(FloatingPointError):
        opt.step()
    torch.testing.assert_close(site.atoms.p, before)
    assert not opt.state


def test_closure_and_checkpoint_validation():
    site = model()
    opt = CSTParameterAdam(site)
    calls = []

    def closure():
        calls.append(1)
        opt.zero_grad()
        loss = site.atoms.p.square().sum()
        loss.backward()
        return loss

    assert opt.step(closure).ndim == 0
    assert len(calls) == 1
    state = copy.deepcopy(opt.state_dict())
    state["param_groups"][0]["param_shapes"] = [(100, 100)]
    with pytest.raises(ValueError, match="param_shapes"):
        opt.load_state_dict(state)
    state = copy.deepcopy(opt.state_dict())
    state["state"][0]["exp_avg"] = torch.zeros(2)
    with pytest.raises(ValueError, match="moment shape"):
        opt.load_state_dict(state)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": float("nan")},
        {"betas": (0.9, 1.0)},
        {"decay_steps": 1},
        {"decay_steps": True},
        {"min_lr_ratio": 1.1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ParameterAdamConfig(**kwargs)


def test_rejects_dense_backend_and_unconfigured_dense_parameters():
    site = model()
    site.backend = "auto"
    with pytest.raises(ValueError, match="factored"):
        CSTParameterAdam(site)
    site.backend = "factored"
    with pytest.raises(ValueError, match="ordinary"):
        CSTParameterAdam(nn.Sequential(site, nn.Linear(4, 2)))
