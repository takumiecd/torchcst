"""Optional checks follow ordinary PyTorch step semantics without host waits."""

import copy
import json

import pytest
import torch
from test_cell_mass_polar import layer

from torchcst import CSTOptimizer
from torchcst._backends.torch import finite
from torchcst._backends.torch.geometry.base import validate_points


def optimizer(model, *, check_finite):
    kwargs = {}
    if model.atoms.p.is_cuda:
        from torchcst._backends.catalog import get_registry
        from torchcst._backends.cuda.algorithms.polar_update.plans import (
            FLAT_TORUS_FUSED,
        )
        from torchcst._backends.dispatch import FixedSelector

        kwargs["update_selector"] = FixedSelector(
            FLAT_TORUS_FUSED, registry=get_registry()
        )
    return CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=0.02, fused=model.atoms.p.is_cuda
        ),
        model=model,
        check_finite=check_finite,
        **kwargs,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_finite_updates_and_moments_are_bitwise_equal(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    a = layer(count=257, dtype=torch.float32, device=device)
    b = copy.deepcopy(a)
    oa, ob = optimizer(a, check_finite=True), optimizer(b, check_finite=False)
    for step in range(4):
        if step == 3:
            for model in (a, b):
                model.kernel.scalar("radial_regularization").mul_(1.1)
        gradient = torch.randn_like(a.atoms.p)
        a.atoms.p.grad, b.atoms.p.grad = gradient.clone(), gradient.clone()
        oa.step()
        ob.step()
        torch.testing.assert_close(a.atoms.p, b.atoms.p, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                oa.state[a.atoms.p][key], ob.state[b.atoms.p][key], rtol=0, atol=0
            )
    assert finite.checks_enabled()


def test_context_is_reset_after_errors_and_closure_keeps_public_checks():
    model = layer(count=3)
    opt = optimizer(model, check_finite=False)
    model.atoms.p.grad = torch.ones_like(model.atoms.p)
    called = []

    def closure():
        called.append(finite.checks_enabled())
        return torch.tensor(2.0)

    assert opt.step(closure).item() == 2
    assert called == [True]
    model.kernel.scalar("sigma_min_input").fill_(-1)
    before = model.atoms.p.detach().clone()
    state = {k: v.clone() for k, v in opt.state[model.atoms.p].items()}
    with pytest.raises(ValueError):
        opt.step()
    assert finite.checks_enabled()
    torch.testing.assert_close(before, model.atoms.p, rtol=0, atol=0)
    for k, v in state.items():
        torch.testing.assert_close(v, opt.state[model.atoms.p][k], rtol=0, atol=0)
    with pytest.raises(ValueError, match="finite"):
        validate_points(model.input_chart.geometry, torch.full((1, 1), float("nan")))


def test_no_implicit_nonfinite_rejection_matches_standard_adamw():
    a = torch.nn.Linear(2, 1)
    b = copy.deepcopy(a)
    oa = CSTOptimizer(torch.optim.AdamW(a.parameters()), model=a, check_finite=False)
    ob = torch.optim.AdamW(b.parameters())
    for pa, pb in zip(a.parameters(), b.parameters()):
        pa.grad = torch.full_like(pa, float("nan"))
        pb.grad = pa.grad.clone()
    oa.step()
    ob.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("value", [0, None, "false"])
def test_invalid_check_policy_is_rejected(value):
    with pytest.raises(TypeError, match="check_finite"):
        optimizer(layer(count=3), check_finite=value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_steady_optimizer_has_no_sync_or_device_to_host_copy(tmp_path):
    model = layer(count=257, dtype=torch.float32, device="cuda")
    opt = optimizer(model, check_finite=False)
    model.atoms.p.grad = torch.randn_like(model.atoms.p)
    opt.step()
    opt.step()
    torch.cuda.synchronize()
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile,
        torch.profiler.record_function("pytorch-mode-step"),
    ):
        opt.step()
    path = tmp_path / "trace.json"
    profile.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    span = next(e for e in events if e.get("name") == "pytorch-mode-step")
    runtime = [
        e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        and e["pid"] == span["pid"]
        and e["tid"] == span["tid"]
        and span["ts"] <= e["ts"] < span["ts"] + span["dur"]
    ]
    assert not [e["name"] for e in runtime if "Synchronize" in e["name"]]
    correlations = {e.get("args", {}).get("correlation") for e in runtime}
    correlations.discard(None)
    assert not [
        e["name"]
        for e in events
        if e.get("cat") == "gpu_memcpy"
        and "DtoH" in e.get("name", "")
        and e.get("args", {}).get("correlation") in correlations
    ]
