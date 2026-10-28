"""Batching preserves finite-value rejection and all optimizer update bits."""

import copy
from unittest.mock import patch

import pytest
import torch
from test_cell_mass_polar import layer

from torchcst import CSTOptimizer
from torchcst._backends.torch import finite


def per_tensor(tensors):
    for index, value in enumerate(tensors):
        if not bool(torch.isfinite(value).all()):
            return index
    return None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_first_invalid_matches_reference_without_overflow(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    good = torch.randn(9, 7, device=device, dtype=dtype).T
    # A squared norm would overflow; a finite maximum must still pass.
    extreme = torch.full((257,), torch.finfo(dtype).max, device=device, dtype=dtype)
    empty = good[:0]
    values = [good, extreme, empty]
    assert finite.first_nonfinite(values) is None
    for index in (0, 1):
        for bad in (float("inf"), -float("inf"), float("nan")):
            broken = [v.clone() for v in values]
            broken[index][(0,) * broken[index].ndim] = bad
            assert finite.first_nonfinite(broken) == per_tensor(broken) == index
    assert finite.first_nonfinite([]) is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complex_finite_components_may_have_infinite_magnitude(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    component = torch.full((5,), 3e38, device=device)
    value = torch.complex(component, component)
    assert torch.isinf(value.abs()).all()
    assert finite.first_nonfinite([value]) is None
    value[3] = complex(float("nan"), 0)
    assert finite.first_nonfinite([value]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_dtype_groups_preserve_first_index():
    a = torch.tensor([float("nan")], device="cuda", dtype=torch.float64)
    b = torch.tensor([float("inf")], device="cuda", dtype=torch.float32)
    assert finite.first_nonfinite([a, b]) == 0
    assert finite.first_nonfinite([b, a]) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optimizer_parameters_moments_and_step_are_bitwise_unchanged(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(939)
    a = torch.nn.Sequential(
        torch.nn.Linear(3, 128), layer(count=29, dtype=torch.float32)
    ).to(device)
    b = copy.deepcopy(a)
    kwargs = {}
    if device == "cuda":
        from torchcst._backends.catalog import get_registry
        from torchcst._backends.cuda.algorithms.polar_update.plans import (
            FLAT_TORUS_FUSED,
        )
        from torchcst._backends.dispatch import FixedSelector

        kwargs["update_selector"] = FixedSelector(
            FLAT_TORUS_FUSED, registry=get_registry()
        )
    oa = CSTOptimizer(
        torch.optim.AdamW(a.parameters(), lr=0.001, weight_decay=0.02),
        model=a,
        **kwargs,
    )
    ob = CSTOptimizer(
        torch.optim.AdamW(b.parameters(), lr=0.001, weight_decay=0.02),
        model=b,
        **kwargs,
    )
    for step in range(4):
        if step == 3:
            for model in (a, b):
                model[1].kernel.scalar("radial_regularization").mul_(1.1)
        for pa, pb in zip(a.parameters(), b.parameters()):
            gradient = torch.randn_like(pa)
            pa.grad, pb.grad = gradient.clone(), gradient.clone()
        with patch.object(finite, "first_nonfinite", per_tensor):
            oa.step()
        ob.step()
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(
                    oa.state[pa][key], ob.state[pb][key], rtol=0, atol=0
                )
    # Ordinary Parameters must also reject NaN before ANY parameter/state update.
    before = [p.detach().clone() for p in b.parameters()]
    moments = [
        {key: value.clone() for key, value in ob.state[p].items()}
        for p in b.parameters()
    ]
    next(b.parameters()).grad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="parameter gradient"):
        ob.step()
    for p, old, state in zip(b.parameters(), before, moments):
        torch.testing.assert_close(p, old, rtol=0, atol=0)
        for key, old_value in state.items():
            torch.testing.assert_close(ob.state[p][key], old_value, rtol=0, atol=0)
