"""Actual CUDA gates; CPU collection never counts skips as qualification."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires NVIDIA CUDA"
)


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("strategy", ["direct", "shared"])
@pytest.mark.parametrize("cached", [False, True])
def test_centered_support_dynamic_width_input_and_all_gradients(
    modes, strategy, cached
):
    from benchmarks.cuda.linear.hierarchical_fourier import independent_gate
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
        FourierLinearSpec,
        FourierRecipe,
    )
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        apply,
    )

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        spec = FourierLinearSpec(256, modes)
        recipe = FourierRecipe(
            strategy=strategy,
            atom_chunk=256,
            minimum_sites=32 if modes == 4 else 64,
            native_cache=cached,
        )
        p = torch.tensor(
            [
                [0.4, 12.713, 10.137, 0.0],
                [0.0, 3.25, 11.3, -4.0],
                [-0.2, -30.0, 17.0, -4.0],
                [0.7, -0.02, 25.31, -6.0],
                [0.5, 12.011, 12.713, math.log((6.5 - 0.1) / (10 - 6.5))],
            ],
            device="cuda",
            requires_grad=True,
        )
        x = torch.randn(3, 256, device="cuda", requires_grad=True)
        dy = torch.randn_like(x)
        for step in range(2):
            if step:
                with torch.no_grad():
                    p[:, 3].add_(0.13)
                    p[:, 1:3].add_(0.013)
                    x.add_(0.007)
            y = apply(x, p, spec, recipe)
            dx, dp = torch.autograd.grad(y, (x, p), dy, retain_graph=True)
            repeated = torch.autograd.grad(y, (x, p), dy)
            target, tx, tp = independent_gate(x, p, dy, spec)
            for a, b in ((y, target), (dx, tx), (dp, tp), *zip((dx, dp), repeated)):
                torch.testing.assert_close(a.double(), b.double(), atol=4e-4, rtol=4e-4)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_saved_forward_parameter_version_is_checked():
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
        FourierLinearSpec,
        FourierRecipe,
    )
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        apply,
    )

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        p = torch.tensor([[0.5, 3.0, 2.0, -3.0]], device="cuda", requires_grad=True)
        x = torch.randn(2, 64, device="cuda", requires_grad=True)
        y = apply(x, p, FourierLinearSpec(64), FourierRecipe(strategy="direct"))
        with torch.no_grad():
            p.add_(0.001)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            y.sum().backward()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
