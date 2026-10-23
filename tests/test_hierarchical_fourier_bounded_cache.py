"""A bounded cache is optional storage, never a support truncation."""

import math
from dataclasses import asdict, replace

import pytest
import torch

from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.algorithm import (
    BoundedCacheHierarchicalFourierAlgorithm,
    FourierContext,
)
from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
    BoundedCacheFourierRecipe,
    FourierLinearSpec,
    PackedFourierRecipe,
)


def test_bounded_cache_declaration_and_old_schema():
    from torchcst._backends.registry import Registry
    from torchcst._backends.schema import ExecutionPlan

    alg = BoundedCacheHierarchicalFourierAlgorithm()
    reg = Registry()
    reg.register(alg)
    assert "cache_sites" not in asdict(PackedFourierRecipe())
    for sites in (32, 64, 128):
        recipe = BoundedCacheFourierRecipe(cache_sites=sites)
        plan = ExecutionPlan(alg.id, alg.revision, recipe)
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
        ctx = FourierContext(FourierLinearSpec(128), "cuda", torch.float32, 32, 128)
        assert alg.supports(ctx, recipe).supported
        assert not alg.supports(replace(ctx, atoms=129), recipe).supported
    for change in (
        {"cache_sites": True},
        {"cache_sites": 16},
        {"cache_sites": 32.0},
        {"sample_cache": 512},
    ):
        with pytest.raises(ValueError):
            BoundedCacheFourierRecipe(**change)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("sites", [32, 64])
def test_cache_overflow_empty_support_floor_and_all_gradients(sites):
    from benchmarks.cuda.linear.hierarchical_fourier import independent_gate
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        _pack,
        _prepare,
        _routes,
        apply,
        tables,
    )

    def logit(width):
        return math.log((width - 0.1) / (10 - width))

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        spec = FourierLinearSpec(128)
        recipe = BoundedCacheFourierRecipe(cache_sites=sites, local_overlap=True)
        p = torch.tensor(
            [
                [0.4, 6.713, 6.137, logit(9.9)],
                [0.0, 3.25, 6.3, logit(0.2)],
                [-0.2, -30.0, 7.0, logit(0.2)],
                [0.7, -0.02, 12.61, logit(0.12)],
                [0.5, 6.011, 6.713, logit(3.4)],
                [0.3, -0.999 * 0.25, 6.713, logit(0.105)],
                [0.2, 6.11, 5.07, logit(1.8)],
            ],
            device="cuda",
            requires_grad=True,
        )
        gen = torch.Generator().manual_seed(132)
        x = torch.randn(3, 128, generator=gen).cuda().requires_grad_()
        dy = torch.randn(3, 128, generator=gen).cuda()
        tab = tables(spec, recipe, p.device)
        routes = _routes(p, spec, recipe, tab)
        packed, packed_routes, *_ = _pack(p, routes, spec, recipe, tab)
        prep = _prepare(packed, spec, recipe, tab, packed_routes)
        assert prep[7].shape == (7, 2, 3, sites)
        assert prep[7].dtype == torch.float64
        counts = prep[8][:, :, 1]
        assert (counts == -1).any()  # Wider supports use the complete fallback.
        assert (counts >= 0).any()  # Narrow direct profiles remain cached.
        base_recipe = PackedFourierRecipe(local_overlap=True)
        for step in range(2):
            if step:
                with torch.no_grad():
                    p[0, 3] = logit(0.2)
                    p[1, 3] = logit(9.9)
            y = apply(x, p, spec, recipe)
            grad = torch.autograd.grad(y, (x, p), dy, retain_graph=True)
            truth = independent_gate(x, p, dy, spec, chunk=7)
            base_y = apply(x, p, spec, base_recipe)
            base_grad = torch.autograd.grad(base_y, (x, p), dy)
            for actual, expected in zip((y, *grad), truth):
                torch.testing.assert_close(
                    actual.double(), expected, atol=4e-4, rtol=4e-4
                )
            for actual, expected in zip((y, *grad), (base_y, *base_grad)):
                torch.testing.assert_close(actual, expected, atol=4e-4, rtol=4e-4)
            for actual, expected in zip(torch.autograd.grad(y, (x, p), dy), grad):
                torch.testing.assert_close(actual, expected, atol=4e-4, rtol=4e-4)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
