"""Local-owner declaration checks and actual dynamic CUDA correctness gates."""

from dataclasses import replace

import pytest
import torch

from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.algorithm import (
    FourierContext,
    LocalHierarchicalFourierAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
    FourierLinearSpec,
    LocalFourierRecipe,
)


def local_recipe(**kwargs):
    return LocalFourierRecipe(
        atom_chunk=1024,
        minimum_sites=32,
        width_cutoff=0.1,
        native_cache=True,
        table_coefficients=True,
        atom_block=4,
        batch_block=4,
        moment_policy="hierarchical",
        sample_cache=512,
        boundary_cache=True,
        local_blocks=8,
        **kwargs,
    )


def test_local_recipe_and_support_bound():
    from torchcst._backends.registry import Registry
    from torchcst._backends.schema import ExecutionPlan

    alg = LocalHierarchicalFourierAlgorithm()
    reg = Registry()
    reg.register(alg)
    recipe = local_recipe(local_overlap=True)
    plan = ExecutionPlan(alg.id, alg.revision, recipe)
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    ctx = FourierContext(FourierLinearSpec(1024), "cuda", torch.float32, 32, 1024)
    assert alg.supports(ctx, recipe).supported
    assert not alg.supports(replace(ctx, atoms=1025), recipe).supported
    for kw in (
        {"local_blocks": True},
        {"local_blocks": -1},
        {"local_overlap": True},
        {"local_width_cutoff": float("nan")},
        {"local_width_cutoff": 0.51},
    ):
        with pytest.raises(ValueError):
            LocalFourierRecipe(**kw)
    for kw in ({"reuse": "h"}, {"sort_atoms": True}, {"atom_block": 1}):
        with pytest.raises(ValueError):
            replace(recipe, **kw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("overlap", [False, True])
def test_local_owner_all_gradients_and_retained(modes, overlap):
    from test_hierarchical_fourier_cuda import (
        test_centered_support_dynamic_width_input_and_all_gradients,
    )

    test_centered_support_dynamic_width_input_and_all_gradients(
        modes,
        "shared",
        True,
        True,
        True,
        "none",
        "hierarchical",
        sample_cache=512,
        local_blocks=8,
        local_overlap=overlap,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_local_layout_memberships_and_cutoff_change():
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier import (
        local_kernels as lk,
    )
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        _prepare,
        tables,
    )

    spec, recipe = FourierLinearSpec(256, 8), local_recipe()
    widths = torch.tensor([0.105, 0.2, 0.349, 0.351, 3.0, 0.2, 0.2], device="cuda")
    p = torch.zeros(7, 4, device="cuda")
    p[:, 0] = 0.4
    p[:, 1] = torch.tensor([-0.01, 3.19, 8.71, 3.2, 12.7, -100, 6.4], device="cuda")
    p[:, 2] = 10.137
    p[:, 3] = torch.logit((widths - 0.1) / 9.9)
    tab = tables(spec, recipe, p.device)
    for changed in (False, True):
        if changed:
            p[2, 3] += 0.2
        prep = _prepare(p, spec, recipe, tab)
        local = torch.empty(len(p), device=p.device, dtype=torch.int32)
        keys = torch.empty(2 * len(p), device=p.device, dtype=torch.int32)
        lk.layout[(1,)](
            p,
            prep[0],
            prep[8],
            local,
            keys,
            len(p),
            spec.size,
            recipe.local_width_cutoff,
            128,
            num_warps=4,
        )
        mask = local.cpu().tolist()
        assert mask[0] == mask[1] == mask[5] == mask[6] == 1
        assert mask[3] == mask[4] == 0
        assert mask[2] == int(not changed)
        bounds = prep[8].cpu().tolist()
        actual = keys.cpu().reshape(-1, 2).tolist()
        for a in range(len(p)):
            lo, count = bounds[a][0]
            expected = (
                sorted({i // 32 for i in range(lo, lo + max(count, 0))})
                if mask[a]
                else []
            )
            assert [k for k in actual[a] if k < tr.cdiv(spec.size, 32)] == expected
