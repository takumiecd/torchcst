"""The same independent oracle/snapshot contracts with compact support storage."""

import pytest
import test_bounded_product_cuda as cases
import torch

from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.cached_recipe import (
    CachedMatrixProductRecipe,
)
from torchcst._backends.schema import ExecutionPlan


def plan(**kwargs):
    return ExecutionPlan(
        "research_profile_product_bounded_matrix",
        "v2",
        CachedMatrixProductRecipe(**kwargs),
    )


@pytest.fixture(autouse=True)
def cached(monkeypatch):
    monkeypatch.setattr(cases, "plan", plan)


@pytest.fixture
def bounded(monkeypatch):
    monkeypatch.setattr(cases.scenarios, "run", cases.run)


@cases.scenarios.GPU
def test_compact_support_roundtrip_is_bitwise_lossless():
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.local_product.preparation import (
        polar_scalars,
    )
    from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.bounded_executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.cache_kernels import (
        decode,
        encode,
    )

    a, offset, total = 257, 17, 300
    p = torch.tensor([[0.3, 1.5, 2.4, 24]]).repeat(a, 1).cuda()
    layer = cases.scenarios.model(p, "global", device="cuda")
    packed = p.new_empty((13, a))
    decoded = torch.empty_like(packed)
    factors = p.new_empty((5, total))
    ends = torch.empty((4, total), dtype=torch.int16, device="cuda")
    flags = torch.empty(total, dtype=torch.uint8, device="cuda")
    recipe = CachedMatrixProductRecipe()
    scalars = polar_scalars(layer.kernel)
    _prepare(
        p,
        packed,
        scalars,
        layer.kernel.spec.normalization.floor,
        (7, 65, 33, 0, 1, 0, 0),
        recipe,
    )
    encode[(tr.cdiv(a, 256),)](
        packed, factors, ends, flags, a, total, offset, 256, enable_fp_fusion=False
    )
    decode[(tr.cdiv(a, 256),)](
        factors,
        ends,
        flags,
        p,
        scalars[0],
        decoded,
        a,
        total,
        offset,
        256,
        enable_fp_fusion=False,
    )
    torch.testing.assert_close(decoded, packed, rtol=0, atol=0)


class TestCached:
    test_boundaries = staticmethod(cases.test_boundaries_and_recipe_roundtrip)
    test_snapshots = staticmethod(cases.test_comparison_snapshots)
    test_oracle = staticmethod(cases.test_every_chunk_and_tail_matches_full_fp64_oracle)
    test_floor = staticmethod(cases.test_full_norm_floor_and_precision_fallback)
    test_gradients = staticmethod(cases.test_empty_atoms_and_gradient_branches)
    test_graph = staticmethod(cases.test_twenty_graph_updates_and_moments)
    test_old_forward = staticmethod(
        cases.test_old_forward_retains_all_decoder_scalars_and_lattice
    )
