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
@pytest.mark.parametrize("ni,no,offset", [(65, 33, 17), (8192, 8192, 33)])
def test_compact_support_roundtrip_is_bitwise_lossless(ni, no, offset):
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

    a, total = 257, 330
    p = torch.tensor([[0.3, 1.5, 2.4, 24]]).repeat(a, 1).cuda()
    p[:, 2] = torch.linspace(-4, no + 4, a, device="cuda")
    p[:, 3] = torch.linspace(-4, ni + 4, a, device="cuda")
    p[0, :2] = 0
    p[0, 0] = -0.0
    p[1, 2], p[1, 3] = no - 0.25, ni - 0.25
    layer = cases.scenarios.model(p, "global", n=ni, out=no, device="cuda")
    packed = p.new_empty((13, a))
    decoded = torch.empty_like(packed)
    factors = p.new_full((5, total), float("nan"))
    ends = torch.full((4, total), -32768, dtype=torch.int16, device="cuda")
    flags = torch.full((total,), 255, dtype=torch.uint8, device="cuda")
    recipe = CachedMatrixProductRecipe()
    scalars = polar_scalars(layer.kernel)
    _prepare(
        p,
        packed,
        scalars,
        layer.kernel.spec.normalization.floor,
        (7, ni, no, 0, 1, 0, 0),
        recipe,
    )
    encode[(tr.cdiv(a, 256),)](
        packed, factors, ends, flags, a, total, offset, 256, enable_fp_fusion=False
    )
    for untouched in (slice(None, offset), slice(offset + a, None)):
        assert torch.isnan(factors[:, untouched]).all()
        assert (ends[:, untouched] == -32768).all()
        assert (flags[untouched] == 255).all()
    torch.testing.assert_close(
        factors[:, offset : offset + a].view(torch.int32),
        packed[[1, 4, 5, 6, 7]].view(torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        ends[:, offset : offset + a], packed[9:13].to(torch.int16), rtol=0, atol=0
    )
    assert int(ends[1, offset + 1]) == ni
    assert int(ends[3, offset + 1]) == no
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
    torch.testing.assert_close(
        decoded.view(torch.int32), packed.view(torch.int32), rtol=0, atol=0
    )


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
