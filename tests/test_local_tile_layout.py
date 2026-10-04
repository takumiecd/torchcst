"""Physical tile layouts retain canonical atom IDs and current support."""

import pytest
import torch

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from torchcst._backends.cuda.algorithms.local_product.contract import Domain
from torchcst._backends.cuda.algorithms.local_product.recipe import Recipe


@pytest.mark.parametrize("stage", ["early", "middle", "late", "narrow-only"])
def test_packed_plan_snapshot(stage):
    run = load_run(
        f"benchmarks/cuda/linear/cases/local-128-packed-{stage}.json",
        "benchmarks/cuda/linear/plans-local-packed.json",
    )
    assert decode_snapshot(run.snapshot()) == run
    assert run.entry("hybrid-packed-mid4").plan.recipe.route == "hybrid_packed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_two_physical_layouts_and_inactive_tail():
    from torchcst._backends.cuda.algorithms.local_product.executor import tile_layout

    # Canonical fields are deliberately unsorted. The floor-active one-site
    # atom (ID6) belongs to a general band, not the certified singleton prefix.
    p = torch.zeros(13, 7, device="cuda")
    p[0] = torch.arange(7, device="cuda")
    p[1] = p.new_tensor([16, 16, 1 / 0.75**2, 0.25, 1 / 64, 16, 16])
    p[8] = p.new_tensor([3, 3, 0, 0, 0, 3, 0])
    p[9:13] = p.new_tensor(
        [
            [2, 19, 0, 1, 0, 32, 0],
            [3, 20, 2, 6, 32, 32, 1],
            [17, 1, 14, 8, 0, 8, 17],
            [18, 2, 18, 13, 32, 9, 18],
        ]
    )
    original = p.clone()
    views, orders, offsets = tile_layout(
        p, Domain(32, 32), Recipe(pack=False, rho_upper=(1, 4, 16))
    )
    for axis, expected in enumerate(([1, 0, 2, 6, 3, 4, 5], [0, 1, 2, 6, 3, 4, 5])):
        assert orders[axis].tolist() == expected
        assert offsets[axis].tolist() == [0, 1, 2, 4, 5, 6, 7]
        torch.testing.assert_close(views[axis], p[:, expected], rtol=0, atol=0)
    torch.testing.assert_close(p, original, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_empty_packed_operator_and_gradients():
    from benchmarks.cuda.linear.fixtures import local_product_state
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    x = torch.randn(3, 32, device="cuda", requires_grad=True)
    p = torch.empty(0, 4, device="cuda", requires_grad=True)
    y = local_h(
        x,
        p,
        local_product_state().cuda(),
        Domain(32, 32),
        hybrid=True,
        sparse=True,
        fused_polar=True,
        three_band=True,
        singletons=True,
        tile_packed=True,
        recipe=Recipe(pack=False, rho_upper=(1, 4, 16)),
    )
    dx, dp = torch.autograd.grad(y.sum(), (x, p))
    assert torch.count_nonzero(y) == 0
    assert torch.count_nonzero(dx) == 0
    assert dp.shape == p.shape
