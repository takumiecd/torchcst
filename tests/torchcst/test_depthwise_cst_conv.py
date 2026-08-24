"""Contract tests for DepthwiseCSTConv2d — the measured winning conv form.

Each test pins one property the FC-7t validation relied on: the composition
semantics (depthwise then CSTLinear-as-1×1, exactly), lawful proposed charts
with the 2×-cells capacity rule, live seeded atoms inside the birth domains,
gradient flow to every learnable part, and the engine lifecycle on the CST
half.
"""

import pytest
import torch

from torchcst.compute import CSTLinear, DepthwiseCSTConv2d
from torchcst.engine import StructuralEngine
from torchcst.policy import (
    ConstantQuota,
    EvenBudgetDistributor,
    PeriodicCadence,
    QuotaRegime,
    StructuralQuota,
    cSET,
)
from torchcst.representation import survey_chart
from torchcst.storage import SynapseStore

SIGMA = 0.1


def _block(**kwargs) -> DepthwiseCSTConv2d:
    return DepthwiseCSTConv2d.propose(
        "mix", 16, 24, 3, SIGMA,
        generator=torch.Generator().manual_seed(0), **kwargs,
    )


def test_forward_is_depthwise_then_cst_rows():
    block = _block()
    x = torch.randn(2, 16, 8, 8, generator=torch.Generator().manual_seed(1))
    out = block(x)
    assert out.shape == (2, 24, 8, 8)

    h = block.depthwise(x)
    rows = h.permute(0, 2, 3, 1).reshape(-1, 16)
    expected = block.mix(rows).reshape(2, 8, 8, 24).permute(0, 3, 1, 2)
    assert torch.allclose(out, expected, atol=1e-6)


def test_propose_builds_lawful_charts_with_capacity_rule():
    block = _block()
    assert survey_chart(block.in_neurons.mu, SIGMA).notes == ()
    assert survey_chart(block.out_neurons.mu, SIGMA).notes == ()
    base = SynapseStore.between(
        "ref", block.in_neurons, block.out_neurons, SIGMA
    )
    assert block.synapses.capacity == 2 * base.capacity
    # Seeded full and live, inside the birth domains.
    assert block.synapses.k_live == block.synapses.capacity
    view = block.synapses.view()
    lo, hi = block.synapses.spec.domain_in.bounds
    assert float(view.s.min()) >= min(lo) and float(view.s.max()) <= max(hi)


def test_propose_dials():
    lean = _block(capacity_scale=0.5)
    fat = _block(capacity_scale=2.0)
    assert lean.synapses.capacity < fat.synapses.capacity
    empty = _block(seed_atoms=False)
    assert empty.synapses.k_live == 0
    with pytest.raises(ValueError):
        _block(factor="delta")
    with pytest.raises(ValueError):
        _block(capacity_scale=0.0)


def test_gradients_reach_every_learnable_part():
    block = _block()
    x = torch.randn(2, 16, 6, 6, generator=torch.Generator().manual_seed(2))
    block(x).square().mean().backward()
    assert block.depthwise.weight.grad is not None
    store = block.synapses
    assert store.w.grad is not None and store.w.grad.abs().sum() > 0
    assert store.s.grad is not None
    assert store.t.grad is not None


def test_engine_lifecycle_on_the_mix_half():
    block = DepthwiseCSTConv2d.propose(
        "mix", 16, 24, 3, SIGMA, seed_atoms=False,
        generator=torch.Generator().manual_seed(0),
    )
    root = QuotaRegime(
        budget=8,
        method=cSET(initial_weight=1e-2),
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(synapse_birth=8)),
        distributor=EvenBudgetDistributor(),
    )
    optimizer = torch.optim.Adam(block.parameters(), lr=1e-3)
    engine = StructuralEngine(
        block.stores(), root, modules={block.capture_site: block.mix},
        optimizer=optimizer, seed=0,
    )
    rng = torch.Generator().manual_seed(3)
    for _ in range(3):
        x = torch.randn(2, 16, 6, 6, generator=rng)
        engine.begin_update()
        optimizer.zero_grad()
        block(x).square().mean().backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        optimizer.step()
        engine.step()
    assert block.synapses.k_live > 0
    assert isinstance(block.mix, CSTLinear)
