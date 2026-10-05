"""Update fusion must preserve production geometry and captured training."""

import pytest
import torch

from benchmarks.cuda.linear.fixtures import local_product_state
from torchcst._backends.cuda.algorithms.polar_update.executor import fused_update_
from torchcst._backends.torch.algorithms.polar_update.executor import graph_update


def test_fused_update_rejects_cpu_and_aliases():
    s = local_product_state()
    p = torch.ones(7, 4)
    with pytest.raises(ValueError):
        fused_update_(s, p, p, step_size=0.001)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["finite_chord", "time_energy"])
@pytest.mark.parametrize("step_size", [0.0001, 0.01, 0.7])
def test_fused_update_matches_production_and_live_scalars(mode, step_size):
    from torchcst._backends.cuda.algorithms.linear.local_product.contract import Domain
    from torchcst._backends.torch.kernels import execution
    from torchcst.kernels import BandwidthBounds, TriweightSpec, presets
    from torchcst.kernels.state import KernelState

    s = KernelState(
        presets.polar_activity(
            amplitude_max=2.0,
            input_bounds=BandwidthBounds(
                minimum=0.25, birth=1, maximum=16, upper_floor=1
            ),
            w_c=0.2,
            profile=presets.profile(TriweightSpec()),
            activity_mode=mode,
            activity_gain=0.8,
            radial_regularization=0.3,
            dormant_expansion_rate=0.05,
        )
    ).cuda()
    gen = torch.Generator().manual_seed(871)
    old = (torch.randn(259, 4, generator=gen) * 2).cuda()
    old[:6, :2] = old.new_tensor(
        [[0, 0], [0, 1e-30], [0.3, 0.4], [0, 1], [0, 2], [3, 4]]
    )
    old[:, 2:] *= 64
    proposal = old + (torch.randn(259, 4, generator=gen) * 0.2).cuda()
    initial = proposal.clone()
    charts = Domain(32, 32).charts(device="cuda")
    expected = execution.apply_parameter_update(
        s, *charts, old, initial - old, step_size=step_size
    )
    v = proposal._version
    fused_update_(s, old, proposal, step_size=step_size)
    assert proposal._version == v + 1
    torch.testing.assert_close(proposal, expected, atol=2e-6, rtol=2e-6)
    # Graph replay must read updated scalar values, rather than constants captured at compile.
    proposal.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_update_(s, old, proposal, step_size=step_size)
    s.scalar("radial_regularization").fill_(0.6)
    s.scalar("w_c").fill_(0.4)
    proposal.copy_(initial)
    graph.replay()
    torch.cuda.synchronize()
    expected = graph_update(s, old, initial - old, step_size=step_size)
    torch.testing.assert_close(proposal, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size", [32, 64, 128])
def test_fused_update_captured_training(size):
    from test_local_product_research import (
        test_graph_training_updates_width_and_matches_public_optimizer as check,
    )

    check(
        "hybrid-persistent-mid4", size_override=size, polar_update="fused", updates=20
    )
