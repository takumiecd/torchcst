"""Bootstrap accounting and a non-cold Adam reference for the long-run study."""

import pytest
import torch

from benchmarks.training.full_step_graph_learning import loop_ranges
from benchmarks.training.owner_window_fp32 import adam_direction


def test_exactly_45000_real_updates_with_eight_counted_fp32_bootstrap_steps():
    indices = [
        (epoch, step)
        for epoch in range(3)
        for lo, hi in loop_ranges(epoch, 15000)
        for step in range(lo, hi)
    ]
    assert len(indices) == len(set(indices)) == 45000
    assert indices[:8] == [(0, i) for i in range(8)]
    assert indices[8] == (0, 8)
    assert indices[-1] == (2, 14999)
    with pytest.raises(ValueError):
        loop_ranges(0, 4)


def test_learned_moment_direction_agrees_with_next_adamw_proposal():
    p = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64))
    g = torch.tensor([1e-8, -2e-7, 3e-6], dtype=torch.float64)
    m = torch.tensor([0.02, -0.003, 0.0004], dtype=torch.float64)
    v = torch.tensor([0.3, 0.02, 0.001], dtype=torch.float64)
    opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0, foreach=False)
    opt.state[p] = {
        "step": torch.tensor(7.0),
        "exp_avg": m.clone(),
        "exp_avg_sq": v.clone(),
    }
    before = p.detach().clone()
    p.grad = g
    opt.step()
    expected = adam_direction(g, m, v, step=7)
    torch.testing.assert_close(
        (before - p.detach()) / 1e-3, expected, atol=1e-12, rtol=1e-11
    )
