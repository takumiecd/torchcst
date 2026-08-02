"""Who may move a neuron's gate, and what happens when nobody does.

A gate is an ordinary learnable parameter that the policy tree opens and
ordinary training then shapes. Both halves are load-bearing and each is easy
to lose: a dormant row that received gradient would let SGD wake neurons
behind the policy's back, and a woken row that nobody trains stays at whatever
one solve chose -- live in the chart, absent from the function.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchcst.compute import CSTBoundary, CSTLinear
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

D_IN, H_MAX, H_LIVE, D_OUT = 6, 10, 3, 4


def _world():
    store = SynapseStore(
        "layer", 1, 1, 32,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    grid = torch.linspace(0.1, 0.9, 8, dtype=torch.float64)[:, None]
    store.apply(
        [
            SynapseBirth(
                "layer", grid.clone(), grid.flip(0).clone(),
                torch.full((8,), 0.1, dtype=torch.float64), torch.arange(8),
            )
        ]
    )
    mu = lambda n: torch.linspace(0.0, 1.0, n, dtype=torch.float64)[:, None]  # noqa: E731
    inputs = NeuronStore("x", D_IN, mu=mu(D_IN), initial_live=D_IN, dtype=torch.float64)
    hidden = NeuronStore("h", H_MAX, mu=mu(H_MAX), initial_live=H_LIVE,
                         dtype=torch.float64)
    linear = CSTLinear(inputs, hidden, store, GaussianKernel(0.2).double())
    boundary = CSTBoundary(linear, activation=F.gelu)
    forward = lambda inp: boundary(linear(inp))  # noqa: E731
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(32, D_IN, dtype=torch.float64, generator=generator)
    target = torch.randn(32, H_MAX, dtype=torch.float64, generator=generator)
    return store, hidden, linear, boundary, forward, x, target


def test_a_gate_is_learnable_but_dormant_rows_stay_the_policys_alone() -> None:
    _, hidden, _, _, forward, x, target = _world()
    assert isinstance(hidden.gate, torch.nn.Parameter)

    (forward(x) - target).square().sum().backward()

    assert hidden.gate.grad is not None
    live = hidden.live_ids()
    dormant = hidden.dormant_ids()
    assert float(hidden.gate.grad.index_select(0, live).abs().min()) > 0.0
    # Ordinary training must never be able to wake a neuron behind the policy
    # tree's back: gate_vector() masks dormant rows, so their gradient is zero.
    assert float(hidden.gate.grad.index_select(0, dormant).abs().max()) == 0.0


def test_an_untrained_gate_never_moves_from_what_one_solve_chose() -> None:
    store, hidden, linear, boundary, forward, x, target = _world()
    # A neuron the policy woke at a small solved value, with only the synapse
    # side handed to the optimizer -- the omission this repo's own experiments
    # made first.
    woken = int(hidden.dormant_ids()[0])
    with torch.no_grad():
        hidden.gate[woken] = 0.02
    hidden.state[woken] = hidden.state[hidden.live_ids()[0]].clone()
    optimizer = torch.optim.Adam(store.parameters(), lr=1.0e-2)

    for _ in range(50):
        optimizer.zero_grad()
        linear.zero_grad(set_to_none=True)
        boundary.zero_grad(set_to_none=True)
        (forward(x) - target).square().mean().backward()
        optimizer.step()

    assert float(hidden.gate[woken]) == 0.02


def test_training_the_gate_moves_it_and_widens_the_effective_chart() -> None:
    store, hidden, linear, boundary, forward, x, target = _world()
    woken = int(hidden.dormant_ids()[0])
    with torch.no_grad():
        hidden.gate[woken] = 0.02
    hidden.state[woken] = hidden.state[hidden.live_ids()[0]].clone()

    def effective() -> int:
        live = hidden.live_ids()
        gates = hidden.gate.detach().index_select(0, live).abs()
        return int((gates > 0.1).sum())

    before = effective()
    optimizer = torch.optim.Adam(store.parameters(), lr=1.0e-2)
    # A gate is a low-curvature direction; the rate that suits synapse
    # coordinates drives it to run away, so it gets its own smaller one.
    gate_optimizer = torch.optim.Adam([hidden.gate], lr=1.0e-3)

    for _ in range(600):
        optimizer.zero_grad()
        gate_optimizer.zero_grad()
        linear.zero_grad(set_to_none=True)
        boundary.zero_grad(set_to_none=True)
        (forward(x) - target).square().mean().backward()
        optimizer.step()
        gate_optimizer.step()

    assert float(hidden.gate[woken]) != 0.02
    assert effective() > before
