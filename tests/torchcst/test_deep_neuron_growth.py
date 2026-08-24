"""Neuron growth at a *hidden* boundary, which needs one gate owner.

The field a growth policy ranks on is ``dL/dgamma`` at ``gamma=0``.  A raw
``CSTLinear`` map is purely synaptic and never applies a gate itself; if a
caller gates the shared hidden store twice on the way from one map to the
next, a neuron enters the composed function as ``gamma^2`` and that field is
identically zero -- not small, zero -- and no dormant neuron can ever be woken.
:class:`~torchcst.compute.CSTBoundary` gives the boundary one owner and applies
the gate once, after the activation, which puts ``gamma`` back on the same
footing as a synapse amplitude.

These tests pin the difference directly: the same hidden store, the same data,
the same policy, growing or frozen depending only on who applies the gate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchcst.compute import CSTBoundary, CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import (
    ConstantQuota,
    EvenBudgetDistributor,
    PeriodicCadence,
    ProfitCourt,
    QuotaRegime,
    StructuralQuota,
    cSFW,
    cVP,
    gamma_ungate,
)
from torchcst.representation import GaussianFactor, RepresentationSpec
from torchcst.storage import NeuronStore, NeuronUngate, SynapseBirth, SynapseStore

D_IN, H_MAX, H_LIVE, D_OUT = 8, 16, 4, 4


def _synapses(site: str) -> SynapseStore:
    store = SynapseStore(
        site, 1, 1, 64,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    grid = torch.linspace(0.05, 0.95, 12, dtype=torch.float64)[:, None]
    store.apply(
        [
            SynapseBirth(
                site,
                grid.clone(),
                grid.flip(0).clone(),
                torch.full((12,), 0.02, dtype=torch.float64),
                torch.arange(12),
            )
        ]
    )
    return store


def _chart(site: str, n_max: int, live: int) -> NeuronStore:
    mu = torch.linspace(0.0, 1.0, n_max, dtype=torch.float64)[:, None]
    return NeuronStore(site, n_max, mu=mu, initial_live=live, dtype=torch.float64)


LADDER = (1.0, 0.5, 0.25, 0.1, 0.01)


def _world(*, blocked: bool, damping: tuple[float, ...] = LADDER):
    """Two CST layers over one hidden store, gated once or twice.

    Both endpoints of the stack ("h" and "y") are ``out_neurons`` of a synapse
    child, so both need a :class:`CSTBoundary` owning their output boundary --
    that is what makes the gate-tangent instrument buildable at all, in either
    world.
    """
    first, second = _synapses("layer1"), _synapses("layer2")
    inputs = _chart("x", D_IN, D_IN)
    hidden = _chart("h", H_MAX, H_LIVE)
    outputs = _chart("y", D_OUT, D_OUT)
    factor = GaussianFactor(0.1).double()
    one = CSTLinear(inputs, hidden, first, factor)
    two = CSTLinear(hidden, outputs, second, factor)
    producer = CSTBoundary(one, activation=F.gelu)
    terminal = CSTBoundary(two)
    if blocked:
        forward = lambda x: terminal(two(producer(one(x))))  # noqa: E731
    else:
        # What a user could still wrongly write now that the flags are gone:
        # take the once-gated hidden activation and have the consumer
        # multiply the hidden gate again -- gating the shared store twice.
        def forward(x):
            h = producer(one(x))
            return terminal(two(h * hidden.gate_vector()))

    root = QuotaRegime(
        budget=2,
        method=cSFW(pool_size=256, multistart=2),
        overrides={"layer2": cVP()},
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        distributor=EvenBudgetDistributor(),
        quota=ConstantQuota(StructuralQuota(synapse_birth=2, neuron_birth=1)),
        interface=gamma_ungate(),
        profit=ProfitCourt(min_profit=0.0, damping=damping),
    )
    optimizer = torch.optim.Adam(
        [*first.parameters(), *second.parameters()], lr=0.01
    )
    engine = StructuralEngine(
        {
            "layer1": first,
            "layer2": second,
            "x": inputs,
            "h": hidden,
            "y": outputs,
        },
        root,
        modules={"layer1": one, "layer2": two},
        optimizer=optimizer,
        seed=11,
    )
    return engine, forward, hidden, optimizer, (one, two)


def _problem() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(128, D_IN, dtype=torch.float64, generator=generator)
    dense = torch.randn(D_IN, D_OUT, dtype=torch.float64, generator=generator)
    return x, torch.tanh(x @ dense)


def _train(engine, forward, optimizer, modules, x, y, events: int) -> list[tuple]:
    def objective() -> float:
        with torch.no_grad():
            return float((forward(x) - y).square().mean())

    log: list[tuple] = []
    for _ in range(events):
        for module in modules:
            module.zero_grad(set_to_none=True)
        engine.begin_update()
        (forward(x) - y).square().mean().backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        optimizer.step()
        log.append((engine.step(objective), objective()))
    return log


def test_a_hidden_neuron_grows_only_when_one_owner_applies_its_gate() -> None:
    x, y = _problem()

    engine, forward, hidden, optimizer, modules = _world(blocked=True)
    blocked = _train(engine, forward, optimizer, modules, x, y, 12)
    grown = int(hidden.live_ids().numel())
    woke = sum(
        isinstance(op, NeuronUngate) for ops, _ in blocked for op in ops
    )

    engine, forward, twice_gated, optimizer, modules = _world(blocked=False)
    doubled = _train(engine, forward, optimizer, modules, x, y, 12)
    frozen = int(twice_gated.live_ids().numel())

    assert woke > 0
    assert grown > H_LIVE
    # Same store, same data, same policy: gated twice, the field is identically
    # zero and not one neuron can ever be woken.
    assert frozen == H_LIVE
    assert not any(
        isinstance(op, NeuronUngate) for ops, _ in doubled for op in ops
    )
    assert blocked[-1][1] < blocked[0][1]


def test_an_unbounded_solve_can_still_land_without_a_ladder() -> None:
    # The gate solve is deliberately uncapped (GammaUngate's docstring): a
    # weakly answered row can ask for hundreds of times the live scale, and
    # whether that is safe is left entirely to the root's realized-loss
    # trial rather than any upfront cap. This fixture's synapse grid --
    # twelve atoms spaced closer than the factor bandwidth -- used to make
    # layer1/layer2's periodic tangent backfit solve a near-singular Gram
    # into a huge cancelling amplitude pair (twin-control.md Sec.1); that
    # corrupted state was what made the *undamped* (``damping=()``) gate
    # trial fail here, so only the ladder's smaller rungs ever landed.
    # ``TangentRefit`` now clamps every applied amplitude to the pre-event
    # live scale (matching ``TangentBirth``'s own convention), so that
    # corruption no longer happens: the gate solve is still genuinely
    # uncapped and still lands at hundreds of times the live gate scale, but
    # the realized-loss trial now accepts it at full size, with or without a
    # ladder to fall back on.
    x, y = _problem()

    magnitudes: list[float] = []
    for damping in ((), LADDER):
        engine, forward, hidden, optimizer, modules = _world(
            blocked=True, damping=damping
        )
        log = _train(engine, forward, optimizer, modules, x, y, 12)
        ungates = [op for ops, _ in log for op in ops if isinstance(op, NeuronUngate)]
        assert ungates, f"expected at least one ungate with damping={damping!r}"
        assert int(hidden.live_ids().numel()) > H_LIVE
        magnitudes.extend(float(g) for op in ungates for g in op.gate.flatten())

    # The solve itself remains deliberately unbounded: some accepted gate is
    # large relative to the initial live gate scale of 1.0.
    assert max(abs(m) for m in magnitudes) > 10.0


def test_the_dormant_gate_field_is_nonzero_and_exact_in_a_stack() -> None:
    x, y = _problem()
    _, forward, hidden, _, modules = _world(blocked=True)
    producer = modules[0].out_boundary

    producer.reset_gate_capture()
    (forward(x) - y).square().sum().backward()
    field, _ = producer.gate_tangent(x, producer.take_gate_grad())

    dormant = hidden.dormant_ids()
    assert float(field.index_select(0, dormant).abs().max()) > 0.0

    # Finite differences through the whole two-layer stack.
    with torch.no_grad():
        base = float((forward(x) - y).square().sum())
    live_state = hidden.state[hidden.live_ids()[0]].clone()
    for chart_id in dormant.tolist()[:3]:
        was = hidden.state[chart_id].clone()
        hidden.state[chart_id] = live_state
        with torch.no_grad():
            hidden.gate[chart_id] = 1.0e-6
            numeric = (float((forward(x) - y).square().sum()) - base) / 1.0e-6
            hidden.gate[chart_id] = 0.0
        hidden.state[chart_id] = was
        torch.testing.assert_close(
            float(field[chart_id]), numeric, rtol=1.0e-4, atol=1.0e-4
        )
