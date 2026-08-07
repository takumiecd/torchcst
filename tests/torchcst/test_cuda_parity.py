"""CUDA parity: the same measure on GPU must compute what it computes on CPU.

The stores take a ``device`` at construction and register every coordinate,
gate, and weight as a buffer or parameter, so the whole library is *designed*
to run on an accelerator -- but nothing here ever asserted that it does.  These
tests pin the contract at three depths:

* ``forward``  -- the compute path (gather, kernel, mass) agrees across devices;
* ``backward`` -- the gradients the structural instruments score agree too;
* one full engine update -- observation, scoring, and the applied structural
  operations agree, which is the part that reads gradients back out of a capture
  and could silently see zeros on a device it was never run on;
* cSFW's position backfit, whose family-private damped-Newton solve assembles a
  small float64 system on the *host* while the atoms it moves live on the
  accelerator -- see ``test_csfw_position_backfit_runs_on_cuda``;
* ``AbsorbCourt``, which maps ``GramService`` live-view *positions* (atoms'
  device) through the store's host-side ``ids`` tensor -- see
  ``test_absorb_court_runs_on_cuda``.

The last two are the same defect twice: a host-side index meeting a
device-side tensor in a code path no test had ever run off the CPU.

Skipped when no CUDA device is present, so the suite stays green on a laptop.
"""

from __future__ import annotations

import pytest
import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.instruments import ContinuousGradientRequest
from torchcst.policy import (
    AbsorbCourt,
    ConstantQuota,
    EvenBudgetDistributor,
    MagnitudeCourt,
    PeriodicCadence,
    QuotaRegime,
    ScoredBirth,
    StructuralQuota,
    SynapseLifecycle,
    cSFW,
)
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import (
    NeuronStore,
    SynapseAbsorb,
    SynapseBirth,
    SynapseRefit,
    SynapseStore,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

N_IN, N_OUT, CAPACITY = 4, 3, 8


def build(device: str):
    """The gradient-birth example, pinned to one device."""
    inputs = NeuronStore(
        "inputs",
        N_IN,
        mu=torch.linspace(0.0, 1.0, N_IN)[:, None],
        initial_live=N_IN,
        device=device,
    )
    outputs = NeuronStore(
        "outputs",
        N_OUT,
        mu=torch.linspace(0.0, 1.0, N_OUT)[:, None],
        initial_live=N_OUT,
        device=device,
    )
    synapses = SynapseStore(
        "layer",
        d_in=1,
        d_out=1,
        capacity=CAPACITY,
        spec=RepresentationSpec.continuous(1, 1),
        device=device,
    )
    synapses.apply(
        [
            SynapseBirth(
                "layer",
                torch.tensor([[0.5]]),
                torch.tensor([[0.5]]),
                torch.tensor([0.1]),
                torch.tensor([0], dtype=torch.int64),
            )
        ]
    )
    layer = CSTLinear(inputs, outputs, synapses, GaussianKernel(0.2))
    scorer = ContinuousGradientRequest(
        pool_size=32, decay=0.0, chunk_size=8, timing="backward_inline"
    )
    method = SynapseLifecycle(
        birth_factory=lambda lam: ScoredBirth(scorer, initial_weight=0.0),
        prune_factory=lambda: MagnitudeCourt(drop_fraction=0.0),
        priceable=False,
        label="cuda-parity",
    )
    policy = QuotaRegime(
        budget=1,
        method=method,
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        distributor=EvenBudgetDistributor(),
    )
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    engine = StructuralEngine(
        {"layer": synapses, "inputs": inputs, "outputs": outputs},
        policy,
        modules={"layer": layer},
        optimizer=optimizer,
        seed=7,
    )
    return layer, synapses, optimizer, engine


def batch(device: str):
    """One fixed batch, generated on CPU so both devices see the same numbers."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(8, N_IN, generator=generator)
    target = torch.randn(8, N_OUT, generator=generator)
    return x.to(device), target.to(device)


@cuda_only
def test_forward_matches_cpu() -> None:
    outputs = {}
    for device in ("cpu", "cuda"):
        layer, _, _, _ = build(device)
        x, _ = batch(device)
        outputs[device] = layer(x).detach().cpu()
    torch.testing.assert_close(outputs["cuda"], outputs["cpu"], rtol=1e-5, atol=1e-6)


@cuda_only
def test_backward_matches_cpu() -> None:
    grads = {}
    for device in ("cpu", "cuda"):
        layer, synapses, _, _ = build(device)
        x, target = batch(device)
        (layer(x) - target).square().mean().backward()
        grads[device] = {
            name: parameter.grad.detach().cpu()
            for name, parameter in synapses.named_parameters()
            if parameter.grad is not None
        }
    assert grads["cuda"].keys() == grads["cpu"].keys()
    assert grads["cpu"], "the CPU run produced no synapse gradients to compare"
    for name in grads["cpu"]:
        torch.testing.assert_close(
            grads["cuda"][name], grads["cpu"][name], rtol=1e-5, atol=1e-6,
            msg=lambda formatted, name=name: f"gradient mismatch on {name}: {formatted}",
        )


@cuda_only
def test_one_structural_update_matches_cpu() -> None:
    results = {}
    for device in ("cpu", "cuda"):
        layer, synapses, optimizer, engine = build(device)
        x, target = batch(device)
        engine.begin_update()
        optimizer.zero_grad()
        (layer(x) - target).square().mean().backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        optimizer.step()
        operations = engine.step()
        results[device] = {
            "kinds": [type(op).__name__ for op in operations],
            "live": int(synapses.live_ids().numel()),
            "weights": synapses.w.detach().cpu(),
        }
    assert results["cuda"]["kinds"] == results["cpu"]["kinds"]
    assert results["cuda"]["live"] == results["cpu"]["live"]
    torch.testing.assert_close(
        results["cuda"]["weights"], results["cpu"]["weights"], rtol=1e-5, atol=1e-6
    )


def _csfw_backfit_parts(device: str):
    """A cSFW site whose refit rule actually moves coordinates.

    ``backfit="event"`` with ``backfit_position_iters=1`` is the one
    configuration that reaches ``_polish``; the shipped ``FastConstruction``
    recipe pins ``backfit_position_iters=0``, which is why no existing test
    exercised this path on any device.
    """
    inputs = NeuronStore(
        "inputs", N_IN, mu=torch.linspace(0.0, 1.0, N_IN)[:, None],
        initial_live=N_IN, device=device,
    )
    outputs = NeuronStore(
        "outputs", N_OUT, mu=torch.linspace(0.0, 1.0, N_OUT)[:, None],
        initial_live=N_OUT, device=device,
    )
    synapses = SynapseStore(
        "layer", d_in=1, d_out=1, capacity=CAPACITY,
        spec=RepresentationSpec.continuous(1, 1), device=device,
    )
    synapses.apply(
        [
            SynapseBirth(
                "layer",
                torch.tensor([[0.3], [0.7]]),
                torch.tensor([[0.4], [0.6]]),
                torch.tensor([0.1, -0.2]),
                torch.tensor([0, 1], dtype=torch.int64),
            )
        ]
    )
    layer = CSTLinear(inputs, outputs, synapses, GaussianKernel(0.2))
    policy = QuotaRegime(
        budget=1,
        method=cSFW(
            polish_iters=0,
            backfit="event",
            backfit_position_iters=1,
            pool_size=64,
            multistart=2,
        ),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        distributor=EvenBudgetDistributor(),
    )
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    engine = StructuralEngine(
        {"layer": synapses, "inputs": inputs, "outputs": outputs},
        policy, modules={"layer": layer}, optimizer=optimizer, seed=7,
    )
    return layer, synapses, optimizer, engine


@cuda_only
def test_csfw_position_backfit_runs_on_cuda() -> None:
    """Regression: ``_polish``'s host-side 6x6 Newton step must come back to
    the atoms' device.  Before the fix this raised ``Expected all tensors to
    be on the same device`` on the first refit event of any CUDA run."""
    results = {}
    for device in ("cpu", "cuda"):
        layer, synapses, optimizer, engine = _csfw_backfit_parts(device)
        x, target = batch(device)
        applied = []
        for _ in range(3):
            engine.begin_update()
            optimizer.zero_grad()
            (layer(x) - target).square().mean().backward()
            engine.observe_microbatch()
            engine.finalize_backward()
            optimizer.step()
            applied.extend(engine.step())
        results[device] = {
            "refits": sum(isinstance(op, SynapseRefit) for op in applied),
            "kinds": [type(op).__name__ for op in applied],
            "live": int(synapses.live_ids().numel()),
        }
    assert results["cpu"]["refits"] > 0, "the CPU run never reached the polish path"
    assert results["cuda"]["kinds"] == results["cpu"]["kinds"]
    assert results["cuda"]["live"] == results["cpu"]["live"]


def _absorb_parts(device: str):
    """Three near-duplicate atoms: two of them are absorbable into the third."""
    store = SynapseStore(
        "edge", 1, 1, capacity=12,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)), device=device,
    )
    inputs = NeuronStore(
        "edge_in", 1, mu=torch.zeros(1, 1), initial_live=1, device=device,
    )
    outputs = NeuronStore(
        "edge_out", 2, mu=torch.tensor([[0.0], [1.0]]), initial_live=2, device=device,
    )
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.1))
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.50], [0.5001], [0.4999]]),
                torch.tensor([[0.50], [0.5001], [0.4999]]),
                torch.tensor([1.0, -0.7, 0.4]),
                torch.arange(3, dtype=torch.int64),
            )
        ]
    )
    method = SynapseLifecycle(
        birth_factory=lambda lam: None,
        absorb_factory=lambda lam: AbsorbCourt(rent=1.0, radius=0.01, ridge=1.0e-9),
        priceable=False,
        label="absorb-only",
    )
    root = QuotaRegime(
        budget=0,
        method=method,
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(synapse_absorb=1)),
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        root,
        modules={"edge": module},
    )
    return store, engine


def _path_parts(device: str):
    """One shared site, four CSTLinear execution paths over it."""
    dtype = torch.float32
    store = SynapseStore(
        "paths", 2, 2, capacity=8,
        spec=RepresentationSpec.continuous(2, 2, bounds=(-1.0, 1.0)),
        device=device, dtype=dtype,
    )
    generator = torch.Generator().manual_seed(11)
    store.apply(
        [
            SynapseBirth(
                "paths",
                torch.rand(8, 2, generator=generator, dtype=dtype) * 2 - 1,
                torch.rand(8, 2, generator=generator, dtype=dtype) * 2 - 1,
                torch.randn(8, generator=generator, dtype=dtype),
                torch.arange(8, dtype=torch.int64),
            )
        ]
    )
    inputs = NeuronStore(
        "paths_in", 6,
        mu=torch.rand(6, 2, generator=generator, dtype=dtype) * 2 - 1,
        initial_live=6, device=device, dtype=dtype,
    )
    outputs = NeuronStore(
        "paths_out", 5,
        mu=torch.rand(5, 2, generator=generator, dtype=dtype) * 2 - 1,
        initial_live=5, device=device, dtype=dtype,
    )
    kernel = GaussianKernel(0.35)
    common = (inputs, outputs, store, kernel)
    return store, kernel, {
        "factored": CSTLinear(*common, materialize=False),
        "materialized": CSTLinear(*common, materialize=True),
        "lean": CSTLinear(
            *common, track_mass=False, materialize=True,
            lean_materialize=True,
        ),
        # radius 6 sigma: the dropped tail (exp(-18)) sits far below the
        # float32 comparison tolerance, so all four paths must agree.
        "native": CSTLinear(*common, track_mass=False, support_radius=6.0),
    }


@cuda_only
def test_execution_paths_agree_on_cuda() -> None:
    generator = torch.Generator().manual_seed(5)
    x_cpu = torch.randn(7, 6, generator=generator)
    upstream_cpu = torch.randn(7, 5, generator=generator)
    results: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for device in ("cpu", "cuda"):
        store, kernel, modules = _path_parts(device)
        x = x_cpu.to(device)
        upstream = upstream_cpu.to(device)
        per_path = {}
        for name, module in modules.items():
            for parameter in (store.w, store.s, store.t, kernel.sigma):
                parameter.grad = None
            output = module(x)
            output.backward(upstream)
            per_path[name] = {
                "out": output.detach().cpu(),
                "w": store.w.grad.detach().cpu(),
                "s": store.s.grad.detach().cpu(),
                "t": store.t.grad.detach().cpu(),
            }
        results[device] = per_path
    for device in ("cpu", "cuda"):
        for name in ("materialized", "lean", "native"):
            for key in ("out", "w", "s", "t"):
                torch.testing.assert_close(
                    results[device][name][key],
                    results[device]["factored"][key],
                    rtol=1e-4, atol=1e-5,
                    msg=lambda formatted, name=name, key=key, device=device: (
                        f"{device}:{name}:{key} disagrees with factored: "
                        f"{formatted}"
                    ),
                )
    for key in ("out", "w", "s", "t"):
        torch.testing.assert_close(
            results["cuda"]["factored"][key],
            results["cpu"]["factored"][key],
            rtol=1e-4, atol=1e-5,
        )


@cuda_only
def test_absorb_court_runs_on_cuda() -> None:
    """Regression: ``AbsorbCourt`` maps ``GramService`` positions (atoms'
    device) through the store's host-side ``ids``.  Before the fix this raised
    ``Expected all tensors to be on the same device`` on the first absorb of
    any CUDA run -- the same defect as the cSFW polish one above."""
    results = {}
    for device in ("cpu", "cuda"):
        store, engine = _absorb_parts(device)
        applied = engine.step()
        results[device] = {
            "absorbs": sum(isinstance(op, SynapseAbsorb) for op in applied),
            "live": int(store.view().ids.numel()),
        }
    assert results["cpu"]["absorbs"] == 1, "the CPU run never absorbed"
    assert results["cuda"]["absorbs"] == results["cpu"]["absorbs"]
    assert results["cuda"]["live"] == results["cpu"]["live"]
