"""Polar is an independent operation; updates preserve the existing coordinate law."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from kernel_cases import polar_state

from torchcst import (
    AtomUpdateBinding,
    AtomUpdateInputs,
    BandwidthBounds,
    CSTOptimizer,
    Dispatcher,
    LinearInputs,
    OrderedSelector,
    make_algorithm_registry,
)
from torchcst._backends.cuda.algorithms.polar_update.plans import FUSED
from torchcst._backends.torch.algorithms.polar_update.plans import TORCH
from torchcst._backends.torch.kernels import execution


def layer(*, sphere=False, mode="finite_chord"):
    from test_cst_optimizer import site

    return site(
        sphere=sphere,
        kernel=polar_state(
            amplitude_max=2.0,
            w_c=0.2,
            input_bounds=BandwidthBounds(
                minimum=0.1, birth=0.5, maximum=1.0, upper_floor=0.1
            ),
            activity_mode=mode,
        ),
    )


@pytest.mark.parametrize("sphere", [False, True])
@pytest.mark.parametrize("mode", ["finite_chord", "time_energy"])
def test_update_matches_coordinate_reference_without_linear_inputs(sphere, mode):
    model = layer(sphere=sphere, mode=mode)
    binding = AtomUpdateBinding(model.operator)
    p = model.atoms.p
    with torch.no_grad():
        p[:, :2] = p.new_tensor([[0, 0], [0.3, 0.4], [3, 4]])
    previous = p.detach().clone()
    delta = torch.randn_like(previous) * 0.02
    with torch.no_grad():
        p.add_(delta)
    expected = execution.apply_parameter_update(
        model.kernel,
        *model.cst_charts(),
        previous,
        p.detach() - previous,
        step_size=0.01,
    )
    result = Dispatcher(registry=make_algorithm_registry()).run(
        binding, AtomUpdateInputs(previous, 0.01), plan=TORCH
    )
    assert result.data_ptr() == p.data_ptr()
    assert not result.requires_grad
    torch.testing.assert_close(p, expected, atol=1e-12, rtol=1e-12)
    assert binding.atom_state is model.atom_state
    assert model.atoms.p is p


@pytest.mark.parametrize(
    "fault", ["shape", "dtype", "alias", "alias_view", "grad", "zero", "nan", "bool"]
)
def test_invalid_snapshot_or_step_is_rejected_before_parameter_mutation(fault):
    model = layer()
    p = model.atoms.p
    previous, rate = p.detach().clone(), 0.01
    if fault == "shape":
        previous = previous[:1]
    elif fault == "dtype":
        previous = previous.float()
    elif fault == "alias":
        previous = p.detach()
    elif fault == "alias_view":
        previous = p.detach().view_as(p)
    elif fault == "grad":
        previous.requires_grad_()
    elif fault == "zero":
        rate = 0
    elif fault == "nan":
        rate = float("nan")
    else:
        rate = True
    initial = p.detach().clone()
    with pytest.raises(ValueError):
        Dispatcher(registry=make_algorithm_registry()).run(
            AtomUpdateBinding(model.operator),
            AtomUpdateInputs(previous, rate),
            plan=TORCH,
        )
    torch.testing.assert_close(p, initial, atol=0, rtol=0)


def test_polar_fallback_and_linear_plan_are_separate():
    from torchcst import DefaultRecipe, ExecutionPlan

    model = layer()
    binding = AtomUpdateBinding(model.operator)
    registry = make_algorithm_registry()
    inputs = AtomUpdateInputs(model.atoms.p.detach().clone(), 0.01)
    selector = OrderedSelector((FUSED,), fallback_plan=TORCH, registry=registry)
    assert selector.select(binding.build_context(inputs)).plan == TORCH
    dispatcher = Dispatcher(selector=selector)
    with pytest.raises(ValueError, match="CUDA"):
        dispatcher.run(binding, inputs, plan=FUSED)
    with pytest.raises(ValueError, match="contract"):
        dispatcher.run(
            binding,
            inputs,
            plan=ExecutionPlan("torch_materialized", "v1", DefaultRecipe()),
        )
    with pytest.raises(TypeError, match="inputs type"):
        dispatcher.run(binding, LinearInputs(torch.zeros(2, model.in_features)))
    dispatcher.run(binding, inputs)


def test_optimizer_rejects_incompatible_update_plan_before_base_step():
    from torchcst import FixedSelector

    model = layer()
    base = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer = CSTOptimizer(
        base,
        model=model,
        update_selector=FixedSelector(FUSED, registry=make_algorithm_registry()),
    )
    model.atoms.p.grad = torch.randn_like(model.atoms.p)
    initial = model.atoms.p.detach().clone()
    with pytest.raises(ValueError, match="CUDA"):
        optimizer.step()
    torch.testing.assert_close(model.atoms.p, initial, atol=0, rtol=0)
    assert not any(base.state.values())


def test_torus_uses_general_torch_update_and_keeps_centers_on_geometry():
    from torchcst import CSTLinear
    from torchcst._backends.torch.algorithms.atom_update.plans import REFERENCE
    from torchcst._backends.torch.charts import construction
    from torchcst._backends.torch.geometry import execution as geometry

    space = construction.torus(2, major_radius=3, minor_radius=1).double()
    angle = torch.linspace(0, 1.5, 5, dtype=torch.float64)
    sites = torch.stack(
        ((3 + angle.cos()) * angle.cos(), (3 + angle.cos()) * angle.sin(), angle.sin()),
        -1,
    )
    charts = (
        construction.points(sites, geometry=space),
        construction.points(sites[:4], geometry=space),
    )
    model = CSTLinear(
        *charts, atoms=3, kernel=layer().kernel.declaration(), dtype=torch.float64
    )
    selector = OrderedSelector(
        (FUSED,), fallback_plan=REFERENCE, registry=make_algorithm_registry()
    )
    base = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer = CSTOptimizer(base, model=model, update_selector=selector)
    for _ in range(3):
        old = model.atoms.p.detach().clone()
        model.atoms.p.grad = torch.randn_like(old)
        optimizer.step()
        point = model.atoms.p
        for chart, center in zip(charts, point[:, 2:].split(3, dim=-1)):
            geometry.validate_points(chart.geometry, center, name="updated center")
        state = optimizer._update_bindings[model]._algorithm_states[0]
        assert state.algorithm.id == "torch_atom_update"
    assert base.state[model.atoms.p]["step"] == 3


@pytest.mark.parametrize("sphere", [False, True])
def test_optimizer_matches_proposal_update_and_moment_transport(sphere):
    model = layer(sphere=sphere)
    oracle = copy.deepcopy(model)
    base = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.02)
    optimizer = CSTOptimizer(base, model=model)
    reference = torch.optim.AdamW(oracle.parameters(), lr=0.01, weight_decay=0.02)
    for _ in range(4):
        gradient = torch.randn_like(model.atoms.p)
        model.atoms.p.grad = gradient.clone()
        oracle.atoms.p.grad = execution.project_parameter_gradient(
            oracle.kernel, *oracle.cst_charts(), oracle.atoms.p, gradient
        )
        old = oracle.atoms.p.detach().clone()
        reference.step()
        with torch.no_grad():
            new = execution.apply_parameter_update(
                oracle.kernel,
                *oracle.cst_charts(),
                old,
                oracle.atoms.p - old,
                step_size=0.01,
            )
            moment = reference.state[oracle.atoms.p]["exp_avg"]
            moment.copy_(
                execution.transport_parameter_state(
                    oracle.kernel, *oracle.cst_charts(), old, new, moment
                )
            )
            oracle.atoms.p.copy_(new)
        optimizer.step()
        torch.testing.assert_close(
            model.atoms.p, oracle.atoms.p, atol=1e-12, rtol=1e-12
        )
        for name, value in base.state[model.atoms.p].items():
            torch.testing.assert_close(
                value, reference.state[oracle.atoms.p][name], atol=1e-12, rtol=1e-12
            )
    binding = optimizer._update_bindings[model]
    state = binding._algorithm_states[0]
    assert state.atom_state is model.atom_state
    model.atom_state.relayout(torch.tensor([2, 0, 1]))
    assert not state.is_current()
    model.atoms.p.grad = torch.randn_like(model.atoms.p)
    optimizer.step()
    assert state.is_current()
    assert not copy.deepcopy(binding)._algorithm_states


def test_polar_registration_does_not_load_linear_or_update_computation():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from torchcst import make_algorithm_registry
from torchcst._backends.cuda.algorithms.polar_update.plans import FUSED
from torchcst._backends.torch.algorithms.polar_update.plans import TORCH
r = make_algorithm_registry()
for plan in (FUSED, TORCH):
    assert r.loads_plan(r.dumps_plan(plan)) == plan
assert 'triton' not in sys.modules
assert not any(n.endswith('.executor') or n.endswith('.kernels') for n in sys.modules if n.startswith('torchcst._backends.'))
""",
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        check=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("plan", [TORCH, FUSED])
@pytest.mark.parametrize("mode", ["finite_chord", "time_energy"])
def test_dispatch_capture_replays_with_live_polar_scalars(plan, mode):
    model = layer(mode=mode).float().cuda()
    binding = AtomUpdateBinding(model.operator)
    old = model.atoms.p.detach().clone()
    proposal = old + torch.randn_like(old) * 0.02
    dispatcher = Dispatcher(registry=make_algorithm_registry())
    with torch.no_grad():
        model.atoms.p.copy_(proposal)
        dispatcher.run(binding, AtomUpdateInputs(old, 0.01), plan=plan)
        model.atoms.p.copy_(proposal)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dispatcher.run(binding, AtomUpdateInputs(old, 0.01), plan=plan)
        model.kernel.scalar("radial_regularization").fill_(0.6)
        model.kernel.scalar("w_c").fill_(0.4)
        model.atoms.p.copy_(proposal)
        graph.replay()
        torch.cuda.synchronize()
        expected = execution.apply_parameter_update(
            model.kernel, *model.cst_charts(), old, proposal - old, step_size=0.01
        )
    torch.testing.assert_close(model.atoms.p, expected, atol=2e-6, rtol=2e-6)
