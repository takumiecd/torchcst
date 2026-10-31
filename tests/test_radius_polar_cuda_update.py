"""Explicit radius GPU updates retain public optimizer arithmetic and state."""

import copy
from dataclasses import replace

import pytest
import torch
from test_radius_compatibility import model, polar_kernel

from torchcst import AtomUpdateBinding, AtomUpdateInputs, CSTOptimizer
from torchcst._backends.catalog import get_registry
from torchcst._backends.cuda.algorithms.polar_update.plans import (
    FLAT_TORUS_FUSED,
    FUSED,
    RADIUS_FLAT_TORUS_FUSED,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.schema import DeviceInfo
from torchcst._backends.torch.algorithms.atom_update.algorithm import (
    AtomUpdateAlgorithm,
)


def radius_model(mode="finite_chord"):
    result = model(polar_kernel(radius=True, activity_mode=mode), atom_count=7).float()
    with torch.no_grad():
        result.atoms.p[:, :2] = result.atoms.p.new_tensor(
            [[0, 0], [0, 1e-30], [0.3, 0.4], [0, 1], [0, 2], [3, 4], [-1, 0]]
        )
        result.atoms.p[:, 2:] = result.atoms.p.new_tensor(
            [
                [0, 16],
                [-16, -1e-7],
                [15.99, 16.01],
                [64, -64],
                [-0.0, 0.0],
                [32, -32],
                [7.3, 8.2],
            ]
        )
    return result


def test_radius_plan_is_explicit_registered_and_metadata_only():
    layer = radius_model()
    context = AtomUpdateBinding(layer.operator).build_context(
        AtomUpdateInputs(layer.atoms.p.detach().clone(), 0.01)
    )
    registry = get_registry()
    algorithm = registry.validate_plan(RADIUS_FLAT_TORUS_FUSED)
    assert not algorithm.supports(context, RADIUS_FLAT_TORUS_FUSED.recipe).supported
    cuda = DeviceInfo("cuda", 0)
    context = replace(
        context,
        device=cuda,
        scalar_metadata=tuple(
            (name, dtype, cuda, size)
            for name, dtype, _, size in context.scalar_metadata
        ),
    )
    assert algorithm.supports(context, RADIUS_FLAT_TORUS_FUSED.recipe).supported
    assert (
        not AtomUpdateAlgorithm()
        .supports(context, RADIUS_FLAT_TORUS_FUSED.recipe)
        .supported
    )
    for plan in (FUSED, FLAT_TORUS_FUSED):
        assert not registry.validate_plan(plan).supports(context, plan.recipe).supported
    assert (
        registry.load_plan(registry.dump_plan(RADIUS_FLAT_TORUS_FUSED))
        == RADIUS_FLAT_TORUS_FUSED
    )
    assert not algorithm.supports(
        replace(context, execution_mode="cuda_graph"), RADIUS_FLAT_TORUS_FUSED.recipe
    ).supported
    old = model(polar_kernel(radius=False)).float()
    old_context = AtomUpdateBinding(old.operator).build_context(
        AtomUpdateInputs(old.atoms.p.detach().clone(), 0.01)
    )
    assert not algorithm.supports(old_context, RADIUS_FLAT_TORUS_FUSED.recipe).supported


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["finite_chord", "time_energy"])
@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
@pytest.mark.parametrize("check_finite", [True, False])
def test_public_radius_optimizer_matches_cpu(mode, optimizer, check_finite):
    cpu = radius_model(mode)
    gpu = copy.deepcopy(cpu).cuda()

    def base(layer):
        if optimizer == "sgd":
            return torch.optim.SGD(
                layer.parameters(),
                lr=0.01,
                momentum=0.8,
                weight_decay=0.02,
                foreach=False,
            )
        return torch.optim.AdamW(
            layer.parameters(), lr=0.01, weight_decay=0.02, foreach=False
        )

    reference = CSTOptimizer(base(cpu), model=cpu, check_finite=check_finite)
    candidate = CSTOptimizer(
        base(gpu),
        model=gpu,
        check_finite=check_finite,
        update_selector=FixedSelector(RADIUS_FLAT_TORUS_FUSED, registry=get_registry()),
    )
    generator = torch.Generator().manual_seed(1729)
    for step in range(5):
        gradient = torch.randn(cpu.atoms.p.shape, generator=generator) * 0.03
        cpu.atoms.p.grad = gradient.clone()
        gpu.atoms.p.grad = gradient.cuda()
        if step == 2:
            for layer in (cpu, gpu):
                layer.kernel.scalar("radial_regularization").mul_(1.1)
        reference.step()
        candidate.step()
        torch.testing.assert_close(gpu.atoms.p.cpu(), cpu.atoms.p, rtol=4e-6, atol=4e-6)
        expected, actual = reference.state[cpu.atoms.p], candidate.state[gpu.atoms.p]
        assert actual.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(
                actual[key].cpu(), expected[key], rtol=3e-6, atol=3e-7
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fault", ["gradient", "period", "radius"])
def test_radius_invalid_changes_reject_before_optimizer_mutation(fault):
    layer = radius_model().cuda()
    optimizer = CSTOptimizer(
        torch.optim.AdamW(layer.parameters(), lr=0.01, foreach=False),
        model=layer,
        update_selector=FixedSelector(RADIUS_FLAT_TORUS_FUSED, registry=get_registry()),
    )
    layer.atoms.p.grad = torch.ones_like(layer.atoms.p)
    optimizer.step()
    previous = layer.atoms.p.detach().clone()
    state = {
        key: value.clone() for key, value in optimizer.state[layer.atoms.p].items()
    }
    if fault == "gradient":
        layer.atoms.p.grad[0, 0] = float("nan")
    elif fault == "period":
        layer.input_chart.geometry.periods.fill_(0)
    else:
        layer.kernel.scalar("radius_min_input").fill_(-1)
    with pytest.raises((ValueError, FloatingPointError)):
        optimizer.step()
    torch.testing.assert_close(layer.atoms.p, previous, rtol=0, atol=0)
    for key, value in state.items():
        torch.testing.assert_close(
            optimizer.state[layer.atoms.p][key], value, rtol=0, atol=0
        )


def test_radius_layout_cache_tracks_fixed_state_but_not_atom_writes(monkeypatch):
    from torchcst._backends.torch.kernels import execution
    from torchcst.kernels import radius_cell_mass

    layer = radius_model()
    charts = layer.cst_charts()
    layer.kernel.__dict__.pop("_radius_layout_signature", None)
    original = radius_cell_mass.radius_cell_mass_axes
    calls = []

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(radius_cell_mass, "radius_cell_mass_axes", counted)

    def validate():
        execution.validate_layout(layer.kernel, charts)

    validate()
    validate()
    assert len(calls) == 1
    with torch.no_grad():
        layer.atoms.p.add_(0.1)
    validate()
    assert len(calls) == 1

    layer.kernel.scalar("radius_min_input").mul_(1.1)
    validate()
    assert len(calls) == 2
    validate()
    assert len(calls) == 2

    # Preserve valid chart spacing/period agreement while changing live metadata.
    for chart in {id(chart): chart for chart in charts}.values():
        chart.geometry.periods.mul_(0.5)
        chart.spacing.mul_(0.5)
    validate()
    assert len(calls) == 3
    certified = layer.kernel.__dict__["_radius_layout_signature"]
    layer.input_chart.geometry.periods.fill_(0)
    with pytest.raises(ValueError):
        validate()
    assert layer.kernel.__dict__["_radius_layout_signature"] == certified
    with pytest.raises(ValueError):
        validate()
    assert layer.kernel.__dict__["_radius_layout_signature"] == certified


@pytest.mark.parametrize("operation", ["project", "transport"])
def test_related_polar_tables_validate_live_layout_once(monkeypatch, operation):
    from torchcst._backends.torch.algorithms.polar_update import executor
    from torchcst._backends.torch.kernels import execution

    layer = radius_model()
    charts = layer.cst_charts()
    p = layer.atoms.p.detach()
    original = execution.validate_layout
    calls = []

    def validate(state, charts):
        calls.append((state, charts))
        return original(state, charts)

    monkeypatch.setattr(execution, "validate_layout", validate)
    if operation == "project":
        value = executor.project_parameter_gradient(
            layer.kernel, *charts, p, torch.ones_like(p)
        )
    else:
        value = executor.transport_parameter_state(
            layer.kernel, *charts, p, p.clone(), torch.ones_like(p)
        )
    assert value.shape == p.shape
    assert len(calls) == 1
    # The next invocation must still see changed live configuration despite
    # sharing validation within each operation.
    layer.kernel.scalar("radius_min_input").fill_(-1)
    with pytest.raises(ValueError):
        if operation == "project":
            executor.project_parameter_gradient(
                layer.kernel, *charts, p, torch.ones_like(p)
            )
        else:
            executor.transport_parameter_state(
                layer.kernel, *charts, p, p.clone(), torch.ones_like(p)
            )
    assert len(calls) == 2


@pytest.mark.parametrize(
    "operation,table_index",
    [
        ("project", 0),
        ("project", 1),
        ("transport", 0),
        ("transport", 1),
        ("transport", 2),
    ],
)
@pytest.mark.parametrize("invalid", ["ndim", "width"])
def test_each_related_polar_table_shape_is_checked(operation, table_index, invalid):
    from torchcst._backends.torch.algorithms.polar_update import executor

    layer = radius_model()
    p = layer.atoms.p.detach()
    tables = [p.clone() for _ in range(2 if operation == "project" else 3)]
    tables[table_index] = p.flatten() if invalid == "ndim" else p[:, :-1]
    function = (
        executor.project_parameter_gradient
        if operation == "project"
        else executor.transport_parameter_state
    )
    with pytest.raises(ValueError, match="p must have shape"):
        function(layer.kernel, *layer.cst_charts(), *tables)
