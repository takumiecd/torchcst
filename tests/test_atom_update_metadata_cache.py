"""Update metadata reuse keeps fresh configuration and invocation boundaries."""

import copy
import pickle
from dataclasses import replace

import pytest
import torch
from test_radius_polar_cuda_update import radius_model

from torchcst import AtomUpdateBinding, AtomUpdateInputs, CSTOptimizer
from torchcst._backends.schema import DeviceInfo


def setup():
    model = radius_model()
    binding = AtomUpdateBinding(model.operator)
    return model, binding


def inputs(model, *, contiguous=True, decay=0.0):
    p = model.atoms.p.detach()
    previous = (
        p.clone()
        if contiguous
        else torch.zeros(p.shape[0], p.shape[1] * 2, dtype=p.dtype)[:, ::2]
    )
    return AtomUpdateInputs(previous, 0.01, decay)


def scalar(context, name):
    return {entry[0]: entry[1:] for entry in context.scalar_metadata}[name]


def test_hot_tuple_reuse_reads_fresh_declaration_without_extra_walk(monkeypatch):
    from torchcst.operators import metadata

    model, binding = setup()
    original = metadata.fixed_state_signature
    calls = []

    def signature(roots):
        calls.append(roots)
        return original(roots)

    monkeypatch.setattr(metadata, "fixed_state_signature", signature)
    first = binding.build_context(inputs(model))
    second = binding.build_context(inputs(model))
    assert second is not first
    assert second.scalar_metadata is first.scalar_metadata
    assert len(calls) == 2  # Exactly one fresh declaration guard per context.
    with torch.no_grad():
        model.atoms.p.add_(0.03)
    third = binding.build_context(inputs(model))
    assert third.scalar_metadata is first.scalar_metadata
    assert len(calls) == 3


@pytest.mark.parametrize("change", ["scalar_value", "setting"])
def test_configuration_changes_rebuild_metadata(change):
    model, binding = setup()
    first = binding.build_context(inputs(model))
    if change == "scalar_value":
        model.kernel.scalar("activity_gain").mul_(0.9)
    else:
        policy = model.kernel.spec.update
        settings = tuple(
            (name, "time_energy" if name == "activity_mode" else value)
            for name, value in policy.settings
        )
        model.kernel.spec = replace(
            model.kernel.spec, update=replace(policy, settings=settings)
        )
    second = binding.build_context(inputs(model))
    assert second.scalar_metadata == first.scalar_metadata
    assert second.scalar_metadata is not first.scalar_metadata
    assert second.kernel != first.kernel
    assert (
        binding.build_context(inputs(model)).scalar_metadata is second.scalar_metadata
    )


@pytest.mark.parametrize(
    "change", ["replacement", "rename", "numel", "dtype", "device"]
)
def test_buffer_topology_and_placement_changes_are_visible(change):
    model, binding = setup()
    module = model.kernel
    prefix = ""
    module.register_buffer("metadata_probe", torch.zeros(1))
    first = binding.build_context(inputs(model))
    if change == "replacement":
        module.metadata_probe = module.metadata_probe.clone()
    elif change == "rename":
        value = module._buffers.pop("metadata_probe")
        module.register_buffer("renamed_probe", value)
    elif change == "numel":
        module.metadata_probe.resize_(3)
    elif change == "dtype":
        module.metadata_probe = module.metadata_probe.double()
    else:
        # Metadata placement can be exercised without allocating a GPU Tensor.
        module.metadata_probe = torch.zeros(1, device="meta")
    second = binding.build_context(inputs(model))
    assert second.scalar_metadata is not first.scalar_metadata
    if change == "rename":
        assert prefix + "metadata_probe" not in {
            entry[0] for entry in second.scalar_metadata
        }
        assert scalar(second, prefix + "renamed_probe") == (
            torch.float32,
            DeviceInfo("cpu"),
            1,
        )
    else:
        value = module.metadata_probe
        assert scalar(second, prefix + "metadata_probe") == (
            value.dtype,
            DeviceInfo(value.device.type, value.device.index),
            value.numel(),
        )
    assert (
        binding.build_context(inputs(model)).scalar_metadata is second.scalar_metadata
    )


def test_previous_contiguity_workspace_and_decay_flag_stay_live():
    model, binding = setup()
    first = binding.build_context(inputs(model))
    second = binding.build_context(inputs(model, contiguous=False, decay=0.3))
    assert first.contiguous and not second.contiguous
    assert not first.amplitude_decay and second.amplitude_decay
    assert second.scalar_metadata is first.scalar_metadata
    binding.workspace_limit_bytes = 1024
    third = binding.build_context(inputs(model))
    assert third.workspace_limit_bytes == 1024
    assert third.contiguous
    assert third.scalar_metadata is first.scalar_metadata


def test_inference_buffer_bypasses_tuple_cache():
    model, binding = setup()
    binding.build_context(inputs(model))
    assert "_scalar_metadata_cache" in vars(binding)
    with torch.inference_mode():
        value = torch.zeros(1)
    model.kernel.register_buffer("metadata_probe", value)
    first = binding.build_context(inputs(model))
    second = binding.build_context(inputs(model))
    assert first.scalar_metadata == second.scalar_metadata
    assert first.scalar_metadata is not second.scalar_metadata
    assert binding.__dict__["_execution_declaration"][0] is None
    assert "_scalar_metadata_cache" not in vars(binding)


@pytest.mark.parametrize(
    "serializer", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_serialization_clears_scalar_metadata_cache(serializer):
    model, binding = setup()
    first = binding.build_context(inputs(model))
    clone = serializer(binding)
    assert "_scalar_metadata_cache" in vars(binding)
    assert "_scalar_metadata_cache" not in vars(clone)
    assert "_execution_declaration" not in vars(clone)
    assert not clone._algorithm_states
    p = clone.execution_parameters()
    rebuilt = clone.build_context(AtomUpdateInputs(p.detach().clone(), 0.01))
    assert rebuilt.scalar_metadata == first.scalar_metadata
    assert rebuilt.scalar_metadata is not first.scalar_metadata
    assert (
        clone.build_context(AtomUpdateInputs(p.detach().clone(), 0.01)).scalar_metadata
        is rebuilt.scalar_metadata
    )


def test_warm_metadata_does_not_certify_invalid_configuration_before_step():
    model = radius_model()
    opt = CSTOptimizer(
        torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0),
        model=model,
        check_finite=False,
    )
    model.atoms.p.grad = torch.ones_like(model.atoms.p) * 0.01
    opt.step()
    binding = opt._update_bindings[model]
    first = binding.build_context(inputs(model))
    before = model.atoms.p.detach().clone()
    states = {name: value.clone() for name, value in opt.state[model.atoms.p].items()}
    model.kernel.scalar("radius_min_input").fill_(-1)
    with pytest.raises(ValueError):
        opt.step()
    torch.testing.assert_close(model.atoms.p, before, atol=0, rtol=0)
    for name, value in states.items():
        torch.testing.assert_close(
            opt.state[model.atoms.p][name], value, atol=0, rtol=0
        )
    assert first.scalar_metadata


@pytest.mark.parametrize("change", ["replacement", "dtype"])
def test_geometry_period_buffer_placement_refreshes_metadata(change):
    model, binding = setup()
    first = binding.build_context(inputs(model))
    geometry = model.input_chart.geometry
    geometry.periods = (
        geometry.periods.clone()
        if change == "replacement"
        else geometry.periods.double()
    )
    second = binding.build_context(inputs(model))
    assert second.scalar_metadata is not first.scalar_metadata
    assert scalar(second, "geometry.0.periods") == (
        geometry.periods.dtype,
        DeviceInfo("cpu"),
        1,
    )
    assert (
        binding.build_context(inputs(model)).scalar_metadata is second.scalar_metadata
    )


@pytest.mark.parametrize("change", ["numel", "device"])
def test_invalid_geometry_declaration_cannot_use_warm_metadata(change):
    model, binding = setup()
    binding.build_context(inputs(model))
    if change == "numel":
        model.input_chart.geometry.periods = model.input_chart.geometry.periods.repeat(
            2
        )
    else:
        model.input_chart.geometry.periods = torch.ones(1, device="meta")
    with pytest.raises((ValueError, RuntimeError)):
        binding.build_context(inputs(model))


def test_metadata_requires_current_validated_declaration_entry(monkeypatch):
    model, binding = setup()
    binding.build_context(inputs(model))
    previous = binding.execution_declaration()
    detached = replace(previous)
    assert detached == previous and detached is not previous
    monkeypatch.setattr(binding, "execution_declaration", lambda: detached)
    first = binding.build_context(inputs(model))
    second = binding.build_context(inputs(model))
    assert first.scalar_metadata == second.scalar_metadata
    assert first.scalar_metadata is not second.scalar_metadata
    assert "_scalar_metadata_cache" not in vars(binding)
