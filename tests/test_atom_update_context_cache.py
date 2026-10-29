"""Context interning follows fresh declarations and current invocation flags."""

import copy
import pickle
import weakref
from dataclasses import replace

import pytest
import torch
from test_atom_update_metadata_cache import inputs, setup

from torchcst import AtomUpdateInputs


def test_context_identity_ignores_atom_values_snapshots_and_rate_magnitude():
    model, binding = setup()
    previous = inputs(model).previous
    weak = weakref.ref(previous)
    first = binding.build_context(AtomUpdateInputs(previous, 0.01, 0.2))
    del previous
    assert weak() is None  # Interning must never retain optimizer snapshots.
    with torch.no_grad():
        model.atoms.p.add_(0.03)
    second = binding.build_context(inputs(model, decay=0.7))
    third = binding.build_context(
        AtomUpdateInputs(model.atoms.p.detach().clone(), 0.2, 0.1)
    )
    assert second is third is first
    assert binding.build_context(inputs(model)) is not first  # positive -> zero decay


@pytest.mark.parametrize(
    "change",
    [
        "previous_contiguous",
        "workspace",
        "decay",
        "shape",
        "dtype",
        "device",
        "contiguous",
    ],
)
def test_context_rebuilds_for_current_layout_and_invocation_flags(monkeypatch, change):
    model, binding = setup()
    first = binding.build_context(inputs(model))
    current = inputs(model)
    if change == "previous_contiguous":
        current = inputs(model, contiguous=False)
    elif change == "workspace":
        binding.workspace_limit_bytes = 1024
    elif change == "decay":
        current = inputs(model, decay=0.3)
    else:
        p = model.atoms.p.detach()
        value = {
            "shape": torch.empty(p.shape[0] + 1, p.shape[1], dtype=p.dtype),
            "dtype": p.double(),
            "device": torch.empty(p.shape, dtype=p.dtype, device="meta"),
            "contiguous": torch.empty(p.shape[0], 2 * p.shape[1], dtype=p.dtype)[
                :, ::2
            ],
        }[change]
        monkeypatch.setattr(binding, "execution_parameters", lambda: value)
    second = binding.build_context(current)
    assert second is not first
    assert binding.build_context(current) is second
    if "contiguous" in change:
        assert not second.contiguous
    elif change == "workspace":
        assert second.workspace_limit_bytes == 1024
    elif change == "decay":
        assert second.amplitude_decay
    elif change == "shape":
        assert second.parameter_shape != first.parameter_shape
    elif change == "device":
        assert second.device.type == "meta"
    else:
        assert second.dtype == torch.float64


def test_capture_flag_changes_context_without_gpu_allocation(monkeypatch):
    model, binding = setup()
    eager = binding.build_context(inputs(model))
    # Only metadata queries are mocked; tensors stay on CPU and no CUDA API runs.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    captured = binding.build_context(inputs(model))
    assert captured is not eager
    assert captured.execution_mode == "cuda_graph"
    assert binding.build_context(inputs(model)) is captured
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    refreshed = binding.build_context(inputs(model))
    assert refreshed is not captured
    assert refreshed.execution_mode == "eager"


@pytest.mark.parametrize(
    "mutation", ["invalid_kernel", "invalid_geometry", "valid_value", "spec"]
)
def test_warm_context_does_not_hide_configuration_changes(mutation):
    model, binding = setup()
    first = binding.build_context(inputs(model))
    if mutation == "invalid_kernel":
        model.kernel.scalar("radius_min_input").fill_(-1)
    elif mutation == "invalid_geometry":
        model.input_chart.geometry.periods.fill_(0)
    elif mutation == "valid_value":
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
    if mutation.startswith("invalid"):
        with pytest.raises(ValueError):
            binding.build_context(inputs(model))
    else:
        second = binding.build_context(inputs(model))
        assert second is not first
        assert second.kernel != first.kernel
        assert binding.build_context(inputs(model)) is second


@pytest.mark.parametrize("bypass", ["inference", "unvalidated_declaration"])
def test_uncertified_declaration_cannot_reuse_interned_context(monkeypatch, bypass):
    model, binding = setup()
    binding.build_context(inputs(model))
    assert "_context_cache" in vars(binding)
    if bypass == "inference":
        with torch.inference_mode():
            probe = torch.zeros(1)
        model.kernel.register_buffer("context_probe", probe)
    else:
        detached = replace(binding.execution_declaration())
        monkeypatch.setattr(binding, "execution_declaration", lambda: detached)
    first = binding.build_context(inputs(model))
    second = binding.build_context(inputs(model))
    assert second == first and second is not first
    assert "_context_cache" not in vars(binding)


@pytest.mark.parametrize(
    "serializer", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_context_cache_is_transient_during_serialization(serializer):
    model, binding = setup()
    original = binding.build_context(inputs(model))
    clone = serializer(binding)
    assert "_context_cache" not in vars(clone)
    rebuilt = clone.build_context(
        AtomUpdateInputs(clone.execution_parameters().detach().clone(), 0.01)
    )
    assert rebuilt == original and rebuilt is not original
    assert (
        clone.build_context(
            AtomUpdateInputs(clone.execution_parameters().detach().clone(), 0.01)
        )
        is rebuilt
    )


def test_fresh_declaration_callback_runs_before_context_reuse(monkeypatch):
    model, binding = setup()
    binding.build_context(inputs(model))
    original = binding.execution_declaration
    calls = []

    def mutate_then_declare():
        calls.append(1)
        model.kernel.scalar("radius_min_input").fill_(-1)
        return original()

    monkeypatch.setattr(binding, "execution_declaration", mutate_then_declare)
    with pytest.raises(ValueError):
        binding.build_context(inputs(model))
    assert calls == [1]


@pytest.mark.parametrize("fault", ["alias", "requires_grad", "step", "decay"])
def test_interned_context_never_certifies_new_invocation_inputs(fault):
    from torchcst import Dispatcher
    from torchcst._backends.catalog import get_registry
    from torchcst._backends.torch.algorithms.atom_update.plans import REFERENCE

    model, binding = setup()
    dispatcher = Dispatcher(registry=get_registry())
    dispatcher.prepare(binding, inputs(model), plan=REFERENCE)
    p = model.atoms.p
    value = {
        "alias": AtomUpdateInputs(p.detach(), 0.01),
        "requires_grad": AtomUpdateInputs(p.detach().clone().requires_grad_(), 0.01),
        "step": AtomUpdateInputs(p.detach().clone(), 0.0),
        "decay": AtomUpdateInputs(p.detach().clone(), 0.01, float("nan")),
    }[fault]
    with pytest.raises(ValueError):
        dispatcher.prepare(binding, value, plan=REFERENCE)


def test_geometry_identity_is_not_certified_by_custom_equality(monkeypatch):
    model, binding = setup()
    first = binding.build_context(inputs(model))
    declaration = binding.execution_declaration()

    class EqualToAnything:
        def __eq__(self, other):
            return True

    replacement = EqualToAnything()
    # Pin the validated declaration to isolate the Context key's own guard;
    # normal declaration validation additionally rejects unsupported specs.
    monkeypatch.setattr(binding, "execution_declaration", lambda: declaration)
    monkeypatch.setattr(model.input_chart.geometry, "spec", replacement)
    second = binding.build_context(inputs(model))
    assert second is not first
    assert second.geometries[0] is replacement
    assert binding.build_context(inputs(model)) is second


@pytest.mark.parametrize("field", ["workspace", "decay"])
def test_invalid_tensor_metadata_cannot_enter_context_cache(field):
    model, binding = setup()
    binding.build_context(inputs(model))
    current = inputs(model)
    # Direct Context construction is not invocation validation. These invalid
    # API inputs must not make interning evaluate Tensor equality or retain it.
    if field == "workspace":
        binding.workspace_limit_bytes = torch.tensor(1024)
    else:
        current = AtomUpdateInputs(current.previous, 0.01, torch.tensor(0.2))
    first = binding.build_context(current)
    second = binding.build_context(current)
    assert second is not first
    assert "_context_cache" not in vars(binding)
