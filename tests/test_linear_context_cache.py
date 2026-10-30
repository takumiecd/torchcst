"""Linear Context reuse never certifies live tensors or changed settings."""

import copy
import pickle
import weakref
from dataclasses import replace

import pytest
import torch
from test_radius_polar_cuda_update import radius_model

from torchcst import (
    DefaultRecipe,
    Dispatcher,
    ExecutionPlan,
    FixedSelector,
    LinearBinding,
    LinearInputs,
)
from torchcst._backends.catalog import get_registry

MATERIALIZED = ExecutionPlan("torch_materialized", "v1", DefaultRecipe())


def setup(kind="module"):
    model = radius_model()
    binding = model if kind == "module" else LinearBinding(model.operator)
    inputs = LinearInputs(torch.randn(4, model.in_features, requires_grad=True))
    return model, binding, inputs


@pytest.mark.parametrize("kind", ["module", "direct"])
def test_interned_context_does_not_retain_inputs_or_freeze_atom_values(kind):
    model, binding, inputs = setup(kind)
    first = binding.build_context(inputs)
    weak = weakref.ref(inputs.x)
    del inputs
    assert weak() is None
    with torch.no_grad():
        model.atoms.p.add_(0.01)
    _, _, fresh = setup()
    assert binding.build_context(fresh) is first


@pytest.mark.parametrize("kind", ["module", "direct"])
@pytest.mark.parametrize(
    "change",
    ["shape", "stride", "x_grad", "p_grad", "no_grad", "deterministic", "dtype"],
)
def test_context_reads_current_tensor_metadata_and_global_flags(kind, change):
    model, binding, inputs = setup(kind)
    first = binding.build_context(inputs)
    if change == "shape":
        inputs = LinearInputs(torch.randn(7, model.in_features, requires_grad=True))
    elif change == "stride":
        inputs = LinearInputs(
            torch.randn(4, 2 * model.in_features, requires_grad=True)[:, ::2]
        )
    elif change == "x_grad":
        inputs.x.requires_grad_(False)
    elif change == "p_grad":
        model.atoms.p.requires_grad_(False)
    elif change == "dtype":
        model.double()
        inputs = LinearInputs(inputs.x.double())
    elif change == "no_grad":
        with torch.no_grad():
            second = binding.build_context(inputs)
            assert second is not first
            assert not second.required_grads.inputs
            assert not second.required_grads.parameters
            assert binding.build_context(inputs) is second
        return
    elif change == "deterministic":
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(not previous)
            second = binding.build_context(inputs)
            assert second is not first
            assert second.deterministic is not previous
            assert binding.build_context(inputs) is second
        finally:
            torch.use_deterministic_algorithms(previous)
        return
    second = binding.build_context(inputs)
    assert second is not first
    assert binding.build_context(inputs) is second


@pytest.mark.parametrize("kind", ["module", "direct"])
@pytest.mark.parametrize(
    "change",
    ["valid_scalar", "same_value_write", "spec", "invalid_kernel", "invalid_chart"],
)
def test_warm_context_keeps_full_configuration_change_validation(kind, change):
    model, binding, inputs = setup(kind)
    first = binding.build_context(inputs)
    if change == "valid_scalar":
        model.kernel.scalar("activity_gain").mul_(0.9)
    elif change == "same_value_write":
        model.kernel.scalar("activity_gain").add_(0)
    elif change == "spec":
        policy = model.kernel.spec.update
        settings = tuple(
            (n, "time_energy" if n == "activity_mode" else v)
            for n, v in policy.settings
        )
        model.kernel.spec = replace(
            model.kernel.spec, update=replace(policy, settings=settings)
        )
    elif change == "invalid_kernel":
        model.kernel.scalar("radius_min_input").fill_(-1)
    else:
        model.input_chart.geometry.periods.fill_(0)
    if change.startswith("invalid"):
        with pytest.raises(ValueError):
            binding.build_context(inputs)
    else:
        second = binding.build_context(inputs)
        assert second is not first
        assert binding.build_context(inputs) is second


@pytest.mark.parametrize("kind", ["module", "direct"])
@pytest.mark.parametrize(
    "serializer", [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_context_cache_is_transient(kind, serializer):
    _, binding, inputs = setup(kind)
    first = binding.build_context(inputs)
    clone = serializer(binding)
    assert "_linear_context_cache" not in vars(clone)
    second = clone.build_context(inputs)
    assert second == first and second is not first
    assert clone.build_context(inputs) is second


@pytest.mark.parametrize("kind", ["module", "direct"])
@pytest.mark.parametrize(
    "bypass", ["inference_buffer", "unvalidated_declaration", "tensor_subclass"]
)
def test_uncertified_metadata_uses_fresh_context(kind, bypass, monkeypatch):
    model, binding, inputs = setup(kind)
    binding.build_context(inputs)
    if bypass == "inference_buffer":
        with torch.inference_mode():
            probe = torch.zeros(1)
        model.kernel.register_buffer("context_probe", probe)
    elif bypass == "unvalidated_declaration":
        declaration = replace(binding.execution_declaration())
        monkeypatch.setattr(binding, "execution_declaration", lambda: declaration)
    else:

        class CustomTensor(torch.Tensor):
            pass

        inputs = LinearInputs(inputs.x.as_subclass(CustomTensor))
    first = binding.build_context(inputs)
    second = binding.build_context(inputs)
    assert second == first and second is not first
    assert "_linear_context_cache" not in vars(binding)


def test_workspace_and_invalid_workspace_remain_live():
    _, binding, inputs = setup("direct")
    first = binding.build_context(inputs)
    binding.workspace_limit_bytes = 1024
    second = binding.build_context(inputs)
    assert second is not first
    assert second.workspace_limit_bytes == 1024
    for invalid in (-1, True, torch.tensor(1024)):
        binding.workspace_limit_bytes = invalid
        with pytest.raises(ValueError, match="workspace"):
            binding.build_context(inputs)


@pytest.mark.parametrize("fault", ["feature", "dtype", "parameter_dim"])
def test_warm_context_never_certifies_new_inputs(fault):
    model, binding, inputs = setup()
    dispatcher = Dispatcher(
        selector=FixedSelector(MATERIALIZED, registry=get_registry())
    )
    dispatcher.prepare(binding, inputs)
    if fault == "feature":
        inputs = LinearInputs(torch.randn(4, model.in_features + 1))
    elif fault == "dtype":
        inputs = LinearInputs(inputs.x.double())
    else:
        # Only validation metadata is changed, not the owning Parameter.
        binding.execution_parameters = lambda: torch.zeros(7, 0)
    with pytest.raises(ValueError):
        dispatcher.prepare(binding, inputs)


@pytest.mark.parametrize(
    "callback", ["validate_inputs", "build_context", "execution_declaration"]
)
def test_custom_callbacks_keep_their_order_and_see_mutation(callback, monkeypatch):
    model, binding, inputs = setup()
    dispatcher = Dispatcher(
        selector=FixedSelector(MATERIALIZED, registry=get_registry())
    )
    dispatcher.prepare(binding, inputs)
    original = getattr(binding, callback)
    calls = []

    def mutate(*args):
        calls.append(callback)
        model.kernel.scalar("radius_min_input").fill_(-1)
        return original(*args)

    monkeypatch.setattr(binding, callback, mutate)
    with pytest.raises(ValueError):
        dispatcher.prepare(binding, inputs)
    assert calls == [callback]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_context_tracks_capture_autocast_and_tf32(monkeypatch):
    model, binding, inputs = setup()
    model.cuda()
    inputs = LinearInputs(inputs.x.cuda())
    first = binding.build_context(inputs)
    assert binding.build_context(inputs) is first
    prior = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = not prior
        second = binding.build_context(inputs)
        assert second is not first
        assert second.precision.allow_tf32 is not prior
        with torch.autocast("cuda"):
            third = binding.build_context(inputs)
            assert third is not second
            assert third.precision.autocast
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        captured = binding.build_context(inputs)
        assert captured.execution_mode == "cuda_graph"
        assert binding.build_context(inputs) is captured
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior
