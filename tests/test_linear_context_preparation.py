"""Sharing declaration reads never suppresses live/custom validation."""

import gc
import weakref

import pytest
import torch
from test_cst_optimizer import site

from torchcst import CSTLinear, LinearInputs


def invocation(model):
    return LinearInputs(torch.ones(2, model.in_features, dtype=model.atoms.p.dtype))


def test_combined_context_matches_fresh_callbacks_and_keeps_no_input():
    model = site()
    inputs = invocation(model)
    expected = model.build_context(inputs)
    assert model.prepare_context(inputs) == expected
    reference = weakref.ref(inputs.x)
    del inputs
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    "name",
    [
        "validate_inputs",
        "build_context",
        "execution_declaration",
        "execution_parameters",
    ],
)
def test_custom_callback_keeps_original_boundaries(monkeypatch, name):
    model = site()
    inputs = invocation(model)
    original = getattr(model, name)
    calls = []

    def callback(*args):
        calls.append(name)
        return original(*args)

    monkeypatch.setattr(model, name, callback)
    model.validate_inputs(inputs)
    expected = model.build_context(inputs)
    expected_calls = list(calls)
    calls.clear()
    assert model.prepare_context(inputs) == expected
    assert calls == expected_calls


def test_in_place_validation_code_change_is_not_bypassed(monkeypatch):
    model = site()

    def changed(self, inputs):
        raise RuntimeError("changed validation")

    monkeypatch.setattr(CSTLinear.validate_inputs, "__code__", changed.__code__)
    with pytest.raises(RuntimeError, match="changed validation"):
        model.prepare_context(invocation(model))


def test_custom_validation_can_replace_the_following_context_callback(monkeypatch):
    model = site()
    inputs = invocation(model)
    expected = object()
    original = model.validate_inputs

    def validate(value):
        original(value)
        model.build_context = lambda _: expected

    monkeypatch.setattr(model, "validate_inputs", validate)
    assert model.prepare_context(inputs) is expected


def test_subclass_custom_context_is_called():
    class Custom(CSTLinear):
        def validate_inputs(self, inputs):
            self.called = True

        def build_context(self, inputs):
            return "custom context"

    model = site()
    model.__class__ = Custom
    assert model.prepare_context(invocation(model)) == "custom context"
    assert model.called


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 5)])
def test_combined_context_rejects_bad_input_every_call(shape):
    model = site()
    model.prepare_context(invocation(model))
    with pytest.raises(ValueError, match="flattened"):
        model.prepare_context(
            LinearInputs(torch.ones(shape, dtype=model.atoms.p.dtype))
        )


def test_tensor_mode_preserves_both_callbacks(monkeypatch):
    from torch.overrides import TorchFunctionMode

    from torchcst.operators import metadata

    model = site()
    inputs = invocation(model)
    calls = []
    original = metadata.operator_declaration

    def declaration(binding):
        calls.append(binding)
        return original(binding)

    monkeypatch.setattr(metadata, "operator_declaration", declaration)

    class Passthrough(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with Passthrough():
        model.prepare_context(inputs)
    assert len(calls) == 2
