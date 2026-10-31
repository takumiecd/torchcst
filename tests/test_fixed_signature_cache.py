"""Cached topology traversal never certifies stale fixed configuration."""

import copy
import gc
import importlib
import pickle
import weakref

import pytest
import torch
from test_dispatcher import Add, Binding, Inputs
from test_radius_polar_cuda_update import radius_model

from torchcst import DefaultRecipe
from torchcst.operators import metadata


@pytest.fixture(scope="module", autouse=True)
def load_optional_native_guard_api():
    # Feature qualification belongs in tests, never in production metadata.
    try:
        importlib.import_module("torch._dynamo.guards")
    except (ImportError, AttributeError, RuntimeError):
        pass


def has_native_guard(owner):
    cache = owner.__dict__.get("_fixed_signature_cache")
    return cache is not None and cache.guard is not None


def roots():
    chart, kernel = torch.nn.Module(), torch.nn.Module()
    chart.child = torch.nn.Module()
    chart.child.register_buffer("coordinate", torch.tensor([0.0, 1.0]))
    kernel.register_buffer("scale", torch.tensor(1.0))
    return chart, kernel


def count_walks(monkeypatch):
    original = metadata._flatten_modules
    calls = []

    def flatten(values):
        calls.append(tuple(id(root) for root in values))
        return original(values)

    monkeypatch.setattr(metadata, "_flatten_modules", flatten)
    return calls


def test_hot_signature_reuses_tuple_without_recursive_walk(monkeypatch):
    values = roots()
    calls = count_walks(monkeypatch)
    first = metadata.fixed_state_signature(values)
    assert first is not None
    if not has_native_guard(values[-1]):
        pytest.skip("optional PyTorch metadata guard is unavailable")
    for _ in range(5):
        assert metadata.fixed_state_signature(values) is first
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["add", "remove", "replace", "rename", "nested"])
def test_child_topology_changes_rebuild_signature(monkeypatch, change):
    chart, kernel = roots()
    calls = count_walks(monkeypatch)
    first = metadata.fixed_state_signature((chart, kernel))
    if change == "add":
        chart.extra = torch.nn.Module()
    elif change == "remove":
        del chart.child
    elif change == "replace":
        chart.child = torch.nn.Module()
    elif change == "rename":
        chart.add_module("renamed", chart._modules.pop("child"))
    else:
        chart.child.nested = torch.nn.Module()
    second = metadata.fixed_state_signature((chart, kernel))
    assert second is not first
    assert len(calls) == 2
    native = has_native_guard(kernel)
    assert metadata.fixed_state_signature((chart, kernel)) is second
    assert len(calls) == (2 if native else 3)


@pytest.mark.parametrize(
    "change",
    ["value", "same_value", "dtype", "device", "numel", "rename", "replacement"],
)
def test_buffer_mutations_rebuild_signature(monkeypatch, change):
    chart, kernel = roots()
    calls = count_walks(monkeypatch)
    first = metadata.fixed_state_signature((chart, kernel))
    if change == "value":
        kernel.scale.fill_(2)
    elif change == "same_value":
        kernel.scale.fill_(1)
    elif change == "dtype":
        kernel.scale = kernel.scale.double()
    elif change == "device":
        kernel.scale = torch.ones((), device="meta")
    elif change == "numel":
        kernel.scale.resize_(3)
    elif change == "rename":
        kernel.register_buffer("renamed", kernel._buffers.pop("scale"))
    else:
        kernel.scale = kernel.scale.clone()
    second = metadata.fixed_state_signature((chart, kernel))
    assert second is not first
    assert second != first
    assert len(calls) == 2
    native = has_native_guard(kernel)
    assert metadata.fixed_state_signature((chart, kernel)) is second
    assert len(calls) == (2 if native else 3)


def test_spec_and_binding_identity_changes_are_observed(monkeypatch):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Declaration:
        value: int = 1

    chart, kernel = roots()
    chart.spec = Declaration()
    kernel.binding = Declaration()
    calls = count_walks(monkeypatch)
    first = metadata.fixed_state_signature((chart, kernel))
    chart.spec = Declaration()  # Equal values, distinct declaration object.
    second = metadata.fixed_state_signature((chart, kernel))
    assert second is not first
    kernel.binding = Declaration()
    third = metadata.fixed_state_signature((chart, kernel))
    assert third is not second
    assert len(calls) == 3


def test_actual_atom_value_writes_do_not_invalidate_fixed_signature(monkeypatch):
    model = radius_model()
    values = (*model.cst_charts(), model.kernel)
    model.kernel.__dict__.pop("_fixed_signature_cache", None)
    calls = count_walks(monkeypatch)
    first = metadata.fixed_state_signature(values)
    with torch.no_grad():
        model.atoms.p.add_(0.1)
    native = has_native_guard(model.kernel)
    assert metadata.fixed_state_signature(values) is first
    assert len(calls) == (1 if native else 2)


def test_cache_does_not_retain_chart_roots_or_buffer_tensors():
    chart, kernel = roots()
    chart_ref, child_ref = weakref.ref(chart), weakref.ref(chart.child)
    buffer_ref = weakref.ref(kernel.scale)
    signature = metadata.fixed_state_signature((chart, kernel))
    assert signature is not None
    kernel.scale = torch.tensor(2.0)
    del chart
    gc.collect()
    assert chart_ref() is None
    assert child_ref() is None
    assert buffer_ref() is None
    assert "_fixed_signature_cache" in vars(kernel)


@pytest.mark.parametrize(
    "serializer", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_serialized_cache_rebuilds_then_returns_stable_identity(
    monkeypatch, serializer
):
    values = roots()
    calls = count_walks(monkeypatch)
    first = metadata.fixed_state_signature(values)
    assert "_fixed_signature_cache" in vars(values[-1])
    clones = serializer(values)
    before = len(calls)
    rebuilt = metadata.fixed_state_signature(clones)
    assert rebuilt is not first
    assert len(calls) == before + 1
    native = has_native_guard(clones[-1])
    assert metadata.fixed_state_signature(clones) is rebuilt
    assert len(calls) == before + (1 if native else 2)


def test_inference_buffer_bypasses_cache_and_normal_buffer_recovers():
    chart, kernel = roots()
    first = metadata.fixed_state_signature((chart, kernel))
    with torch.inference_mode():
        kernel.scale = torch.tensor(1.0)
    assert metadata.fixed_state_signature((chart, kernel)) is None
    assert metadata.fixed_state_signature((chart, kernel)) is None
    kernel.scale = torch.tensor(1.0)
    recovered = metadata.fixed_state_signature((chart, kernel))
    assert recovered is not None and recovered is not first
    assert metadata.fixed_state_signature((chart, kernel)) is recovered


@pytest.mark.parametrize("walker", ["modules", "named_modules"])
@pytest.mark.parametrize("position", ["root", "child"])
def test_custom_walkers_are_never_skipped(walker, position):
    class Custom(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.walk_calls = 0

    if walker == "modules":

        def modules(self):
            self.walk_calls += 1
            return torch.nn.Module.modules(self)

        Custom.modules = modules
    else:

        def named_modules(self, *args, **kwargs):
            self.walk_calls += 1
            return torch.nn.Module.named_modules(self, *args, **kwargs)

        Custom.named_modules = named_modules
    chart, kernel = roots()
    custom = Custom()
    if position == "root":
        chart = custom
    else:
        chart.child = custom
    for _ in range(3):
        metadata.fixed_state_signature((chart, kernel))
    # A child.modules override is not called by the standard root walker;
    # inspect it as a root too so its callback boundary is actually exercised.
    if walker == "modules" and position == "child":
        for _ in range(3):
            metadata.fixed_state_signature((custom, kernel))
    assert custom.walk_calls >= 3


@pytest.mark.parametrize("callback", ["validate_inputs", "validate_recipe"])
def test_algorithm_callbacks_cannot_hide_fixed_config_changes(callback):
    chart, kernel = roots()

    class FixedBinding(Binding):
        def state_signature(self):
            return metadata.fixed_state_signature((chart, kernel))

    class MutatingAdd(Add):
        def validate_inputs(self, state, inputs):
            super().validate_inputs(state, inputs)
            if callback == "validate_inputs" and self.mutate:
                kernel.scale.add_(1)

        def validate_recipe(self, recipe):
            super().validate_recipe(recipe)
            if callback == "validate_recipe" and self.mutate:
                kernel.scale.add_(1)

    algorithm = MutatingAdd()
    object.__setattr__(algorithm, "mutate", False)
    binding = FixedBinding()
    state = binding.algorithm_state(algorithm, recipe=DefaultRecipe())
    algorithm.run(state, Inputs(torch.ones(2)))
    assert algorithm.preparations == 1
    object.__setattr__(algorithm, "mutate", True)
    if callback == "validate_inputs":
        algorithm.validate_inputs(state, Inputs(torch.ones(2)))
    else:
        algorithm.validate_recipe(DefaultRecipe())
    assert not state.is_current()
    object.__setattr__(algorithm, "mutate", False)
    algorithm.run(state, Inputs(torch.ones(2)))
    assert algorithm.preparations == 2
    assert state.is_current()


@pytest.mark.parametrize("attribute", ["spec", "binding"])
@pytest.mark.parametrize("kind", ["module", "tensor"])
def test_nonmetadata_declaration_attributes_are_not_retained(attribute, kind):
    chart, kernel = roots()
    metadata.fixed_state_signature((chart, kernel))
    value = torch.nn.Module() if kind == "module" else torch.ones(1)
    reference = weakref.ref(value)
    # Deliberately expose a custom declaration attribute without registering
    # it as a child/buffer: the generic guard must bypass metadata caching.
    object.__setattr__(chart, attribute, value)
    metadata.fixed_state_signature((chart, kernel))
    delattr(chart, attribute)
    del value
    gc.collect()
    assert reference() is None
    recovered = metadata.fixed_state_signature((chart, kernel))
    assert metadata.fixed_state_signature((chart, kernel)) is recovered


def test_moving_same_named_buffer_between_owners_changes_signature(monkeypatch):
    chart, kernel = roots()
    calls = count_walks(monkeypatch)
    first = metadata.fixed_state_signature((chart, kernel))
    # The flattened order of actual buffers stays coordinate,scale; only the
    # registration owner changes. Values/id/version/name stay unchanged.
    chart.child.register_buffer("scale", kernel._buffers.pop("scale"))
    second = metadata.fixed_state_signature((chart, kernel))
    assert second is not first and second != first
    assert len(calls) == 2
    native = has_native_guard(kernel)
    assert metadata.fixed_state_signature((chart, kernel)) is second
    assert len(calls) == (2 if native else 3)


def test_custom_module_hash_callbacks_are_not_skipped(monkeypatch):
    class CustomHash(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hash_calls = 0

        def __hash__(self):
            self.hash_calls += 1
            return object.__hash__(self)

    chart, kernel = roots()
    custom = CustomHash()
    chart.child = custom
    walks = count_walks(monkeypatch)
    counts = []
    for _ in range(3):
        previous = custom.hash_calls
        metadata.fixed_state_signature((chart, kernel))
        counts.append(custom.hash_calls - previous)
    assert all(count > 0 for count in counts)
    assert len(walks) == 3


def test_forced_unavailable_native_api_keeps_full_validation_and_tuple_intern(
    monkeypatch,
):
    from torchcst.operators import _native_metadata_guard

    monkeypatch.setattr(_native_metadata_guard, "guard_api", lambda: None)
    chart, kernel = roots()
    walks = count_walks(monkeypatch)
    first = metadata.fixed_state_signature((chart, kernel))
    assert not has_native_guard(kernel)
    assert metadata.fixed_state_signature((chart, kernel)) is first
    assert len(walks) == 2
    kernel.scale.add_(0)
    changed = metadata.fixed_state_signature((chart, kernel))
    assert changed is not first and changed != first
    assert len(walks) == 3
    assert metadata.fixed_state_signature((chart, kernel)) is changed
    assert len(walks) == 4


@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_buffer_data_replacement_without_version_still_invalidates(change):
    chart, kernel = roots()
    first = metadata.fixed_state_signature((chart, kernel))
    value = kernel.scale
    identity, version = id(value), value._version
    value.data = torch.ones(3) if change == "shape" else value.double()
    assert id(kernel.scale) == identity and value._version == version
    second = metadata.fixed_state_signature((chart, kernel))
    assert second is not first and second != first
    assert metadata.fixed_state_signature((chart, kernel)) is second


def test_replaced_buffer_dictionary_cannot_reuse_tensor_match_only_guards():
    chart, kernel = roots()
    first = metadata.fixed_state_signature((chart, kernel))
    old = kernel.scale
    kernel._buffers = {name: value.clone() for name, value in kernel._buffers.items()}
    assert kernel.scale is not old
    second = metadata.fixed_state_signature((chart, kernel))
    assert second is not first and second != first


def test_later_buffer_type_change_is_observed_by_tensor_match():
    class CustomTensor(torch.Tensor):
        pass

    chart, kernel = roots()
    metadata.fixed_state_signature((chart, kernel))
    cache = kernel.__dict__["_fixed_signature_cache"]
    if cache.guard is None:
        pytest.skip("optional PyTorch metadata guard is unavailable")
    assert cache.guard.check((chart, kernel))
    # Keep object, parent dictionary and numerical metadata unchanged. The
    # earlier chart buffer has already supplied the shared Tensor class tree.
    kernel.scale.__class__ = CustomTensor
    assert not cache.guard.check((chart, kernel))


def test_unavailable_tensor_match_keeps_identity_type_and_metadata_checks(monkeypatch):
    from torchcst.operators import _native_metadata_guard

    monkeypatch.setattr(
        _native_metadata_guard, "tensor_metadata_guard", lambda *a: False
    )
    chart, kernel = roots()
    first = metadata.fixed_state_signature((chart, kernel))
    assert metadata.fixed_state_signature((chart, kernel)) is first
    kernel.scale = kernel.scale.clone()
    second = metadata.fixed_state_signature((chart, kernel))
    assert second is not first
    kernel.scale.data = kernel.scale.double()
    third = metadata.fixed_state_signature((chart, kernel))
    assert third is not second


def test_native_guard_does_not_keep_any_root_or_tensor_alive():
    chart, kernel = roots()
    references = [
        weakref.ref(value)
        for value in (chart, chart.child, kernel, kernel.scale, chart.child.coordinate)
    ]
    metadata.fixed_state_signature((chart, kernel))
    # Hold the guard/cache themselves after their owner has gone, so a cycle
    # being collected would not conceal strong example references in the guard.
    cache = kernel.__dict__["_fixed_signature_cache"]
    guard = cache.guard
    del chart, kernel
    gc.collect()
    assert all(reference() is None for reference in references)
    assert cache is not None
    assert guard is None or guard is cache.guard


@pytest.mark.parametrize("override", ["property", "named_modules"])
def test_class_attribute_overrides_invalidate_native_guard_and_keep_callbacks(
    monkeypatch, override
):
    class Changeable(torch.nn.Module):
        pass

    chart, kernel = Changeable(), torch.nn.Module()
    chart.calls = 0
    first = metadata.fixed_state_signature((chart, kernel))
    if override == "property":

        def spec(self):
            self.calls += 1
            return "custom fixed declaration"

        monkeypatch.setattr(Changeable, "spec", property(spec), raising=False)
    else:

        def named_modules(self, *args, **kwargs):
            self.calls += 1
            return torch.nn.Module.named_modules(self, *args, **kwargs)

        monkeypatch.setattr(Changeable, "named_modules", named_modules)
    counts = []
    for _ in range(3):
        before = chart.calls
        second = metadata.fixed_state_signature((chart, kernel))
        counts.append(chart.calls - before)
    assert all(count > 0 for count in counts)
    if override == "property":
        assert second != first


def test_canonical_modules_code_change_does_not_bypass_new_callbacks(monkeypatch):
    chart, kernel = roots()
    metadata.fixed_state_signature((chart, kernel))

    def changed(self):
        self.__dict__["_cst_walk_calls"] = self.__dict__.get("_cst_walk_calls", 0) + 1
        for _, module in self.named_modules():
            yield module

    monkeypatch.setattr(torch.nn.Module.modules, "__code__", changed.__code__)
    counts = []
    for _ in range(3):
        before = chart.__dict__.get("_cst_walk_calls", 0)
        metadata.fixed_state_signature((chart, kernel))
        counts.append(chart.__dict__.get("_cst_walk_calls", 0) - before)
    assert all(count > 0 for count in counts)


def test_canonical_named_modules_default_change_uses_fresh_walker_semantics(
    monkeypatch,
):
    chart, kernel = roots()
    chart.alias = chart.child
    metadata.fixed_state_signature((chart, kernel))
    original_defaults = torch.nn.Module.named_modules.__defaults__
    assert original_defaults[-1] is True
    original_flatten = metadata._flatten_modules
    calls = count_walks(monkeypatch)
    monkeypatch.setattr(
        torch.nn.Module.named_modules, "__defaults__", (*original_defaults[:-1], False)
    )
    for _ in range(3):
        # Some Torch versions' modules() explicitly passes remove_duplicate;
        # others inherit named_modules' default. Compare with the actual fresh
        # walker rather than requiring aliases to appear on every version.
        expected_modules = original_flatten((chart, kernel))
        before = len(calls)
        signature = metadata.fixed_state_signature((chart, kernel))
        module_records = signature[: len(expected_modules)]
        assert all(isinstance(record[1], tuple) for record in module_records)
        assert [record[0] for record in module_records] == [
            id(module) for module in expected_modules
        ]
        assert len(calls) == before + 1
        assert not has_native_guard(kernel)


def test_custom_dict_property_keeps_callbacks_and_uses_python_guard():
    descriptor = torch.nn.Module.__dict__["__dict__"]

    class CustomDictionary(torch.nn.Module):
        @property
        def __dict__(self):
            namespace = descriptor.__get__(self, type(self))
            namespace["_dict_reads"] = namespace.get("_dict_reads", 0) + 1
            return namespace

    chart, kernel = CustomDictionary(), torch.nn.Module()
    counts = []
    for _ in range(3):
        namespace = descriptor.__get__(chart, type(chart))
        before = namespace.get("_dict_reads", 0)
        signature = metadata.fixed_state_signature((chart, kernel))
        assert signature is not None
        counts.append(namespace["_dict_reads"] - before)
        assert not has_native_guard(kernel)
    assert all(count > 0 for count in counts)


def test_constructed_guard_that_fails_qualification_uses_bounded_fallback(monkeypatch):
    from torchcst.operators import _native_metadata_guard

    constructions, checks = [], []

    class RejectingGuard:
        def __init__(self):
            constructions.append(self)

        def check(self, values):
            checks.append(tuple(id(value) for value in values))
            return False

    api = (RejectingGuard, object())
    monkeypatch.setattr(_native_metadata_guard, "guard_api", lambda: api)
    monkeypatch.setattr(
        _native_metadata_guard, "_build_guard", lambda guard, roots, kind: guard
    )
    chart, kernel = roots()
    walks = count_walks(monkeypatch)
    first = metadata.fixed_state_signature((chart, kernel))
    assert not has_native_guard(kernel)
    for _ in range(5):
        assert metadata.fixed_state_signature((chart, kernel)) is first
    assert len(walks) == 6  # All calls still validate through the fresh walker.
    assert len(constructions) == len(checks) == 1
    # Changed configuration warrants one new qualification attempt. A failed
    # accessor never certifies new values or triggers a retry on every hot call.
    kernel.scale.add_(0)
    changed = metadata.fixed_state_signature((chart, kernel))
    assert changed is not first and changed != first
    assert len(constructions) == len(checks) == 2
    assert metadata.fixed_state_signature((chart, kernel)) is changed
    assert len(walks) == 8
    assert len(constructions) == len(checks) == 2
    assert not has_native_guard(kernel)
