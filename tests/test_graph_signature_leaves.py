"""CPU shadow, raw topology and lifetime requirements of three G1 columns."""

import gc
import os
import weakref
from types import MethodType, SimpleNamespace

import pytest
import torch

from benchmarks.training import graph_signature_leaves as leaves
from benchmarks.training.full_step_graph import metadata


def columns(model):
    return (
        tuple((name, metadata(p)) for name, p in model.named_parameters()),
        tuple(
            (name, metadata(b), b._version)
            for name, b in model.named_buffers()
            if ".optimizer_state." not in name
        ),
        tuple(
            (id(m), type(m), m.training, m.__dict__.get("forward"), type(m).forward)
            for m in model.modules()
        ),
    )


@pytest.fixture
def certified():
    if (
        not torch.cuda.is_available()
        and os.environ.get("CST_TEST_NATIVE_FUNCTION_GUARD") != "1"
    ):
        pytest.skip("explicit CPU native-host qualification or CUDA required")
    assert leaves.host.get_factory() is not None
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)
    )
    model[0].register_buffer("constant", torch.ones(2))
    return model, leaves.GraphSignatureLeaves(model), columns(model)


def test_accepts_numeric_learning_without_mutating_state(certified):
    model, certificate, before = certified
    for p in model.parameters():
        with torch.no_grad():
            p.add_(0.25)
    versions = tuple(p._version for p in model.parameters())
    assert certificate.check() and columns(model) == before
    assert tuple(p._version for p in model.parameters()) == versions


@pytest.mark.parametrize(
    "change",
    [
        "storage",
        "shape",
        "stride",
        "dtype",
        "requires_grad",
        "buffer_version",
        "training",
        "instance_forward",
        "class_forward",
        "parameter_none",
        "buffer_none",
        "module_none",
        "alias",
        "parameter_order",
        "dict_rebind",
        "module_dict_rebind",
    ],
)
def test_changes_reject_and_never_reanchor(certified, change, monkeypatch):
    model, certificate, before = certified
    p = model[0].weight
    if change == "storage":
        p.data = p.data.clone()
    elif change == "shape":
        p.data = p.data.reshape(1, 9)
    elif change == "stride":
        p.data = p.data.t()
    elif change == "dtype":
        p.data = p.data.double()
    elif change == "requires_grad":
        p.requires_grad_(False)
    elif change == "buffer_version":
        model[0].constant.add_(1)
    elif change == "training":
        model.eval()
    elif change == "instance_forward":
        model[0].forward = lambda x: x
    elif change == "class_forward":
        monkeypatch.setattr(torch.nn.Linear, "forward", lambda self, x: x)
    elif change == "parameter_none":
        model[0]._parameters["bias"] = None
    elif change == "buffer_none":
        model[0]._buffers["constant"] = None
    elif change == "module_none":
        model._modules["1"] = None
    elif change == "alias":
        model._modules["1"] = model[0]
    elif change == "parameter_order":
        values = model[0]._parameters
        values["weight"] = values.pop("weight")
    elif change == "dict_rebind":
        model[0]._parameters = dict(model[0]._parameters)
    else:
        model[0].__dict__ = dict(model[0].__dict__)
        model[0]._parameters = dict(model[0]._parameters)
    assert not certificate.check()
    assert not certificate.check()
    if change not in ("dict_rebind", "module_dict_rebind"):
        assert columns(model) != before


def test_all_raw_alias_and_none_edges_with_first_path_dedup(certified):
    shared = torch.nn.Linear(2, 2)
    shared.register_buffer("none", None)
    model = torch.nn.Sequential(shared, shared)
    model.add_module("empty", None)
    certificate = leaves.GraphSignatureLeaves(model)
    assert certificate.parameter_names == ("0.weight", "0.bias")
    assert certificate.module_names == ("", "0")
    before = columns(model)
    model._modules["1"] = None
    assert columns(model) == before  # Legacy dedup does not see this raw edge.
    assert not certificate.check()


@pytest.mark.parametrize("optimizer_first", [True, False])
def test_buffer_filter_follows_first_path_before_dedup(certified, optimizer_first):
    model = torch.nn.Module()
    core, other = torch.nn.Module(), torch.nn.Module()
    core.optimizer_state = torch.nn.Module()
    value = torch.ones(2)
    core.optimizer_state.register_buffer("shared", value)
    other.register_buffer("shared", value)
    for key, module in (
        (("core", core), ("other", other))
        if optimizer_first
        else (("other", other), ("core", core))
    ):
        model.add_module(key, module)
    certificate = leaves.GraphSignatureLeaves(model)
    expected = tuple(
        name for name, _ in model.named_buffers() if ".optimizer_state." not in name
    )
    assert certificate.buffer_names == expected
    before = columns(model)
    value.add_(1)
    assert certificate.check() is optimizer_first
    assert (columns(model) == before) is optimizer_first


@pytest.mark.parametrize("field", ["_parameters", "_buffers", "_modules"])
def test_removed_registration_dict_and_examples_are_not_owned(certified, field):
    model = torch.nn.Module()
    value = (
        torch.nn.Linear(2, 2)
        if field == "_modules"
        else torch.nn.Parameter(torch.ones(2))
        if field == "_parameters"
        else torch.ones(2)
    )
    model.__dict__[field]["value"] = value
    certificate = leaves.GraphSignatureLeaves(model)
    reference = weakref.ref(value)
    model.__dict__[field] = {}
    del value
    gc.collect()
    assert reference() is None
    assert not certificate.check()


def test_root_and_bound_forward_are_not_owned(certified):
    model = torch.nn.Linear(2, 2)
    model.forward = MethodType(lambda self, x: x, model)
    refs = [weakref.ref(model), weakref.ref(model.weight), weakref.ref(model.forward)]
    certificate = leaves.GraphSignatureLeaves(model)
    assert certificate.check()
    del model
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert not certificate.check()


def test_owner_dict_rebind_detected_while_old_dict_lives(certified):
    model, certificate, before = certified
    old = model[0]._parameters
    model[0]._parameters = dict(old)
    assert old["weight"] is model[0].weight and columns(model) == before
    assert not certificate.check()


def test_storage_override_is_rejected_without_calling_it(certified):
    model, certificate, _ = certified
    calls = []
    model[0].weight.data_ptr = lambda: calls.append(True) or 0
    assert not certificate.check() and not calls


def test_small_cpu_cst_metadata_and_weak_lifetime(certified):
    from benchmarks.cuda.linear.radius_cell_mass import fixture

    model, *_ = fixture(SimpleNamespace(size=1024, atoms=3, rows=1, seed=41))
    before = columns(model)
    certificate = leaves.GraphSignatureLeaves(model)
    assert certificate.check() and columns(model) == before
    assert certificate.parameter_names == tuple(
        name for name, _ in model.named_parameters()
    )
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None and not certificate.check()


def test_unavailable_is_explicit_not_success(monkeypatch):
    monkeypatch.setattr(leaves, "guard_api", lambda: None)
    with pytest.raises(leaves.LeavesUnavailable, match="RootGuardManager"):
        leaves.GraphSignatureLeaves(torch.nn.Linear(2, 2))


def test_tensor_match_unavailable_is_explicit(certified, monkeypatch):
    monkeypatch.setattr(leaves, "tensor_metadata_guard", lambda *args: False)
    with pytest.raises(leaves.LeavesUnavailable, match="TensorMatch"):
        leaves.GraphSignatureLeaves(torch.nn.Linear(2, 2))
