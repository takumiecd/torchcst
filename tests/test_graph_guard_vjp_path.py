"""The Stage1 inspector must understand real native read-lease autograd edges."""

import os

import pytest
import torch

from benchmarks.training.graph_guard_gate import READ_LEASE, native_read_lease_path
from torchcst.atoms import Atoms, AtomState
from torchcst.atoms import _native_read_lease as native


@pytest.fixture(scope="module")
def factory():
    if (
        not torch.cuda.is_available()
        and os.environ.get("CST_TEST_NATIVE_READ_LEASE") != "1"
    ):
        pytest.skip("explicit native-host compilation or CUDA required")
    result = native.prepare()
    assert result is not None
    return result


def cohort(factory, monkeypatch):
    monkeypatch.setattr(native, "_prepared", factory)
    state = AtomState(Atoms(torch.randn(3, 4)))
    p = state.atoms.p
    x = torch.randn_like(p, requires_grad=True)
    raw = x * state.protect_tensor(p)
    output = state.protect_tensor(raw)
    return state, p, x, raw, output


def test_inspector_follows_real_two_input_native_lease(factory, monkeypatch):
    state, p, x, raw, output = cohort(factory, monkeypatch)
    names, parameter_names = native_read_lease_path(output, p)
    assert names == [READ_LEASE, raw.grad_fn.name()]
    assert parameter_names == [READ_LEASE, "torch::autograd::AccumulateGrad"]
    assert output.grad_fn.next_functions[1] == (None, 0)
    dx, dp = torch.autograd.grad(output.sum(), (x, p))
    torch.testing.assert_close(dx, p, rtol=0, atol=0)
    torch.testing.assert_close(dp, x, rtol=0, atol=0)
    assert not any(reference() is not None for reference in state._readers)


@pytest.mark.parametrize(
    "change", ["extra_wrapper", "missing_lease", "wrong_parameter"]
)
def test_inspector_rejects_an_unexpected_route(factory, monkeypatch, change):
    _state, p, _x, raw, output = cohort(factory, monkeypatch)
    if change == "extra_wrapper":
        output = output.clone()
    elif change == "missing_lease":
        output = raw
    else:
        p = torch.nn.Parameter(p.detach().clone())
    with pytest.raises(AssertionError):
        native_read_lease_path(output, p)
