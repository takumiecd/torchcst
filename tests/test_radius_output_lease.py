"""Release-time protection uses the existing saved graph, including hook errors."""

import gc
import hashlib
import os
import weakref
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from torchcst.atoms import Atoms, AtomState
from torchcst.atoms import _native_read_lease as leases

# CPU arithmetic stands in only for the CUDA stage calls. These tests validate
# the actual host graph/ownership code; CUDA profile mathematics has separate
# independent-oracle tests, not this matrix-product stand-in.
STUB = r"""
#include <torch/extension.h>
namespace radius_cell_mass {
using Tensor=torch::Tensor; using Config=std::vector<double>;
Tensor classify_atoms_async(Tensor p,Config,Config) {
    return torch::zeros({p.size(0)},p.options().dtype(torch::kLong));
}
std::vector<Tensor> autograd_ranges(Tensor k,Config,Config,bool) { return {k,k}; }
std::vector<Tensor> autograd_transform(Tensor x,Tensor p,Tensor,Tensor,
    Config,Config,int,int,bool,int,int,bool reverse) {
    return {reverse ? x.matmul(p) : x.matmul(p.t()),x.unsqueeze(1),x};
}
std::vector<Tensor> autograd_project(Tensor dy,Tensor,Config,int,bool,int) {
    return {dy.unsqueeze(1),dy};
}
Tensor autograd_parameter_grad(Tensor,Tensor x,Tensor dy,Tensor,Tensor,
    Tensor,Tensor,Config,Config,int,int,int) { return dy.t().matmul(x); }
void bind_host_autograd(pybind11::module_ &);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { radius_cell_mass::bind_host_autograd(m); }
"""


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    if not torch.cuda.is_available() and os.getenv("CST_TEST_NATIVE_READ_LEASE") != "1":
        pytest.skip("explicit optional native-host build or CUDA required")
    from torch.utils.cpp_extension import load

    source = (
        Path(__file__).resolve().parents[1]
        / "src/torchcst/_backends/cuda/algorithms/linear/radius_cell_mass/host_autograd.cpp"
    )
    stub = tmp_path_factory.mktemp("host-lease") / "arithmetic.cpp"
    stub.write_text(STUB)
    digest = hashlib.sha256(source.read_bytes() + STUB.encode()).hexdigest()[:12]
    native = load(
        name="torchcst_host_lease_test_" + digest,
        sources=[str(source), str(stub)],
        with_cuda=False,
        extra_cflags=["-O2"],
    )
    proof = leases.prepare()
    assert proof is not None
    return native, proof


@pytest.fixture
def owned(host, monkeypatch):
    native, proof = host
    monkeypatch.setattr(leases, "_completion_guard", native.radius_lease_guard(proof))
    monkeypatch.setattr(leases, "_completion_factory", proof)
    return native, proof


def forward(owned, state, x):
    native, proof = owned
    return native.autograd_linear_leased(
        x,
        state.parameters_for_execution(),
        [1.0],
        [1.0],
        0,
        0,
        True,
        32,
        1,
        state,
        proof,
    )


@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (True, False), (False, True)]
)
@pytest.mark.parametrize("offload", [False, True])
def test_owned_graph_vjps_saved_hooks_and_retained_normal_release(
    owned, needs_x, needs_p, offload
):
    state = AtomState(Atoms(torch.randn(3, 4, dtype=torch.float64)))
    state.atoms.p.requires_grad_(needs_p)
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=needs_x)

    def pack(t):
        return t.detach().clone() if offload else t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = forward(owned, state, x)
        assert y.grad_fn.name() == "RadiusLeaseNode"
        assert state.protect_tensor(y) is y
    selected = tuple(t for t in (x, state.atoms.p) if t.requires_grad)
    actual = torch.autograd.grad(y.square().sum(), selected, retain_graph=True)
    expected = torch.autograd.grad((x @ state.atoms.p.T).square().sum(), selected)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="backward"):
        state.relayout(torch.tensor([2, 0, 1]))
    torch.autograd.grad(y.square().sum(), selected)
    assert not any(r() is not None for r in state._readers)
    with pytest.raises(RuntimeError, match="second time|freed"):
        torch.autograd.grad(y.square().sum(), selected)
    state.relayout(torch.tensor([2, 0, 1]))


def test_hook_error_does_not_pretend_saved_variables_were_released(owned):
    state = AtomState(Atoms(torch.randn(3, 4, dtype=torch.float64)))
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    y = forward(owned, state, x)

    def fail(*args):
        raise RuntimeError("deliberate hook failure")

    handle = y.grad_fn.register_hook(fail)
    with pytest.raises(RuntimeError, match="deliberate hook failure"):
        y.sum().backward()
    with pytest.raises(RuntimeError, match="backward"):
        state.relayout(torch.tensor([2, 0, 1]))
    handle.remove()
    y.sum().backward()
    state.relayout(torch.tensor([2, 0, 1]))


def test_abandoned_graph_releases_numerical_resources_even_with_weak_observers(owned):
    state = AtomState(Atoms(torch.randn(3, 4, dtype=torch.float64)))
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    y = forward(owned, state, x)
    references = [weakref.ref(state), weakref.ref(state.atoms.p), weakref.ref(x)]
    # Keep its reader list alive: weak graph observers must not keep the inner
    # node's numeric caches alive on intrusive_ptr versions of PyTorch either.
    readers = state._readers
    del state, x, y
    gc.collect()
    assert all(r() is None for r in references)
    assert all(r() is None for r in readers)


def test_other_owner_and_changed_reader_callbacks_keep_identity_fallback(
    owned, monkeypatch
):
    from torchcst.atoms import state as module

    state = AtomState(Atoms(torch.randn(3, 4, dtype=torch.float64)))
    other = AtomState(Atoms(torch.randn(3, 4, dtype=torch.float64)))
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    y = forward(owned, state, x)
    assert other.protect_tensor(y) is not y
    original = module._ReadParameter.forward
    calls = []

    def custom(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(module._ReadParameter, "forward", staticmethod(custom))
    assert state.protect_tensor(y) is not y
    assert calls == [True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (True, False), (False, True)]
)
@pytest.mark.parametrize("offload", [False, True])
def test_public_cuda_owned_lease_all_gradients_and_placement_boundary(
    monkeypatch, needs_x, needs_p, offload
):
    import copy

    from benchmarks.cuda.linear.manifest import load_run
    from benchmarks.cuda.linear.radius_cell_mass import configure, fixture
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
        BLOCK_LOCAL_CACHED_PARALLEL,
    )

    monkeypatch.setattr(leases, "_completion_guard", leases._completion_guard)
    monkeypatch.setattr(leases, "_completion_factory", leases._completion_factory)
    root = Path(__file__).resolve().parents[1]
    case = load_run(
        root / "benchmarks/cuda/linear/cases/radius-cell-mass-128-initial.json",
        root / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    ).case
    cpu, x, _, dy = fixture(replace(case, atoms=8, rows=2))
    cpu = cpu.double()
    cpu.atoms.p.requires_grad_(needs_p)
    gpu = copy.deepcopy(cpu).cuda()
    plan = replace(
        BLOCK_LOCAL_CACHED_PARALLEL,
        recipe=replace(
            BLOCK_LOCAL_CACHED_PARALLEL.recipe,
            native_autograd=True,
            native_read_lease=True,
        ),
    )
    configure(gpu, plan)
    x = x.double().requires_grad_(needs_x)
    gx = x.detach().cuda().requires_grad_(needs_x)
    expected = cpu(x)
    selected = tuple(t for t in (x, cpu.atoms.p) if t.requires_grad)
    target = torch.autograd.grad(expected, selected, dy.double())

    def pack(t):
        return (t.detach().cpu().clone(), t.device) if offload else t

    def unpack(t):
        return t[0].to(t[1]) if offload else t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = gpu(gx)
    gpu_selected = tuple(t for t in (gx, gpu.atoms.p) if t.requires_grad)
    for retain in (True, False):
        actual = torch.autograd.grad(
            y, gpu_selected, dy.double().cuda(), retain_graph=retain
        )
        for a, b in zip((y, *actual), (expected, *target)):
            torch.testing.assert_close(a.cpu(), b, atol=2e-9, rtol=2e-8)
        if retain:
            with pytest.raises(RuntimeError, match="backward"):
                gpu.atom_state.relayout(torch.arange(7, -1, -1))
    gpu.atom_state.relayout(torch.arange(7, -1, -1))
