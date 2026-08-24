"""CSTLinear native truncated path: exactness, retraction, approximation."""

from __future__ import annotations

import pytest
import torch

from torchcst.compute import CSTLinear, Factored, NativeTruncated
from torchcst.compute.backends.native import NativeTruncatedFunction
from torchcst.representation import GaussianFactor, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

N_IN, N_OUT, N_ATOMS = 6, 5, 11
RADIUS = 1.6


def _parts(**module_kw):
    gen = torch.Generator().manual_seed(3)
    dtype = torch.float64
    store = SynapseStore(
        "continuous",
        2,
        2,
        N_ATOMS,
        spec=RepresentationSpec.continuous(2, 2, bounds=(-1.0, 1.0)),
        dtype=dtype,
    )
    store.apply(
        [
            SynapseBirth(
                "continuous",
                torch.rand(N_ATOMS, 2, generator=gen, dtype=dtype) * 2 - 1,
                torch.rand(N_ATOMS, 2, generator=gen, dtype=dtype) * 2 - 1,
                torch.randn(N_ATOMS, generator=gen, dtype=dtype),
                torch.arange(N_ATOMS, dtype=torch.int64),
            )
        ]
    )
    inputs = NeuronStore(
        "inputs",
        N_IN,
        mu=torch.rand(N_IN, 2, generator=gen, dtype=dtype) * 2 - 1,
        initial_live=N_IN,
        dtype=dtype,
    )
    outputs = NeuronStore(
        "outputs",
        N_OUT,
        mu=torch.rand(N_OUT, 2, generator=gen, dtype=dtype) * 2 - 1,
        initial_live=N_OUT,
        dtype=dtype,
    )
    factor = GaussianFactor(0.35).double()
    return store, factor, CSTLinear(inputs, outputs, store, factor, **module_kw)


def _oracle_forward(module, x, radius):
    """Brute-force truncated map built independently of the native path."""
    store = module.synapses
    view = store.view()
    sigma = module.factor_in.sigma
    k_in = module.factor_in(module.in_neurons.mu, store.s)
    k_out = module.factor_out(module.out_neurons.mu, store.t)
    d2_in = (store.s[None] - module.in_neurons.mu[:, None]).square().sum(-1)
    d2_out = (store.t[None] - module.out_neurons.mu[:, None]).square().sum(-1)
    k_in = k_in * (d2_in <= (radius * sigma).square()).detach()
    k_out = k_out * (d2_out <= (radius * sigma).square()).detach()
    del view
    return ((x @ k_in) * store.w) @ k_out.transpose(0, 1)


def _grads_of(forward, module, x, upstream):
    module.zero_grad()
    if x.grad is not None:
        x.grad = None
    forward(x).backward(upstream)
    store = module.synapses
    return {
        "w": store.w.grad.clone(),
        "s": store.s.grad.clone(),
        "t": store.t.grad.clone(),
        "sigma": module.factor_in.sigma.grad.clone(),
        "x": x.grad.clone(),
    }


def test_native_matches_bruteforce_truncation_values_and_grads() -> None:
    store, factor, module = _parts(
        track_mass=False, backend=NativeTruncated(radius=RADIUS)
    )
    x = torch.randn(7, N_IN, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(7, N_OUT, dtype=torch.float64)
    torch.testing.assert_close(
        module(x), _oracle_forward(module, x, RADIUS)
    )
    g_native = _grads_of(module, module, x, upstream)
    g_oracle = _grads_of(
        lambda inp: _oracle_forward(module, inp, RADIUS), module, x, upstream
    )
    for key in g_oracle:
        torch.testing.assert_close(g_native[key], g_oracle[key])


def test_native_chunked_accumulation_is_exact(monkeypatch) -> None:
    store, factor, module = _parts(track_mass=False, backend=NativeTruncated(radius=RADIUS))
    x = torch.randn(4, N_IN, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(4, N_OUT, dtype=torch.float64)
    whole = _grads_of(module, module, x, upstream)
    monkeypatch.setattr(NativeTruncatedFunction, "CHUNK", 3)
    chunked = _grads_of(module, module, x, upstream)
    for key in whole:
        torch.testing.assert_close(chunked[key], whole[key])


def test_wide_radius_recovers_the_full_map() -> None:
    store, factor, ref = _parts(backend=Factored())
    native = CSTLinear(
        ref.in_neurons, ref.out_neurons, store, factor,
        track_mass=False, backend=NativeTruncated(radius=100.0),
    )
    x = torch.randn(5, N_IN, dtype=torch.float64)
    torch.testing.assert_close(native(x), ref(x))


def test_tight_radius_is_a_bounded_approximation() -> None:
    store, factor, ref = _parts(backend=Factored())
    native = CSTLinear(
        ref.in_neurons, ref.out_neurons, store, factor,
        track_mass=False, backend=NativeTruncated(radius=3.0),
    )
    x = torch.randn(5, N_IN, dtype=torch.float64)
    full = ref(x)
    rel = ((native(x) - full).norm() / full.norm()).detach()
    assert float(rel) < 2e-2  # dropped tails are exp(-4.5) per entry


def test_out_of_support_atom_contributes_nothing_and_gets_no_pull() -> None:
    store, factor, module = _parts(track_mass=False, backend=NativeTruncated(radius=RADIUS))
    with torch.no_grad():
        store.s[0] = torch.tensor([50.0, 50.0], dtype=torch.float64)
        store.t[0] = torch.tensor([50.0, 50.0], dtype=torch.float64)
    x = torch.randn(4, N_IN, dtype=torch.float64)
    module(x).sum().backward()
    # Retraction semantics: an atom outside every neuron's support is
    # invisible to the map and feels no position gradient from it.
    assert torch.all(store.s.grad[0] == 0.0)
    assert torch.all(store.t.grad[0] == 0.0)
    assert float(store.w.grad[0]) == 0.0


def test_batched_rows_reshape_round_trip() -> None:
    store, factor, module = _parts(track_mass=False, backend=NativeTruncated(radius=RADIUS))
    x = torch.randn(2, 3, N_IN, dtype=torch.float64)
    out = module(x)
    assert out.shape == (2, 3, N_OUT)
    torch.testing.assert_close(
        out.reshape(-1, N_OUT), module(x.reshape(-1, N_IN))
    )


def test_native_validation() -> None:
    store, factor, module = _parts()
    with pytest.raises(ValueError, match="positive"):
        NativeTruncated(radius=0.0)
    with pytest.raises(TypeError, match="number"):
        NativeTruncated(radius=True)
    with pytest.raises(ValueError, match="track_mass=False"):
        CSTLinear(
            module.in_neurons, module.out_neurons, store, factor,
            backend=NativeTruncated(radius=2.0),
        )
