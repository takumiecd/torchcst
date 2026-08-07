"""CSTLinear materialized-path contracts: parity, auto crossover, lean backward."""

from __future__ import annotations

import pytest
import torch

from torchcst.compute import BackwardContext, CSTLinear
from torchcst.compute.cst_linear import _LeanLinearMaterialize
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

N_IN, N_OUT, N_ATOMS = 4, 3, 10


def _parts(
    *,
    dtype: torch.dtype = torch.float64,
    kernel_name: str = "gaussian",
    **module_kw,
) -> tuple[SynapseStore, GaussianKernel, CSTLinear]:
    gen = torch.Generator().manual_seed(0)
    store = SynapseStore(
        "continuous",
        2,
        2,
        N_ATOMS,
        spec=RepresentationSpec.continuous(
            2, 2, bounds=(-1.0, 1.0), kernel=kernel_name
        ),
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
    kernel = GaussianKernel(0.55).to(dtype)
    return store, kernel, CSTLinear(inputs, outputs, store, kernel, **module_kw)


def _grads_of(module, x, upstream):
    module.zero_grad()
    if x.grad is not None:
        x.grad = None
    module(x).backward(upstream)
    store = module.synapses
    return {
        "w": store.w.grad.clone(),
        "s": store.s.grad.clone(),
        "t": store.t.grad.clone(),
        "sigma": module.kernel_in.sigma.grad.clone(),
        "x": x.grad.clone(),
    }


def test_materialized_forward_matches_factored_values_and_grads() -> None:
    store, kernel, ref = _parts(materialize=False)
    mat = CSTLinear(
        ref.in_neurons, ref.out_neurons, store, kernel, materialize=True
    )
    x = torch.randn(5, N_IN, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(5, N_OUT, dtype=torch.float64)
    torch.testing.assert_close(ref(x), mat(x))
    g_ref = _grads_of(ref, x, upstream)
    g_mat = _grads_of(mat, x, upstream)
    for key in g_ref:
        torch.testing.assert_close(g_mat[key], g_ref[key])


def test_lean_materialize_matches_default_path_values_and_grads() -> None:
    store, kernel, ref = _parts(materialize=True)
    lean = CSTLinear(
        ref.in_neurons,
        ref.out_neurons,
        store,
        kernel,
        track_mass=False,
        materialize=True,
        lean_materialize=True,
    )
    torch.testing.assert_close(ref.dense_weight(), lean.dense_weight())
    x = torch.randn(5, N_IN, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(5, N_OUT, dtype=torch.float64)
    g_ref = _grads_of(ref, x, upstream)
    g_lean = _grads_of(lean, x, upstream)
    for key in g_ref:
        torch.testing.assert_close(g_lean[key], g_ref[key])


def test_lean_chunked_accumulation_is_exact(monkeypatch) -> None:
    # CHUNK smaller than K exercises the multi-chunk accumulation in both
    # directions of the Function.
    store, kernel, lean = _parts(
        track_mass=False, materialize=True, lean_materialize=True
    )
    x = torch.randn(5, N_IN, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(5, N_OUT, dtype=torch.float64)
    whole = _grads_of(lean, x, upstream)
    monkeypatch.setattr(_LeanLinearMaterialize, "CHUNK", 3)
    chunked = _grads_of(lean, x, upstream)
    for key in whole:
        torch.testing.assert_close(chunked[key], whole[key])


def test_lean_backward_passes_gradcheck() -> None:
    store, kernel, lean = _parts(
        track_mass=False, materialize=True, lean_materialize=True
    )
    lean._view()
    source, target, weights = lean._live_factors()
    mu_in = lean.in_neurons.mu.clone()
    mu_out = lean.out_neurons.mu.clone()
    inputs = tuple(
        t.detach().clone().requires_grad_(True)
        for t in (source, target, weights, kernel.sigma, kernel.sigma.clone())
    )
    assert torch.autograd.gradcheck(
        lambda s, t, w, si, so: _LeanLinearMaterialize.apply(
            s, t, w, mu_in, mu_out, si, so, None
        ),
        inputs,
    )


def test_auto_picks_the_flop_crossover() -> None:
    _, _, module = _parts()  # d_in=4, d_out=3: crossover at K > 12/7
    assert module._materialize_now(1) is False
    assert module._materialize_now(2) is True
    _, _, pinned_off = _parts(materialize=False)
    assert pinned_off._materialize_now(10 ** 6) is False
    _, _, pinned_on = _parts(materialize=True)
    assert pinned_on._materialize_now(0) is True


def test_materialized_path_still_refreshes_mass() -> None:
    store, kernel, module = _parts(materialize=True)
    module(torch.randn(2, N_IN, dtype=torch.float64))
    view = store.view()
    k_in = kernel(module.in_neurons.mu, view.s)
    k_out = kernel(module.out_neurons.mu, view.t)
    expected = view.w.abs() * k_in.norm(dim=0) * k_out.norm(dim=0)
    torch.testing.assert_close(view.mass, expected)


def test_materialized_path_queues_capture() -> None:
    store, kernel, module = _parts(materialize=True)
    context = BackwardContext(0)
    module.set_backward_context(context)
    x = torch.randn(2, N_IN, dtype=torch.float64, requires_grad=True)
    module(x).sum().backward()
    assert context.raw_queued == 1


def test_compute_dtype_materialization_close_to_full_precision() -> None:
    store, kernel, module = _parts(dtype=torch.float32, materialize=True)
    x = torch.randn(4, N_IN)
    full = module(x)
    module.compute_dtype = torch.bfloat16
    reduced = module(x)
    rel = ((full - reduced).abs().max() / full.abs().max()).detach()
    assert float(rel) < 5e-2  # bf16 mantissa; GEMM accumulates fp32
    module(x).square().mean().backward()
    assert store.s.grad is not None  # grads flow through the cast


def test_validation() -> None:
    store, kernel, module = _parts()
    build = lambda **kw: CSTLinear(  # noqa: E731
        module.in_neurons, module.out_neurons, store, kernel, **kw
    )
    with pytest.raises(ValueError, match="materialize must be"):
        build(materialize="sometimes")
    with pytest.raises(TypeError, match="compute_dtype"):
        build(compute_dtype=torch.int32)
    with pytest.raises(ValueError, match="track_mass=False"):
        build(lean_materialize=True)  # track_mass defaults True
    with pytest.raises(ValueError, match="materialized path"):
        build(track_mass=False, materialize=False, lean_materialize=True)


def test_lean_requires_gaussian_kernels() -> None:
    from torchcst.representation import TriangularKernel

    gen = torch.Generator().manual_seed(0)
    store = SynapseStore(
        "continuous",
        2,
        2,
        N_ATOMS,
        spec=RepresentationSpec.continuous(
            2, 2, bounds=(-1.0, 1.0), kernel="triangular"
        ),
        dtype=torch.float64,
    )
    inputs = NeuronStore(
        "inputs",
        N_IN,
        mu=torch.rand(N_IN, 2, generator=gen, dtype=torch.float64) * 2 - 1,
        initial_live=N_IN,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "outputs",
        N_OUT,
        mu=torch.rand(N_OUT, 2, generator=gen, dtype=torch.float64) * 2 - 1,
        initial_live=N_OUT,
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="Gaussian"):
        CSTLinear(
            inputs,
            outputs,
            store,
            TriangularKernel(0.55).double(),
            track_mass=False,
            lean_materialize=True,
        )
