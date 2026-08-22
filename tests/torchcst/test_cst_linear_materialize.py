"""CSTLinear materialized backend: parity, auto crossover, lean backward."""

from __future__ import annotations

import pytest
import torch

from torchcst.compute import (
    BackwardContext,
    CSTLinear,
    Factored,
    Materialized,
)
from torchcst.compute.backends.materialize import (
    LeanL2LinearMaterialize,
    LeanLinearMaterialize,
)
from torchcst.representation import (
    GaussianKernel,
    L2NormalizedColumns,
    RepresentationSpec,
)
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
    store, kernel, ref = _parts(backend=Factored())
    mat = CSTLinear(
        ref.in_neurons, ref.out_neurons, store, kernel, backend=Materialized()
    )
    x = torch.randn(5, N_IN, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(5, N_OUT, dtype=torch.float64)
    torch.testing.assert_close(ref(x), mat(x))
    g_ref = _grads_of(ref, x, upstream)
    g_mat = _grads_of(mat, x, upstream)
    for key in g_ref:
        torch.testing.assert_close(g_mat[key], g_ref[key])


def test_lean_materialize_matches_default_path_values_and_grads() -> None:
    store, kernel, ref = _parts(backend=Materialized())
    lean = CSTLinear(
        ref.in_neurons,
        ref.out_neurons,
        store,
        kernel,
        track_mass=False,
        backend=Materialized(lean=True),
    )
    torch.testing.assert_close(ref.dense_weight(), lean.dense_weight())
    x = torch.randn(5, N_IN, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(5, N_OUT, dtype=torch.float64)
    g_ref = _grads_of(ref, x, upstream)
    g_lean = _grads_of(lean, x, upstream)
    for key in g_ref:
        torch.testing.assert_close(g_lean[key], g_ref[key])


@pytest.mark.parametrize("compile_l2", [False, True])
def test_l2_lean_materialize_matches_default_path_values_and_grads(
    compile_l2,
) -> None:
    store, kernel, ref = _parts(
        gauge=L2NormalizedColumns(), track_mass=False, backend=Materialized()
    )
    lean = CSTLinear(
        ref.in_neurons,
        ref.out_neurons,
        store,
        kernel,
        gauge=L2NormalizedColumns(),
        track_mass=False,
        backend=Materialized(lean=True, compile_l2=compile_l2),
    )
    torch.testing.assert_close(ref.dense_weight(), lean.dense_weight())
    x = torch.randn(5, N_IN, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(5, N_OUT, dtype=torch.float64)
    g_ref = _grads_of(ref, x, upstream)
    g_lean = _grads_of(lean, x, upstream)
    for key in g_ref:
        torch.testing.assert_close(g_lean[key], g_ref[key])


def test_l2_lean_uses_the_single_build_path_without_grad(monkeypatch) -> None:
    _, _, lean = _parts(
        gauge=L2NormalizedColumns(),
        track_mass=False,
        backend=Materialized(lean=True),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the chunked autograd Function ran under no_grad")

    monkeypatch.setattr(LeanL2LinearMaterialize, "apply", forbidden)
    with torch.no_grad():
        weight = lean.dense_weight()
    assert weight.shape == (N_OUT, N_IN)


def test_eval_reuses_the_graph_free_weight_until_training_resumes(monkeypatch) -> None:
    _, _, module = _parts(
        gauge=L2NormalizedColumns(),
        track_mass=False,
        backend=Materialized(lean=True),
    )
    original = module.dense_weight
    calls = 0

    def counted_dense_weight():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(module, "dense_weight", counted_dense_weight)
    x = torch.randn(5, N_IN, dtype=torch.float64)
    module.eval()
    with torch.no_grad():
        first = module(x)
        second = module(x)
    torch.testing.assert_close(first, second)
    assert calls == 1

    module.train()
    module.eval()
    with torch.no_grad():
        module(x)
    assert calls == 2


def test_eval_weight_cache_notices_an_in_place_parameter_write(monkeypatch) -> None:
    store, _, module = _parts(
        gauge=L2NormalizedColumns(),
        track_mass=False,
        backend=Materialized(lean=True),
    )
    original = module.dense_weight
    calls = 0

    def counted_dense_weight():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(module, "dense_weight", counted_dense_weight)
    x = torch.randn(5, N_IN, dtype=torch.float64)
    module.eval()
    with torch.no_grad():
        before = module(x)
        store.w.mul_(1.01)
        after = module(x)
    assert calls == 2
    assert not torch.equal(before, after)


def test_lean_chunked_accumulation_is_exact(monkeypatch) -> None:
    # CHUNK smaller than K exercises the multi-chunk accumulation in both
    # directions of the Function.
    store, kernel, lean = _parts(
        track_mass=False, backend=Materialized(lean=True)
    )
    x = torch.randn(5, N_IN, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(5, N_OUT, dtype=torch.float64)
    whole = _grads_of(lean, x, upstream)
    monkeypatch.setattr(LeanLinearMaterialize, "CHUNK", 3)
    chunked = _grads_of(lean, x, upstream)
    for key in whole:
        torch.testing.assert_close(chunked[key], whole[key])


def test_l2_lean_chunked_accumulation_is_exact(monkeypatch) -> None:
    _, _, lean = _parts(
        gauge=L2NormalizedColumns(),
        track_mass=False,
        backend=Materialized(lean=True),
    )
    x = torch.randn(5, N_IN, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(5, N_OUT, dtype=torch.float64)
    whole = _grads_of(lean, x, upstream)
    monkeypatch.setattr(LeanL2LinearMaterialize, "CHUNK", 3)
    chunked = _grads_of(lean, x, upstream)
    for key in whole:
        torch.testing.assert_close(chunked[key], whole[key])


def test_lean_backward_passes_gradcheck() -> None:
    store, kernel, lean = _parts(
        track_mass=False, backend=Materialized(lean=True)
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
        lambda s, t, w, si, so: LeanLinearMaterialize.apply(
            s, t, w, mu_in, mu_out, si, so, None
        ),
        inputs,
    )


def test_l2_lean_backward_passes_gradcheck() -> None:
    _, kernel, lean = _parts(
        gauge=L2NormalizedColumns(),
        track_mass=False,
        backend=Materialized(lean=True),
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
        lambda s, t, w, si, so: LeanL2LinearMaterialize.apply(
            s, t, w, mu_in, mu_out, si, so, None, False
        ),
        inputs,
    )


def test_auto_picks_the_flop_crossover() -> None:
    _, _, module = _parts()  # d_in=4, d_out=3: crossover at K > 12/7
    assert isinstance(module._resolved_backend(1), Factored)
    assert isinstance(module._resolved_backend(2), Materialized)
    _, _, pinned_off = _parts(backend=Factored())
    assert isinstance(pinned_off._resolved_backend(10 ** 6), Factored)
    _, _, pinned_on = _parts(backend=Materialized())
    assert isinstance(pinned_on._resolved_backend(0), Materialized)


def test_materialized_path_still_refreshes_mass() -> None:
    store, kernel, module = _parts(backend=Materialized())
    module(torch.randn(2, N_IN, dtype=torch.float64))
    view = store.view()
    k_in = kernel(module.in_neurons.mu, view.s)
    k_out = kernel(module.out_neurons.mu, view.t)
    expected = view.w.abs() * k_in.norm(dim=0) * k_out.norm(dim=0)
    torch.testing.assert_close(view.mass, expected)


def test_materialized_path_queues_capture() -> None:
    store, kernel, module = _parts(backend=Materialized())
    context = BackwardContext(0)
    module.set_backward_context(context)
    x = torch.randn(2, N_IN, dtype=torch.float64, requires_grad=True)
    module(x).sum().backward()
    assert context.raw_queued == 1


def test_compute_dtype_materialization_close_to_full_precision() -> None:
    store, kernel, module = _parts(
        dtype=torch.float32, backend=Materialized()
    )
    x = torch.randn(4, N_IN)
    full = module(x)
    reduced_module = CSTLinear(
        module.in_neurons,
        module.out_neurons,
        store,
        kernel,
        backend=Materialized(compute_dtype=torch.bfloat16),
    )
    reduced = reduced_module(x)
    rel = ((full - reduced).abs().max() / full.abs().max()).detach()
    assert float(rel) < 5e-2  # bf16 mantissa; GEMM accumulates fp32
    reduced_module(x).square().mean().backward()
    assert store.s.grad is not None  # grads flow through the cast


def test_backend_validation() -> None:
    store, kernel, module = _parts()
    build = lambda **kw: CSTLinear(  # noqa: E731
        module.in_neurons, module.out_neurons, store, kernel, **kw
    )
    with pytest.raises(TypeError, match="backend must be"):
        build(backend="sometimes")
    with pytest.raises(TypeError, match="compute_dtype"):
        Materialized(compute_dtype=torch.int32)
    with pytest.raises(TypeError, match="lean"):
        Materialized(lean=1)
    with pytest.raises(TypeError, match="compile_l2"):
        Materialized(lean=True, compile_l2=1)
    with pytest.raises(ValueError, match="lean=True"):
        Materialized(compile_l2=True)
    with pytest.raises(ValueError, match="track_mass=False"):
        build(backend=Materialized(lean=True))  # track_mass defaults True


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
            backend=Materialized(lean=True),
        )


def test_lean_refuses_a_learnable_chart() -> None:
    """Silence is the failure mode worth guarding.

    The lean backward returns ``None`` for the chart coordinates, and autograd
    reads that as "no gradient" -- so a learnable chart on this path would
    train nothing and report nothing.  It must say so instead.
    """
    from torchcst.compute.backends.materialize import reject_learnable_chart

    frozen = torch.zeros(3, 2)
    learnable = torch.nn.Parameter(torch.zeros(4, 2))

    reject_learnable_chart(frozen, frozen)  # both fixed: nothing to refuse

    with pytest.raises(NotImplementedError, match="lean=False"):
        reject_learnable_chart(frozen, learnable)
    with pytest.raises(NotImplementedError, match="lean=False"):
        reject_learnable_chart(learnable, frozen)


def test_lean_l2_chart_gradient_matches_autograd() -> None:
    """The closed-form backward now carries the chart, exactly.

    ``diff`` is ``mu - coord``: one distance, two endpoints.  The atom's
    gradient sums it over the chart's points and the chart point's sums it
    over the atoms, so the lean path can carry both for one extra reduction
    of a tensor it already built.  A learnable chart therefore does not have
    to fall back to the memory-hungry autograd build.
    """
    from torchcst.compute.backends.materialize import (
        LeanL2LinearMaterialize,
        linear_weight,
        normalized_gaussian_columns,
    )

    torch.manual_seed(0)
    n_in, n_out, atoms, dim = 7, 5, 9, 3
    tensors = {
        "source": torch.randn(atoms, dim, dtype=torch.float64),
        "target": torch.randn(atoms, dim, dtype=torch.float64),
        "weights": torch.randn(atoms, dtype=torch.float64),
        "mu_in": torch.randn(n_in, dim, dtype=torch.float64),
        "mu_out": torch.randn(n_out, dim, dtype=torch.float64),
        "sigma_in": torch.tensor(0.7, dtype=torch.float64),
        "sigma_out": torch.tensor(0.6, dtype=torch.float64),
    }
    for tensor in tensors.values():
        tensor.requires_grad_(True)
    seed = torch.randn(n_out, n_in, dtype=torch.float64)

    def run(lean: bool):
        for tensor in tensors.values():
            tensor.grad = None
        if lean:
            weight = LeanL2LinearMaterialize.apply(
                tensors["source"], tensors["target"], tensors["weights"],
                tensors["mu_in"], tensors["mu_out"],
                tensors["sigma_in"], tensors["sigma_out"], None, False,
            )
        else:
            k_in = normalized_gaussian_columns(
                tensors["mu_in"], tensors["source"], tensors["sigma_in"]
            )
            k_out = normalized_gaussian_columns(
                tensors["mu_out"], tensors["target"], tensors["sigma_out"]
            )
            weight = linear_weight(k_in, k_out, tensors["weights"], None)
        (weight * seed).sum().backward()
        return weight.detach().clone(), {
            name: tensor.grad.clone() for name, tensor in tensors.items()
        }

    lean_weight, lean_grads = run(lean=True)
    plain_weight, plain_grads = run(lean=False)

    torch.testing.assert_close(lean_weight, plain_weight, rtol=0, atol=1e-12)
    for name in tensors:
        torch.testing.assert_close(
            lean_grads[name], plain_grads[name], rtol=1e-9, atol=1e-11,
            msg=lambda text, name=name: f"{name}: {text}",
        )


def test_a_frozen_chart_costs_the_lean_backward_nothing() -> None:
    """The chart gradient is only accumulated when someone asked for it."""
    from torchcst.compute.backends.materialize import LeanL2LinearMaterialize

    torch.manual_seed(1)
    source = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
    target = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(6, dtype=torch.float64, requires_grad=True)
    mu_in = torch.randn(4, 2, dtype=torch.float64)
    mu_out = torch.randn(3, 2, dtype=torch.float64)
    sigma = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    weight = LeanL2LinearMaterialize.apply(
        source, target, weights, mu_in, mu_out, sigma, sigma, None, False
    )
    weight.sum().backward()

    assert mu_in.grad is None and mu_out.grad is None
    assert source.grad is not None
