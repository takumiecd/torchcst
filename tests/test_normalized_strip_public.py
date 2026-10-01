"""Public API checks against an independent dense sampled-site oracle."""

import math

import pytest
import torch

from torchcst._backends.torch.charts import construction as _construction


def public_class():
    from torchcst.nn import NormalizedStripLinear

    return NormalizedStripLinear


def chart(
    sizes=(64, 4, 4), origin=(0.0, 0.0, 0.0), *, dtype=torch.float64, device="cpu"
):
    n, h, j = sizes
    return _construction.strip(
        shape=(n, h * j),
        tile_shape=(n, h * j),
        axis=0,
        tile_pitch=float(n),
        axes=(
            _construction.line_pattern(n, low=origin[0], high=origin[0] + n - 1),
            _construction.grid_pattern(
                (h, j),
                low=origin[1:],
                high=(origin[1] + (h - 1) * 0.5, origin[2] + (j - 1) * 0.5),
            ),
        ),
    ).to(device=device, dtype=dtype)


def oracle(p, sizes=(64, 4, 4), origin=(0.0, 0.0, 0.0), stored_dtype=None):
    # Lift stored FP32 clamp endpoints; high-coordinate CUDA tests use F64 truth.
    bounds = torch.tensor(
        [math.log(0.03), math.log(3.25)], dtype=stored_dtype or p.dtype
    ).to(p)
    sigma = p[:, 1].clamp(bounds[0], bounds[1]).exp()
    sites = torch.cartesian_prod(
        *[
            torch.arange(n, device=p.device, dtype=p.dtype) * s + o
            for n, s, o in zip(sizes, (1.0, 0.5, 0.5), origin)
        ]
    )
    delta = sites[None] - p[:, None, 2:]
    k = (1 - delta.square().sum(-1) / sigma[:, None].square()).clamp_min(0).pow(3)
    norm = torch.linalg.vector_norm(k, dim=1).clamp_min(
        float(torch.tensor(1e-6, dtype=stored_dtype or p.dtype))
    )
    return (p[:, 0, None] * k / norm[:, None]).sum(0).reshape(sizes[0], -1)


def mixed(dtype=torch.float64, device="cpu"):
    return torch.tensor(
        [
            [-0.3, math.log(3.0), 31.25, 0.63, 0.77],
            [0.2, math.log(3.25), 32.5, 0.53, 0.61],
            [0.1, math.log(0.03), 32.02978515625, 1.50355, 1.5],
            [0.0, math.log(0.03), 32.02978515625, 1.50355, 1.5],
            [-0.2, math.log(0.03), 16.0, 1.5, 1.5],
            [0.3, math.log(0.03), 7.5, 1.25, 1.25],
            [0.1, math.log(0.003), 8.0, 0.5, 0.5],
            [-0.1, math.log(10.0), 63.5, 0.5, 0.5],
        ],
        dtype=dtype,
        device=device,
    )


def parameter(model):
    return model.p


@pytest.mark.parametrize("memory", ["full", "window"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_cpu_independent_y_dx_all_five(memory, dtype):
    cls = public_class()
    p = mixed(dtype)
    model = cls(chart(dtype=dtype), p, memory=memory)
    actualp = parameter(model)
    truthp = actualp.detach().double().requires_grad_()
    x = torch.linspace(-0.3, 0.7, 96, dtype=dtype).reshape(2, 3, 16).requires_grad_()
    xx = x.detach().double().requires_grad_()
    dy = torch.linspace(-1, 0.6, 384, dtype=dtype).reshape(2, 3, 64)
    actual = model(x)
    truth = xx @ oracle(truthp, stored_dtype=dtype).T
    ga = torch.autograd.grad(actual, (x, actualp), dy)
    ge = torch.autograd.grad(truth, (xx, truthp), dy.double())
    for a, b in [(actual, truth), *zip(ga, ge)]:
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(
            a.double(),
            b,
            atol=3e-4 if dtype == torch.float32 else 1e-9,
            rtol=3e-4 if dtype == torch.float32 else 1e-9,
        )
    assert ga[1][2, 2:].abs().max() > 0
    assert ga[1][3, 1:].abs().max() == 0 and ga[1][3, 0].abs() > 0
    assert ga[1][4, 1:].abs().max() == 0 and ga[1][5].abs().max() == 0
    assert ga[1][6, 1] == 0 and ga[1][7, 1] == 0


def test_default_full_checkpoint_and_arbitrary_leading_dimensions():
    cls = public_class()
    p = mixed()
    a = cls(chart(), p)
    b = cls(chart(), p, memory="full")
    b.load_state_dict(a.state_dict())
    for shape in [(16,), (3, 16), (2, 3, 16)]:
        x = torch.linspace(-1, 1, math.prod(shape), dtype=p.dtype).reshape(shape)
        torch.testing.assert_close(a(x), b(x), atol=0, rtol=0)
        assert a(x).shape == (*shape[:-1], 64)
    assert set(a.state_dict()) == set(b.state_dict())


@pytest.mark.parametrize("memory", ["bad", "", None])
def test_invalid_memory_deterministic(memory):
    with pytest.raises((ValueError, TypeError), match="memory"):
        public_class()(chart(), mixed(), memory=memory)


@pytest.mark.parametrize("shape", [(2, 4), (2, 6), (5,)])
def test_invalid_parameter_shape(shape):
    with pytest.raises((ValueError, TypeError), match="(shape|atom|parameter|five|5)"):
        public_class()(chart(), torch.zeros(shape, dtype=torch.float64))


def test_invalid_input_feature_shape_and_dtype():
    model = public_class()(chart(), mixed())
    with pytest.raises(
        (ValueError, RuntimeError), match="(feature|shape|column|dimension|input)"
    ):
        model(torch.zeros(2, 15, dtype=torch.float64))
    with pytest.raises(
        (ValueError, TypeError, RuntimeError),
        match="(dtype|floating|Float|Double|type)",
    ):
        model(torch.zeros(2, 16, dtype=torch.int64))


def test_empty_atoms_zero_operator():
    model = public_class()(chart(), torch.empty(0, 5, dtype=torch.float64))
    x = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
    y = model(x)
    assert torch.equal(y, torch.zeros_like(y))
    y.sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))


def test_owns_parameter_storage_and_dtype_mismatch():
    p = mixed()
    model = public_class()(chart(), p)
    old = model.p.detach().clone()
    p.zero_()
    assert torch.equal(model.p, old)
    assert model.p.data_ptr() != p.data_ptr()
    with pytest.raises(
        (ValueError, TypeError, RuntimeError), match="(dtype|Float|Double|type)"
    ):
        model(torch.zeros(2, 16, dtype=torch.float32))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for device mismatch"
)
def test_input_device_mismatch():
    model = public_class()(chart(), mixed())
    with pytest.raises((ValueError, RuntimeError), match="device"):
        model(torch.zeros(2, 16, dtype=torch.float64, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("memory", ["full", "window"])
def test_cuda_public_window_boundary_independent_all_five(memory):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        sizes = (1024, 4, 4)
        p = mixed(torch.float32, "cuda")
        p[0, 2] = 511.25
        p[1, 2] = 512.5
        p[2:4, 2] = 512.02978515625
        model = public_class()(
            chart(sizes, dtype=torch.float32, device="cuda"), p, memory=memory
        )
        g = torch.Generator(device="cuda").manual_seed(21)
        x = torch.randn(2, 3, 16, device="cuda", generator=g, requires_grad=True)
        dy = torch.randn(2, 3, 1024, device="cuda", generator=g)
        pp = model.p.detach().double().requires_grad_()
        xx = x.detach().double().requires_grad_()
        weight = oracle(pp, sizes, stored_dtype=torch.float32)
        observed_weight = model(torch.eye(16, device="cuda")).T
        torch.testing.assert_close(
            observed_weight.double(), weight, atol=2e-5, rtol=2e-5
        )
        actual = model(x)
        truth = xx @ weight.T
        a = torch.autograd.grad(actual, (x, model.p), dy)
        b = torch.autograd.grad(truth, (xx, pp), dy.double())
        for left, right in [(actual, truth), *zip(a, b)]:
            assert torch.isfinite(left).all() and torch.isfinite(right).all()
            torch.testing.assert_close(left.double(), right, atol=3e-4, rtol=3e-4)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_parameter_rejected(value):
    p = mixed()
    p[0, 2] = value
    with pytest.raises(ValueError, match="finite"):
        public_class()(chart(), p)


def test_dtype_roundtrip_keeps_fixed_bounds_and_checkpoint():
    model = public_class()(chart(), mixed()).float().double()
    torch.testing.assert_close(
        model.sigma_bounds,
        torch.tensor([0.03, 3.25], dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    x = torch.linspace(-1, 1, 32, dtype=torch.float64).reshape(2, 16)
    restored = public_class()(chart(), model.p.detach(), memory="full")
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_rejects_autocast_and_tf32():
    model = public_class()(
        chart(dtype=torch.float32, device="cuda"), mixed(torch.float32, "cuda")
    )
    x = torch.zeros(2, 16, device="cuda")
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with (
            torch.autocast("cuda", dtype=torch.float16),
            pytest.raises((ValueError, RuntimeError), match="autocast"),
        ):
            model(x)
        torch.backends.cuda.matmul.allow_tf32 = True
        with pytest.raises((ValueError, RuntimeError), match="TF32"):
            model(x)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_replaced_metadata_buffer_refreshes_geometry_and_validates_contract():
    model = public_class()(chart(), mixed())
    x = torch.linspace(-1, 1, 32, dtype=torch.float64).reshape(2, 16)
    before = model(x).detach().clone()
    model.origin = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    actual = model(x)
    expected = x @ oracle(model.p, origin=(1.0, 0.0, 0.0)).T
    assert not torch.equal(actual, before)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    model.sigma_bounds = torch.tensor([0.01, 3.25], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="sigma bounds"):
        model(x)
