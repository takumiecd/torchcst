"""Public API checks against an independent dense sampled-site oracle."""

import math

import pytest
import torch

from benchmarks.cuda.linear.fixtures import normalized_chart as chart
from benchmarks.cuda.linear.reference import mixed, oracle
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip import REGISTRY
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.plans import (
    FULL,
    WINDOW,
)
from torchcst._backends.cuda.dispatch import FixedSelector


def public_class():
    from torchcst import CSTLinear, presets

    def build(chart, p, *, selector=None):
        return CSTLinear(
            chart=chart,
            atoms=p,
            kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
            selector=selector,
        )

    return build


def parameter(model):
    return model.atoms.p


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_cpu_independent_y_dx_all_five(dtype):
    cls = public_class()
    p = mixed(dtype)
    model = cls(chart(dtype=dtype), p)
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


def test_checkpoint_and_arbitrary_leading_dimensions():
    cls = public_class()
    p = mixed()
    a = cls(chart(), p)
    b = cls(chart(), p)
    b.load_state_dict(a.state_dict())
    for shape in [(16,), (3, 16), (2, 3, 16)]:
        x = torch.linspace(-1, 1, math.prod(shape), dtype=p.dtype).reshape(shape)
        torch.testing.assert_close(a(x), b(x), atol=0, rtol=0)
        assert a(x).shape == (*shape[:-1], 64)
    assert set(a.state_dict()) == set(b.state_dict())


@pytest.mark.parametrize("selector", ["bad", {}, 1])
def test_invalid_selector(selector):
    with pytest.raises(TypeError, match="selector"):
        public_class()(chart(), mixed(), selector=selector)


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
    old = model.atoms.p.detach().clone()
    p.zero_()
    assert torch.equal(model.atoms.p, old)
    assert model.atoms.p.data_ptr() != p.data_ptr()
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
@pytest.mark.parametrize("plan", [FULL, WINDOW])
def test_cuda_public_algorithm_boundary_independent_all_five(plan):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        sizes = (1024, 4, 4)
        p = mixed(torch.float32, "cuda")
        p[0, 2] = 511.25
        p[1, 2] = 512.5
        p[2:4, 2] = 512.02978515625
        model = public_class()(
            chart(sizes, dtype=torch.float32, device="cuda"),
            p,
            selector=FixedSelector(plan, registry=REGISTRY),
        )
        g = torch.Generator(device="cuda").manual_seed(21)
        x = torch.randn(2, 3, 16, device="cuda", generator=g, requires_grad=True)
        dy = torch.randn(2, 3, 1024, device="cuda", generator=g)
        pp = model.atoms.p.detach().double().requires_grad_()
        xx = x.detach().double().requires_grad_()
        weight = oracle(pp, sizes, stored_dtype=torch.float32)
        observed_weight = model(torch.eye(16, device="cuda")).T
        torch.testing.assert_close(
            observed_weight.double(), weight, atol=2e-5, rtol=2e-5
        )
        actual = model(x)
        truth = xx @ weight.T
        a = torch.autograd.grad(actual, (x, model.atoms.p), dy)
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
        torch.stack((model.kernel.sigma_min, model.kernel.sigma_max)),
        torch.tensor([0.03, 3.25], dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    x = torch.linspace(-1, 1, 32, dtype=torch.float64).reshape(2, 16)
    restored = public_class()(chart(), model.atoms.p.detach())
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
    model.chart.axes[0].start = torch.tensor([1.0], dtype=torch.float64)
    actual = model(x)
    expected = x @ oracle(model.atoms.p, origin=(1.0, 0.0, 0.0)).T
    assert not torch.equal(actual, before)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    model.kernel.sigma_min = torch.tensor(0.01, dtype=torch.float64)
    # A changed kernel contract uses its general Torch reference, not the fixed CUDA algorithm.
    assert model._resolved_backend() == "materialized"
    torch.testing.assert_close(model(x), model.operator.apply(x), atol=0, rtol=0)
