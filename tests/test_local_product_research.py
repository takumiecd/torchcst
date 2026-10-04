"""Production PolarAmpWidth equivalence for local H and full-domain slicing."""

import math

import pytest
import torch

from benchmarks.cuda.linear.fixtures import local_product_reference as reference
from benchmarks.cuda.linear.fixtures import local_product_state as state
from torchcst._backends.cuda.algorithms.local_product.contract import Domain
from torchcst._backends.cuda.algorithms.local_product.preparation import (
    decode,
    validate_state,
)
from torchcst._backends.cuda.algorithms.local_product.recipe import Recipe
from torchcst._backends.torch.kernels import execution


def fixture(value, domain, *, device="cpu", dtype=torch.float64, atoms=19):
    torch.manual_seed(31)
    amplitude = torch.linspace(-0.12, 0.12, atoms, device=device, dtype=dtype)
    radius = torch.linspace(1, 2, atoms, device=device, dtype=dtype)
    p = torch.stack((amplitude, (1 - amplitude.square()).sqrt()), 1) * radius[:, None]
    center = torch.rand(atoms, 2, device=device, dtype=dtype)
    center[:, 0] = (
        domain.input_origin + center[:, 0] * (domain.input_size - 1) * domain.spacing
    )
    center[:, 1] = (
        domain.output_origin + center[:, 1] * (domain.output_size - 1) * domain.spacing
    )
    p = torch.cat((p, center), 1)
    # Exact singleton, two-site, empty, norm-floor, and zero-amplitude support.
    if atoms >= 5:
        p[:5, :2] = p.new_tensor(
            [
                [0.1, math.sqrt(0.99)],
                [-0.1, math.sqrt(0.99)],
                [0.1, math.sqrt(0.99)],
                [0.1, math.sqrt(0.99)],
                [0.0, 1.0],
            ]
        )
        p[0, 2:] = p.new_tensor(
            [
                domain.input_origin + 4 * domain.spacing,
                domain.output_origin + 5 * domain.spacing,
            ]
        )
        p[1, 2:] = p.new_tensor(
            [
                domain.input_origin + 4.5 * domain.spacing,
                domain.output_origin + 5.5 * domain.spacing,
            ]
        )
        p[2, 2] = domain.input_origin - 50 * domain.spacing
        p[3, 2:] = p.new_tensor(
            [
                domain.input_origin - 0.999 * domain.spacing,
                domain.output_origin - 0.999 * domain.spacing,
            ]
        )
    return p.requires_grad_()


def scalar_oracle(x, p, value, domain):
    q = decode(value, p)
    j = torch.arange(domain.input_size, device=p.device, dtype=p.dtype)
    i = torch.arange(domain.output_size, device=p.device, dtype=p.dtype)
    y = x.new_zeros((len(x), domain.output_count))
    for amp, inv, ci, co in q:
        v = (
            (1 - (domain.input_origin + j * domain.spacing - ci).square() * inv)
            .clamp_min(0)
            .pow(3)
        )
        u = (
            (1 - (domain.output_origin + i * domain.spacing - co).square() * inv)
            .clamp_min(0)
            .pow(3)
        )
        v = v / torch.linalg.vector_norm(v).clamp_min(1e-6)
        u = u / torch.linalg.vector_norm(u).clamp_min(1e-6)
        v = v[domain.input_start : domain.input_start + domain.input_count]
        u = u[domain.output_start : domain.output_start + domain.output_count]
        y = y + amp * (x * v[None]).sum(1)[:, None] * u[None]
    return y


def test_reference_slice_uses_full_domain_norms():
    d = Domain(
        32,
        32,
        spacing=0.5,
        input_start=5,
        output_start=3,
        input_count=9,
        output_count=11,
    )
    s = state(minimum=0.5, birth=0.5, maximum=8, w_c=0.2).double()
    p = fixture(s, d)
    x = torch.randn(7, d.input_count, dtype=p.dtype, requires_grad=True)
    charts = d.charts(dtype=p.dtype)
    actual, truth = reference(x, p, s, charts, d), scalar_oracle(x, p, s, d)
    dy = torch.randn_like(actual)
    ag, eg = (
        torch.autograd.grad(actual, (x, p), dy),
        torch.autograd.grad(truth, (x, p), dy),
    )
    for a, e in [(actual, truth), *zip(ag, eg)]:
        torch.testing.assert_close(a, e, atol=2e-11, rtol=2e-11)
    # Width is activity state: the task gradient is angular, not radial.
    assert (ag[1][:, :2] * p[:, :2]).sum(1).abs().max() < 1e-10


def test_recipe_and_shared_contract():
    validate_state(state())
    for upper in [(16.0,), (2.0, 16.0), (1.0, 2.0, 4.0, 16.0)]:
        Recipe(rho_upper=upper)
    with pytest.raises(ValueError):
        Recipe(rho_upper=(2.0, 1.0))
    with pytest.raises(ValueError):
        Domain(32, 32, input_start=25, input_count=16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize(
    "batch,spacing,sliced", [(1, 1.0, False), (7, 0.5, True), (32, 1.0, True)]
)
def test_local_h_y_dx_polar_gradient_update(saved, batch, spacing, sliced):
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    d = Domain(
        32,
        32,
        spacing=spacing,
        input_start=3 if sliced else 0,
        output_start=5 if sliced else 0,
        input_count=17 if sliced else 32,
        output_count=19 if sliced else 32,
    )
    s = state(minimum=spacing, birth=spacing, maximum=16 * spacing, w_c=0.2).cuda()
    validate_state(s)
    p = fixture(s, d, device="cuda", dtype=torch.float32)
    x = torch.randn(batch, d.input_count, device="cuda", requires_grad=True)
    dy = torch.randn(batch, d.output_count, device="cuda")
    actual = local_h(x, p, s, d, saved=saved)
    ax, ap = torch.autograd.grad(actual, (x, p), dy)
    pp, xx = p.detach().double().requires_grad_(), x.detach().double().requires_grad_()
    ss = (
        state(minimum=spacing, birth=spacing, maximum=16 * spacing, w_c=0.2)
        .double()
        .cuda()
    )
    truth = scalar_oracle(xx, pp, ss, d)
    ex, ep = torch.autograd.grad(truth, (xx, pp), dy.double())
    for a, e in [(actual, truth), (ax, ex), (ap, ep)]:
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)
    # Preserve the actual projected polar activity update, not plain log-width SGD.
    charts = d.charts(device="cuda")
    updated = execution.apply_parameter_update(
        s, *charts, p.detach(), -1e-4 * ap, step_size=1e-4
    )
    expected = execution.apply_parameter_update(
        ss,
        *d.charts(device="cuda", dtype=torch.float64),
        pp.detach(),
        -1e-4 * ep,
        step_size=1e-4,
    )
    torch.testing.assert_close(updated.double(), expected, atol=4e-4, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("saved", [False, True])
def test_graph_replay_refreshes_polar_decode_and_packing(saved):
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    d = Domain(32, 32)
    s = state(minimum=1, birth=1, w_c=0.2).cuda()
    p = fixture(s, d, device="cuda", dtype=torch.float32)
    x = torch.randn(16, 32, device="cuda", requires_grad=True)

    def step():
        y = local_h(x, p, s, d, saved=saved)
        return y, *torch.autograd.grad(y, (x, p), torch.ones_like(y))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    with torch.no_grad():
        p[0, :2] *= 2
        p[0, 2:] = p.new_tensor([19.3, 23.2])
    graph.replay()
    pp, xx = p.detach().double().requires_grad_(), x.detach().double().requires_grad_()
    ss = state(minimum=1, birth=1, w_c=0.2).double().cuda()
    y = scalar_oracle(xx, pp, ss, d)
    expected = (y, *torch.autograd.grad(y, (xx, pp), torch.ones_like(y)))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)


def test_graph_update_and_dense_baseline_match_production():
    from benchmarks.cuda.linear.fixtures import (
        local_product_dense_factors as dense_factors,
    )
    from torchcst._backends.cuda.algorithms.local_product.polar import graph_update

    d = Domain(32, 32, input_start=4, input_count=17, output_start=2, output_count=19)
    s = state(minimum=1, birth=1, w_c=0.2).double()
    p = fixture(s, d)
    displacement = torch.randn_like(p) * 0.01
    charts = d.charts(dtype=p.dtype)
    actual = graph_update(s, p, displacement, step_size=0.01)
    expected = execution.apply_parameter_update(
        s, *charts, p, displacement, step_size=0.01
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    v, u = dense_factors(p, s, d)
    x = torch.randn(7, d.input_count, dtype=p.dtype)
    torch.testing.assert_close(
        x @ v @ u.T, reference(x, p, s, charts, d), atol=1e-12, rtol=1e-12
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("saved", [False, True])
def test_max_batch_atom32_and_unsorted_preserve_gradients(saved):
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    d = Domain(64, 64, input_start=7, output_start=3, input_count=33, output_count=47)
    s = state(minimum=1, birth=1, w_c=0.2).cuda()
    p = fixture(s, d, device="cuda", dtype=torch.float32, atoms=41)
    x = torch.randn(64, d.input_count, device="cuda", requires_grad=True)
    dy = torch.randn(64, d.output_count, device="cuda")
    actual = local_h(x, p, s, d, saved=saved, recipe=Recipe(atom_block=32, pack=False))
    ax, ap = torch.autograd.grad(actual, (x, p), dy)
    pp, xx = p.detach().double().requires_grad_(), x.detach().double().requires_grad_()
    truth = scalar_oracle(xx, pp, state(minimum=1, birth=1, w_c=0.2).double().cuda(), d)
    ex, ep = torch.autograd.grad(truth, (xx, pp), dy.double())
    for a, e in [(actual, truth), (ax, ex), (ap, ep)]:
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)
