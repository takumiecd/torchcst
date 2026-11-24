"""Channel-chart weight assembly and its adjoint on an irregular grid."""

import pytest
import torch
import torch.nn.functional as F
from test_radius_cell_mass import oracle_mass

from torchcst import ChebyshevSpec, kernels
from torchcst._backends.cuda.algorithms.conv2d.radius_channel import (
    channel_fourier_weights,
    channel_weights,
)
from torchcst._backends.torch.kernels.radius_cell_mass import axis_masses
from torchcst.kernels.radius_cell_mass import RadiusCellMassGrid


def test_channel_weight_rejects_unsupported_metadata_before_compiling():
    p = torch.empty(1, 1, 4)
    with pytest.raises(ValueError, match="96 or 128"):
        channel_weights(p, 864, first_fourier_level=4)
    with pytest.raises(ValueError, match="first_fourier_level"):
        channel_weights(p, 96, first_fourier_level=2)
    with pytest.raises(ValueError, match="CUDA FP32"):
        channel_weights(p, 96, first_fourier_level=4)
    with pytest.raises(ValueError, match="sort_owners"):
        channel_fourier_weights(p, 96, sort_owners=1)
    with pytest.raises(ValueError, match="active_banks_only"):
        channel_fourier_weights(p, 96, active_banks_only=1)
    with pytest.raises(ValueError, match="output_sites"):
        channel_fourier_weights(p, 32, output_sites=10)
    with pytest.raises(ValueError, match="radius_cap"):
        channel_fourier_weights(p, 16, output_sites=32, radius_cap=16)
    with pytest.raises(ValueError, match="radius_cap"):
        channel_fourier_weights(p, 32, output_sites=64, radius_cap=12)
    with pytest.raises(ValueError, match="CUDA FP32"):
        channel_fourier_weights(p, 16, output_sites=32)


@pytest.fixture
def ieee_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    previous = torch.get_float32_matmul_precision()
    previous_conv = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
        torch.backends.cudnn.allow_tf32 = previous_conv


def atoms(taps):
    rows = torch.tensor(
        [
            [0.22, 0.5, -0.19, 95.77],
            [0.13, 0.24, -0.21, 95.79],
            [-0.31, 1, 127.81, -0.38],
            [0.73, 1.0000001192092896, 0.19, 63.83],
            [-0.07, 1.8, 0.23, 63.91],
            [0.17, 4, 89.25, 31.78],
            [-0.5, 8.2, 0.37, 95.91],
            [0.42, 16.21, 125.69, 13.06],
            [0.3, 32, 128.18, 98.13],
        ],
        dtype=torch.float32,
    ).repeat(taps, 1, 1)
    rows[:, :, 2:] += torch.arange(taps)[:, None, None] * 0.19
    return rows


def conv(x, weights, k):
    return F.conv2d(
        x,
        weights.permute(1, 2, 0).reshape(128, x.size(1), k, k),
        stride=2,
        padding=k // 2,
    )


def assemble(p, input_sites, method):
    if isinstance(method, str):
        return channel_fourier_weights(
            p,
            input_sites,
            sort_owners=method.endswith("-sorted"),
            active_banks_only="active" in method,
        )
    return channel_weights(p, input_sites, first_fourier_level=method)


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("input_sites", [96, 128])
@pytest.mark.parametrize(
    "method",
    [
        3,
        4,
        5,
        6,
        "fourier",
        "fourier-sorted",
        "fourier-active",
        "fourier-active-sorted",
    ],
)
def test_channel_weight_and_adjoint(ieee_cuda, k, input_sites, method):
    binding = kernels.cell_mass(ChebyshevSpec(modes=8))
    ai = RadiusCellMassGrid(input_sites, 1.0, 0.0, binding)
    ao = RadiusCellMassGrid(128, 1.0, 0.0, binding)
    p = atoms(k * k)
    p[:, (0, 1, 6, 8), 3] += input_sites - 96
    pref = p.double().requires_grad_()
    references = []
    quadrature = []
    for tap in pref:
        a, r, co, ci = tap.unbind(1)
        vi, vo = axis_masses(ai, r, ci), axis_masses(ao, r, co)
        references.append((vo * a[:, None]).T @ vi)
        weights = [
            amplitude
            * torch.outer(
                torch.from_numpy(oracle_mass(ao, radius, out_center)),
                torch.from_numpy(oracle_mass(ai, radius, in_center)),
            )
            for amplitude, radius, out_center, in_center in tap.tolist()
        ]
        quadrature.append(torch.stack(weights).sum(0))
    weights_ref = torch.stack(references)
    weights_quad = torch.stack(quadrature)
    torch.testing.assert_close(weights_ref, weights_quad, atol=1e-12, rtol=3e-12)

    g = torch.Generator().manual_seed(17)
    x = torch.randn(2, input_sites, 6, 7, generator=g)
    x_ref = x.double().requires_grad_()
    y_ref = conv(x_ref, weights_ref, k)
    dy = torch.randn(y_ref.shape, generator=g)
    dx_ref, dp_ref = torch.autograd.grad(y_ref, (x_ref, pref), dy.double())

    p = p.cuda().requires_grad_()
    x = x.cuda().requires_grad_()
    weights = assemble(p, input_sites, method)
    y = conv(x, weights, k)
    dx, dp = torch.autograd.grad(y, (x, p), dy.cuda())
    for have, want in ((weights, weights_quad), (y, y_ref), (dx, dx_ref), (dp, dp_ref)):
        torch.testing.assert_close(have.cpu().double(), want, atol=2e-5, rtol=5e-4)


@pytest.mark.parametrize("method", [4, 6, "fourier", "fourier-sorted"])
def test_channel_physical_weight_graph_replay(ieee_cuda, method):
    p = atoms(9).cuda().requires_grad_()
    x = torch.randn(2, 96, 6, 7, device="cuda").requires_grad_()
    dy = torch.randn(2, 128, 3, 4, device="cuda")

    def forward_and_grad():
        y = conv(x, assemble(p, 96, method), 3)
        return (y, *torch.autograd.grad(y, (x, p), dy))

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            forward_and_grad()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward_and_grad()
    with torch.no_grad():
        p[..., 1].copy_(p[..., 1].roll(1, dims=1))
        p[:, :, 3].add_(0.31)
        x.add_(0.1)
    graph.replay()
    for have, want in zip(captured, forward_and_grad(), strict=True):
        torch.testing.assert_close(have, want, atol=2e-5, rtol=5e-4)


@pytest.mark.parametrize("sort_owners", [False, True])
def test_channel_fourier_center_shifts_follow_the_torus(ieee_cuda, sort_owners):
    rows = atoms(9).cuda()
    rows[..., 2:] = (rows[..., 2:] * 4).round() / 4
    shifted = rows.clone()
    shifted[..., 2] += 128
    shifted[..., 3] -= 96
    generator = torch.Generator(device="cuda").manual_seed(33)
    dw = torch.randn(9, 128, 96, device="cuda", generator=generator)

    def weight_and_grad(p):
        p.requires_grad_()
        weight = channel_fourier_weights(p, 96, sort_owners=sort_owners)
        (dp,) = torch.autograd.grad(weight, p, dw)
        return weight, dp

    for have, want in zip(weight_and_grad(rows), weight_and_grad(shifted), strict=True):
        torch.testing.assert_close(have, want, atol=2e-5, rtol=5e-4)


@pytest.mark.parametrize("sort_owners", [False, True])
def test_channel_fourier_active_banks_follow_the_saved_geometry(ieee_cuda, sort_owners):
    inputs = tuple(atoms(9).cuda() for _ in range(2))
    inputs[0][..., 1].fill_(0.34)
    inputs[1][..., 1].fill_(22)
    generator = torch.Generator(device="cuda").manual_seed(42)
    dws = tuple(
        torch.randn(9, 128, 96, device="cuda", generator=generator) for _ in inputs
    )

    def two_forwards(active):
        operands = tuple(p.clone().requires_grad_() for p in inputs)
        weights = tuple(
            channel_fourier_weights(
                p, 96, sort_owners=sort_owners, active_banks_only=active
            )
            for p in operands
        )
        gradients = torch.autograd.grad(weights, operands, dws)
        return (*weights, *gradients)

    for have, want in zip(two_forwards(True), two_forwards(False), strict=True):
        torch.testing.assert_close(have, want, atol=2e-5, rtol=5e-4)


@pytest.mark.parametrize(
    "k,ci,co,size,stride",
    [
        (1, 16, 32, 32, 1),
        (1, 32, 64, 32, 2),
        (1, 64, 128, 16, 2),
        (3, 16, 32, 32, 1),
        (3, 32, 32, 32, 1),
        (3, 32, 64, 32, 2),
        (3, 64, 64, 16, 1),
        (3, 64, 128, 16, 2),
        (3, 128, 128, 8, 1),
    ],
)
def test_wideresnet_fourier_tori_and_gradients(ieee_cuda, k, ci, co, size, stride):
    # Exercise the largest declared bank, including repeated images on the
    # shorter torus, and all dyadic boundaries below it.
    cap = min(ci, co) // 2
    radius = torch.tensor([0.24, 0.5, 0.73] + [2**j for j in range(cap.bit_length())])
    generator = torch.Generator().manual_seed(55)
    rows = torch.randn(k * k, len(radius), 4, generator=generator) * 0.3
    rows[..., 1] = radius
    rows[:, ::2, 2] += co
    rows[:, 1::2, 3] -= ci
    rows = rows.double().requires_grad_()
    amplitude, r, out_center, in_center = rows.reshape(-1, 4).unbind(-1)
    binding = kernels.cell_mass(ChebyshevSpec(modes=8))
    ao, ai = (RadiusCellMassGrid(c, 1.0, 0.0, binding) for c in (co, ci))
    mi = axis_masses(ai, r, in_center).view(k * k, -1, ci)
    mo = axis_masses(ao, r, out_center).view(k * k, -1, co)
    expected = (mo * amplitude.view(k * k, -1, 1)).transpose(1, 2) @ mi
    quadrature = [
        a
        * torch.outer(
            torch.from_numpy(oracle_mass(ao, radius, o)),
            torch.from_numpy(oracle_mass(ai, radius, i)),
        )
        for a, radius, o, i in rows[0].tolist()
    ]
    torch.testing.assert_close(expected[0], torch.stack(quadrature).sum(0))

    def apply_conv(x, weights):
        padding = k // 2
        if k == 3 and stride == 2:
            x = F.pad(x, (0, 1, 0, 1))
            padding = 0
        return F.conv2d(
            x,
            weights.permute(1, 2, 0).reshape(co, ci, k, k),
            stride=stride,
            padding=padding,
        )

    x = torch.randn(1, ci, size, size, generator=generator, dtype=torch.float64)
    x.requires_grad_()
    expected_y = apply_conv(x, expected)
    dy = torch.randn(expected_y.shape, generator=generator, dtype=torch.float64)
    expected_dx, expected_dp = torch.autograd.grad(expected_y, (x, rows), dy)
    for active, sorted_owners in ((False, False), (True, True)):
        p = rows.detach().to(device="cuda", dtype=torch.float32).requires_grad_()
        gx = x.detach().to(device="cuda", dtype=torch.float32).requires_grad_()
        w = channel_fourier_weights(
            p,
            ci,
            output_sites=co,
            radius_cap=cap,
            sort_owners=sorted_owners,
            active_banks_only=active,
        )
        y = apply_conv(gx, w)
        dx, dp = torch.autograd.grad(y, (gx, p), dy.to(y))
        for have, want in (
            (w, expected),
            (y, expected_y),
            (dx, expected_dx),
            (dp, expected_dp),
        ):
            torch.testing.assert_close(
                have.cpu().double(), want, atol=1.2e-4, rtol=5e-4
            )
