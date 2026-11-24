"""Channel-chart weight assembly and its adjoint on an irregular grid."""

import pytest
import torch
import torch.nn.functional as F
from test_radius_cell_mass import oracle_mass

from torchcst import ChebyshevSpec, kernels
from torchcst._backends.cuda.algorithms.conv2d.radius_channel import channel_weights
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
            [-0.31, 1, 127.81, -0.38],
            [0.73, 1.0000001192092896, 0.19, 63.83],
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
        x, weights.permute(1, 2, 0).reshape(128, 96, k, k), stride=2, padding=k // 2
    )


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("first_level", [3, 4, 5, 6])
def test_channel_weight_and_adjoint(ieee_cuda, k, first_level):
    binding = kernels.cell_mass(ChebyshevSpec(modes=8))
    ai = RadiusCellMassGrid(96, 1.0, 0.0, binding)
    ao = RadiusCellMassGrid(128, 1.0, 0.0, binding)
    p = atoms(k * k)
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
    x = torch.randn(2, 96, 6, 7, generator=g)
    x_ref = x.double().requires_grad_()
    y_ref = conv(x_ref, weights_ref, k)
    dy = torch.randn(y_ref.shape, generator=g)
    dx_ref, dp_ref = torch.autograd.grad(y_ref, (x_ref, pref), dy.double())

    p = p.cuda().requires_grad_()
    x = x.cuda().requires_grad_()
    weights = channel_weights(p, 96, first_fourier_level=first_level)
    y = conv(x, weights, k)
    dx, dp = torch.autograd.grad(y, (x, p), dy.cuda())
    for have, want in ((weights, weights_quad), (y, y_ref), (dx, dx_ref), (dp, dp_ref)):
        torch.testing.assert_close(have.cpu().double(), want, atol=2e-5, rtol=5e-4)


@pytest.mark.parametrize("first_level", [4, 6])
def test_channel_physical_weight_graph_replay(ieee_cuda, first_level):
    p = atoms(9).cuda().requires_grad_()
    x = torch.randn(2, 96, 6, 7, device="cuda").requires_grad_()
    dy = torch.randn(2, 128, 3, 4, device="cuda")

    def forward_and_grad():
        y = conv(x, channel_weights(p, 96, first_fourier_level=first_level), 3)
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
        p[:, :, 3].add_(0.31)
        x.add_(0.1)
    graph.replay()
    for have, want in zip(captured, forward_and_grad(), strict=True):
        torch.testing.assert_close(have, want, atol=2e-5, rtol=5e-4)
