"""Canonical dense/autograd checks for the experimental atom contraction."""

import pytest
import torch
import torch.nn.functional as F
from test_triton_linear import _model

from prototypes.atom_prefix_linear import atom_prefix_linear, prepare_prefix
from torchcst import Triangle


@pytest.mark.parametrize("representation", ["intrinsic", "ambient"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("atoms,chunk", [(1, 1), (13, 4)])
@pytest.mark.parametrize("sigma_min", [0.3, 3.5])
def test_prefix_matches_canonical_value_and_gradients(
    representation, dtype, atoms, chunk, sigma_min
):
    torch.manual_seed(38)
    layer = _model(
        representation=representation, atoms=max(1, atoms), sigma_min=sigma_min
    ).to(dtype)
    p = layer.atoms.p[:atoms].detach().requires_grad_()
    with torch.no_grad():
        p[:, 0].uniform_(-0.8, 0.8)
        p[:, 1].uniform_(1, 4)
        if atoms:
            p[0, 0] = 1.4  # amplitude clamp
            # Torus seam and a center not snapped to any sampled row.
            point = p.new_tensor([[-0.031, 0.17, -0.13]])
            center = layer.chart.geometry.lift_chart_coordinates(point)
            p[0, 2:] = layer.chart.geometry.encode_centers(center)[0]
    x = torch.randn(2, 21, 3, dtype=dtype).transpose(1, 2).requires_grad_()
    expected = F.linear(x, layer.kernel.weight(layer.chart, p))
    actual = atom_prefix_linear(layer, x, p, atom_chunk=chunk)
    grad = torch.randn_like(expected)
    reference = torch.autograd.grad(expected, (x, p), grad)
    result = torch.autograd.grad(actual, (x, p), grad)
    atol = 2e-10 if dtype == torch.float64 else 5e-5
    torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)
    for got, wanted in zip(result, reference, strict=True):
        torch.testing.assert_close(got, wanted, atol=atol, rtol=atol)
    torch.testing.assert_close(result[1][:, 1], torch.zeros_like(p[:, 1]))


def test_prefix_empty_batch_and_recomputed_plan():
    layer = _model().double()
    p = layer.atoms.p
    x = torch.empty(0, 21, dtype=p.dtype, requires_grad=True)
    atom_prefix_linear(layer, x, p).sum().backward()
    assert p.grad is not None and not bool(p.grad.any())
    x = torch.randn(21, dtype=p.dtype)
    before = atom_prefix_linear(layer, x, p)
    with torch.no_grad():
        p[:, 0].mul_(0.5)
        p[:, 2].add_(0.1)
    after = atom_prefix_linear(layer, x, p)
    torch.testing.assert_close(
        after, F.linear(x, layer.dense_weight()), atol=2e-10, rtol=2e-10
    )
    assert not torch.equal(before, after)


def test_prefix_rejects_unsupported_profile():
    layer = _model(profile=Triangle)
    with pytest.raises(ValueError, match="Triweight"):
        prepare_prefix(layer, layer.atoms.p)
