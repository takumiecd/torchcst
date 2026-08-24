"""The squared-distance spelling is a code-generation choice, not a numeric one.

``_geometry`` unrolls the coordinate axis so the compiler emits a pointwise
factor instead of a reduction over an axis of length three or four.  These
tests pin what that costs numerically: nothing in float32, at most a couple
of ULP in float64.  If a future torch regroups the float32 reduction too,
this file fails rather than a training run drifting quietly.
"""

import pytest
import torch

from torchcst._geometry import squared_distance_matrix, squared_norm_last

CHART_DIMS = (1, 2, 3, 4)


def _reference(query, centers):
    return (query[:, None, :] - centers[None, :, :]).square().sum(-1)


@pytest.mark.parametrize("dim", CHART_DIMS)
def test_float32_is_bitwise_equal_to_the_reduction(dim: int) -> None:
    generator = torch.Generator().manual_seed(dim)
    query = torch.randn(97, dim, generator=generator)
    centers = torch.randn(151, dim, generator=generator)

    assert torch.equal(
        squared_distance_matrix(query, centers), _reference(query, centers)
    )
    displacement = query[:, None, :] - centers[None, :, :]
    assert torch.equal(
        squared_norm_last(displacement), displacement.square().sum(-1)
    )


@pytest.mark.parametrize("dim", CHART_DIMS)
def test_float64_agrees_to_a_couple_of_ulp(dim: int) -> None:
    generator = torch.Generator().manual_seed(dim)
    query = torch.randn(97, dim, generator=generator, dtype=torch.float64)
    centers = torch.randn(151, dim, generator=generator, dtype=torch.float64)

    reference = _reference(query, centers)
    relative = (
        (squared_distance_matrix(query, centers) - reference).abs()
        / reference.abs().clamp_min(1e-30)
    ).max()
    assert relative <= 4.0 * torch.finfo(torch.float64).eps


def test_leading_batch_axes_are_preserved() -> None:
    # batched_conv stacks layers, so the displacement carries an extra axis.
    generator = torch.Generator().manual_seed(11)
    displacement = torch.randn(5, 7, 13, 3, generator=generator)

    result = squared_norm_last(displacement)

    assert result.shape == (5, 7, 13)
    assert torch.equal(result, displacement.square().sum(-1))


def test_gradients_match_the_reduction() -> None:
    generator = torch.Generator().manual_seed(3)
    centers = torch.randn(31, 3, generator=generator, dtype=torch.float64)
    query = torch.randn(17, 3, generator=generator, dtype=torch.float64)

    grads = []
    for build in (squared_distance_matrix, _reference):
        moving = centers.clone().requires_grad_(True)
        build(query, moving).square().sum().backward()
        grads.append(moving.grad)

    torch.testing.assert_close(grads[0], grads[1], rtol=1e-12, atol=0.0)
