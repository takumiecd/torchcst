"""Squared distances, spelled so the compiler emits a pointwise factor.

The obvious spelling of a squared distance -- broadcast the coordinate axis
and ``.square().sum(-1)`` -- makes inductor emit a *reduction* factor over an
axis of length three or four.  A three-term sum then pays the whole reduction
machinery, and it dominates: on the lm1 sites one such factor
(``triton_red_fused_pow_sub_sum_unsqueeze_0``) was 113 ms of a 177 ms step,
64% of the whole thing, and it is why that run's backward cost ten times its
forward.

Unrolling the coordinate axis makes the same arithmetic pointwise.  In
float32 -- every training run -- the result is bitwise equal to the spelling
these functions replace, for the coordinate dimensions CST charts use.  In
float64 it can differ by one or two ULP, because torch's reduction groups the
terms differently than a left fold does; the difference is the grouping, not
the arithmetic.  ``tests/torchcst/test_squared_distance.py`` holds both
statements as a contract, so a future torch that regroups float32 too is
caught here rather than in a run.

The Gram identity ``|q|^2 - 2 q.c + |c|^2`` is faster still, but it cancels,
and under TF32 the dot product carries far more error than a Gaussian
exponent can absorb.  Not used.
"""

from __future__ import annotations

from torch import Tensor

__all__ = ["squared_norm_last", "squared_distance_matrix"]


def squared_norm_last(displacement: Tensor) -> Tensor:
    """``|displacement|^2`` over the last axis, without a reduction factor."""
    total = displacement[..., 0].square()
    for axis in range(1, displacement.shape[-1]):
        total = total + displacement[..., axis].square()
    return total


def squared_distance_matrix(query: Tensor, centers: Tensor) -> Tensor:
    """``[N, K]`` squared distances between ``[N, d]`` and ``[K, d]`` rows.

    Never forms the ``[N, K, d]`` cube, so the compiler has nothing to fuse
    away in the first place.
    """
    total = (query[:, None, 0] - centers[None, :, 0]).square()
    for axis in range(1, query.shape[-1]):
        total = total + (query[:, None, axis] - centers[None, :, axis]).square()
    return total
