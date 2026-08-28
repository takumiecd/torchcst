"""Spectral diagnostics for a factorized CST map.

The participation rank of ``K K.T`` is an *effective dimension*, not a hard
rank ceiling.  Smooth Gaussian columns can make it much smaller than the
ordinary numerical rank even when the factor matrix, and the represented
map, have full row/column rank.  This module reports both quantities from one
shared tolerance so experiments cannot silently substitute one for the
other.

These functions are deliberately diagnostic: they materialize singular
values (and, for :func:`survey_factorized_map`, the dense represented map).
They do not belong on a training hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from torchcst._validation import require_real

__all__ = [
    "FactorizedSpectrumSurvey",
    "SpectrumSurvey",
    "survey_factorized_map",
    "survey_spectrum",
]


@dataclass(frozen=True)
class SpectrumSurvey:
    """Hard-threshold and energy-weighted readings of one matrix spectrum.

    ``participation_rank`` is ``(sum s^2)^2 / sum s^4`` and
    ``stable_rank`` is ``sum s^2 / max(s)^2``.  Neither is an algebraic rank:
    both intentionally fall when energy concentrates in a few singular
    directions.  ``numerical_rank`` counts singular values above the reported
    ``tolerance``.
    """

    singular_values: Tensor
    numerical_rank: int
    participation_rank: float
    stable_rank: float
    condition_number: float
    tolerance: float


@dataclass(frozen=True)
class FactorizedSpectrumSurvey:
    """Endpoint and represented-map spectra for ``U diag(w) V.T``.

    ``input_factor`` surveys ``V`` and ``output_factor`` surveys ``U``.
    ``numerical_rank_upper_bound`` is the corresponding finite-precision
    counterpart of

    ``rank(W) <= min(rank(U), rank(V), count_nonzero(w))``.

    It is an upper bound, not a prediction of the map's effective rank.
    """

    input_factor: SpectrumSurvey
    output_factor: SpectrumSurvey
    represented_map: SpectrumSurvey
    numerical_rank_upper_bound: int


def _validate_matrix(matrix: Tensor, name: str) -> None:
    if not isinstance(matrix, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be rank 2")
    if not matrix.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must be finite")


@torch.no_grad()
def survey_spectrum(
    matrix: Tensor,
    *,
    atol: float = 0.0,
    rtol: float | None = None,
) -> SpectrumSurvey:
    """Measure numerical and energy-weighted ranks of a finite matrix.

    The default relative tolerance is the convention used by
    :func:`torch.linalg.matrix_rank`: ``max(m, n) * eps``.  ``atol`` and
    ``rtol`` combine as ``max(atol, rtol * sigma_max)``.
    """

    _validate_matrix(matrix, "matrix")
    atol = require_real(atol, "atol", nonnegative=True)
    if rtol is None:
        rtol = max(matrix.shape) * torch.finfo(matrix.dtype).eps
    else:
        rtol = require_real(rtol, "rtol", nonnegative=True)

    singular = torch.linalg.svdvals(matrix.detach())
    sigma_max = float(singular[0]) if singular.numel() else 0.0
    tolerance = max(atol, rtol * sigma_max)
    numerical_rank = int((singular > tolerance).sum())

    energy = singular.square()
    total = float(energy.sum())
    participation_denominator = float(energy.square().sum())
    participation = (
        total * total / participation_denominator
        if participation_denominator > 0.0
        else 0.0
    )
    stable = total / (sigma_max * sigma_max) if sigma_max > 0.0 else 0.0
    if numerical_rank == 0:
        condition = math.inf
    else:
        sigma_min = float(singular[numerical_rank - 1])
        condition = sigma_max / sigma_min if sigma_min > 0.0 else math.inf
    return SpectrumSurvey(
        singular_values=singular.detach().clone(),
        numerical_rank=numerical_rank,
        participation_rank=participation,
        stable_rank=stable,
        condition_number=condition,
        tolerance=tolerance,
    )


@torch.no_grad()
def survey_factorized_map(
    input_factor: Tensor,
    output_factor: Tensor,
    amplitudes: Tensor,
    *,
    atol: float = 0.0,
    rtol: float | None = None,
) -> FactorizedSpectrumSurvey:
    """Survey ``W = output_factor @ diag(amplitudes) @ input_factor.T``.

    Factor columns are atoms: both factor matrices therefore have ``K``
    columns and ``amplitudes`` has ``K`` entries.  The represented map is
    materialized once; use this at checkpoints, not per optimization step.
    """

    _validate_matrix(input_factor, "input_factor")
    _validate_matrix(output_factor, "output_factor")
    if not isinstance(amplitudes, Tensor):
        raise TypeError("amplitudes must be a Tensor")
    if amplitudes.ndim != 1:
        raise ValueError("amplitudes must be rank 1")
    atoms = amplitudes.numel()
    if input_factor.shape[1] != atoms or output_factor.shape[1] != atoms:
        raise ValueError("factor matrices must have one column per amplitude")
    if not amplitudes.is_floating_point():
        raise TypeError("amplitudes must have a floating dtype")
    for name, value in (
        ("input_factor", input_factor),
        ("output_factor", output_factor),
    ):
        if value.dtype != amplitudes.dtype:
            raise TypeError(f"{name} must share amplitudes' dtype")
        if value.device != amplitudes.device:
            raise TypeError(f"{name} must share amplitudes' device")
    if not bool(torch.isfinite(amplitudes).all()):
        raise ValueError("amplitudes must be finite")

    input_reading = survey_spectrum(
        input_factor, atol=atol, rtol=rtol
    )
    output_reading = survey_spectrum(
        output_factor, atol=atol, rtol=rtol
    )
    represented = (output_factor * amplitudes.unsqueeze(0)) @ input_factor.T
    map_reading = survey_spectrum(represented, atol=atol, rtol=rtol)
    nonzero_amplitudes = int(torch.count_nonzero(amplitudes))
    upper_bound = min(
        input_reading.numerical_rank,
        output_reading.numerical_rank,
        nonzero_amplitudes,
    )
    return FactorizedSpectrumSurvey(
        input_factor=input_reading,
        output_factor=output_reading,
        represented_map=map_reading,
        numerical_rank_upper_bound=upper_bound,
    )
