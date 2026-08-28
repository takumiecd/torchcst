"""Spectral diagnostics distinguish rank from energy concentration."""

from __future__ import annotations

import pytest
import torch

from torchcst.representation import survey_factorized_map, survey_spectrum


def test_participation_dimension_is_not_reported_as_a_rank_ceiling() -> None:
    # Full-rank, but with almost all energy in one singular direction.
    factor = torch.diag(
        torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
    )
    reading = survey_spectrum(factor)

    assert reading.numerical_rank == 3
    assert 1.0 < reading.participation_rank < 1.03
    assert 1.0 < reading.stable_rank < 1.02
    assert reading.condition_number == pytest.approx(100.0)


def test_factorized_survey_reports_actual_map_rank_and_its_upper_bound() -> None:
    input_factor = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float64
    )
    output_factor = torch.eye(3, dtype=torch.float64)
    amplitudes = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)

    reading = survey_factorized_map(input_factor, output_factor, amplitudes)
    expected = (output_factor * amplitudes) @ input_factor.T

    assert reading.input_factor.numerical_rank == 2
    assert reading.output_factor.numerical_rank == 3
    assert reading.numerical_rank_upper_bound == 2
    assert reading.represented_map.numerical_rank == int(
        torch.linalg.matrix_rank(expected)
    )
    torch.testing.assert_close(
        reading.represented_map.singular_values,
        torch.linalg.svdvals(expected),
    )


def test_zero_amplitude_tightens_the_factorized_rank_bound() -> None:
    identity = torch.eye(3, dtype=torch.float64)
    reading = survey_factorized_map(
        identity,
        identity,
        torch.tensor([1.0, 0.0, -2.0], dtype=torch.float64),
    )

    assert reading.numerical_rank_upper_bound == 2
    assert reading.represented_map.numerical_rank == 2


def test_factorized_survey_rejects_misaligned_atoms() -> None:
    with pytest.raises(ValueError, match="one column per amplitude"):
        survey_factorized_map(
            torch.ones(2, 3), torch.ones(4, 2), torch.ones(3)
        )
