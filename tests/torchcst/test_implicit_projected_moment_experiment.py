"""Mechanism checks for the rank-one square-root D-side experiment."""

from __future__ import annotations

from experiments.implicit_projected_moment.rank_one_d import (
    ExperimentConfig,
    run_trial,
    summarize,
)


def _config(**overrides) -> ExperimentConfig:
    values = {
        "atoms": 2,
        "amplitude": 1.0e-2,
        "residual_scale": 1.0e-1,
        "spread": 0.25,
        "batch_size": 7,
        "n_in": 5,
        "n_out": 4,
        "directions": 5,
        "seed": 3,
        "device": "cpu",
        "dtype": "float64",
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def test_rank_one_square_root_is_recovered_from_p2_statistics():
    result = run_trial(_config())

    assert result.rank_pullback_effective_rank == 1
    assert result.p2_linear_relative_error < 1.0e-10
    assert result.p2_curvature_relative_error < 1.0e-10
    assert result.factored_gradient_norm_relative_error < 1.0e-10
    assert result.rank_sqrt_identity_relative_error < 1.0e-10
    assert result.jtdj_relative_error < 1.0e-10
    assert result.jtdh_relative_error < 1.0e-10
    assert result.htdh_relative_error < 1.0e-10
    assert result.maximum_compact_energy_relative_error < 1.0e-10
    assert result.maximum_compact_force_relative_error < 1.0e-10


def test_summary_separates_identities_from_diagonal_comparison():
    results = [run_trial(_config(seed=seed)) for seed in (0, 1)]
    summary = summarize(results)

    assert summary["trials"] == 2
    assert set(summary) == {
        "trials",
        "mechanism_checks",
        "rank_one_vs_elementwise_diagonal",
        "by_atoms",
        "by_amplitude",
    }
    assert (
        summary["mechanism_checks"]["max_compact_force_relative_error"]
        < 1.0e-10
    )
