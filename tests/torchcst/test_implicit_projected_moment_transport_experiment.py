"""Checks for the projected EMA transport fidelity experiment."""

from __future__ import annotations

from experiments.implicit_projected_moment.ema_transport_fidelity import (
    ExperimentConfig,
    run_trial,
    summarize,
)


def _config(**overrides) -> ExperimentConfig:
    values = {
        "atoms": 2,
        "beta1": 0.9,
        "beta2": 0.99,
        "radius": 1.0e-2,
        "steps": 6,
        "batch_size": 7,
        "n_in": 5,
        "n_out": 4,
        "seed": 3,
        "device": "cpu",
        "dtype": "float64",
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def test_recompression_preserves_its_current_observable_numerator():
    result = run_trial(_config())

    assert result.maximum_recompression_identity_error < 1.0e-10


def test_no_history_has_no_cross_time_compression_error():
    result = run_trial(_config(beta1=0.0, beta2=0.0))

    assert result.accepted_first_maximum_relative_error < 1.0e-10
    assert result.accepted_second_maximum_relative_error < 1.0e-10
    assert result.base_first_maximum_relative_error < 1.0e-10
    assert result.base_second_maximum_relative_error < 1.0e-10


def test_summary_separates_moving_ema_from_zero_history_controls():
    results = [
        run_trial(_config(beta1=0.0, beta2=0.0)),
        run_trial(_config(beta1=0.9, beta2=0.99)),
    ]
    summary = summarize(results)

    assert summary["trials"] == 2
    assert summary["moving_ema_trials"] == 1
    assert summary["zero_history_controls"] == 1
    assert set(summary["by_radius"]) == {"0.01"}
    assert set(summary["by_atoms"]) == {"2"}
