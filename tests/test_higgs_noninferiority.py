"""Finite-family, paired uncertainty and minimum-density interpretation."""

import math

import numpy as np
import pytest

from benchmarks.training.higgs_noninferiority import (
    CONFIRMATION_SEEDS,
    DIAGNOSTIC_ENDPOINTS,
    PRIMARY_CONFIGS,
    _prepare_auc,
    _weighted_auc,
    compare_predictions,
    configuration_status,
    minimum_alpha_summary,
    paired_case_bootstrap,
    seed_difference_statistics,
    student_t_df2_ppf,
    student_t_df2_sf,
)


def test_df2_quantile_matches_known_reference_and_cdf():
    assert student_t_df2_ppf(0.975) == pytest.approx(4.3026527299, abs=1e-9)
    assert student_t_df2_ppf(0.95) == pytest.approx(2.91998558035, abs=1e-9)
    assert student_t_df2_ppf(1 - 0.05 / (2 * DIAGNOSTIC_ENDPOINTS)) == pytest.approx(
        17.8465902198, rel=1e-11
    )
    for probability in (0.001, 0.025, 0.5, 0.95, 0.999):
        value = student_t_df2_ppf(probability)
        cdf = 0.5 + value / (2 * math.sqrt(value * value + 2))
        assert cdf == pytest.approx(probability, abs=1e-15)


@pytest.mark.parametrize("value", (-1, 0, 1, 2, True, float("nan"), float("inf")))
def test_invalid_df2_quantile_probabilities_rejected(value):
    with pytest.raises(ValueError):
        student_t_df2_ppf(value)


def test_weighted_auc_ties_match_explicit_resampled_pairs():
    scores = np.array([1.0, 1.0, 2.0, -1.0, 2.0, 0.0])
    targets = np.array([1, 0, 0, 0, 1, 1])
    weights = np.array([2, 3, 1, 2, 3, 1])
    positive = np.repeat(scores[targets == 1], weights[targets == 1])
    negative = np.repeat(scores[targets == 0], weights[targets == 0])
    reference = (positive[:, None] > negative).mean() + 0.5 * (
        positive[:, None] == negative
    ).mean()
    actual = _weighted_auc(
        _prepare_auc(scores, targets),
        weights,
        int(weights @ targets),
        int(weights.sum() - weights @ targets),
    )
    assert actual == reference


def _fixture():
    targets = np.array([0, 0, 0, 1, 1, 1])
    dense = np.tile(np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]), (3, 1))
    return dense, targets


def test_identical_scores_have_zero_paired_uncertainty_and_are_supported():
    dense, targets = _fixture()
    result = compare_predictions(
        dense, dense, targets, width=1024, alpha=0.001, bootstrap_replicates=50
    )
    assert result["status"] == "supported"
    assert result["primary_configurations"] == 8
    assert result["diagnostic_simultaneous_endpoint_count"] == 16
    assert result["empirical_all_three_seed_margin_pass_candidate"]
    assert not result["combined_joint_seed_example_confidence_claim"]
    for name in ("accuracy", "roc_auc"):
        assert result["metrics"][name]["difference_per_seed"] == [0, 0, 0]
        assert result["metrics"][name]["simultaneous_seed_t_lower"] == 0
        assert result["conditional_paired_case_bootstrap"][name] == {
            "lower": 0,
            "upper": 0,
        }


def test_consistently_reversed_predictions_are_statistically_inferior():
    dense, targets = _fixture()
    result = compare_predictions(
        -dense, dense, targets, width=2048, alpha=0.01, bootstrap_replicates=50
    )
    assert result["status"] == "inferior"
    assert not result["empirical_all_three_seed_margin_pass_candidate"]
    assert result["metrics"]["accuracy"]["mean_difference"] == -1
    assert result["metrics"]["roc_auc"]["mean_difference"] == -1


def test_seed_variation_can_be_inconclusive_despite_nonnegative_mean():
    dense, targets = _fixture()
    baseline = np.zeros_like(dense)
    cst = np.stack((dense[0], -dense[1], dense[2]))
    result = compare_predictions(
        cst, baseline, targets, width=1024, alpha=0.002, bootstrap_replicates=50
    )
    assert result["status"] == "inconclusive"
    assert result["metrics"]["accuracy"]["mean_difference"] > 0
    assert result["metrics"]["accuracy"]["simultaneous_seed_t_lower"] < -0.002
    assert result["metrics"]["accuracy"]["simultaneous_seed_t_upper"] > -0.002


def test_accuracy_ties_predict_class_zero_like_argmax():
    scores = np.zeros((3, 4))
    targets = np.array([0, 0, 0, 1])
    result = compare_predictions(
        scores, scores, targets, width=1024, alpha=0.005, bootstrap_replicates=30
    )
    assert result["metrics"]["accuracy"]["cst_per_seed"] == [0.75] * 3
    assert result["metrics"]["roc_auc"]["cst_per_seed"] == [0.5] * 3


def test_paired_bootstrap_is_deterministic_and_preserves_pairing():
    dense, targets = _fixture()
    cst = dense.copy()
    cst[0, 0] = 1
    first = paired_case_bootstrap(cst, dense, targets, replicates=60, random_seed=73)
    second = paired_case_bootstrap(cst, dense, targets, replicates=60, random_seed=73)
    assert first == second
    identical = paired_case_bootstrap(cst, cst, targets, replicates=60, random_seed=73)
    assert identical["accuracy"]["lower"] == identical["accuracy"]["upper"] == 0
    assert identical["roc_auc"]["lower"] == identical["roc_auc"]["upper"] == 0


def test_seed_input_order_is_canonicalized():
    dense, targets = _fixture()
    cst = dense.copy()
    cst[1] *= -1
    first = compare_predictions(
        cst, dense, targets, width=1024, alpha=0.001, bootstrap_replicates=40
    )
    perm = [2, 0, 1]
    second = compare_predictions(
        cst[perm],
        dense[perm],
        targets,
        seeds=tuple(CONFIRMATION_SEEDS[index] for index in perm),
        width=1024,
        alpha=0.001,
        bootstrap_replicates=40,
    )
    assert first == second


@pytest.mark.parametrize(
    "change",
    ("shape", "nonfinite", "labels", "one_class", "seeds", "width", "alpha"),
)
def test_undeclared_or_malformed_confirmation_inputs_rejected(change):
    dense, targets = _fixture()
    options = {"width": 1024, "alpha": 0.001, "bootstrap_replicates": 30}
    cst = dense.copy()
    if change == "shape":
        cst = cst[:2]
    elif change == "nonfinite":
        cst[1, 2] = np.nan
    elif change == "labels":
        targets = targets.astype(float)
    elif change == "one_class":
        targets[:] = 0
    elif change == "seeds":
        options["seeds"] = (17, 29, 43)
    elif change == "width":
        options["width"] = 4096
    else:
        options["alpha"] = 0.0015
    with pytest.raises(ValueError):
        compare_predictions(cst, dense, targets, **options)


def _comparison(alpha, status):
    return {
        "width": 1024,
        "alpha": alpha,
        "status": status,
        "primary_configurations": 8,
        "diagnostic_simultaneous_endpoint_count": 16,
        "seeds": list(CONFIRMATION_SEEDS),
        "examples": 6,
        "targets_sha256": "fixture_same_ordered_targets",
        "empirical_all_three_seed_margin_pass_candidate": status != "inferior",
    }


def test_smaller_inconclusive_density_does_not_identify_a_threshold():
    result = minimum_alpha_summary(
        [
            _comparison(0.001, "inconclusive"),
            _comparison(0.002, "supported"),
            _comparison(0.005, "supported"),
            _comparison(0.01, "supported"),
        ]
    )["widths"][0]
    assert result["minimum_supported_tested_alpha"] == 0.002
    assert not result["threshold_identified"]
    assert result["smaller_inconclusive_alphas"] == [0.001]
    assert not result["monotonic_quality_assumed"]


def test_lower_range_endpoint_support_does_not_claim_a_lower_bound():
    result = minimum_alpha_summary(
        [
            _comparison(0.001, "supported"),
            _comparison(0.002, "supported"),
            _comparison(0.005, "supported"),
            _comparison(0.01, "supported"),
        ]
    )["widths"][0]
    assert result["minimum_supported_tested_alpha"] == 0.001
    assert result["lower_range_endpoint_supported"]
    assert result["minimum_alpha_leq_within_declared_protocol"] == 0.001
    assert not result["threshold_identified"]


def test_finite_grid_boundary_requires_smaller_points_to_be_inferior():
    result = minimum_alpha_summary(
        [
            _comparison(0.001, "inferior"),
            _comparison(0.002, "supported"),
            _comparison(0.005, "inferior"),
            _comparison(0.01, "supported"),
        ]
    )["widths"][0]
    assert result["threshold_identified"]
    assert result["minimum_supported_tested_alpha"] == 0.002
    assert not result["monotonic_quality_assumed"]


def test_missing_grid_points_and_duplicates_are_not_hidden():
    result = minimum_alpha_summary([_comparison(0.002, "supported")])["widths"][0]
    assert result["missing_alphas"] == [0.001, 0.005, 0.01]
    assert not result["threshold_identified"]
    with pytest.raises(ValueError):
        minimum_alpha_summary(
            [_comparison(0.001, "supported"), _comparison(0.001, "supported")]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("primary_configurations", 16),
        ("seeds", [17, 29, 43]),
        ("targets_sha256", "different"),
    ],
)
def test_minimum_summary_refuses_mixed_families_and_benchmarks(field, value):
    first = _comparison(0.001, "supported")
    second = _comparison(0.002, "supported")
    second[field] = value
    with pytest.raises(ValueError):
        minimum_alpha_summary([first, second])


def test_iut_primary_can_support_when_diagnostic_joint_band_crosses_margin():
    endpoint = seed_difference_statistics([0.0022, 0.0032, 0.0042], 0.002)
    assert endpoint["noninferiority_one_sided_p"] < 0.05 / PRIMARY_CONFIGS
    assert endpoint["primary_one_sided_seed_t_lower"] > -0.002
    assert endpoint["simultaneous_seed_t_lower"] < -0.002
    assert (
        configuration_status({"accuracy": endpoint, "roc_auc": endpoint}) == "supported"
    )


def test_iut_requires_both_coprimary_metrics_and_zero_variance_boundary_is_null():
    supported = seed_difference_statistics([0, 0, 0], 0.002)
    boundary = seed_difference_statistics([-0.002] * 3, 0.002)
    assert supported["noninferiority_one_sided_p"] == 0
    assert boundary["noninferiority_one_sided_p"] == 1
    assert boundary["noninferiority_t_statistic"] is None
    assert (
        configuration_status({"accuracy": supported, "roc_auc": boundary})
        == "inconclusive"
    )
    inferior = seed_difference_statistics([-0.003] * 3, 0.002)
    assert (
        configuration_status({"accuracy": supported, "roc_auc": inferior}) == "inferior"
    )


def test_exact_df2_survival_matches_quantiles_and_remains_stable():
    critical = student_t_df2_ppf(1 - 0.05 / PRIMARY_CONFIGS)
    assert critical == pytest.approx(8.8602000344, rel=1e-10)
    for probability in (0.001, 0.025, 0.5, 0.95, 0.999):
        assert student_t_df2_sf(student_t_df2_ppf(probability)) == pytest.approx(
            1 - probability
        )
    assert 0 < student_t_df2_sf(1e100) < 1e-190
