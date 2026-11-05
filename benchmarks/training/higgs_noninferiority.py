"""Preregistered HIGGS confirmation statistics; no training or GPU dependency.

The primary family contains 2 widths * 4 densities = 8 fixed configurations.
Each configuration must reject both co-primary noninferiority nulls (IUT), with
Bonferroni adjustment across configurations. Diagnostic simultaneous two-sided
95% Bonferroni bands cover 16 endpoints. Seed-t inference conditions on the
fixed benchmark and assumes approximately normal IID paired seed differences.
Three seeds do not establish that approximation. Conditional
paired-example bootstrap intervals are a separate diagnostic, not a joint
seed-and-example confidence guarantee.

References:
https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
https://pubmed.ncbi.nlm.nih.gov/3203132/
https://www.fda.gov/media/162416/download?attachment=
https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm

The bootstrap below recomputes exact tie-aware weighted empirical AUC rather
than using the DeLong asymptotic covariance estimator.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

CONFIRMATION_WIDTHS = (1024, 2048)
CONFIRMATION_ALPHAS = (0.001, 0.002, 0.005, 0.01)
CONFIRMATION_SEEDS = (29, 43, 71)
CONFIDENCE = 0.95
MARGINS = {"accuracy": 0.002, "roc_auc": 0.002}
PRIMARY_CONFIGS = len(CONFIRMATION_WIDTHS) * len(CONFIRMATION_ALPHAS)
DIAGNOSTIC_ENDPOINTS = PRIMARY_CONFIGS * len(MARGINS)
BOOTSTRAP_SEED = 20261105
BOOTSTRAP_REPLICATES = 2000
_SCOPE = (
    "Primary familywise IUT paired-seed t tests are conditional on this fixed benchmark "
    "and rely on approximately normal IID paired seed differences. Three seeds "
    "provide little power and cannot validate normality. Paired-example "
    "bootstrap intervals condition on the six learned models and do not "
    "provide a combined joint seed/example confidence guarantee."
)


def student_t_df2_ppf(probability):
    """Exact inverse CDF for Student t with df=2, without SciPy."""
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 < probability < 1
    ):
        raise ValueError("probability must be finite and in (0,1)")
    return (2 * probability - 1) / math.sqrt(2 * probability * (1 - probability))


def student_t_df2_sf(value):
    """Stable exact upper-tail probability for a finite df=2 t statistic."""
    if not math.isfinite(value):
        raise ValueError("t statistic must be finite")
    if value < 0:
        return 1 - student_t_df2_sf(-value)
    root = math.hypot(value, math.sqrt(2))
    return (1 / root) / (root + value)


def seed_difference_statistics(differences, margin):
    """Paired-seed endpoint statistics, with conservative zero-variance nulls."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.shape != (3,) or not np.isfinite(differences).all():
        raise ValueError("three finite paired seed differences are required")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    mean = float(differences.mean())
    deviation = float(differences.std(ddof=1))
    error = deviation / math.sqrt(3)
    shifted = mean + margin
    statistic = shifted / error if error else None
    p_value = student_t_df2_sf(statistic) if error else (0.0 if shifted > 0 else 1.0)
    diagnostic_critical = student_t_df2_ppf(
        1 - (1 - CONFIDENCE) / (2 * DIAGNOSTIC_ENDPOINTS)
    )
    return {
        "margin": margin,
        "difference_per_seed": differences.tolist(),
        "mean_difference": mean,
        "seed_standard_deviation": deviation,
        "seed_standard_error": error,
        "degenerate_seed_variance": deviation == 0,
        "noninferiority_t_statistic": statistic,
        "noninferiority_one_sided_p": p_value,
        "primary_one_sided_seed_t_lower": mean
        - student_t_df2_ppf(1 - (1 - CONFIDENCE) / PRIMARY_CONFIGS) * error,
        "simultaneous_seed_t_lower": mean - diagnostic_critical * error,
        "simultaneous_seed_t_upper": mean + diagnostic_critical * error,
        "pointwise_one_sided_95_lower_diagnostic": mean
        - student_t_df2_ppf(CONFIDENCE) * error,
        "all_three_seed_point_estimates_within_margin": bool(
            (differences >= -margin).all()
        ),
    }


def configuration_status(metrics):
    """IUT support; strong diagnostic inferiority; otherwise inconclusive."""
    configuration_p = max(
        metric["noninferiority_one_sided_p"] for metric in metrics.values()
    )
    if configuration_p <= (1 - CONFIDENCE) / PRIMARY_CONFIGS:
        return "supported"
    if any(
        metric["simultaneous_seed_t_upper"] < -metric["margin"]
        for metric in metrics.values()
    ):
        return "inferior"
    return "inconclusive"


def _arrays(cst_scores, dense_scores, targets, seeds):
    raw_cst, raw_dense = np.asarray(cst_scores), np.asarray(dense_scores)
    y = np.asarray(targets)
    if raw_cst.dtype.kind not in "fiu" or raw_dense.dtype.kind not in "fiu":
        raise ValueError("scores must be real numerical arrays")
    if y.dtype.kind not in "biu" or y.ndim != 1 or len(y) < 2:
        raise ValueError("targets must be a one-dimensional integer binary array")
    if not np.isin(y, (0, 1)).all() or not (y == 0).any() or not (y == 1).any():
        raise ValueError("both binary target classes are required")
    cst = np.asarray(raw_cst, dtype=np.float64)
    dense = np.asarray(raw_dense, dtype=np.float64)
    if cst.shape != dense.shape or cst.shape != (3, len(y)):
        raise ValueError("scores must have shape [three paired seeds, examples]")
    if not np.isfinite(cst).all() or not np.isfinite(dense).all():
        raise ValueError("scores must be finite")
    if (
        len(seeds) != 3
        or any(type(seed) is not int for seed in seeds)
        or set(seeds) != set(CONFIRMATION_SEEDS)
    ):
        raise ValueError("the preregistered confirmation seeds are 29,43,71")
    order = [tuple(seeds).index(seed) for seed in CONFIRMATION_SEEDS]
    return cst[order], dense[order], y.astype(np.int64, copy=False)


def _prepare_auc(scores, targets):
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    return order, targets[order], starts


def _weighted_auc(cache, weights, positives, negatives):
    order, sorted_targets, starts = cache
    sorted_weights = weights[order]
    positive = np.add.reduceat(sorted_weights * sorted_targets, starts)
    negative = np.add.reduceat(sorted_weights * (1 - sorted_targets), starts)
    # Every positive receives credit for negatives strictly below its score
    # plus half of the negatives tied within the same score group.
    concordance = (positive * (np.cumsum(negative) - 0.5 * negative)).sum(
        dtype=np.float64
    )
    return float(concordance / (positives * negatives))


def paired_case_bootstrap(
    cst_scores,
    dense_scores,
    targets,
    *,
    seeds=CONFIRMATION_SEEDS,
    replicates=BOOTSTRAP_REPLICATES,
    random_seed=BOOTSTRAP_SEED,
):
    """Exact paired percentile intervals conditional on fixed learned models.

    One resampled example-weight vector is shared by both methods and all
    three seeds. The statistic is the mean of the three single-model metrics,
    not the metric of an ensemble. Missing-class draws are rejected; this is
    negligible for the declared 50,000-example HIGGS benchmark.
    """
    cst, dense, y = _arrays(cst_scores, dense_scores, targets, seeds)
    if type(replicates) is not int or replicates < 20:
        raise ValueError("at least twenty bootstrap replicates are required")
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError("bootstrap random_seed must be a nonnegative integer")
    n = len(y)
    caches = [
        [_prepare_auc(method[seed], y) for seed in range(3)] for method in (cst, dense)
    ]
    correctness_difference = (
        ((cst > 0) == y[None]).astype(np.float64)
        - ((dense > 0) == y[None]).astype(np.float64)
    ).mean(axis=0)
    rng = np.random.default_rng(random_seed)
    probabilities = np.full(n, 1 / n, dtype=np.float64)
    differences = np.empty((replicates, 2), dtype=np.float64)
    completed = rejected = 0
    while completed < replicates:
        if rejected > 100 + 10 * replicates:
            raise RuntimeError("too many bootstrap draws lack a target class")
        weights = rng.multinomial(n, probabilities)
        positives = int(weights @ y)
        negatives = n - positives
        if not positives or not negatives:
            rejected += 1
            continue
        differences[completed, 0] = float(weights @ correctness_difference / n)
        aucs = [
            [
                _weighted_auc(cache, weights, positives, negatives)
                for cache in method_caches
            ]
            for method_caches in caches
        ]
        differences[completed, 1] = np.mean(np.asarray(aucs[0]) - np.asarray(aucs[1]))
        completed += 1
    bounds = np.quantile(differences, (0.025, 0.975), axis=0, method="linear")
    return {
        "scope": (
            "conditional on fixed learned models; paired-example uncertainty "
            "only; no seed-population or combined confidence guarantee"
        ),
        "algorithm": (
            "NumPy default_rng multinomial paired nonparametric percentile "
            "bootstrap; shared weights across methods and all seeds; exact "
            "weighted tie-aware empirical AUC; mean single-seed metrics"
        ),
        "confidence": CONFIDENCE,
        "numpy": np.__version__,
        "bit_generator": type(rng.bit_generator).__name__,
        "replicates": replicates,
        "random_seed": random_seed,
        "missing_class_draws_rejected": rejected,
        "accuracy": {
            "lower": float(bounds[0, 0]),
            "upper": float(bounds[1, 0]),
        },
        "roc_auc": {
            "lower": float(bounds[0, 1]),
            "upper": float(bounds[1, 1]),
        },
    }


def compare_predictions(
    cst_scores,
    dense_scores,
    targets,
    *,
    width,
    alpha,
    seeds=CONFIRMATION_SEEDS,
    bootstrap_replicates=BOOTSTRAP_REPLICATES,
    bootstrap_seed=BOOTSTRAP_SEED,
):
    """Classify one fixed width/density in the preregistered family.

    Scores must be the retained FP32 logit margin, class1 minus class0. Class1
    is predicted only for score>0; exact ties select class0, like argmax.
    """
    if type(width) is not int or width not in CONFIRMATION_WIDTHS:
        raise ValueError("width is outside the preregistered family")
    if type(alpha) not in (float, int) or alpha not in CONFIRMATION_ALPHAS:
        raise ValueError("alpha is outside the preregistered family")
    cst, dense, y = _arrays(cst_scores, dense_scores, targets, seeds)
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    weights = np.ones(len(y), dtype=np.int64)
    metrics = {}
    simultaneous_critical = student_t_df2_ppf(
        1 - (1 - CONFIDENCE) / (2 * DIAGNOSTIC_ENDPOINTS)
    )
    for name, margin in MARGINS.items():
        if name == "accuracy":
            cst_metric = ((cst > 0) == y[None]).mean(axis=1)
            dense_metric = ((dense > 0) == y[None]).mean(axis=1)
        else:
            values = [
                [
                    _weighted_auc(
                        _prepare_auc(method[seed], y),
                        weights,
                        n_positive,
                        n_negative,
                    )
                    for seed in range(3)
                ]
                for method in (cst, dense)
            ]
            cst_metric, dense_metric = map(np.asarray, values)
        differences = cst_metric - dense_metric
        metrics[name] = {
            **seed_difference_statistics(differences, margin),
            "cst_per_seed": cst_metric.tolist(),
            "dense_per_seed": dense_metric.tolist(),
        }
    status = configuration_status(metrics)
    configuration_p = max(
        metric["noninferiority_one_sided_p"] for metric in metrics.values()
    )
    bootstrap = paired_case_bootstrap(
        cst,
        dense,
        y,
        seeds=CONFIRMATION_SEEDS,
        replicates=bootstrap_replicates,
        random_seed=bootstrap_seed,
    )
    return {
        "schema_version": 2,
        "width": width,
        "alpha": float(alpha),
        "seeds": list(CONFIRMATION_SEEDS),
        "examples": len(y),
        "class_counts": {"positive": n_positive, "negative": n_negative},
        "targets_sha256": hashlib.sha256(
            np.ascontiguousarray(y, dtype="<i8").tobytes()
        ).hexdigest(),
        "score_definition": "retained FP32 class1-class0 logit margin; ties predict class0",
        "primary_method": (
            "intersection-union of two co-primary one-sided paired-seed "
            "noninferiority t tests; df=2; configuration p=max(endpoint p); "
            "Bonferroni over 8 fixed width/density configurations"
        ),
        "primary_sources": [
            "https://www.fda.gov/media/162416/download?attachment=",
            "https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm",
        ],
        "primary_configurations": PRIMARY_CONFIGS,
        "primary_configuration_p": configuration_p,
        "primary_bonferroni_adjusted_p": min(1.0, PRIMARY_CONFIGS * configuration_p),
        "primary_configuration_significance": (1 - CONFIDENCE) / PRIMARY_CONFIGS,
        "primary_seed_t_critical": student_t_df2_ppf(
            1 - (1 - CONFIDENCE) / PRIMARY_CONFIGS
        ),
        "family_confidence": CONFIDENCE,
        "diagnostic_simultaneous_endpoint_count": DIAGNOSTIC_ENDPOINTS,
        "diagnostic_simultaneous_method": (
            "two-sided Bonferroni simultaneous 95% paired-seed Student-t "
            "bands; df=2; 16 endpoints; separate from primary IUT support"
        ),
        "simultaneous_seed_t_critical": simultaneous_critical,
        "scope": _SCOPE,
        "status": status,
        "metrics": metrics,
        "empirical_all_three_seed_margin_pass_candidate": all(
            metric["all_three_seed_point_estimates_within_margin"]
            for metric in metrics.values()
        ),
        "conditional_paired_case_bootstrap": bootstrap,
        "conditional_case_bootstrap_margin_pass_diagnostic": all(
            bootstrap[name]["lower"] >= -MARGINS[name] for name in MARGINS
        ),
        "combined_joint_seed_example_confidence_claim": False,
    }


def minimum_alpha_summary(comparisons):
    """Summarize supported tested points without assuming monotonic quality."""
    identities = set()
    benchmark_identity = None
    by_width = {width: [] for width in CONFIRMATION_WIDTHS}
    for comparison in comparisons:
        if (
            comparison.get("primary_configurations") != PRIMARY_CONFIGS
            or comparison.get("diagnostic_simultaneous_endpoint_count")
            != DIAGNOSTIC_ENDPOINTS
            or comparison.get("seeds") != list(CONFIRMATION_SEEDS)
            or type(comparison.get("examples")) is not int
            or comparison["examples"] < 2
            or type(comparison.get("targets_sha256")) is not str
        ):
            raise ValueError(
                "comparison lacks the preregistered family/benchmark identity"
            )
        current_benchmark = comparison["examples"], comparison["targets_sha256"]
        if benchmark_identity is None:
            benchmark_identity = current_benchmark
        elif current_benchmark != benchmark_identity:
            raise ValueError("comparisons use different confirmation benchmarks")
        identity = comparison["width"], comparison["alpha"]
        if (
            identity[0] not in CONFIRMATION_WIDTHS
            or identity[1] not in CONFIRMATION_ALPHAS
            or comparison["status"] not in ("supported", "inferior", "inconclusive")
            or identity in identities
        ):
            raise ValueError("duplicate or undeclared confirmation comparison")
        identities.add(identity)
        by_width[identity[0]].append(comparison)
    result = []
    for width, rows in by_width.items():
        rows = sorted(rows, key=lambda row: row["alpha"])
        statuses = {row["alpha"]: row["status"] for row in rows}
        missing = [alpha for alpha in CONFIRMATION_ALPHAS if alpha not in statuses]
        supported = [row["alpha"] for row in rows if row["status"] == "supported"]
        minimum = min(supported) if supported else None
        smaller = (
            [alpha for alpha in CONFIRMATION_ALPHAS if alpha < minimum]
            if minimum is not None
            else []
        )
        endpoint = minimum == CONFIRMATION_ALPHAS[0]
        lower_inconclusive = [
            alpha for alpha in smaller if statuses.get(alpha) == "inconclusive"
        ]
        lower_missing = [alpha for alpha in smaller if alpha not in statuses]
        threshold_identified = bool(
            minimum is not None
            and not endpoint
            and not missing
            and all(statuses[alpha] == "inferior" for alpha in smaller)
        )
        result.append(
            {
                "width": width,
                "preset_alphas": list(CONFIRMATION_ALPHAS),
                "labels": [
                    {
                        "alpha": row["alpha"],
                        "status": row["status"],
                        "empirical_candidate": row.get(
                            "empirical_all_three_seed_margin_pass_candidate"
                        ),
                    }
                    for row in rows
                ],
                "missing_alphas": missing,
                "minimum_supported_tested_alpha": minimum,
                "threshold_identified": threshold_identified,
                "threshold_scope": (
                    "finite preregistered grid only; no continuous or universal minimum"
                ),
                "threshold_identified_definition": (
                    "the smallest supported preset point is distinguished from all "
                    "smaller preset points; no monotone threshold is asserted"
                ),
                "lower_range_endpoint_supported": endpoint,
                "minimum_alpha_leq_within_declared_protocol": minimum
                if endpoint
                else None,
                "smaller_inconclusive_alphas": lower_inconclusive,
                "smaller_missing_alphas": lower_missing,
                "monotonic_quality_assumed": False,
                "statement": (
                    "The lower range endpoint is supported; alpha_min may be below 0.001."
                    if endpoint
                    else "A boundary is distinguished only among the preset grid points."
                    if threshold_identified
                    else "No minimum-density threshold is identified."
                ),
            }
        )
    return {
        "schema_version": 1,
        "scope": _SCOPE,
        "primary_configurations": PRIMARY_CONFIGS,
        "diagnostic_simultaneous_endpoint_count": DIAGNOSTIC_ENDPOINTS,
        "combined_joint_seed_example_confidence_claim": False,
        "widths": result,
    }
