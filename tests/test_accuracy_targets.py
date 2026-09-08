"""Threshold reporting must preserve oscillations, censoring and confirmation time."""

from experiments.accuracy_targets import summarize_targets


def test_confirm_only_after_streak_and_keep_first_observation():
    evaluations = [
        {"step": step, "accuracy": accuracy, "training_seconds": step / 2}
        for step, accuracy in [
            (0, 0.1),
            (8, 0.8),
            (16, 0.79),
            (24, 0.81),
            (32, 0.82),
            (40, 0.8),
        ]
    ]
    result = summarize_targets(evaluations, [0.8, 0.85], consecutive=3)
    assert result["0.8"]["first_observed"]["step"] == 8
    confirmed = result["0.8"]["consecutive_confirmed"]
    assert confirmed["beginning"]["step"] == 24
    assert confirmed["confirmation"]["step"] == 40
    assert confirmed["confirmation"]["training_seconds"] == 20
    assert result["0.85"]["first_observed"] is None
    assert result["0.85"]["consecutive_confirmed"] is None
    # Truncated/failed trials do not borrow confirmation from an imagined future.
    short = summarize_targets(evaluations[:-1], [0.8], consecutive=3)
    assert short["0.8"]["first_observed"]["step"] == 8
    assert short["0.8"]["consecutive_confirmed"] is None


def test_aggregate_keeps_nonarrival_out_of_time_estimate(tmp_path):
    import json

    from experiments.summarize_accuracy_time import aggregate

    paths = []
    for seed, accuracy in [(17, 0.81), (29, 0.79)]:
        evaluations = [
            {
                "step": 8,
                "accuracy": accuracy,
                "training_seconds": 4,
                "wall_seconds": 5,
                "training_seconds_after_two_updates": 2,
                "cumulative_peak_allocated_mib": 60,
                "cumulative_warm_peak_allocated_mib": None,
            }
        ]
        report = {
            "status": "passed",
            "protocol": {"batch": 128},
            "config": {
                "seed": seed,
                "solver": "spectral",
                "recompression_action": "pair",
                "steps": 8,
                "eval_every": 8,
                "targets": [0.8],
                "consecutive": 1,
                "atoms": 64,
            },
            "initial_parameter_sha256": str(seed),
            "batch_permutation_sha256": str(seed),
            "evaluations": evaluations,
            "target_times": summarize_targets(evaluations, [0.8], 1),
            "source_sha256": {"fixture": "same"},
            "rows": [
                {"seconds": 4, "passed": True, "memory": {"peak_allocated_mib": 60}}
            ],
        }
        path = tmp_path / f"s{seed}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    result = aggregate(paths)["summary"]["spectral-pair"]["targets"]["0.8"][
        "consecutive_confirmed"
    ]
    assert result["reached"] == 1 and result["trials"] == 2
    assert result["reached_only_median_training_seconds"] == 4
    assert result["reached_only_warm_peak_allocated_mib_range"] is None
