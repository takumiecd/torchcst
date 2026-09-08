"""Observed threshold times; never interpolate or replace censored trials."""


def summarize_targets(evaluations, targets, consecutive=3):
    if consecutive < 1:
        raise ValueError("consecutive must be positive")
    result = {}
    for target in targets:
        if not 0 < target <= 1:
            raise ValueError("accuracy target must be in (0, 1]")
        first = confirmed = beginning = None
        streak = 0
        for row in evaluations:
            if row["accuracy"] >= target:
                if first is None:
                    first = row
                if streak == 0:
                    beginning = row
                streak += 1
                if streak >= consecutive and confirmed is None:
                    confirmed = {"beginning": beginning, "confirmation": row}
            else:
                streak = 0
        result[str(target)] = {
            "first_observed": first,
            "consecutive_confirmed": confirmed,
            "required_consecutive_evaluations": consecutive,
        }
    return result
