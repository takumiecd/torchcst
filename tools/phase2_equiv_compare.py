"""Phase 2 S0 comparator: PASS/FAIL two fingerprints against preregistered tolerances.

Takes two fingerprint JSON files produced by ``phase2_equiv_harness.py`` --
conventionally a baseline (captured on pre-migration main,
``tools/phase2_baseline.json``) and a candidate (captured on a later Phase 2
stage's branch) -- and decides, fixture by fixture, whether the candidate's
behavior is statistically equivalent to the baseline's.

``docs/policy-tree-phase2.md`` retires op_log bit-compatibility as the
Phase 2 acceptance gate; this comparator is what replaces it. Every stage
S1..S5 must PASS this comparator against the S0 baseline before it lands.

------------------------------------------------------------------------
PREREGISTERED TOLERANCES -- DO NOT RELAX. Per the design note's discipline
("事前登録してから着手"), these numbers were chosen when this file was
written, before any Phase 2 semantics changed. If a later stage fails a
check, the fix is to fix that stage's semantics (or, in a rare and
deliberate case, to have the user explicitly re-register a new tolerance in
a follow-up commit that says so) -- never to quietly loosen the constant
here to make a failing diff go away. Tightening a tolerance is always fine;
only loosening needs a fresh, explicit decision.
------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Relative difference in the seed-cross mean of the fixture's final training
# loss (baseline denominator; see _rel_diff).
FINAL_LOSS_REL_TOL = 0.05

# Relative difference, at every event index, of the seed-cross mean live-atom
# count (K trajectory).
K_TRAJECTORY_REL_TOL = 0.10

# Relative difference in the seed-cross mean total count of each op kind
# (SynapseBirth, SynapseDeath, SynapseAbsorb, NeuronUngate, NeuronRetire, ...).
OP_COUNT_REL_TOL = 0.15

# Structural event totals (how many events actually fired, per
# docs/policy-tree-phase2.md's cadence-owns-"is this an event?" semantics)
# must match exactly when the fixture's cadence is unchanged between the two
# runs -- there is no preregistered slack here by design.
EVENT_TOTAL_EXACT = True

_EPS = 1e-12


def _rel_diff(baseline: float, candidate: float) -> float:
    """Relative difference of ``candidate`` from ``baseline``.

    Both being (numerically) zero is a perfect match. Only the baseline being
    zero while the candidate is not is a maximal mismatch -- reported as
    ``inf`` so it always fails a finite tolerance check, rather than being
    silently swallowed by an epsilon-padded denominator.
    """
    if abs(baseline) < _EPS and abs(candidate) < _EPS:
        return 0.0
    if abs(baseline) < _EPS:
        return float("inf")
    return abs(candidate - baseline) / abs(baseline)


class FixtureReport:
    def __init__(self, name: str) -> None:
        self.name = name
        self.failures: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)

    @property
    def passed(self) -> bool:
        return not self.failures


def _compare_final_loss(baseline: dict, candidate: dict, report: FixtureReport) -> None:
    base_mean = baseline["summary"]["final_loss"]["mean"]
    cand_mean = candidate["summary"]["final_loss"]["mean"]
    diff = _rel_diff(base_mean, cand_mean)
    if diff > FINAL_LOSS_REL_TOL:
        report.fail(
            f"final_loss seed-cross mean: baseline={base_mean!r} "
            f"candidate={cand_mean!r} rel_diff={diff:.4f} > tol={FINAL_LOSS_REL_TOL}"
        )


def _compare_event_totals(baseline: dict, candidate: dict, report: FixtureReport) -> None:
    base_mean = baseline["summary"]["event_total"]["mean"]
    cand_mean = candidate["summary"]["event_total"]["mean"]
    if EVENT_TOTAL_EXACT:
        if base_mean != cand_mean:
            report.fail(
                f"event_total seed-cross mean: baseline={base_mean!r} "
                f"candidate={cand_mean!r} (exact match required -- cadence "
                "unchanged means structural event count must be identical)"
            )


def _compare_k_trajectory(baseline: dict, candidate: dict, report: FixtureReport) -> None:
    base_traj = baseline["summary"]["k_trajectory"]["mean"]
    cand_traj = candidate["summary"]["k_trajectory"]["mean"]
    if len(base_traj) != len(cand_traj):
        report.fail(
            f"k_trajectory length differs: baseline={len(base_traj)} "
            f"candidate={len(cand_traj)} events -- cannot compare pointwise"
        )
        return
    worst_index = -1
    worst_diff = 0.0
    for index, (base_value, cand_value) in enumerate(zip(base_traj, cand_traj)):
        diff = _rel_diff(base_value, cand_value)
        if diff > K_TRAJECTORY_REL_TOL and diff > worst_diff:
            worst_diff = diff
            worst_index = index
    if worst_index >= 0:
        report.fail(
            f"k_trajectory[{worst_index}] seed-cross mean: "
            f"baseline={base_traj[worst_index]!r} candidate={cand_traj[worst_index]!r} "
            f"rel_diff={worst_diff:.4f} > tol={K_TRAJECTORY_REL_TOL} "
            f"(worst of {len(base_traj)} event indices)"
        )


def _compare_op_counts(baseline: dict, candidate: dict, report: FixtureReport) -> None:
    base_counts = baseline["summary"]["op_counts"]
    cand_counts = candidate["summary"]["op_counts"]
    keys = sorted(set(base_counts) | set(cand_counts))
    for key in keys:
        base_mean = base_counts.get(key, {"mean": 0.0})["mean"]
        cand_mean = cand_counts.get(key, {"mean": 0.0})["mean"]
        diff = _rel_diff(base_mean, cand_mean)
        if diff > OP_COUNT_REL_TOL:
            report.fail(
                f"op_counts[{key!r}] seed-cross mean: baseline={base_mean!r} "
                f"candidate={cand_mean!r} rel_diff={diff:.4f} > tol={OP_COUNT_REL_TOL}"
            )


def compare_fixture(name: str, baseline: dict, candidate: dict) -> FixtureReport:
    report = FixtureReport(name)
    _compare_final_loss(baseline, candidate, report)
    _compare_event_totals(baseline, candidate, report)
    _compare_k_trajectory(baseline, candidate, report)
    _compare_op_counts(baseline, candidate, report)
    return report


def compare_fingerprints(baseline: dict, candidate: dict) -> tuple[bool, list[FixtureReport], list[str]]:
    """Return ``(overall_pass, per_fixture_reports, structural_errors)``.

    ``structural_errors`` covers mismatches in the fixture *set* itself
    (missing/extra fixtures between the two fingerprints) -- distinct from a
    fixture that is present in both but fails its tolerance checks.
    """
    base_fixtures = baseline.get("fixtures", {})
    cand_fixtures = candidate.get("fixtures", {})
    base_names = set(base_fixtures)
    cand_names = set(cand_fixtures)

    structural_errors: list[str] = []
    missing = base_names - cand_names
    extra = cand_names - base_names
    if missing:
        structural_errors.append(f"candidate is missing fixture(s): {sorted(missing)!r}")
    if extra:
        structural_errors.append(f"candidate has unexpected extra fixture(s): {sorted(extra)!r}")

    reports = [
        compare_fixture(name, base_fixtures[name], cand_fixtures[name])
        for name in sorted(base_names & cand_names)
    ]
    overall_pass = not structural_errors and all(report.passed for report in reports)
    return overall_pass, reports, structural_errors


def _load(path: Path) -> dict[str, Any]:
    if str(path) == "-":
        return json.loads(sys.stdin.read())
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="baseline fingerprint JSON path (or '-' for stdin)")
    parser.add_argument("candidate", type=Path, help="candidate fingerprint JSON path (or '-' for stdin)")
    args = parser.parse_args(argv)

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    overall_pass, reports, structural_errors = compare_fingerprints(baseline, candidate)

    for error in structural_errors:
        print(f"STRUCTURAL FAIL: {error}")
    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        print(f"[{status}] {report.name}")
        for failure in report.failures:
            print(f"    - {failure}")

    print()
    print("OVERALL:", "PASS" if overall_pass else "FAIL")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
