"""Versioned admission of fixed-shape Graph lifecycle evidence (no CUDA imports).

The original per-epoch equality flags remain historical evidence. This rule
requires exactly eight cycles; it never searches for a convenient flat suffix.
"""

from collections.abc import Mapping

GATE_VERSION = "graph-lifecycle-memory-v2"
ENDPOINT_PHASE = "after_eval_train_sync_before_checkpoint_and_graph_retirement"
ALLOCATED_BUDGET = 2 * 1024**2
RESERVED_BUDGET = 8 * 1024**2


def evaluate_lifecycle(
    baseline: Mapping,
    candidate: Mapping,
    *,
    evidence_scope: str,
    baseline_run: str,
    candidate_run: str,
    endpoint_phase: str,
) -> dict:
    """Evaluate recorded hashes/endpoints, not the underlying Tensor bytes.

    ``endpoint_phase`` is explicit because old reports did not name their
    sampling phase. A caller interpreting old evidence must verify that phase
    against its runner. Missing evidence is indeterminate, never a pass.
    """
    if evidence_scope not in ("retrospective_supplemental", "prospective"):
        raise ValueError("unknown evidence scope")
    if not baseline_run or not candidate_run:
        raise ValueError("named baseline and candidate runs are required")
    result = {
        "gate_version": GATE_VERSION,
        "evidence_scope": evidence_scope,
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
        "endpoint_phase": endpoint_phase,
        "numerical_evidence": "recorded Tensor hashes/dtypes/shapes and metrics; no Tensor loading",
        "cycles": 8,
        "stable_cycles": [3, 4, 5, 6, 7, 8],
        "allocated_budget_bytes": ALLOCATED_BUDGET,
        "reserved_budget_bytes": RESERVED_BUDGET,
        "status": "indeterminate",
        "passed": None,
        "checks": {},
        "reasons": [],
    }
    if endpoint_phase != ENDPOINT_PHASE:
        result["reasons"] = ["endpoint phase is not the versioned measurement phase"]
        return result
    try:
        groups = [baseline["epochs"], candidate["epochs"]]
        if any(len(rows) < 8 for rows in groups):
            result["reasons"] = ["eight complete cycles are required"]
            return result
        if any(len(rows) != 8 for rows in groups):
            result.update(
                status="fail",
                passed=False,
                reasons=["exactly eight cycles required; no suffix selection"],
            )
            return result
        checks = result["checks"]
        checks["fixed_cycles_and_updates"] = all(
            r["total_updates"] == 128
            and r["replay_updates"] == 120
            and [e["epoch"] for e in r["epochs"]] == list(range(1, 9))
            and all(
                e["steps"] == 16 and e["total_updates"] == 16 * e["epoch"]
                for e in r["epochs"]
            )
            for r in (baseline, candidate)
        )
        checks["recorded_phase_matches"] = all(
            r.get("memory_endpoint_phase", endpoint_phase) == endpoint_phase
            for r in (baseline, candidate)
        )
        checks["same_start_and_workload"] = all(
            baseline[k] == candidate[k]
            for k in (
                "initial_state",
                "initialization_sha256",
                "optimizer_contract",
                "data",
            )
        )
        checks["recorded_states_gradients_metrics_equal"] = all(
            len(a["state"]) == len(b["state"]) == 42
            and len(a["gradients"]) == len(b["gradients"]) == 5
            and all(
                a[k] == b[k]
                for k in (
                    "state",
                    "gradients",
                    "totals",
                    "train",
                    "test",
                    "data_order_sha256",
                )
            )
            for a, b in zip(*groups)
        )
        checks["no_unexpected_fallback"] = all(
            r["unexpected_fallbacks"] == 0 for r in (baseline, candidate)
        )
        for kind, budget in (
            ("allocated", ALLOCATED_BUDGET),
            ("reserved", RESERVED_BUDGET),
        ):
            peak = f"peak_{kind}_bytes"
            endpoint = f"end_{kind}_bytes"
            values = [r[peak] for r in (baseline, candidate)] + [
                e[endpoint] for rows in groups for e in rows
            ]
            if any(type(v) is not int or v < 0 for v in values):
                raise ValueError("invalid byte count")
            delta = candidate[peak] - baseline[peak]
            result[f"{kind}_peak_delta_bytes"] = delta
            checks[f"{kind}_peak_budget"] = delta <= budget
            checks[f"{kind}_peak_covers_endpoints"] = all(
                r[peak] >= max(e[endpoint] for e in r["epochs"])
                for r in (baseline, candidate)
            )
            for label, rows in zip(("baseline", "candidate"), groups):
                result[f"{label}_{kind}_endpoints_bytes"] = [e[endpoint] for e in rows]
                checks[f"{label}_{kind}_cycles_3_to_8_constant"] = (
                    len({e[endpoint] for e in rows[2:]}) == 1
                )
    except (KeyError, TypeError, ValueError) as error:
        result["reasons"] = [f"incomplete or invalid evidence: {error}"]
        return result
    failed = [name for name, passed in checks.items() if not passed]
    result.update(
        status="fail" if failed else "pass", passed=not failed, reasons=failed
    )
    return result
