"""Fixed cohorts → generic metrics → scores → leaderboard → exact artifact."""

import copy
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from benchmarks.cuda.linear.protocol import LOCAL_OPTIMIZER_POLICY, measurement_operator
from benchmarks.database.model import digest
from torchcst._backends.cuda.dispatch.conditions import condition_key, dump_condition
from torchcst._backends.cuda.dispatch.exact import validate_selector_artifact
from torchcst._backends.cuda.schema import DeviceInfo, DispatchContext, RequiredGrads

from .model import Candidate, MetricKey, Summary
from .request import validate_request


@dataclass(frozen=True)
class ScorePolicy:
    id: str
    revision: str
    parameters: dict
    function: Callable[[Candidate], float | None]

    def declaration(self):
        return {"id": self.id, "revision": self.revision, "parameters": self.parameters}


def preset_policy(declaration, mode):
    name, revision, params = (declaration[k] for k in ("id", "revision", "parameters"))
    if revision != "v1" or name not in ("speed", "memory"):
        raise ValueError(
            "unknown preset; custom functions use the Python ScorePolicy API"
        )
    permitted = {"max_peak_allocated_bytes"} if name == "speed" else {"max_time_ms"}
    if set(params) - permitted:
        raise ValueError("unknown preset parameter")
    for value in params.values():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("preset limits must be finite and positive")
    scope = "graph" if mode == "cuda_graph" else "eager"

    def score(candidate):
        time = candidate.metric("step.time", scope, "ms", "median")
        memory = candidate.metric("memory.allocated", "capture_replay", "bytes", "peak")
        if time is None or memory is None:
            return None
        if memory.maximum > params.get(
            "max_peak_allocated_bytes", math.inf
        ) or time.median > params.get("max_time_ms", math.inf):
            return None
        return time.median if name == "speed" else memory.median

    return ScorePolicy(name, revision, copy.deepcopy(params), score)


def _context(row, mode):
    """Protocol-specific reconstruction, never an arbitrary JSON type loader."""
    case, env = row["case_declaration"], row["environment"]
    n = case["size"]
    local = case["fixture"] == "local_polar_product"
    if row["adapter_revision"] != (3 if local else 2):
        raise ValueError("generation fixture/adapter revision differs")
    operator = measurement_operator(case)
    result = row["payload"]["result"]
    if (
        result["operator"] != json.loads(json.dumps(asdict(operator), allow_nan=False))
        or result["rows"] != case["rows"]
        or result["atoms"] != case["atoms"]
    ):
        raise ValueError("stored worker differs from generation fixture")
    if local and (
        row["protocol"].get("optimizer_policy") != LOCAL_OPTIMIZER_POLICY
        or row["protocol"].get("polar_update") not in ("torch", "fused")
        or row["payload"]["metadata"].get("polar_update")
        != row["protocol"]["polar_update"]
        or result.get("optimizer_policy") != LOCAL_OPTIMIZER_POLICY
    ):
        raise ValueError("stored polar optimizer contract differs")
    if env["dtype"] != "float32" or env["tf32"] is not False:
        raise ValueError("unsupported fixture precision")
    return DispatchContext(
        operator=operator,
        input_shape=(case["rows"], n),
        input_strides=(n, 1),
        dtype=torch.float32,
        atom_count=case["atoms"],
        parameter_dim=4 if local else 5,
        device=DeviceInfo(
            "cuda", None, env["gpu"], tuple(env["compute_capability"]), env["sm_count"]
        ),
        required_grads=RequiredGrads(True, True),
        execution_mode=mode,
    )


def _candidate(context, cohort, rows, registry):
    declaration = rows[0]["plan_declaration"]
    plan = registry.load_plan(declaration)
    registry.validate(plan, context)
    values = {}
    for row in rows:
        seen = set()
        if (
            row["plan_declaration"] != declaration
            or digest(declaration) != row["plan_id"]
        ):
            raise ValueError("Plan identity differs from declaration")
        for metric in row["metrics"]:
            key = MetricKey(
                *(metric[k] for k in ("name", "scope", "unit", "statistic"))
            )
            if key in seen:
                raise ValueError("duplicate metric in an observation")
            seen.add(key)
            value = metric["value"]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid observation metric")
            values.setdefault(key, []).append(float(value))
    # Missing measurements stay missing; do not rank a partial subset of runs.
    metrics = {key: Summary.of(v) for key, v in values.items() if len(v) == len(rows)}
    return Candidate(context, plan, rows[0]["plan_id"], cohort, tuple(rows), metrics)


@dataclass(frozen=True)
class Generation:
    artifact: dict
    leaderboard: dict


def generate(dataset, request, *, registry, policy=None):
    request = validate_request(request)
    exact_body = {k: dataset[k] for k in ("schema_version", "selection", "records")}
    if (
        set(dataset) != {*exact_body, "id"}
        or type(dataset["schema_version"]) is not int
        or dataset["schema_version"] != 1
        or dataset["selection"] != request["dataset"]
        or digest(exact_body) != dataset["id"]
    ):
        raise ValueError("dataset snapshot/selection differs")
    fallback = registry.load_plan(request["fallback_plan"])
    policy = policy or preset_policy(request["score_policy"], request["execution_mode"])
    if policy.declaration() != request["score_policy"] or not callable(policy.function):
        raise ValueError("score function metadata differs from request")
    groups, contexts, records_seen, runs_seen, skipped = {}, {}, {}, set(), []
    excluded = set(request["excluded_run_ids"])
    for original in dataset["records"]:
        row = copy.deepcopy(original)
        identity = f"{row['projection_id']}:{row['ordinal']}"
        if identity in records_seen:
            if records_seen[identity] != row:
                raise ValueError("conflicting duplicate observation")
            continue
        records_seen[identity] = row
        if (
            row["kind"] != "measure"
            or row["status"] != "PASS"
            or row["run_status"] != "PASS"
            or row["run_id"] in excluded
        ):
            skipped.append(
                {
                    "evidence_id": identity,
                    "reason": "failed, excluded or not a Plan measurement",
                }
            )
            continue
        if (
            any(row.get(k) != v for k, v in request["dataset"].items() if k != "gpu")
            or row["environment"]["gpu"] != request["dataset"]["gpu"]
        ):
            raise ValueError("record differs from dataset filters")
        # One adapter revision / one Plan / one run contributes only once.
        run_key = (row["run_id"], row["plan_id"])
        if run_key in runs_seen:
            raise ValueError("run/Plan occurs in multiple projections")
        runs_seen.add(run_key)
        context = _context(row, request["execution_mode"])
        env = {k: v for k, v in row["environment"].items() if k != "device_index"}
        cohort = {
            "case_id": row["case_id"],
            "protocol_id": row["protocol_id"],
            "source_id": row["source_id"],
            "environment": env,
            "adapter": row["adapter"],
            "adapter_revision": row["adapter_revision"],
            "initial_inputs": row["payload"]["result"]["initial_inputs"],
            "initial_p_sha256": row["payload"]["result"]["initial_p_sha256"],
        }
        key = condition_key(dump_condition(context))
        cohort_id = digest(cohort)
        if key in contexts and contexts[key][1] != cohort_id:
            raise ValueError(
                "multiple comparison cohorts map to one exact condition; narrow case/environment/protocol filters"
            )
        contexts[key] = (context, cohort_id)
        group = groups.setdefault((cohort_id, row["plan_id"]), (context, cohort, []))
        group[2].append(row)
    candidates = {}
    for (cohort_id, _), (context, cohort, rows) in sorted(groups.items()):
        rows.sort(key=lambda r: (r["run_id"], r["projection_id"], r["ordinal"]))
        if len(rows) < request["min_runs"]:
            skipped.append(
                {
                    "plan_id": rows[0]["plan_id"],
                    "cohort_id": cohort_id,
                    "reason": "insufficient independent runs",
                    "run_count": len(rows),
                }
            )
            continue
        candidate = _candidate(context, cohort, rows, registry)
        # Trusted callbacks may use all fields, but cannot mutate output evidence.
        score = policy.function(copy.deepcopy(candidate))
        if score is None:
            skipped.append(
                {
                    "plan_id": candidate.plan_id,
                    "cohort_id": cohort_id,
                    "reason": "score function abstained",
                }
            )
            continue
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("score must be a finite number or None")
        candidates.setdefault(cohort_id, []).append((float(score), candidate))
    if not candidates:
        raise ValueError(
            "no eligible measured conditions; check filters, min_runs and score limits"
        )
    plans = {digest(request["fallback_plan"]): request["fallback_plan"]}
    entries, boards, runtimes = [], [], []
    for cohort_id, ranking in sorted(candidates.items()):
        ranking.sort(key=lambda pair: (pair[0], pair[1].plan_id))
        winner = ranking[0][1]
        registry.validate(fallback, winner.context)
        env = winner.cohort["environment"]
        runtimes.append({"torch": env["torch"], "triton": env["triton"]})
        evidence = [f"{r['projection_id']}:{r['ordinal']}" for r in winner.observations]
        plans[winner.plan_id] = registry.dump_plan(winner.plan)
        entries.append(
            {
                "condition": dump_condition(winner.context),
                "plan_id": winner.plan_id,
                "evidence_ids": evidence,
            }
        )
        boards.append(
            {
                "cohort_id": cohort_id,
                "cohort": winner.cohort,
                "condition": dump_condition(winner.context),
                "winner_plan_id": winner.plan_id,
                "candidates": [
                    {
                        "rank": i + 1,
                        "plan_id": c.plan_id,
                        "score": score,
                        "run_count": c.run_count,
                        "evidence_ids": [
                            f"{r['projection_id']}:{r['ordinal']}"
                            for r in c.observations
                        ],
                        "metrics": [
                            {**asdict(key), "summary": asdict(summary)}
                            for key, summary in sorted(c.metrics.items())
                        ],
                    }
                    for i, (score, c) in enumerate(ranking)
                ],
            }
        )
    if any(runtime != runtimes[0] for runtime in runtimes):
        raise ValueError("one artifact must target one Torch/Triton runtime")
    artifact = validate_selector_artifact(
        {
            "schema_version": 1,
            "selector": {"kind": "exact_table", "revision": request["revision"]},
            "score_policy": policy.declaration(),
            "dataset_snapshot": dataset["id"],
            "runtime_versions": runtimes[0],
            "plans": plans,
            "entries": entries,
            "fallback_plan_id": digest(request["fallback_plan"]),
        },
        registry=registry,
    )
    leaderboard = {
        "schema_version": 1,
        "dataset_snapshot": dataset["id"],
        "request": request,
        "artifact_sha256": digest(artifact),
        "score_direction": "lower_is_better",
        "certification": "not assessed",
        "leaderboards": boards,
        "skipped": skipped,
    }
    return Generation(artifact, leaderboard)
