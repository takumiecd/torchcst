"""Synthetic product artifacts exercise the adapter and selector contract."""

import copy
import json
from dataclasses import asdict, replace

import pytest

from benchmarks.cuda.linear.manifest import load_run
from benchmarks.cuda.linear.protocol import (
    LOCAL_OPTIMIZER_POLICY,
    PRODUCT_ORACLE_SCOPE,
    measurement_operator,
)
from benchmarks.database.adapters.linear import project
from benchmarks.database.model import digest
from benchmarks.dispatch.generate import _context
from tests.benchmark_database_fixtures import artifact
from tests.test_dispatch_generation import dataset_for


def product_artifact():
    run = load_run(
        "benchmarks/cuda/linear/cases/profile-product-64-rho3.json",
        "benchmarks/cuda/linear/plans-profile-product.json",
    )
    run = replace(run, case=replace(run.case, rounds=3))
    snapshot = json.loads(json.dumps(run.snapshot()))
    old = artifact()
    meta = old["records"][0]["metadata"] | {
        "case": snapshot["case"],
        "input_hashes": snapshot["input_hashes"],
        "snapshot_sha256": digest(snapshot),
        "seed": run.case.seed,
        "polar_update": "torch",
    }
    result = old["records"][2]["result"] | {
        "operator": json.loads(
            json.dumps(asdict(measurement_operator(snapshot["case"])))
        ),
        "rows": run.case.rows,
        "atoms": run.case.atoms,
        "profile": run.case.profile,
        "optimizer": snapshot["case"]["optimizer"],
        "optimizer_policy": LOCAL_OPTIMIZER_POLICY,
    }
    records = []
    for kind in ("correctness", "measure"):
        for entry in snapshot["plans"]:
            records.append(
                {
                    "metadata": meta
                    | {"worker": kind, "plan_id": entry["id"], "plan": entry["plan"]},
                    "result": (
                        {
                            "status": "PASS",
                            "scope": PRODUCT_ORACLE_SCOPE,
                            **{
                                k: {"max": 1e-6, "rel_l2": 1e-7}
                                for k in ("y", "dx", "dp", "polar_update")
                            },
                        }
                        if kind == "correctness"
                        else result | {"reference": entry["id"]}
                    ),
                }
            )
    records.append(
        {
            "metadata": meta | {"worker": "dense", "plan_id": None, "plan": None},
            "result": result
            | {
                "reference": "dense_linear",
                "atoms": None,
                "profile": None,
                "initial_p_sha256": None,
                "optimizer_policy": "ordinary AdamW",
            },
        }
    )
    return old | {
        "run": snapshot,
        "snapshot_sha256": digest(snapshot),
        "records": copy.deepcopy(records),
    }


def test_new_projection_and_generation_keep_product_contract():
    value = product_artifact()
    projection = project(value)
    assert projection.revision == 4
    assert projection.protocol["revision"] == 4
    assert projection.protocol["correctness"] == PRODUCT_ORACLE_SCOPE
    assert projection.protocol["polar_update"] == "torch"
    dataset, _ = dataset_for(value)
    row = dataset["records"][0]
    context = _context(row, "cuda_graph")
    assert context.operator.kernel.composition == "profile_product"
    assert context.parameter_dim == 4


@pytest.mark.parametrize("part", ["scope", "floor", "update"])
def test_rejects_mixed_legacy_oracle_and_normalization(part):
    value = product_artifact()
    if part == "scope":
        value["records"][0]["result"]["scope"] = (
            "independent FP64 scalar Y/dX/all atom gradients; production polar update"
        )
    elif part == "floor":
        next(r for r in value["records"] if r["metadata"]["worker"] == "measure")[
            "result"
        ]["operator"]["kernel"]["normalization"]["floor"] = 0.5
    else:
        next(r for r in value["records"] if r["metadata"]["worker"] == "measure")[
            "result"
        ]["optimizer_policy"] = "ordinary AdamW"
    with pytest.raises(ValueError):
        project(value)
