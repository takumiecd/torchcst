"""Real PostgreSQL → fixed dataset → leaderboard → dispatcher integration."""

import copy
import hashlib
import json

import pytest

from benchmarks.database.model import digest
from benchmarks.dispatch import generate
from benchmarks.dispatch.database import read_dataset
from tests.benchmark_database_fixtures import raw_artifact
from tests.test_benchmark_database_postgres import db as database_fixture
from tests.test_dispatch_generation import measured, request_for
from torchcst._backends.catalog import REGISTRY

psycopg = pytest.importorskip("psycopg")
from benchmarks.database.postgres import Database

db = database_fixture


def prepare(database):
    database.import_linear(raw_artifact(measured()))
    row = database.list_records(kind="measure")[0]
    return request_for(row)


def test_local_product_import_export_and_scoped_dispatch(db):
    from benchmarks.cuda.linear.manifest import REGISTRY as research_registry
    from tests.test_local_product_database import local_artifact

    database, _ = db
    value = local_artifact()
    raw = raw_artifact(value)
    first = database.import_linear(raw)
    again = database.import_linear(raw_artifact(value))
    assert first["run_inserted"] and first["projection_inserted"]
    assert not again["run_inserted"] and not again["projection_inserted"]
    assert database.export_run(first["run_id"]) == raw
    assert database.counts()["metrics"] == 83
    row = database.list_records(kind="measure", adapter_revision=3)[0]
    request = request_for(row)
    request["dataset"]["case_id"] = row["case_id"]
    request["fallback_plan"] = value["run"]["plans"][0]["plan"]
    dataset = read_dataset(database, request, page_size=1)
    assert len(dataset["records"]) == 6
    speed = generate(dataset, request, registry=research_registry)
    request["score_policy"]["parameters"] = {"max_peak_allocated_bytes": 3000}
    bounded = generate(dataset, request, registry=research_registry)
    for result, route in (
        (speed, "persistent_supportprep_g"),
        (bounded, "persistent_supportprep_band"),
    ):
        entry = result.artifact["entries"][0]
        assert result.artifact["plans"][entry["plan_id"]]["recipe"]["route"] == route
    database.import_linear(raw_artifact(local_artifact("narrow-only")))
    assert len(read_dataset(database, request)["records"]) == 6
    request["dataset"].pop("case_id")
    with pytest.raises(ValueError, match="multiple comparison cohorts"):
        generate(read_dataset(database, request), request, registry=research_registry)


def test_paginated_db_generation_keeps_raw_evidence_and_uses_no_writes(db):
    database, _ = db
    request = prepare(database)
    database.import_linear(raw_artifact(measured(full=4, window=2)))
    before = database.counts()
    dataset = read_dataset(database, request, page_size=1)
    assert len(dataset["records"]) == 4
    assert all(
        type(m["value"]) is float for r in dataset["records"] for m in r["metrics"]
    )
    result = generate(dataset, request, registry=REGISTRY)
    assert len(result.artifact["entries"]) == 1
    assert result.leaderboard["leaderboards"][0]["candidates"][0]["run_count"] == 2
    assert database.counts() == before
    assert read_dataset(database, request)["id"] == dataset["id"]


def test_new_runs_during_pagination_are_excluded_from_fixed_snapshot(db, monkeypatch):
    database, dsn = db
    request = prepare(database)
    original = database.list_records
    calls = 0

    def read_page(**kwargs):
        nonlocal calls
        page = original(**kwargs)
        if calls == 0:
            with psycopg.connect(dsn, autocommit=True) as other:
                Database(other).import_linear(raw_artifact(measured()))
        calls += 1
        return page

    monkeypatch.setattr(database, "list_records", read_page)
    dataset = read_dataset(database, request, page_size=1)
    assert len(dataset["records"]) == 2
    assert database.counts()["runs"] == 2
    assert len(read_dataset(database, request)["records"]) == 4


def test_database_display_timezone_does_not_change_dataset_identity(db):
    database, _ = db
    request = prepare(database)
    database.connection.execute("SET TIME ZONE 'UTC'")
    utc = read_dataset(database, request)
    database.connection.execute("SET TIME ZONE 'Asia/Tokyo'")
    tokyo = read_dataset(database, request)
    assert utc == tokyo
    assert all(row["started_at"].endswith("+00:00") for row in tokyo["records"])


def test_sql_filters(db):
    database, _ = db
    request = prepare(database)
    altered = measured()
    for record in altered["records"]:
        record["metadata"]["source_commit"] = "f" * 40
    database.import_linear(raw_artifact(altered))
    filtered = read_dataset(database, request)
    assert len(filtered["records"]) == 2
    request["dataset"]["case_id"] = "0" * 64
    assert not read_dataset(database, request)["records"]
    request["dataset"].pop("case_id")
    request["dataset"]["protocol_id"] = filtered["records"][0]["protocol_id"]
    request["dataset"]["environment_id"] = filtered["records"][0]["environment_id"]
    assert len(read_dataset(database, request)["records"]) == 2


def test_multiple_measured_batch_shapes_and_runtime_conflict(db):
    database, _ = db
    request = prepare(database)
    value = measured()
    value["run"]["case"]["rows"] = 64
    sha = hashlib.sha256(
        (json.dumps(value["run"], indent=2) + "\n").encode()
    ).hexdigest()
    value["snapshot_sha256"] = sha
    for record in value["records"]:
        record["metadata"]["case"] = copy.deepcopy(value["run"]["case"])
        record["metadata"]["snapshot_sha256"] = sha
        if record["metadata"]["worker"] != "correctness":
            record["result"]["rows"] = 64
    database.import_linear(raw_artifact(value))
    dataset = read_dataset(database, request)
    result = generate(dataset, request, registry=REGISTRY)
    assert {
        entry["condition"]["input_shape"][0] for entry in result.artifact["entries"]
    } == {64, 128}
    for row in dataset["records"]:
        if row["case_declaration"]["rows"] == 64:
            row["environment"]["torch"] = "different-torch"
    dataset["id"] = digest({k: v for k, v in dataset.items() if k != "id"})
    with pytest.raises(ValueError, match="one Torch/Triton"):
        generate(dataset, request, registry=REGISTRY)


def test_malformed_plan_identity_cannot_export(db):
    database, _ = db
    request = prepare(database)
    dataset = read_dataset(database, request)
    dataset["records"][0]["plan_declaration"]["algorithm_revision"] = "missing"
    dataset["id"] = digest({k: v for k, v in dataset.items() if k != "id"})
    with pytest.raises(ValueError, match="unknown"):
        generate(dataset, request, registry=REGISTRY)
