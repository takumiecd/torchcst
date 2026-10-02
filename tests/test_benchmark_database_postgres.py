"""Real PostgreSQL integration: explicit admin DSN, isolated owned databases.

TORCHCST_TEST_DATABASE_URL must point to a disposable local test server with
CREATE DATABASE permission. Each test creates/drops its own randomly named DB.
"""

import json
import hashlib
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from benchmarks.database.model import Metric, digest
from tests.benchmark_database_fixtures import artifact, raw_artifact

psycopg = pytest.importorskip("psycopg")
postgres = pytest.importorskip("benchmarks.database.postgres")
sql = psycopg.sql
make_conninfo = psycopg.conninfo.make_conninfo
dict_row = psycopg.rows.dict_row
Database = postgres.Database


@pytest.fixture
def db():
    admin_dsn = os.environ.get("TORCHCST_TEST_DATABASE_URL")
    if not admin_dsn:
        pytest.skip("set TORCHCST_TEST_DATABASE_URL to run real PostgreSQL tests")
    name = "torchcst_test_" + uuid.uuid4().hex
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        dsn = make_conninfo(admin_dsn, dbname=name)
        try:
            with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
                database = Database(conn)
                database.migrate()
                yield database, dsn
        finally:
            admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )


def test_repeated_observations_idempotence_and_exact_restoration(db):
    database, _ = db
    raw = raw_artifact()
    first = database.import_linear(raw, provenance={"provider": "local-test"})
    again = database.import_linear(
        json.dumps(artifact(), sort_keys=True).encode(),
        provenance={"provider": "local-test"},
    )
    assert first["run_inserted"] and first["projection_inserted"]
    assert again == first | {"run_inserted": False, "projection_inserted": False}
    assert database.export_run(first["run_id"]) == raw
    data = artifact()
    data["records"][2]["result"]["eager"] = {
        "median_ms": 4.0,
        "samples_ms": [3.0, 4.0, 5.0],
    }
    second = database.import_linear(raw_artifact(data))
    assert second["run_id"] != first["run_id"]
    assert database.counts() == {
        "plans": 2,
        "cases": 1,
        "runs": 2,
        "projections": 2,
        "run_plans": 4,
        "records": 10,
        "metrics": 62,
    }
    records = database.list_records(
        plan_id=digest(data["run"]["plans"][0]["plan"]), gpu="Test GPU", kind="measure"
    )
    assert len(records) == 2
    assert sorted(
        m["value"]
        for r in records
        for m in r["metrics"]
        if m["name"] == "step.time" and m["scope"] == "eager"
    ) == [2.0, 4.0]
    assert records[0]["source"]["commit_is_full_sha"] is True
    assert records[0]["metrics"][0]["unit"]


def test_failure_evidence_is_not_discarded_or_projected_as_measurement(db):
    database, _ = db
    data = artifact()
    data["status"] = "FAIL"
    data["records"] = [
        {
            "metadata": {"worker": "measure", "plan_id": "full"},
            "result": {"status": "FAIL", "error": "GPU unavailable"},
        }
    ]
    result = database.import_linear(raw_artifact(data))
    assert database.counts()["metrics"] == 0
    (row,) = database.list_records()
    assert row["run_status"] == "FAIL" and not row["metrics"]
    assert json.loads(database.export_run(result["run_id"])) == data


def test_execution_ids_distinguish_repetitions_and_reject_changed_evidence(db):
    database, _ = db
    data = artifact() | {
        "execution_id": str(uuid.uuid4()),
        "started_at": "2026-10-02T00:00:00+00:00",
    }
    database.import_linear(raw_artifact(data))
    changed = data | {"extra": "changed payload for the same execution"}
    with pytest.raises(psycopg.errors.UniqueViolation):
        database.import_linear(raw_artifact(changed))
    # Identical numbers from a different actual execution are another observation.
    database.import_linear(raw_artifact(data | {"execution_id": str(uuid.uuid4())}))
    assert database.counts()["runs"] == 2
    assert all(r["execution_id"] for r in database.list_records())


def test_reordered_resubmission_uses_original_snapshot_byte_order(db):
    database, _ = db
    data = artifact()
    for entry in data["run"]["plans"]:
        entry["plan"] = dict(reversed(list(entry["plan"].items())))
    snapshot_hash = hashlib.sha256(
        (json.dumps(data["run"], indent=2) + "\n").encode()
    ).hexdigest()
    data["snapshot_sha256"] = snapshot_hash
    for record in data["records"]:
        record["metadata"]["snapshot_sha256"] = snapshot_hash
    raw = raw_artifact(data)
    result = database.import_linear(raw)
    reordered = json.dumps(data, sort_keys=True).encode()
    assert not database.import_linear(reordered)["run_inserted"]
    assert database.export_run(result["run_id"]) == raw


def test_cli_import_query_and_export_restore_original_bytes(db, tmp_path):
    _, dsn = db
    raw = raw_artifact()
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    env = dict(os.environ, DATABASE_URL=dsn)

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "benchmarks.database", *args],
            env=env,
            capture_output=True,
            check=True,
        ).stdout

    result = json.loads(cli("import-linear", str(path)))
    assert json.loads(cli("status"))["runs"] == 1
    rows = json.loads(
        cli("records", "--gpu", "Test GPU", "--kind", "measure", "--limit", "1")
    )
    assert len(rows["records"]) == 1 and rows["next_after"]
    output = tmp_path / "restored.json"
    cli("export-run", result["run_id"], str(output))
    assert output.read_bytes() == raw


def test_exported_artifact_can_rebuild_another_database(db):
    database, dsn = db
    result = database.import_linear(raw_artifact())
    before = database.counts()
    raw = database.export_run(result["run_id"])
    other_name = "torchcst_restore_" + uuid.uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(other_name)))
        try:
            with psycopg.connect(
                make_conninfo(dsn, dbname=other_name), autocommit=True
            ) as conn:
                other = Database(conn)
                other.migrate()
                assert other.import_linear(raw)["run_id"] == result["run_id"]
                assert other.counts() == before
                assert other.export_run(result["run_id"]) == raw
        finally:
            admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                    sql.Identifier(other_name)
                )
            )


def test_transaction_rolls_back_all_tables_after_late_metric_failure(db, monkeypatch):
    database, _ = db
    original = postgres.project

    def broken(value):
        p = original(value)
        record = p.records[0]
        record = replace(record, metrics=record.metrics + (record.metrics[0],))
        return replace(p, records=(record, *p.records[1:]))

    monkeypatch.setattr(postgres, "project", broken)
    with pytest.raises(psycopg.errors.UniqueViolation):
        database.import_linear(raw_artifact())
    assert all(n == 0 for n in database.counts().values())


def test_append_only_enforcement_including_truncate(db):
    database, _ = db
    database.import_linear(raw_artifact())
    for command in (
        "UPDATE benchmark.runs SET status='FAIL'",
        "DELETE FROM benchmark.records",
        "TRUNCATE benchmark.metrics",
    ):
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            database.connection.execute(command)
    assert database.counts()["runs"] == 1


def test_concurrent_import_is_one_observation_set(db):
    database, dsn = db
    raw = raw_artifact()

    def ingest(_):
        with psycopg.connect(dsn, autocommit=True) as conn:
            return Database(conn).import_linear(raw)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(ingest, range(4)))
    assert sum(r["run_inserted"] for r in results) == 1
    assert sum(r["projection_inserted"] for r in results) == 1
    assert database.counts()["metrics"] == 31


def test_new_projection_revision_adds_metrics_without_overwriting_evidence(
    db, monkeypatch
):
    database, _ = db
    raw = raw_artifact()
    initial = database.import_linear(raw)
    original = postgres.project

    def newer(value):
        p = original(value)
        record = p.records[2]
        record = replace(
            record,
            metrics=record.metrics
            + (Metric("forward.time", "kernel", "us", "p95", 10, [8, 10]),),
        )
        return replace(p, revision=2, records=(*p.records[:2], record, *p.records[3:]))

    monkeypatch.setattr(postgres, "project", newer)
    second = database.import_linear(raw)
    assert not second["run_inserted"] and second["projection_inserted"]
    assert second["projection_id"] != initial["projection_id"]
    assert database.export_run(initial["run_id"]) == raw
    assert database.counts()["runs"] == 1 and database.counts()["projections"] == 2
    assert database.counts()["metrics"] == 63
    assert {r["adapter_revision"] for r in database.list_records()} == {1, 2}
    assert len(database.list_records(adapter_revision=2)) == 5


def test_read_snapshot_and_keyset_pagination(db):
    database, dsn = db
    database.import_linear(raw_artifact())
    with database.snapshot():
        first = database.list_records(limit=2)
        with psycopg.connect(dsn, autocommit=True) as conn:
            data = artifact() | {"session": "another actual measurement"}
            Database(conn).import_linear(raw_artifact(data))
        rest = database.list_records(
            after=(first[-1]["projection_id"], first[-1]["ordinal"])
        )
        assert len(first) + len(rest) == 5
    assert len(database.list_records()) == 10


def test_provenance_conflict_and_bad_artifact_leave_no_new_data(db):
    database, _ = db
    database.import_linear(raw_artifact(), provenance={"job": "original"})
    before = database.counts()
    with pytest.raises(ValueError, match="provenance"):
        database.import_linear(raw_artifact(), provenance={"job": "other"})
    for raw in (
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        raw_artifact(artifact() | {"schema_version": 2}),
    ):
        with pytest.raises(ValueError):
            database.import_linear(raw)
    assert database.counts() == before


def test_migration_history_is_checked_and_migration_is_idempotent(db):
    database, _ = db
    assert database.migrate() == 1
    database.connection.execute(
        "UPDATE benchmark.schema_migrations SET checksum = 'changed'"
    )
    with pytest.raises(ValueError, match="migration history"):
        database.migrate()
    with pytest.raises(ValueError, match="schema differs"):
        database.import_linear(raw_artifact())


def test_unversioned_schema_is_rejected(db):
    database, _ = db
    database.connection.execute("DROP SCHEMA benchmark CASCADE")
    database.connection.execute("CREATE SCHEMA benchmark")
    with pytest.raises(ValueError, match="unversioned"):
        database.migrate()


def test_integrity_verification_on_export(db):
    database, _ = db
    result = database.import_linear(raw_artifact())
    # Simulate corruption as the owner; normal evidence writes cannot do this.
    database.connection.execute(
        "ALTER TABLE benchmark.runs DISABLE TRIGGER immutable_evidence"
    )
    database.connection.execute("UPDATE benchmark.runs SET artifact = %s", (b"{}",))
    with pytest.raises(ValueError, match="integrity"):
        database.export_run(result["run_id"])


def test_reader_and_append_writer_roles_need_no_schema_owner_privileges(db):
    database, dsn = db
    role = "torchcst_writer_" + uuid.uuid4().hex
    # Role is global; always remove only this explicitly owned role.
    database.connection.execute(
        sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(role))
    )
    try:
        database.connection.execute(
            sql.SQL("GRANT USAGE ON SCHEMA benchmark TO {}").format(
                sql.Identifier(role)
            )
        )
        database.connection.execute(
            sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA benchmark TO {}").format(
                sql.Identifier(role)
            )
        )
        for table in postgres.TABLES:
            database.connection.execute(
                sql.SQL("GRANT INSERT ON benchmark.{} TO {}").format(
                    sql.Identifier(table), sql.Identifier(role)
                )
            )
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role)))
            store = Database(conn)
            store.import_linear(raw_artifact())
            assert store.counts()["runs"] == 1
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute("DELETE FROM benchmark.runs")
    finally:
        database.connection.execute(
            sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role))
        )
        database.connection.execute(
            sql.SQL("DROP ROLE {}").format(sql.Identifier(role))
        )
