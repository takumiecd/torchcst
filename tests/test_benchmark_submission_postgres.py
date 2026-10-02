"""Real DB admission: quota races, immutable policy history, attribution, replay."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from benchmarks.submissions.github import submission_from_issue
from benchmarks.submissions.policy import Policy, load_policy
from tests.test_benchmark_database_postgres import db as database_fixture
from tests.test_benchmark_submissions import issue, new_run, recognized

psycopg = pytest.importorskip("psycopg")
from benchmarks.database.postgres import Database, SubmissionQuotaExceeded

db = database_fixture


def test_append_with_account_attribution_policy_and_exact_bytes(db):
    database, _ = db
    policy = load_policy()
    raw = new_run()
    submitted = submission_from_issue(issue(policy), policy)
    receipt = database.import_submission(raw, submitted, policy)
    assert receipt["submission_inserted"] and receipt["trust_points"] == 1
    assert database.export_run(receipt["run_id"]) == raw
    (row,) = database.list_submissions(submitter_id=123)
    assert row["submitter_login"] == "example-user" and row["tier"] == "general"
    assert (
        row["policy_id"] == policy.id
        and row["measurement_authenticity"] == "self_reported"
    )
    assert all(r["submission"]["trust_points"] == 1 for r in database.list_records())
    second = database.import_submission(
        new_run(), replace(submitted, issue_number=8, issue_id=9008), policy
    )
    assert second["run_id"] != receipt["run_id"]
    assert len(database.list_submissions()) == 2 and database.counts()["runs"] == 2


def test_general_daily_limit_and_recognized_unlimited(db):
    database, _ = db
    policy = load_policy()
    submitted = submission_from_issue(issue(policy), policy)
    first_raw = new_run()
    database.import_submission(first_raw, submitted, policy)
    for number in range(8, 17):
        database.import_submission(
            new_run(),
            replace(submitted, issue_number=number, issue_id=9000 + number),
            policy,
        )
    assert len(database.list_submissions()) == 10
    with pytest.raises(SubmissionQuotaExceeded):
        database.import_submission(
            new_run(), replace(submitted, issue_number=17, issue_id=9017), policy
        )
    assert database.counts()["runs"] == 10
    # Replay is free even at the cap; policy changes do not rewrite old points.
    replay = database.import_submission(first_raw, submitted, recognized(policy))
    assert not replay["submission_inserted"] and replay["trust_points"] == 1
    trusted = database.import_submission(
        new_run(),
        replace(submitted, issue_number=17, issue_id=9017),
        recognized(policy),
    )
    assert trusted["tier"] == "recognized" and trusted["trust_points"] == 10
    assert len(database.list_submissions()) == 11


def test_quota_uses_db_receipt_day_not_self_reported_date(db):
    database, _ = db
    value = load_policy().declaration | {"general_daily_runs": 1}
    policy = Policy(value)
    submitted = submission_from_issue(issue(policy), policy)
    raw = new_run()
    database.import_submission(raw, submitted, policy)
    changed = json.loads(new_run())
    changed["started_at"] = "2000-01-01T00:00:00Z"
    with pytest.raises(SubmissionQuotaExceeded):
        database.import_submission(
            json.dumps(changed).encode(),
            replace(submitted, issue_number=8, issue_id=9008),
            policy,
        )


def test_duplicate_from_another_issue_or_account_adds_no_observation_or_credit(db):
    database, _ = db
    policy = load_policy()
    raw = new_run()
    submitted = submission_from_issue(issue(policy), policy)
    first = database.import_submission(raw, submitted, policy)
    copied = database.import_submission(
        raw,
        replace(
            submitted,
            issue_number=8,
            issue_id=9008,
            submitter_id=456,
            submitter_login="other-user",
        ),
        policy,
    )
    assert copied == first | {"submission_inserted": False}
    assert len(database.list_submissions()) == 1 and not database.list_submissions(
        submitter_id=456
    )
    with pytest.raises(ValueError, match="already refers"):
        database.import_submission(new_run(), submitted, policy)


def test_old_run_origin_survives_new_channel(db):
    database, _ = db
    policy = load_policy()
    raw = new_run()
    original = {"origin": "github-actions-hosted-colab", "source": "historical"}
    database.import_linear(raw, provenance=original)
    database.import_submission(
        raw, submission_from_issue(issue(policy), policy), policy
    )
    assert database.counts()["runs"] == 1
    assert all(r["provenance"] == original for r in database.list_records())


def test_failed_evidence_is_preserved_and_policy_history_is_immutable(db):
    database, _ = db
    policy = load_policy()
    data = json.loads(new_run())
    data["status"] = "FAIL"
    data["records"] = [
        {
            "metadata": {"worker": "measure", "plan_id": "full"},
            "result": {"status": "FAIL", "error": "GPU failed"},
        }
    ]
    database.import_submission(
        json.dumps(data).encode(), submission_from_issue(issue(policy), policy), policy
    )
    assert database.counts()["runs"] == 1 and database.counts()["metrics"] == 0
    for table in ("submissions", "submission_policies"):
        with pytest.raises(psycopg.Error, match="append-only"):
            database.connection.execute(f"DELETE FROM benchmark.{table}")


def test_parallel_last_quota_slot_is_admitted_once(db):
    database, dsn = db
    policy = Policy(load_policy().declaration | {"general_daily_runs": 1})
    submitted = submission_from_issue(issue(policy), policy)
    raws = [new_run(), new_run()]

    def submit(index):
        with psycopg.connect(dsn, autocommit=True) as connection:
            try:
                return Database(connection).import_submission(
                    raws[index],
                    replace(submitted, issue_number=7 + index, issue_id=9007 + index),
                    policy,
                )["submission_inserted"]
            except SubmissionQuotaExceeded:
                return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(submit, (0, 1))) == [False, True]
    assert len(database.list_submissions()) == 1 and database.counts()["runs"] == 1


def test_parallel_identical_run_from_different_accounts_is_not_double_counted(db):
    database, dsn = db
    policy = load_policy()
    raw = new_run()
    submitted = submission_from_issue(issue(policy), policy)

    def submit(index):
        with psycopg.connect(dsn, autocommit=True) as connection:
            return Database(connection).import_submission(
                raw,
                replace(
                    submitted,
                    issue_number=7 + index,
                    issue_id=9007 + index,
                    submitter_id=123 + index,
                ),
                policy,
            )["submission_inserted"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(submit, (0, 1))) == [False, True]
    assert database.counts()["runs"] == 1 and len(database.list_submissions()) == 1


def test_new_migration_preserves_original_evidence(db, monkeypatch):
    database, _ = db
    # Rebuild only this owned disposable schema as an existing migration-1 DB.
    database.connection.execute("DROP SCHEMA benchmark CASCADE")
    migrations = database._migrations()
    with monkeypatch.context() as patch:
        patch.setattr(database, "_migrations", lambda: migrations[:1])
        assert database.migrate() == 1
        raw = new_run()
        prior = database.import_linear(raw, provenance={"origin": "old-production"})
        before = database.counts()
    assert database.migrate() == 2
    assert database.counts() == before
    assert database.export_run(prior["run_id"]) == raw
    assert not database.list_submissions()


def test_writer_role_can_submit_but_not_edit_evidence(db):
    import uuid

    from psycopg import sql

    from benchmarks.database.postgres import TABLES

    database, dsn = db
    role = "torchcst_submit_test_" + uuid.uuid4().hex
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
        for table in (*TABLES, "submission_policies", "submissions"):
            database.connection.execute(
                sql.SQL("GRANT INSERT ON benchmark.{} TO {}").format(
                    sql.Identifier(table), sql.Identifier(role)
                )
            )
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role)))
            store = Database(connection)
            policy = load_policy()
            assert store.import_submission(
                new_run(), submission_from_issue(issue(policy), policy), policy
            )["submission_inserted"]
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                connection.execute("UPDATE benchmark.submissions SET trust_points=100")
    finally:
        database.connection.execute(
            sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role))
        )
        database.connection.execute(
            sql.SQL("DROP ROLE {}").format(sql.Identifier(role))
        )


def test_conflicting_execution_rolls_back_policy_and_submission(db):
    database, _ = db
    policy = load_policy()
    raw = new_run()
    submitted = submission_from_issue(issue(policy), policy)
    database.import_submission(raw, submitted, policy)
    changed = json.loads(raw) | {"extra": "same execution, conflicting data"}
    with pytest.raises(psycopg.errors.UniqueViolation):
        database.import_submission(
            json.dumps(changed).encode(),
            replace(submitted, issue_number=8, issue_id=9008),
            recognized(policy),
        )
    assert len(database.list_submissions()) == 1 and database.counts()["runs"] == 1
    assert (
        database.connection.execute(
            "SELECT count(*) AS n FROM benchmark.submission_policies"
        ).fetchone()["n"]
        == 1
    )
