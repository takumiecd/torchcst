"""Transactional PostgreSQL storage, with immutable raw runs and projections."""

import hashlib
import os
import re
from contextlib import contextmanager
from datetime import timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from benchmarks.database.adapters.linear import project
from benchmarks.database.model import digest, execution_identity
from torchcst._backends.cuda.serialization import decode_json, encode_json

MIGRATIONS = Path(__file__).with_name("migrations")
TABLES = ("plans", "cases", "runs", "projections", "run_plans", "records", "metrics")


class SubmissionQuotaExceeded(ValueError):
    pass


def connect_from_env(name="DATABASE_URL"):
    """Never put a connection string in the CLI arguments or printed output."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or not os.environ.get(name):
        raise ValueError("database connection environment variable is missing/invalid")
    return psycopg.connect(os.environ[name], autocommit=True, row_factory=dict_row)


class Database:
    def __init__(self, connection):
        if not connection.autocommit:
            raise ValueError("Database requires an autocommit connection")
        self.connection = connection
        self.connection.row_factory = dict_row

    def _migrations(self):
        return [
            (int(p.name.split("_")[0]), p.read_bytes())
            for p in sorted(MIGRATIONS.glob("[0-9]*_*.sql"))
        ]

    def migrate(self):
        """Serialize migrations; reject unknown or edited migration history."""
        with self.connection.transaction():
            self.connection.execute("SELECT pg_advisory_xact_lock(1937011555, 1)")
            exists = self.connection.execute(
                "SELECT to_regnamespace('benchmark') AS schema"
            ).fetchone()["schema"]
            history = self.connection.execute(
                "SELECT to_regclass('benchmark.schema_migrations') AS history"
            ).fetchone()["history"]
            if exists and not history:
                raise ValueError(
                    "unversioned benchmark schema; use an empty database/schema"
                )
            self.connection.execute("CREATE SCHEMA IF NOT EXISTS benchmark")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS benchmark.schema_migrations (
                version integer PRIMARY KEY, checksum text NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
            )""")
            applied = self.connection.execute(
                "SELECT version, checksum FROM benchmark.schema_migrations ORDER BY version"
            ).fetchall()
            migrations = self._migrations()
            expected = [
                {"version": v, "checksum": hashlib.sha256(raw).hexdigest()}
                for v, raw in migrations
            ]
            if applied != expected[: len(applied)]:
                raise ValueError("unknown/modified database migration history")
            for (version, raw), info in zip(
                migrations[len(applied) :], expected[len(applied) :]
            ):
                self.connection.execute(raw.decode(), prepare=False)
                self.connection.execute(
                    "INSERT INTO benchmark.schema_migrations(version, checksum) VALUES (%s, %s)",
                    (version, info["checksum"]),
                )
        return len(migrations)

    def _check_schema(self):
        history = self.connection.execute(
            "SELECT to_regclass('benchmark.schema_migrations') AS history"
        ).fetchone()["history"]
        if not history:
            raise ValueError(
                "database is not initialized; run migrate with the owner role"
            )
        applied = self.connection.execute(
            "SELECT version, checksum FROM benchmark.schema_migrations ORDER BY version"
        ).fetchall()
        expected = [
            {"version": v, "checksum": hashlib.sha256(raw).hexdigest()}
            for v, raw in self._migrations()
        ]
        if applied != expected:
            raise ValueError(
                "database schema differs; check migration history and run migrate"
            )

    def import_linear(self, raw: bytes, *, provenance=None):
        """Store one completed artifact atomically; return IDs and insertion flags.

        Same parsed JSON means the same run, even if whitespace differs. Preserve
        the first accepted byte encoding. Provenance is assigned by the caller,
        never read from untrusted artifact fields; this API does not attest it.
        """
        if type(raw) is not bytes:
            raise TypeError("artifact must be bytes")
        value = decode_json(raw)
        provenance = {} if provenance is None else provenance
        if type(provenance) is not dict:
            raise ValueError("provenance needs a JSON object")
        encode_json(provenance)
        run_id = digest(value)
        with self.connection.transaction():
            self._check_schema()
            existing = self.connection.execute(
                "SELECT artifact, artifact_sha256 FROM benchmark.runs WHERE id = %s",
                (run_id,),
            ).fetchone()
            if existing:
                # JSONB or a reordered resubmission cannot reproduce historical
                # byte-dependent snapshot hashes. Reproject the preserved original.
                original = bytes(existing["artifact"])
                if hashlib.sha256(original).hexdigest() != existing["artifact_sha256"]:
                    raise ValueError("stored artifact integrity check failed")
                value = decode_json(original)
                if digest(value) != run_id:
                    raise ValueError("stored artifact identity differs")
            projection = project(value)
            execution_id, started_at = execution_identity(value)
            projection_id = digest(
                {
                    "run_id": run_id,
                    "adapter": projection.adapter,
                    "revision": projection.revision,
                }
            )
            inserted = self.connection.execute(
                """INSERT INTO benchmark.runs
                (id, artifact_sha256, artifact, artifact_schema_version, status, provenance, execution_id, started_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING RETURNING id""",
                (
                    run_id,
                    hashlib.sha256(raw).hexdigest(),
                    raw,
                    value["schema_version"],
                    value["status"],
                    Jsonb(provenance),
                    execution_id,
                    started_at,
                ),
            ).fetchone()
            stored = self.connection.execute(
                "SELECT provenance FROM benchmark.runs WHERE id = %s", (run_id,)
            ).fetchone()
            if stored["provenance"] != provenance:
                raise ValueError("run already exists with different provenance")
            plan_ids = {alias: digest(plan) for alias, plan in projection.plans.items()}
            for alias, plan in projection.plans.items():
                self.connection.execute(
                    """INSERT INTO benchmark.plans
                    (id, algorithm_id, algorithm_revision, declaration) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING""",
                    (
                        plan_ids[alias],
                        plan["algorithm_id"],
                        plan["algorithm_revision"],
                        Jsonb(plan),
                    ),
                )
            case = {k: v for k, v in projection.case.items() if k != "id"}
            self.connection.execute(
                "INSERT INTO benchmark.cases(id, declaration) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                (projection.case_id, Jsonb(case)),
            )
            projected = self.connection.execute(
                """INSERT INTO benchmark.projections
                (id, run_id, adapter, adapter_revision, case_id, protocol_id, protocol)
                VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING RETURNING id""",
                (
                    projection_id,
                    run_id,
                    projection.adapter,
                    projection.revision,
                    projection.case_id,
                    digest(projection.protocol),
                    Jsonb(projection.protocol),
                ),
            ).fetchone()
            if projected:
                for alias, plan_id in plan_ids.items():
                    self.connection.execute(
                        "INSERT INTO benchmark.run_plans VALUES (%s, %s, %s, %s)",
                        (projection_id, alias, plan_id, alias == projection.baseline),
                    )
                for ordinal, record in enumerate(projection.records):
                    self.connection.execute(
                        """INSERT INTO benchmark.records
                        (projection_id, ordinal, plan_alias, kind, status, environment_id, environment, source_id, source, payload)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            projection_id,
                            ordinal,
                            record.plan_alias,
                            record.kind,
                            record.status,
                            digest(record.environment),
                            Jsonb(record.environment),
                            digest(record.source),
                            Jsonb(record.source),
                            Jsonb(record.payload),
                        ),
                    )
                    for metric in record.metrics:
                        self.connection.execute(
                            """INSERT INTO benchmark.metrics
                            (projection_id, record_ordinal, name, scope, unit, statistic, value, samples, details)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            (
                                projection_id,
                                ordinal,
                                metric.name,
                                metric.scope,
                                metric.unit,
                                metric.statistic,
                                metric.value,
                                Jsonb(metric.samples),
                                Jsonb(metric.details),
                            ),
                        )
        return {
            "run_id": run_id,
            "projection_id": projection_id,
            "run_inserted": bool(inserted),
            "projection_inserted": bool(projected),
        }

    def import_submission(self, raw, submission, policy):
        """Admit one independently measured run, atomically with its UTC quota.

        Account identity comes from the central GitHub caller, not the artifact.
        All evidence remains append-only; recognition/points are policy snapshots.
        Replays and copied runs do not consume another slot or gain another attribution.
        """
        submission.validate(policy)
        if type(raw) is not bytes or not 0 < len(raw) <= policy.max_file_bytes:
            raise ValueError("empty or oversized result JSON")
        value = decode_json(raw)
        project(value)  # Validate before locks/writes, including FAIL evidence.
        execution_id, _ = execution_identity(value)
        if execution_id is None:
            raise ValueError(
                "new submissions require an execution UUID and UTC start time"
            )
        run_id = digest(value)
        submission_id = digest(
            {"repository": submission.repository, "issue_id": submission.issue_id}
        )

        def receipt(row, inserted):
            return {
                "submission_id": row["id"],
                "run_id": row["run_id"],
                "submission_inserted": inserted,
                "tier": row["tier"],
                "trust_points": row["trust_points"],
                "policy_id": row["policy_id"],
                "submitter_id": row["submitter_id"],
                "submitter_login": row["submitter_login"],
                "validation": "consistency_checked",
                "measurement_authenticity": "self_reported",
            }

        with self.connection.transaction():
            self._check_schema()
            # First serialize a submitter's admissions, then serialize duplicate
            # run identities across accounts. All callers acquire in this order.
            self.connection.execute(
                "SELECT pg_advisory_xact_lock(1937011556, hashtext(%s))",
                (str(submission.submitter_id),),
            )
            self.connection.execute(
                "SELECT pg_advisory_xact_lock(1937011557, hashtext(%s))", (run_id,)
            )
            existing = self.connection.execute(
                "SELECT * FROM benchmark.submissions WHERE repository = %s AND issue_id = %s",
                (submission.repository, submission.issue_id),
            ).fetchone()
            if existing:
                if (
                    existing["submitter_id"] != submission.submitter_id
                    or existing["run_id"] != run_id
                ):
                    raise ValueError(
                        "accepted issue already refers to another submitter or run; use a new issue"
                    )
                return receipt(existing, False)
            existing = self.connection.execute(
                "SELECT * FROM benchmark.submissions WHERE run_id = %s", (run_id,)
            ).fetchone()
            if existing:
                return receipt(existing, False)
            received = self.connection.execute(
                "SELECT clock_timestamp() AS now"
            ).fetchone()["now"]
            utc_day = received.astimezone(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            limit = policy.daily_limit(submission.submitter_id)
            if limit is not None:
                count = self.connection.execute(
                    "SELECT count(*) AS n FROM benchmark.submissions WHERE submitter_id = %s AND received_at >= %s",
                    (submission.submitter_id, utc_day),
                ).fetchone()["n"]
                if count >= limit:
                    raise SubmissionQuotaExceeded(
                        "general account daily submission limit reached (UTC)"
                    )
            self.connection.execute(
                "INSERT INTO benchmark.submission_policies(id, declaration) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (policy.id, Jsonb(policy.declaration)),
            )
            previous = self.connection.execute(
                "SELECT provenance FROM benchmark.runs WHERE id = %s",
                (run_id,),
            ).fetchone()
            # A run's immutable origin survives linkage from a new submission
            # channel, including observations imported by an older hosted workflow.
            provenance = (
                previous["provenance"]
                if previous
                else {
                    "origin": "external-benchmark-json",
                    "measurement_authenticity": "self_reported",
                    "certification": "not assessed",
                }
            )
            self.import_linear(raw, provenance=provenance)
            row = self.connection.execute(
                """INSERT INTO benchmark.submissions
                (id, repository, issue_id, issue_number, issue_body_sha256, submitter_id,
                 submitter_login, tier, trust_points, policy_id, run_id, artifact_url,
                 artifact_bytes, validation, measurement_authenticity, received_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'consistency_checked', 'self_reported', %s) RETURNING *""",
                (
                    submission_id,
                    submission.repository,
                    submission.issue_id,
                    submission.issue_number,
                    submission.issue_body_sha256,
                    submission.submitter_id,
                    submission.submitter_login,
                    policy.tier(submission.submitter_id),
                    policy.trust_points(submission.submitter_id),
                    policy.id,
                    run_id,
                    submission.artifact_url,
                    len(raw),
                    received,
                ),
            ).fetchone()
            return receipt(row, True)

    def list_submissions(self, *, submitter_id=None, after=None, limit=100):
        """Read account attribution without conflating trust with correctness."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be 1..1000")
        from benchmarks.submissions.policy import positive_id

        self._check_schema()
        clauses, params = [], []
        if submitter_id is not None:
            positive_id(submitter_id)
            clauses.append("submitter_id = %s")
            params.append(submitter_id)
        if after is not None:
            if type(after) is not str or not re.fullmatch(r"[0-9a-f]{64}", after):
                raise ValueError("invalid submission cursor")
            clauses.append("id > %s")
            params.append(after)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return self.connection.execute(
            "SELECT id, repository, issue_number, submitter_id, submitter_login, tier, trust_points, "
            "policy_id, run_id, validation, measurement_authenticity, received_at::text "
            "FROM benchmark.submissions" + where + " ORDER BY id LIMIT %s",
            (*params, limit),
        ).fetchall()

    @contextmanager
    def snapshot(self):
        """Several reads use the same PostgreSQL snapshot, excluding later writes."""
        if self.connection.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
            raise ValueError("snapshot must start outside a transaction")
        with self.connection.transaction():
            self.connection.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
            )
            yield self

    def counts(self):
        with self.snapshot():
            self._check_schema()
            return {
                table: self.connection.execute(
                    f"SELECT count(*) AS n FROM benchmark.{table}"
                ).fetchone()["n"]
                for table in TABLES
            }

    def export_run(self, run_id):
        self._check_schema()
        row = self.connection.execute(
            "SELECT artifact, artifact_sha256 FROM benchmark.runs WHERE id = %s",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError("unknown run")
        raw = bytes(row["artifact"])
        if (
            hashlib.sha256(raw).hexdigest() != row["artifact_sha256"]
            or digest(decode_json(raw)) != run_id
        ):
            raise ValueError("stored artifact integrity check failed")
        return raw

    def list_records(
        self,
        *,
        plan_id=None,
        case_id=None,
        gpu=None,
        kind=None,
        adapter_revision=None,
        adapter=None,
        source_id=None,
        environment_id=None,
        protocol_id=None,
        include_declarations=False,
        after=None,
        limit=100,
    ):
        """Bounded, keyset-paginated observations, including metric samples.

        A record is one worker observation. Dense references have no Plan ID.
        Comparability also requires protocol, source, and environment identities.
        """
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be 1..1000")
        if type(include_declarations) is not bool:
            raise ValueError("include_declarations must be bool")
        self._check_schema()
        clauses, params = [], []
        for column, value in (
            ("rp.plan_id", plan_id),
            ("p.case_id", case_id),
            ("r.environment ->> 'gpu'", gpu),
            ("r.kind", kind),
            ("p.adapter_revision", adapter_revision),
            ("p.adapter", adapter),
            ("r.source_id", source_id),
            ("r.environment_id", environment_id),
            ("p.protocol_id", protocol_id),
        ):
            if value is not None:
                clauses.append(column + " = %s")
                params.append(value)
        if after is not None:
            if (
                len(after) != 2
                or type(after[0]) is not str
                or type(after[1]) is not int
                or after[1] < 0
            ):
                raise ValueError("cursor needs a projection ID and nonnegative ordinal")
            clauses.append("(r.projection_id, r.ordinal) > (%s, %s)")
            params.extend(after)
        where = " AND ".join(clauses) or "TRUE"
        params.append(limit)
        declarations = (
            ", c.declaration AS case_declaration, pl.declaration AS plan_declaration, r.payload"
            if include_declarations
            else ""
        )
        return self.connection.execute(
            f"""SELECT r.projection_id, r.ordinal, p.run_id,
            p.adapter, p.adapter_revision, p.case_id, p.protocol_id, p.protocol,
            r.plan_alias, rp.plan_id, rp.is_baseline, r.kind, r.status,
            u.status AS run_status, u.provenance, u.execution_id::text,
            u.started_at::text, u.artifact_schema_version,
            CASE WHEN s.id IS NULL THEN NULL ELSE jsonb_build_object(
                'id', s.id, 'repository', s.repository, 'issue_number', s.issue_number,
                'submitter_id', s.submitter_id, 'submitter_login', s.submitter_login,
                'tier', s.tier, 'trust_points', s.trust_points, 'policy_id', s.policy_id,
                'validation', s.validation, 'measurement_authenticity', s.measurement_authenticity
            ) END AS submission,
            r.environment_id, r.environment, r.source_id, r.source{declarations},
            COALESCE((SELECT jsonb_agg(jsonb_build_object(
                'name', m.name, 'scope', m.scope, 'unit', m.unit,
                'statistic', m.statistic, 'value', m.value,
                'samples', m.samples, 'details', m.details
            ) ORDER BY m.name, m.scope, m.statistic) FROM benchmark.metrics m
                WHERE m.projection_id = r.projection_id AND m.record_ordinal = r.ordinal), '[]'::jsonb) AS metrics
            FROM benchmark.records r JOIN benchmark.projections p ON p.id = r.projection_id
            JOIN benchmark.runs u ON u.id = p.run_id
            LEFT JOIN benchmark.submissions s ON s.run_id = u.id
            LEFT JOIN benchmark.run_plans rp ON rp.projection_id = r.projection_id AND rp.alias = r.plan_alias
            JOIN benchmark.cases c ON c.id = p.case_id
            LEFT JOIN benchmark.plans pl ON pl.id = rp.plan_id
            WHERE {where} ORDER BY r.projection_id, r.ordinal LIMIT %s""",
            params,
        ).fetchall()
