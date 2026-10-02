-- Submission identity and admission policy are separate from immutable run data.
CREATE TABLE benchmark.submission_policies (
    id text PRIMARY KEY CHECK (id ~ '^[0-9a-f]{64}$'),
    declaration jsonb NOT NULL CHECK (jsonb_typeof(declaration) = 'object'),
    received_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE benchmark.submissions (
    id text PRIMARY KEY CHECK (id ~ '^[0-9a-f]{64}$'),
    repository text NOT NULL,
    issue_id bigint NOT NULL CHECK (issue_id > 0),
    issue_number bigint NOT NULL CHECK (issue_number > 0),
    issue_body_sha256 text NOT NULL CHECK (issue_body_sha256 ~ '^[0-9a-f]{64}$'),
    submitter_id bigint NOT NULL CHECK (submitter_id > 0),
    submitter_login text NOT NULL,
    tier text NOT NULL CHECK (tier IN ('general', 'recognized')),
    trust_points integer NOT NULL CHECK (trust_points >= 1),
    policy_id text NOT NULL REFERENCES benchmark.submission_policies(id),
    run_id text NOT NULL UNIQUE REFERENCES benchmark.runs(id),
    artifact_url text NOT NULL,
    artifact_bytes bigint NOT NULL CHECK (artifact_bytes > 0),
    validation text NOT NULL CHECK (validation = 'consistency_checked'),
    measurement_authenticity text NOT NULL CHECK (measurement_authenticity = 'self_reported'),
    received_at timestamptz NOT NULL,
    UNIQUE (repository, issue_id),
    UNIQUE (repository, issue_number)
);

CREATE INDEX submission_quota ON benchmark.submissions(submitter_id, received_at);

CREATE TRIGGER immutable_evidence BEFORE UPDATE OR DELETE OR TRUNCATE
    ON benchmark.submission_policies FOR EACH STATEMENT
    EXECUTE FUNCTION benchmark.reject_evidence_mutation();
CREATE TRIGGER immutable_evidence BEFORE UPDATE OR DELETE OR TRUNCATE
    ON benchmark.submissions FOR EACH STATEMENT
    EXECUTE FUNCTION benchmark.reject_evidence_mutation();
