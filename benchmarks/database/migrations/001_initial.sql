CREATE TABLE benchmark.plans (
    id text PRIMARY KEY CHECK (id ~ '^[0-9a-f]{64}$'),
    algorithm_id text NOT NULL,
    algorithm_revision text NOT NULL,
    declaration jsonb NOT NULL CHECK (jsonb_typeof(declaration) = 'object')
);

CREATE TABLE benchmark.cases (
    id text PRIMARY KEY CHECK (id ~ '^[0-9a-f]{64}$'),
    declaration jsonb NOT NULL CHECK (jsonb_typeof(declaration) = 'object')
);

CREATE TABLE benchmark.runs (
    id text PRIMARY KEY CHECK (id ~ '^[0-9a-f]{64}$'),
    artifact_sha256 text NOT NULL CHECK (artifact_sha256 ~ '^[0-9a-f]{64}$'),
    artifact bytea NOT NULL,
    artifact_schema_version integer NOT NULL CHECK (artifact_schema_version > 0),
    execution_id uuid UNIQUE,
    started_at timestamptz,
    status text NOT NULL CHECK (status IN ('PASS', 'FAIL')),
    provenance jsonb NOT NULL CHECK (jsonb_typeof(provenance) = 'object'),
    received_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CHECK ((execution_id IS NULL) = (started_at IS NULL))
);

-- A later adapter may add a new projection without changing the original run.
CREATE TABLE benchmark.projections (
    id text PRIMARY KEY CHECK (id ~ '^[0-9a-f]{64}$'),
    run_id text NOT NULL REFERENCES benchmark.runs(id),
    adapter text NOT NULL,
    adapter_revision integer NOT NULL CHECK (adapter_revision > 0),
    case_id text NOT NULL REFERENCES benchmark.cases(id),
    protocol_id text NOT NULL CHECK (protocol_id ~ '^[0-9a-f]{64}$'),
    protocol jsonb NOT NULL CHECK (jsonb_typeof(protocol) = 'object'),
    UNIQUE (run_id, adapter, adapter_revision)
);

CREATE TABLE benchmark.run_plans (
    projection_id text NOT NULL REFERENCES benchmark.projections(id),
    alias text NOT NULL,
    plan_id text NOT NULL REFERENCES benchmark.plans(id),
    is_baseline boolean NOT NULL,
    PRIMARY KEY (projection_id, alias),
    UNIQUE (projection_id, plan_id)
);
CREATE UNIQUE INDEX one_baseline ON benchmark.run_plans (projection_id)
    WHERE is_baseline;

CREATE TABLE benchmark.records (
    projection_id text NOT NULL REFERENCES benchmark.projections(id),
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    plan_alias text,
    kind text NOT NULL,
    status text NOT NULL CHECK (status IN ('PASS', 'FAIL')),
    environment_id text NOT NULL CHECK (environment_id ~ '^[0-9a-f]{64}$'),
    environment jsonb NOT NULL CHECK (jsonb_typeof(environment) = 'object'),
    source_id text NOT NULL CHECK (source_id ~ '^[0-9a-f]{64}$'),
    source jsonb NOT NULL CHECK (jsonb_typeof(source) = 'object'),
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    PRIMARY KEY (projection_id, ordinal),
    FOREIGN KEY (projection_id, plan_alias)
        REFERENCES benchmark.run_plans(projection_id, alias)
);

CREATE TABLE benchmark.metrics (
    projection_id text NOT NULL,
    record_ordinal integer NOT NULL,
    name text NOT NULL,
    scope text NOT NULL,
    unit text NOT NULL,
    statistic text NOT NULL,
    value numeric NOT NULL CHECK (value >= 0 AND value::text NOT IN ('NaN', 'Infinity')),
    samples jsonb NOT NULL CHECK (jsonb_typeof(samples) = 'array'),
    details jsonb NOT NULL CHECK (jsonb_typeof(details) = 'object'),
    PRIMARY KEY (projection_id, record_ordinal, name, scope, statistic),
    FOREIGN KEY (projection_id, record_ordinal)
        REFERENCES benchmark.records(projection_id, ordinal)
);

CREATE INDEX projections_case ON benchmark.projections(case_id, protocol_id);
CREATE INDEX plans_observations ON benchmark.run_plans(plan_id, projection_id);
CREATE INDEX records_gpu ON benchmark.records((environment ->> 'gpu'));
CREATE INDEX records_environment ON benchmark.records(environment_id, source_id);
CREATE INDEX metrics_lookup ON benchmark.metrics(name, scope, unit, statistic);

CREATE FUNCTION benchmark.reject_evidence_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'benchmark evidence is append-only';
END;
$$;

DO $$
DECLARE evidence_table text;
BEGIN
    FOREACH evidence_table IN ARRAY ARRAY[
        'plans', 'cases', 'runs', 'projections', 'run_plans', 'records', 'metrics'
    ] LOOP
        EXECUTE format(
            'CREATE TRIGGER immutable_evidence BEFORE UPDATE OR DELETE OR TRUNCATE ON benchmark.%I FOR EACH STATEMENT EXECUTE FUNCTION benchmark.reject_evidence_mutation()',
            evidence_table
        );
    END LOOP;
END;
$$;
