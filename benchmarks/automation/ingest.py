"""Central request-bound validation and transactional ingestion; no provider imports."""

from pathlib import Path

from benchmarks.automation.contract import decode, keys, load_request, sha
from tools.colab.profiles import validate_hardware


def expected_worker_source(request):
    files = request["request"]["source_files"]
    benchmark_names = {
        f"benchmarks/cuda/linear/{name}.py"
        for name in ("run", "reference", "fixtures", "check_normalized", "manifest")
    }
    return {
        name: expected
        for name, expected in files.items()
        if (name.startswith("src/torchcst/") and name.endswith(".py"))
        or name in benchmark_names
    }


def validate_bundle(request_file, directory):
    import hashlib

    from benchmarks.cuda.linear.manifest import decode_snapshot
    from benchmarks.database.adapters.linear import project
    from benchmarks.database.model import execution_identity

    request = load_request(request_file)
    root = Path(directory).resolve()
    bundle = decode((root / "submission.json").read_bytes())
    keys(bundle, ("schema_version", "request_id", "jobs"), "submission")
    if (
        type(bundle["schema_version"]) is not int
        or bundle["schema_version"] != 1
        or bundle["request_id"] != request["request_id"]
    ):
        raise ValueError("submission belongs to another request/version")
    expected_jobs = {j["key"]: j for j in request["request"]["jobs"]}
    if type(bundle["jobs"]) is not list or len(bundle["jobs"]) != len(expected_jobs):
        raise ValueError("incomplete submission")
    body = request["request"]
    source = expected_worker_source(request)
    seen, executions, validated = set(), set(), []
    for entry in bundle["jobs"]:
        keys(
            entry,
            (
                "key",
                "pool_job_id",
                "status",
                "source_archive_sha256",
                "result_archive_sha256",
                "artifact",
                "artifact_sha256",
            ),
            "submission job",
        )
        key = entry["key"]
        if type(key) is not str or key not in expected_jobs or key in seen:
            raise ValueError("unknown/duplicate submitted job")
        seen.add(key)
        if entry["status"] not in ("succeeded", "failed", "cancelled"):
            raise ValueError("submission is not completed")
        sha(entry["source_archive_sha256"])
        if entry["result_archive_sha256"] is not None:
            sha(entry["result_archive_sha256"])
        if entry["artifact"] is None:
            if entry["status"] == "succeeded" or entry["artifact_sha256"] is not None:
                raise ValueError("successful job lacks its benchmark artifact")
            validated.append((entry, None))
            continue
        if (
            entry["artifact"] != f"records/{key}.json"
            or entry["result_archive_sha256"] is None
        ):
            raise ValueError("invalid artifact path / missing retrieval receipt")
        path = root / entry["artifact"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or path.stat().st_size > 32 * 1024**2
        ):
            raise ValueError("unsafe or oversized benchmark artifact")
        raw = path.read_bytes()
        sha(entry["artifact_sha256"])
        if hashlib.sha256(raw).hexdigest() != entry["artifact_sha256"]:
            raise ValueError("benchmark artifact hash mismatch")
        value = decode(raw)
        expected = expected_jobs[key]["run"]
        decode_snapshot(expected)
        if value.get("run") != expected:
            raise ValueError("measured Case/Plans differ from request")
        identity, _ = execution_identity(value)
        if identity is None or identity in executions:
            raise ValueError("missing/duplicate independent execution identity")
        executions.add(identity)
        for record in value["records"]:
            meta = record["metadata"]
            if value["status"] == "FAIL" and "source_hashes" not in meta:
                continue  # Preserve pre-metadata failures without treating them as verified measurements.
            if (
                meta.get("source_commit") != body["source_commit"]
                or meta.get("source_hashes") != source
            ):
                raise ValueError("measured source differs from request")
            validate_hardware(
                expected_jobs[key]["accelerator"],
                meta.get("gpu"),
                meta.get("compute_capability"),
            )
            for field, expected_value in body["environment"].items():
                if meta.get(field) != expected_value:
                    raise ValueError("measured hardware/runtime differs from request")
        projection = project(value)
        if value["status"] == "PASS" and not any(
            r.kind == "measure" for r in projection.records
        ):
            raise ValueError(
                "request requires performance, not correctness-only results"
            )
        if entry["status"] == "succeeded" and value["status"] != "PASS":
            raise ValueError("successful pool job contains a failed benchmark")
        validated.append((entry, raw))
    return request, validated


def ingest(request_file, directory, database, provenance):
    """Caller-supplied provenance is trusted metadata, never read from submission."""
    request, entries = validate_bundle(request_file, directory)
    if type(provenance) is not dict or not provenance:
        raise ValueError("central ingestion needs caller-supplied provenance")
    results = []
    with database.connection.transaction():
        for entry, raw in entries:
            if raw is not None:
                origin = provenance | {
                    "request_id": request["request_id"],
                    "request": request,
                    "measurement": entry,
                    "certification": "not assessed",
                }
                results.append(database.import_linear(raw, provenance=origin))
    return {
        "request_id": request["request_id"],
        "runs": results,
        "jobs_without_observations": [e["key"] for e, raw in entries if raw is None],
    }
