"""Versioned measurement contracts. Reading them needs only the standard library."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from tools.colab.profiles import target

BENCHMARK = "cuda.linear.complete-step"
ENVIRONMENT = {"torch": "2.11.0+cu128", "cuda": "12.8", "triton": "3.6.0"}


def encode(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON number")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    encode(value)  # Also reject overflowing finite-looking number literals.
    return value


def keys(value, fields, name):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"invalid {name} fields")


def sha(value, length=64):
    if type(value) is not str or not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
        raise ValueError("invalid content hash / full source commit")


def relative_path(value):
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or value.startswith("/")
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise ValueError("unsafe relative path")
    return value


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def source_files(root):
    """Require an exact committed checkout; never freeze unreviewed local edits."""
    root = Path(root).resolve()
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("source must be the Git checkout root")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("commit source changes before creating/submitting a request")
    names = git(root, "ls-files", "-z").split("\0")
    result = {}
    for name in names:
        if not name:
            continue
        relative_path(name)
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("source symlinks/submodules are unsupported")
        if name.split("/")[-1].startswith(".env"):
            raise ValueError(
                "credentials must not be tracked in a measurement checkout"
            )
        result[name] = file_hash(path)
    return result


def validate_request(value):
    keys(value, ("schema_version", "request_id", "request"), "request envelope")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported request version")
    body = value["request"]
    keys(
        body,
        (
            "benchmark",
            "source_commit",
            "source_files",
            "environment",
            "targets",
            "configuration",
            "timeout_seconds",
            "jobs",
        ),
        "request",
    )
    sha(value["request_id"])
    if digest(body) != value["request_id"]:
        raise ValueError("request content hash mismatch")
    sha(body["source_commit"], 40)
    if body["benchmark"] != BENCHMARK or body["environment"] != ENVIRONMENT:
        raise ValueError("unsupported benchmark or environment")
    targets = body["targets"]
    if type(targets) is not list or not targets:
        raise ValueError("request needs explicit Colab targets")
    accelerators = []
    for item in targets:
        keys(item, ("provider", "accelerator"), "target")
        if item != target(item["accelerator"]) or item["accelerator"] in accelerators:
            raise ValueError("unsupported/duplicate Colab target")
        accelerators.append(item["accelerator"])
    if (
        type(body["timeout_seconds"]) is not int
        or not 60 <= body["timeout_seconds"] <= 1200
    ):
        raise ValueError("request timeout must be 60..1200 seconds per job")
    files = body["source_files"]
    if type(files) is not dict or not files:
        raise ValueError("missing frozen source inventory")
    for name, expected in files.items():
        relative_path(name)
        sha(expected)
    config = body["configuration"]
    keys(config, ("path", "sha256"), "configuration")
    relative_path(config["path"])
    sha(config["sha256"])
    if files.get(config["path"]) != config["sha256"]:
        raise ValueError("input JSON differs from frozen source")
    jobs = body["jobs"]
    if type(jobs) is not list or not 1 <= len(jobs) <= 6:
        raise ValueError("request must contain 1..6 jobs")
    seen = set()
    for job in jobs:
        keys(job, ("key", "accelerator", "repetition", "run"), "job")
        if (
            type(job["key"]) is not str
            or not re.fullmatch("[a-z][a-z0-9_.-]*", job["key"])
            or job["key"] in seen
            or type(job["repetition"]) is not int
            or not 1 <= job["repetition"] <= 3
        ):
            raise ValueError("invalid/duplicate job key or repetition")
        seen.add(job["key"])
        keys(
            job["run"],
            ("schema_version", "case", "plans", "baseline", "dense", "input_hashes"),
            "run",
        )
        if (
            job["accelerator"] not in accelerators
            or job["key"]
            != f"{job['run']['case']['id']}.{job['accelerator'].lower()}.r{job['repetition']}"
        ):
            raise ValueError("job key differs from case and repetition")
    return value


def load_request(path):
    return validate_request(decode(Path(path).read_bytes()))


def prepare(root, configuration):
    from benchmarks.cuda.linear.manifest import load_run

    root = Path(root).resolve()
    files = source_files(root)
    configuration = Path(configuration)
    if not configuration.is_absolute():
        configuration = root / configuration
    if not configuration.resolve().is_relative_to(root):
        raise ValueError("request JSON must be inside the committed checkout")
    configuration_name = str(configuration.resolve().relative_to(root))
    if configuration_name not in files:
        raise ValueError("request JSON must be tracked and committed")
    spec = decode(configuration.read_bytes())
    keys(
        spec,
        (
            "schema_version",
            "benchmark",
            "catalog",
            "cases",
            "targets",
            "environment",
            "repetitions",
            "timeout_seconds",
        ),
        "measurement JSON",
    )
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["benchmark"] != BENCHMARK
    ):
        raise ValueError("unsupported measurement JSON version/benchmark")
    cases, accelerators, repetitions = (
        spec["cases"],
        spec["targets"],
        spec["repetitions"],
    )
    if (
        type(cases) is not list
        or not cases
        or any(type(c) is not str for c in cases)
        or len(set(cases)) != len(cases)
        or type(accelerators) is not list
        or not accelerators
        or any(type(g) is not str for g in accelerators)
        or len(set(accelerators)) != len(accelerators)
        or type(repetitions) is not int
        or not 1 <= repetitions <= 3
        or len(cases) * len(accelerators) * repetitions > 6
    ):
        raise ValueError(
            "measurement JSON requires explicit unique cases/GPUs, 1..3 repetitions, at most 6 jobs"
        )
    targets = [target(accelerator) for accelerator in accelerators]
    for name in [spec["catalog"], *cases]:
        relative_path(name)
        if name not in files:
            raise ValueError(
                "Case and catalog must be tracked in the same source commit"
            )
    jobs = []
    for case in cases:
        run = load_run(
            root / case,
            root / spec["catalog"],
        ).snapshot()
        for accelerator in accelerators:
            for repetition in range(1, repetitions + 1):
                jobs.append(
                    {
                        "key": f"{run['case']['id']}.{accelerator.lower()}.r{repetition}",
                        "accelerator": accelerator,
                        "repetition": repetition,
                        "run": run,
                    }
                )
    body = {
        "benchmark": BENCHMARK,
        "source_commit": git(root, "rev-parse", "HEAD"),
        "source_files": files,
        "environment": spec["environment"],
        "targets": targets,
        "configuration": {
            "path": configuration_name,
            "sha256": files[configuration_name],
        },
        "timeout_seconds": spec["timeout_seconds"],
        "jobs": jobs,
    }
    return validate_request(
        {"schema_version": 1, "request_id": digest(body), "request": body}
    )


def verify_source(root, request, *, checkout=False):
    body = validate_request(request)["request"]
    root = Path(root).resolve()
    if checkout and (
        git(root, "rev-parse", "HEAD") != body["source_commit"]
        or source_files(root) != body["source_files"]
    ):
        raise ValueError("checkout differs from frozen request")
    for name, expected in body["source_files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"frozen source mismatch: {name}")
