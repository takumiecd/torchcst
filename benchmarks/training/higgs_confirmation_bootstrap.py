"""Pool-driver bootstrap for immutable HIGGS source and recoverable phase state.

Run this file as the shared pool's ``--script``. It never provisions, contacts or
stops a Colab runtime. The pool owns those operations. Immutable source paths
avoid rebuilding the same native extension solely because the phase job ID
changed. Mutable campaign state is outside the source tree and is included in
each phase's verified recovery capsule, including failed phases.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

RECOVERY_MANIFEST = "__higgs_recovery_manifest__.json"
STATE_BINDING = "bootstrap-binding.json"
_HEX256 = re.compile(r"[0-9a-f]{64}")
_HEX160 = re.compile(r"[0-9a-f]{40}")
_INPUT_NAMES = ("source-manifest.json", "screen.npz", "data-manifest.json")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Create a new record; records are never silently overwritten."""
    with Path(path).open("x") as destination:
        json.dump(value, destination, indent=2, allow_nan=False)
        destination.write("\n")


def safe_path(root, name):
    relative = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or relative.is_absolute()
        or ".." in relative.parts
        or str(relative) != name
        or relative == PurePosixPath(".")
    ):
        raise ValueError(f"unsafe relative path: {name!r}")
    root = Path(root)
    path = root.joinpath(*relative.parts)
    if root.is_symlink() or any(
        ancestor.is_symlink()
        for ancestor in (path, *path.parents)
        if ancestor != root.parent
    ):
        raise ValueError(f"symlink is not permitted: {name!r}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path escapes its root: {name!r}")
    return path


def regular_files(root):
    """List only regular files, refusing symlink directories and devices."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("symlink root is not permitted")
    if not root.exists():
        return {}
    result = {}
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"symlink in artifact tree: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"nonregular artifact: {path}")
        result[path.relative_to(root).as_posix()] = path
    return result


def read_source_manifest(source, commit):
    if not _HEX160.fullmatch(commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    manifest_path = safe_path(source, "research-inputs/source-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("commit") != commit:
        raise ValueError("source manifest commit differs from requested source")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source manifest must contain tracked file hashes")
    for name, expected in files.items():
        path = safe_path(source, name)
        if name.startswith(("research-inputs/", ".git/")):
            raise ValueError("source manifest may only enumerate tracked source files")
        if type(expected) is not str or not _HEX256.fullmatch(expected):
            raise ValueError(f"invalid source SHA256: {name}")
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"source file hash mismatch: {name}")
    inputs = {}
    for name in _INPUT_NAMES:
        path = safe_path(source, "research-inputs/" + name)
        if not path.is_file():
            raise ValueError(f"missing required HIGGS input: {name}")
        inputs[name] = file_hash(path)
    data = json.loads(
        safe_path(source, "research-inputs/data-manifest.json").read_text()
    )
    # The prepared dataset manifest binds the actual normalized screen archive.
    prepared_files = data.get("artifacts", {})
    screen = prepared_files.get("screen.npz")
    expected_screen = screen.get("sha256") if isinstance(screen, dict) else screen
    if expected_screen != inputs["screen.npz"]:
        raise ValueError("screen archive differs from prepared data manifest")
    return manifest, inputs


def prepare_source_cache(source, cache, commit):
    """Verify a frozen input tree, then create or verify its immutable copy."""
    source, cache = Path(source), Path(cache)
    manifest, inputs = read_source_manifest(source, commit)
    expected = dict(manifest["files"])
    expected.update(
        {"research-inputs/" + name: value for name, value in inputs.items()}
    )
    marker = {
        "schema_version": 1,
        "source_commit": commit,
        "source_manifest_sha256": inputs["source-manifest.json"],
        "dataset_sha256": inputs["screen.npz"],
        "data_manifest_sha256": inputs["data-manifest.json"],
        "tracked_files": len(manifest["files"]),
    }
    destination, marker_path = cache / "source", cache / "source-binding.json"
    cache.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not marker_path.is_file() or json.loads(marker_path.read_text()) != marker:
            raise ValueError("existing source cache has a different binding")
    else:
        if marker_path.exists():
            raise ValueError("source binding exists without its source tree")
        temporary = Path(tempfile.mkdtemp(prefix=".source-build-", dir=cache))
        try:
            for name in expected:
                target = safe_path(temporary, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(safe_path(source, name), target)
            temporary.rename(destination)
            write_json(marker_path, marker)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    observed = regular_files(destination)
    # Only the isolated dependency target may contain additional runtime files.
    unexpected = sorted(
        name
        for name in observed
        if name not in expected and not name.startswith(".job-deps/")
    )
    missing = sorted(set(expected) - set(observed))
    bad = sorted(
        name
        for name, wanted in expected.items()
        if name in observed and file_hash(observed[name]) != wanted
    )
    if unexpected or missing or bad:
        raise ValueError(
            f"immutable source cache mismatch: unexpected={unexpected}, missing={missing}, changed={bad}"
        )
    return destination, marker


def recovery_binding(commit, dataset_sha256, runtime):
    return {
        "schema_version": 1,
        "source_commit": commit,
        "dataset_sha256": dataset_sha256,
        "runtime_fingerprint": runtime,
    }


def restore_state(archive, state_directory, binding):
    """Validate every capsule entry before merging only absent, compatible files."""
    archive, state_directory = Path(archive), Path(state_directory)
    state_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".state-restore-", dir=state_directory.parent)
    )
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or RECOVERY_MANIFEST not in names:
                raise ValueError("recovery archive lacks a unique manifest")
            for member in members:
                safe_path(temporary, member.name)
                if not member.isfile():
                    raise ValueError("recovery archive must contain regular files only")
            record_member = bundle.getmember(RECOVERY_MANIFEST)
            if record_member.size > 16 * 1024 * 1024:
                raise ValueError("recovery manifest is unexpectedly large")
            record = json.load(bundle.extractfile(record_member))
            if any(record.get(key) != value for key, value in binding.items()):
                raise ValueError("recovery source, dataset or runtime binding mismatch")
            expected = record.get("files")
            if not isinstance(expected, dict) or set(names) != set(expected) | {
                RECOVERY_MANIFEST
            }:
                raise ValueError("recovery archive inventory differs from its manifest")
            for name, metadata in expected.items():
                target = safe_path(temporary, name)
                member = bundle.getmember(name)
                if member.size != metadata.get("bytes"):
                    raise ValueError(f"recovery size mismatch: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as destination:
                    shutil.copyfileobj(bundle.extractfile(member), destination)
                if file_hash(target) != metadata.get("sha256"):
                    raise ValueError(f"recovery hash mismatch: {name}")
        if (
            expected.get(STATE_BINDING) is None
            or json.loads((temporary / STATE_BINDING).read_text()) != binding
        ):
            raise ValueError("recovery capsule lacks its bound state identity")
        existing = regular_files(state_directory)
        for name, path in existing.items():
            if name in expected and file_hash(path) != expected[name]["sha256"]:
                raise ValueError(
                    f"recovery would overwrite different existing state: {name}"
                )
        restored = []
        state_directory.mkdir(exist_ok=True)
        for name in expected:
            target = safe_path(state_directory, name)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temporary / name), target)
                restored.append(name)
        return {
            "archive_sha256": file_hash(archive),
            "files": len(expected),
            "restored_files": restored,
            "compatible_existing_files": sorted(set(expected) & set(existing)),
        }
    finally:
        shutil.rmtree(temporary)


def archive_state(state_directory, output, binding):
    state_directory, output = Path(state_directory), Path(output)
    paths = regular_files(state_directory)
    if RECOVERY_MANIFEST in paths:
        raise ValueError("reserved recovery manifest name appears in campaign state")
    if (
        STATE_BINDING not in paths
        or json.loads(paths[STATE_BINDING].read_text()) != binding
    ):
        raise ValueError("campaign state binding changed before preservation")
    record = dict(binding)
    record["files"] = {
        name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
        for name, path in sorted(paths.items())
    }
    archive = output / "recovery-state.tar.gz"
    if archive.exists():
        raise FileExistsError(archive)
    manifest_bytes = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as bundle:
        for name, path in sorted(paths.items()):
            bundle.add(path, arcname=name, recursive=False)
        member = tarfile.TarInfo(RECOVERY_MANIFEST)
        member.size = len(manifest_bytes)
        bundle.addfile(member, io.BytesIO(manifest_bytes))
    write_json(output / "recovery-state-manifest.json", record)
    result = {
        "file": archive.name,
        "sha256": file_hash(archive),
        "bytes": archive.stat().st_size,
        "files": len(paths),
    }
    write_json(output / "recovery-state-receipt.json", result)
    return result


def probe_runtime():
    code = """import importlib.metadata as m,json,platform,subprocess,sys,torch
assert torch.cuda.get_device_name() == 'NVIDIA L4', 'expected exactly NVIDIA L4'
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability() == (8, 9)
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'triton':m.version('triton'),'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )
    return json.loads(result.stdout)


def prepare_dependencies(source, cache):
    """Install into a temporary private target; publish only a complete target."""
    source, cache = Path(source), Path(cache)
    dependencies, ready = source / ".job-deps", cache / "dependencies.json"
    requested = ["ninja", "pytest>=8,<9"]
    if ready.exists():
        if (
            json.loads(ready.read_text()).get("requested") != requested
            or not dependencies.is_dir()
            or not list(dependencies.glob("ninja-*.dist-info"))
            or not list(dependencies.glob("pytest-*.dist-info"))
        ):
            raise ValueError("dependency cache declaration or package target changed")
        return dependencies
    if dependencies.exists():
        raise ValueError("unbound dependency target already exists")
    temporary = Path(tempfile.mkdtemp(prefix=".job-deps-build-", dir=source))
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--target",
                str(temporary),
                *requested,
            ],
            check=True,
        )
        temporary.rename(dependencies)
        write_json(ready, {"requested": requested})
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return dependencies


@contextmanager
def exclusive_cache(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "bootstrap.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another phase is already using this campaign cache"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(args, *, cache_root=Path("/content/torchcst-higgs-confirmation")):
    if not _HEX160.fullmatch(args.source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    original = Path.cwd()
    output = Path(os.environ["CST_JOB_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_root) / args.source_commit
    with exclusive_cache(cache):
        source, marker = prepare_source_cache(original, cache, args.source_commit)
        supplied = json.loads(
            (original / "research-inputs/source-manifest.json").read_text()
        )
        driver_hash = supplied["files"].get(
            "benchmarks/training/higgs_confirmation_bootstrap.py"
        )
        if not driver_hash or file_hash(__file__) != driver_hash:
            raise ValueError(
                "executed pool driver differs from its tracked bootstrap source"
            )
        runtime = probe_runtime()
        runtime_path = cache / "runtime.json"
        if runtime_path.exists():
            if json.loads(runtime_path.read_text()) != runtime:
                raise ValueError("existing study cache runtime fingerprint differs")
        else:
            write_json(runtime_path, runtime)
        binding = recovery_binding(
            args.source_commit, marker["dataset_sha256"], runtime
        )
        state = cache / "campaign-state"
        capsule = original / "research-inputs/recovery-state.tar.gz"
        restoration = (
            restore_state(capsule, state, binding) if capsule.exists() else None
        )
        state.mkdir(exist_ok=True)
        binding_path = state / STATE_BINDING
        if binding_path.exists():
            if json.loads(binding_path.read_text()) != binding:
                raise ValueError("existing campaign state has a different binding")
        elif regular_files(state):
            raise ValueError("existing campaign state is unbound")
        else:
            write_json(binding_path, binding)
        write_json(
            output / "bootstrap-source-verification.json",
            {
                **marker,
                "executed_source": str(source),
                "runtime": runtime,
                "restoration": restoration,
                "phase": args.stage,
            },
        )
        result, failure = None, None
        try:
            dependencies = prepare_dependencies(source, cache)
            environment = dict(os.environ)
            environment.update(
                PYTHONPATH=os.pathsep.join(
                    (str(dependencies), str(source / "src"), str(source))
                ),
                PATH=str(dependencies / "bin") + os.pathsep + environment["PATH"],
                PYTHONDONTWRITEBYTECODE="1",
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                PYTEST_ADDOPTS="-p no:cacheprovider",
                MAX_JOBS="2",
                TORCH_CUDA_ARCH_LIST="8.9",
                TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"),
                CST_CONFIRM_SOURCE_SHA256=marker["source_manifest_sha256"],
            )
            command = [
                sys.executable,
                "-m",
                "benchmarks.training.higgs_confirmation_campaign",
                "--stage",
                args.stage,
                "--data",
                str(source / "research-inputs/screen.npz"),
                "--output",
                str(output),
                "--campaign-state",
                str(state),
                "--source-identity",
                marker["source_manifest_sha256"],
            ]
            for name in ("width", "kind", "divisor", "seed"):
                value = getattr(args, name)
                if value is not None:
                    command.extend(("--" + name, str(value)))
            write_json(
                output / "bootstrap-command.json",
                {
                    "command": command,
                    "source_commit": args.source_commit,
                    "source_identity": marker["source_manifest_sha256"],
                },
            )
            result = subprocess.run(command, cwd=source, env=environment, check=False)
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, command)
        except BaseException as error:
            failure = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            write_json(
                output / "bootstrap-status.json",
                {
                    "stage": args.stage,
                    "returncode": None if result is None else result.returncode,
                    "failure": failure,
                },
            )
            archive_state(state, output, binding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--stage",
        "--phase",
        choices=("qualify", "tune", "freeze-width", "confirm", "assess"),
        required=True,
    )
    parser.add_argument("--width", type=int, choices=(1024, 2048))
    parser.add_argument("--kind", choices=("dense", "cst"))
    parser.add_argument("--divisor", type=int, choices=(1000, 500, 200, 100))
    parser.add_argument("--seed", type=int, choices=(29, 43, 71))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
