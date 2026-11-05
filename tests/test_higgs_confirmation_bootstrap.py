"""Immutable source and recovery-capsule boundaries without GPU execution."""

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.training import higgs_confirmation_bootstrap as bootstrap

COMMIT = "a" * 40
RUNTIME = {"python": "3.13.16", "torch": "2.11.0+cu130", "gpu": "NVIDIA L4"}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def source_tree(tmp_path):
    source = tmp_path / "original"
    inputs = source / "research-inputs"
    inputs.mkdir(parents=True)
    (source / "module.py").write_text("value = 4\n")
    driver = source / "benchmarks/training/higgs_confirmation_bootstrap.py"
    driver.parent.mkdir(parents=True)
    driver.write_bytes(Path(bootstrap.__file__).read_bytes())
    (inputs / "screen.npz").write_bytes(b"screen payload")
    (inputs / "source-manifest.json").write_text(
        json.dumps(
            {
                "commit": COMMIT,
                "files": {
                    "module.py": digest((source / "module.py").read_bytes()),
                    "benchmarks/training/higgs_confirmation_bootstrap.py": digest(
                        driver.read_bytes()
                    ),
                },
            }
        )
    )
    (inputs / "data-manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {"screen.npz": {"sha256": digest(b"screen payload")}},
            }
        )
    )
    return source


def bound_state(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    binding = bootstrap.recovery_binding(COMMIT, "b" * 64, RUNTIME)
    bootstrap.write_json(state / bootstrap.STATE_BINDING, binding)
    (state / "confirmed").mkdir()
    (state / "confirmed/checkpoint.pt").write_bytes(b"actual checkpoint bytes")
    return state, binding


def capsule(tmp_path):
    state, binding = bound_state(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    result = bootstrap.archive_state(state, output, binding)
    return output / result["file"], binding


def test_source_copy_is_verified_reusable_and_preserves_source_mtime(tmp_path):
    source = source_tree(tmp_path)
    original_time = (source / "module.py").stat().st_mtime_ns
    cached, marker = bootstrap.prepare_source_cache(source, tmp_path / "cache", COMMIT)
    assert (cached / "module.py").read_bytes() == b"value = 4\n"
    assert (cached / "module.py").stat().st_mtime_ns == original_time
    again, second = bootstrap.prepare_source_cache(source, tmp_path / "cache", COMMIT)
    assert again == cached and second == marker
    assert marker["dataset_sha256"] == digest(b"screen payload")


@pytest.mark.parametrize(
    "change", ("supplied_source", "cached_source", "extra_code", "dataset", "commit")
)
def test_source_and_dataset_tampering_cannot_silently_reuse_cache(tmp_path, change):
    source = source_tree(tmp_path)
    cache = tmp_path / "cache"
    cached, _ = bootstrap.prepare_source_cache(source, cache, COMMIT)
    if change == "supplied_source":
        (source / "module.py").write_text("changed source")
    elif change == "cached_source":
        (cached / "module.py").write_text("changed cached source")
    elif change == "extra_code":
        (cached / "injected.py").write_text("arbitrary = 1")
    elif change == "dataset":
        (source / "research-inputs/screen.npz").write_bytes(b"different examples")
    else:
        manifest = source / "research-inputs/source-manifest.json"
        record = json.loads(manifest.read_text())
        record["commit"] = "c" * 40
        manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        bootstrap.prepare_source_cache(source, cache, COMMIT)


def test_source_symlinks_are_rejected_even_when_their_content_hash_matches(tmp_path):
    source = source_tree(tmp_path)
    content = source / "content.py"
    (source / "module.py").rename(content)
    (source / "module.py").symlink_to(content)
    with pytest.raises(ValueError, match="symlink"):
        bootstrap.prepare_source_cache(source, tmp_path / "cache", COMMIT)


def test_recovery_round_trip_preserves_all_files_and_binding(tmp_path):
    archive, binding = capsule(tmp_path)
    target = tmp_path / "restored"
    result = bootstrap.restore_state(archive, target, binding)
    assert result["files"] == 2
    assert (
        target / "confirmed/checkpoint.pt"
    ).read_bytes() == b"actual checkpoint bytes"
    assert json.loads((target / bootstrap.STATE_BINDING).read_text()) == binding
    # Repeated import is read-only for existing, byte-identical state.
    second = bootstrap.restore_state(archive, target, binding)
    assert second["restored_files"] == []


@pytest.mark.parametrize(
    "field", ("source_commit", "dataset_sha256", "runtime_fingerprint")
)
def test_recovery_rejects_different_study_or_runtime_before_copying(tmp_path, field):
    archive, binding = capsule(tmp_path)
    changed = dict(binding)
    changed[field] = "different"
    target = tmp_path / "restored"
    with pytest.raises(ValueError, match="binding mismatch"):
        bootstrap.restore_state(archive, target, changed)
    assert not target.exists()


def test_recovery_never_overwrites_a_newer_or_conflicting_checkpoint(tmp_path):
    archive, binding = capsule(tmp_path)
    target = tmp_path / "restored"
    bootstrap.restore_state(archive, target, binding)
    checkpoint = target / "confirmed/checkpoint.pt"
    checkpoint.write_bytes(b"newer checkpoint")
    with pytest.raises(ValueError, match="overwrite different existing state"):
        bootstrap.restore_state(archive, target, binding)
    assert checkpoint.read_bytes() == b"newer checkpoint"


@pytest.mark.parametrize(
    "kind", ("traversal", "absolute", "symlink", "duplicate", "hash")
)
def test_recovery_rejects_unsafe_or_tampered_archive_entries(tmp_path, kind):
    archive, binding = capsule(tmp_path)
    replacement = tmp_path / "tampered.tar.gz"
    with tarfile.open(archive) as original, tarfile.open(replacement, "w:gz") as output:
        for member in original.getmembers():
            raw = original.extractfile(member).read()
            if kind == "hash" and member.name.endswith("checkpoint.pt"):
                raw = b"changed checkpoint bytes"
                member.size = len(raw)
            output.addfile(member, io.BytesIO(raw))
            if kind == "duplicate" and member.name == bootstrap.STATE_BINDING:
                output.addfile(member, io.BytesIO(raw))
        if kind in ("traversal", "absolute", "symlink"):
            member = tarfile.TarInfo(
                {
                    "traversal": "../outside",
                    "absolute": "/outside",
                    "symlink": "linked",
                }[kind]
            )
            if kind == "symlink":
                member.type, member.linkname = tarfile.SYMTYPE, "../outside"
            output.addfile(member)
    target = tmp_path / "restored"
    with pytest.raises(ValueError):
        bootstrap.restore_state(replacement, target, binding)
    assert not target.exists()


def test_failed_phase_still_preserves_recoverable_state(tmp_path, monkeypatch):
    source = source_tree(tmp_path)
    output = tmp_path / "phase-output"
    output.mkdir()
    monkeypatch.chdir(source)
    monkeypatch.setenv("CST_JOB_OUTPUT", str(output))
    monkeypatch.setattr(bootstrap, "probe_runtime", lambda: RUNTIME)
    monkeypatch.setattr(
        bootstrap, "prepare_dependencies", lambda source, cache: source / ".job-deps"
    )

    def failed_command(command, **options):
        state = tmp_path / "cache" / COMMIT / "campaign-state"
        (state / "partial.json").write_text('{"finished": false}')
        assert options["cwd"] == tmp_path / "cache" / COMMIT / "source"
        assert options["env"]["CST_CONFIRM_SOURCE_SHA256"] == digest(
            (source / "research-inputs/source-manifest.json").read_bytes()
        )
        assert "--stage" in command and "--source-identity" in command
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(bootstrap.subprocess, "run", failed_command)
    args = SimpleNamespace(
        source_commit=COMMIT,
        stage="tune",
        width=1024,
        kind="dense",
        divisor=None,
        seed=None,
    )
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.run(args, cache_root=tmp_path / "cache")
    status = json.loads((output / "bootstrap-status.json").read_text())
    assert status["returncode"] == 9
    assert status["failure"]["type"] == "CalledProcessError"
    binding = bootstrap.recovery_binding(COMMIT, digest(b"screen payload"), RUNTIME)
    bootstrap.restore_state(
        output / "recovery-state.tar.gz", tmp_path / "restore", binding
    )
    assert (tmp_path / "restore/partial.json").read_text() == '{"finished": false}'


def test_invalid_commit_cannot_create_cache_directories(tmp_path):
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        bootstrap.run(SimpleNamespace(source_commit="../outside"), cache_root=cache)
    assert not cache.exists()
    assert not (tmp_path / "outside").exists()
