"""Real archive reconstruction and bounded existing-CLI calls, without Colab."""

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from tools.colab.scripts import remote_runner, source_upload


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    archive = tmp_path / "local.tar.gz"
    archive.write_bytes(bytes(range(251)) * 3)
    job = "l4job-" + "a" * 32
    spec = {
        "id": job,
        "remote_archive": str(tmp_path / f"{job}.tar.gz"),
        "source_sha256": sha(archive.read_bytes()),
    }
    calls = []

    def call(slot, *args, **kwargs):
        assert slot == 2 and args[:3] == ("upload", "-s", "owned")
        calls.append((args, kwargs, Path(args[3]).read_bytes()))
        shutil.copyfile(args[3], args[4])

    monkeypatch.setattr(source_upload, "CHUNK_BYTES", 256)
    monkeypatch.setattr(source_upload, "CONTENT_ROOT", str(tmp_path))
    return archive, spec, calls, call


def test_chunked_upload_reconstructs_original_and_preserves_unrelated(transfer):
    archive, spec, calls, call = transfer
    unrelated = Path(spec["remote_archive"] + ".part-foreign")
    unrelated.write_bytes(b"unrelated")
    parts = source_upload.upload_source(call, 2, "owned", archive, spec, None)
    assert len(parts) == 3 and len(calls) == 3
    assert [part["bytes"] for part in parts] == [256, 256, 241]
    assert all(call[1]["timeout"] <= 180 for call in calls)
    assert spec["source_sha256"] == sha(archive.read_bytes())
    remote_runner.assemble_source(spec, parts, archive.parent)
    assert Path(spec["remote_archive"]).read_bytes() == archive.read_bytes()
    assert all(not Path(part["path"]).exists() for part in parts)
    assert unrelated.read_bytes() == b"unrelated"
    assert all(not Path(call[0][3]).exists() for call in calls)


def test_single_upload_gets_bounded_request_deadline(transfer):
    archive, spec, calls, call = transfer
    archive.write_bytes(b"small")
    spec["source_sha256"] = sha(b"small")
    assert source_upload.upload_source(call, 2, "owned", archive, spec, None) == []
    assert len(calls) == 1 and calls[0][0][4] == spec["remote_archive"]
    assert calls[0][1]["timeout"] == 180


def test_one_total_deadline_not_900_seconds_per_chunk(transfer, monkeypatch):
    archive, spec, calls, call = transfer
    clock = [0.0]
    monkeypatch.setattr(source_upload.time, "monotonic", lambda: clock[0])

    def slow(*args, **kwargs):
        call(*args, **kwargs)
        clock[0] += 600

    with pytest.raises(TimeoutError, match="total deadline"):
        source_upload.upload_source(slow, 2, "owned", archive, spec, None)
    assert len(calls) == 2
    assert calls[0][1]["timeout"] == calls[1][1]["timeout"] == 180


def test_timeout_or_failure_never_retries_upload(transfer):
    archive, spec, calls, call = transfer

    def failed(*args, **kwargs):
        call(*args, **kwargs)
        raise TimeoutError("uncertain upload")

    with pytest.raises(TimeoutError, match="uncertain"):
        source_upload.upload_source(failed, 2, "owned", archive, spec, None)
    assert len(calls) == 1
    assert not Path(spec["remote_archive"]).exists()


def test_local_change_stops_before_remote_driver(transfer):
    archive, spec, _, call = transfer
    spec["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Local source changed"):
        source_upload.upload_source(call, 2, "owned", archive, spec, None)
    assert not Path(spec["remote_archive"]).exists()


@pytest.mark.parametrize("fault", ["outside", "reorder", "bytes", "symlink", "extra"])
def test_invalid_descriptor_cannot_create_archive(transfer, tmp_path, fault):
    archive, spec, _, call = transfer
    parts = source_upload.upload_source(call, 2, "owned", archive, spec, None)
    if fault == "outside":
        parts[0]["path"] = str(tmp_path / "unrelated")
    elif fault == "reorder":
        parts.reverse()
    elif fault == "bytes":
        parts[0]["bytes"] = True
    elif fault == "extra":
        parts[0]["other"] = "unexpected"
    else:
        path = Path(parts[0]["path"])
        target = tmp_path / "unrelated"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    with pytest.raises(ValueError, match="Invalid source chunk"):
        remote_runner.assemble_source(spec, parts, archive.parent)
    assert not Path(spec["remote_archive"]).exists()


@pytest.mark.parametrize("fault", ["chunk", "whole"])
def test_corruption_retains_chunks_and_blocks_extraction(transfer, fault):
    archive, spec, _, call = transfer
    parts = source_upload.upload_source(call, 2, "owned", archive, spec, None)
    if fault == "chunk":
        path = Path(parts[0]["path"])
        path.write_bytes(b"X" * parts[0]["bytes"])
    else:
        spec["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        remote_runner.assemble_source(spec, parts, archive.parent)
    assert all(Path(part["path"]).exists() for part in parts)


def test_existing_archive_never_overwritten(transfer):
    archive, spec, _, call = transfer
    parts = source_upload.upload_source(call, 2, "owned", archive, spec, None)
    target = Path(spec["remote_archive"])
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        remote_runner.assemble_source(spec, parts, archive.parent)
    assert target.read_bytes() == b"keep"
    assert all(Path(part["path"]).exists() for part in parts)


def test_reconstructed_archive_runs_real_driver_with_original_identity(
    transfer, tmp_path, monkeypatch
):
    archive, spec, _, call = transfer
    code = (
        b"import os; from pathlib import Path; "
        b"Path(os.environ['CST_JOB_OUTPUT'], 'answer').write_text('ok')"
    )
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in [
            ("__pool_driver__.py", code),
            ("padding", bytes(range(251)) * 8),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    spec.update(
        id="l4job-" + "a" * 32,
        source_sha256=sha(archive.read_bytes()),
        remote_result=str(tmp_path / "result.tar.gz"),
        timeout=10,
        args=[],
    )
    parts = source_upload.upload_source(call, 2, "owned", archive, spec, None)
    assert len(parts) > 1
    monkeypatch.setattr(
        remote_runner.subprocess,
        "check_output",
        lambda command, **kwargs: (
            "NVIDIA L4, UUID, driver, 23034 MiB"
            if command[0] == "nvidia-smi"
            else '{"torch":"test"}'
        ),
    )
    remote_runner.main(spec, tmp_path, source_parts=parts)
    results = tmp_path / spec["id"] / "results"
    assert (results / "artifacts/answer").read_text() == "ok"
    result = json.loads((results / "result.json").read_text())
    assert result["source_sha256"] == spec["source_sha256"]
    assert result["returncode"] == 0


@pytest.mark.parametrize("fault", ["outside", "other-job", "bad-id"])
def test_owned_target_guard_precedes_upload_and_assembly(transfer, fault):
    archive, spec, calls, call = transfer
    if fault == "outside":
        spec["remote_archive"] = str(archive.parent / "unrelated.tar.gz")
    elif fault == "other-job":
        spec["id"] = "l4job-" + "b" * 32
    else:
        spec["id"] = "../escape"
    with pytest.raises(ValueError, match="exact job-owned"):
        source_upload.upload_source(call, 2, "owned", archive, spec, None)
    assert not calls
    with pytest.raises(ValueError, match="exact job-owned"):
        remote_runner.assemble_source(spec, [], archive.parent)
