"""Mid-upload uncertainty follows the real scheduler path; no CLI or GPU calls."""

import json
import subprocess
from pathlib import Path

import pytest
import test_colab_l4_pool as baseline

from tools.colab.scripts import source_upload

source = baseline.source


@pytest.mark.parametrize("fresh_transport", [False, True])
@pytest.mark.parametrize("stop_fails", [False, True])
def test_mid_upload_timeout_never_dispatches_driver_or_following_job(
    source, tmp_path, monkeypatch, fresh_transport, stop_fails
):
    class UploadTimeoutPool(baseline.FakePool):
        execute = baseline.pool_module.Pool.execute

        def __init__(self, root):
            super().__init__(root, fail="stop" if stop_fails else None)
            self.uploads = []
            self.other_transport = []

        def call(self, slot, *args, timeout=180, log=None):
            if args[0] == "upload":
                self.uploads.append((slot, args, timeout))
                assert timeout <= 180
                if len(self.uploads) == 2:
                    raise subprocess.TimeoutExpired("owned source upload", timeout)
                return "uploaded one chunk"
            if args[0] in ("exec", "download"):
                self.other_transport.append(args[0])
                pytest.fail(
                    "Uncertain source upload must not reach driver or retrieval"
                )
            return super().call(slot, *args, timeout=timeout, log=log)

    # Split the real frozen source archive, without allocating 64MiB or sleeping.
    monkeypatch.setattr(source_upload, "CHUNK_BYTES", 16)
    pool = UploadTimeoutPool(tmp_path / "state")
    job = baseline.submit(pool, source)
    following = baseline.submit(pool, source)
    directory = pool.jobdir(job)
    spec_before = (directory / "spec.json").read_bytes()
    spec = json.loads(spec_before)
    archive_before = baseline.pool_module.file_hash(directory / "source.tar.gz")
    assert archive_before == spec["source_sha256"]
    assert (directory / "source.tar.gz").stat().st_size > 32

    with pytest.raises(RuntimeError):
        pool.serve(1, 0, fresh_proxy_transport=fresh_transport)

    assert len(pool.uploads) == 2
    assert pool.get(job)["status"] == "interrupted"
    assert pool.get(following)["status"] == "queued"
    assert not pool.other_transport
    assert not (directory / "remote.py").exists()
    assert not (directory / "results.tar.gz").exists()
    assert not (directory / "receipt.json").exists()
    assert (directory / "spec.json").read_bytes() == spec_before
    assert baseline.pool_module.file_hash(directory / "source.tar.gz") == archive_before
    # Temporary chunks can be regenerated from the retained original archive.
    assert all(not Path(args[3]).exists() for _, args, _ in pool.uploads)
    assert pool.slotstate(1)["status"] == ("quarantined" if stop_fails else "stopped")
    assert bool(pool.active) is stop_fails
