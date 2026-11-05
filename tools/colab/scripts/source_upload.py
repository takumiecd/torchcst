"""Bound source uploads without placing a large archive in one CLI request."""

import hashlib
import re
import tempfile
import time
from pathlib import Path

CHUNK_BYTES = 32 * 1024 * 1024
UPLOAD_SECONDS = 900
CONTENT_ROOT = "/content"


def upload_source(call, slot, session, archive, spec, log):
    """Return operational chunk descriptors; never alter the frozen job spec."""
    archive = Path(archive)
    job = spec.get("id")
    if (
        not isinstance(job, str)
        or not re.fullmatch(r"(?:l4job|colabjob)-[0-9a-f]{32}", job)
        or spec.get("remote_archive") != str(Path(CONTENT_ROOT) / f"{job}.tar.gz")
    ):
        raise ValueError("Source upload target is not the exact job-owned archive")
    started = time.monotonic()

    def upload(local, remote, ceiling):
        remaining = UPLOAD_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Source upload exceeded its total deadline")
        call(
            slot,
            "upload",
            "-s",
            session,
            local,
            remote,
            timeout=min(ceiling, remaining),
            log=log,
        )
        if time.monotonic() - started > UPLOAD_SECONDS:
            raise TimeoutError("Source upload exceeded its total deadline")

    if archive.stat().st_size <= CHUNK_BYTES:
        upload(archive, spec["remote_archive"], 180)
        return []

    digest = hashlib.sha256()
    parts = []
    # One temporary chunk at a time, rather than another archive-sized copy.
    with tempfile.TemporaryDirectory(prefix="cst-source-upload-") as temporary:
        local = Path(temporary) / "part"
        with archive.open("rb") as handle:
            while data := handle.read(CHUNK_BYTES):
                digest.update(data)
                local.write_bytes(data)
                part = {
                    "path": f"{spec['remote_archive']}.part-{len(parts):05d}",
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                upload(local, part["path"], 180)
                parts.append(part)
    if digest.hexdigest() != spec["source_sha256"]:
        raise ValueError("Local source changed before upload completed")
    return parts
