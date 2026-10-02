"""Noninteractive Colab authentication on a fresh GitHub-hosted runner."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.automation.contract import encode


def install_credentials(raw, home):
    """Keep OAuth outside the checkout and out of source/result snapshots."""
    value = json.loads(raw)
    required = ("refresh_token", "token_uri", "client_id", "client_secret")
    if not isinstance(value, dict) or any(
        not isinstance(value.get(key), str) or not value[key] for key in required
    ):
        raise ValueError("Colab Secret must contain the CLI authorized-user JSON")
    if value["token_uri"] != "https://oauth2.googleapis.com/token":
        raise ValueError("unexpected OAuth token endpoint")
    directory = Path(home) / ".config/colab-cli"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / "token.json"
    with path.open("x", encoding="utf-8") as file:
        os.chmod(path, 0o600)
        json.dump(value, file)
    return path


def bootstrap():
    account = os.environ.get("COLAB_ACCOUNT", "").strip()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+", account):
        raise ValueError("configure the COLAB_ACCOUNT environment variable")
    raw = os.environ.pop("COLAB_AUTH_JSON", "")
    if not raw:
        raise ValueError("configure the COLAB_AUTH_JSON environment Secret")
    path = install_credentials(raw, Path.home())
    try:
        # Refresh directly: the CLI's fallback is an interactive login prompt.
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        credentials = Credentials.from_authorized_user_file(path)
        credentials.refresh(Request())
        path.write_text(credentials.to_json())
        # Mask individual fields too; GitHub already masks the whole Secret.
        for value in (
            credentials.token,
            credentials.refresh_token,
            credentials.client_secret,
        ):
            if value:
                escaped = (
                    value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
                )
                print(f"::add-mask::{escaped}", flush=True)
        completed = subprocess.run(
            ["colab", "--auth", "oauth2", "whoami"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if completed.returncode or not re.search(
            r"Email:\s*" + re.escape(account) + r"\s", completed.stdout
        ):
            raise RuntimeError("Colab identity verification failed")
        subprocess.run(
            [
                sys.executable,
                "tools/colab/scripts/pool.py",
                "configure",
                "--account",
                account,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=30,
        )
    except Exception:  # noqa: BLE001 - provider errors may contain credentials
        path.unlink(missing_ok=True)
        # Provider exceptions can contain credentials; never echo them.
        raise RuntimeError(
            "Colab authentication failed; renew the CLI login and Secret"
        ) from None
    print("Colab account verified; no GPU allocated")


def preserve_receipts(home, output):
    """Export measurement evidence, never live CLI session credentials/cache."""
    root = Path(home) / ".local/state/colab-l4-pool"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    private_values = set()

    def remember(value):
        if isinstance(value, dict):
            for item in value.values():
                remember(item)
        elif isinstance(value, list):
            for item in value:
                remember(item)
        elif isinstance(value, str) and len(value) >= 8:
            private_values.add(value)

    auth = Path(home) / ".config/colab-cli/token.json"
    for path in [auth, *(root / "slots").glob("*/sessions.json")]:
        if path.is_file():
            remember(json.loads(path.read_bytes()))
    if os.environ.get("COLAB_AUTH_JSON"):
        remember(json.loads(os.environ["COLAB_AUTH_JSON"]))

    def sanitized_log(path, destination):
        if path.is_file() and not path.is_symlink():
            content = path.read_text(errors="replace")
            for value in sorted(private_values, key=len, reverse=True):
                content = content.replace(value, "[redacted]")
            content = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[redacted]", content)
            content = re.sub(
                r"(?i)([?&](?:token|key|auth)=)[^\s&#\"']+", r"\1[redacted]", content
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)

    states = []
    for path in sorted((root / "slots").glob("*/state.json")):
        state = json.loads(path.read_bytes())
        states.append(
            {key: state.get(key) for key in ("status", "session", "endpoint")}
        )
        sanitized_log(
            path.parent / "lifecycle.log",
            output / "slots" / path.parent.name / "lifecycle.log",
        )
    (output / "owned-sessions.json").write_bytes(encode(states))
    # OAuth and live session configs and queue DB are deliberately absent.
    for job in (root / "jobs").glob("*"):
        if not job.is_dir() or job.is_symlink():
            continue
        for name in ("spec.json", "receipt.json", "source.tar.gz", "results.tar.gz"):
            path = job / name
            if path.is_file() and not path.is_symlink():
                destination = output / job.name / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        sanitized_log(job / "transport.log", output / job.name / "transport.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bootstrap", "erase", "receipts"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "bootstrap":
            bootstrap()
        elif args.command == "receipts":
            if args.output is None:
                raise ValueError("receipts needs --output")
            preserve_receipts(Path.home(), args.output)
        else:
            (Path.home() / ".config/colab-cli/token.json").unlink(missing_ok=True)
        return 0
    except Exception:  # noqa: BLE001 - provider errors may contain credentials
        print(
            "Colab authentication setup failed; check the account and Secret privately",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
