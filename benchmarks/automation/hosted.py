"""Noninteractive Colab authentication on a fresh GitHub-hosted runner."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bootstrap", "erase"])
    args = parser.parse_args()
    try:
        if args.command == "bootstrap":
            bootstrap()
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
