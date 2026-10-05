"""GitHub issue attachment intake; CPU-only central code, no contributor source."""

import argparse
import hashlib
import os
import re
import sys
import urllib.error
from pathlib import Path

from benchmarks.submissions.github import GitHub, Submission, submission_from_issue
from benchmarks.submissions.policy import DEFAULT_POLICY, load_policy
from torchcst._backends.serialization import decode_json, encode_json


def validate_result(raw, policy):
    from benchmarks.database.adapters.linear import project
    from benchmarks.database.model import execution_identity

    if not 0 < len(raw) <= policy.max_file_bytes:
        raise ValueError("empty or oversized result JSON")
    value = decode_json(raw)
    project(value)
    if execution_identity(value)[0] is None:
        raise ValueError("new submissions require an execution UUID and UTC start time")


def issue_number(args, policy):
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        if not re.fullmatch(r"[1-9][0-9]*", args.issue_number):
            raise ValueError("invalid issue number")
        return int(args.issue_number)
    event = decode_json(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
    if (
        event.get("action") not in ("opened", "edited")
        or event["repository"]["full_name"] != policy.repository
    ):
        raise ValueError("unexpected issue event")
    if "pull_request" in event["issue"]:
        raise ValueError("not a submission issue")
    # An editor/re-runner is not the submitter. Only the original user may edit
    # an automatic intake request; maintainer retries use workflow_dispatch.
    if event["sender"]["id"] != event["issue"]["user"]["id"]:
        raise ValueError(
            "automatic submissions must be posted or edited by their author"
        )
    if args.issue_number and str(event["issue"]["number"]) != args.issue_number:
        raise ValueError("issue number differs from GitHub event")
    return event["issue"]["number"]


def fetch(github, number, policy, output):
    issue = github.issue(policy.repository, number)
    submission = submission_from_issue(issue, policy)
    raw = github.attachment(submission.artifact_url, policy.max_file_bytes)
    validate_result(raw, policy)
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_bytes(raw)
    manifest = {
        "schema_version": 1,
        "policy_id": policy.id,
        "submission": submission.declaration(),
        "result_sha256": hashlib.sha256(raw).hexdigest(),
    }
    (output / "submission.json").write_text(encode_json(manifest))
    return manifest


def load_fetched(directory, policy):
    manifest = decode_json((directory / "submission.json").read_bytes())
    if type(manifest) is not dict or set(manifest) != {
        "schema_version",
        "policy_id",
        "submission",
        "result_sha256",
    }:
        raise ValueError("invalid fetched submission manifest")
    if manifest["schema_version"] != 1 or manifest["policy_id"] != policy.id:
        raise ValueError("fetched submission policy differs")
    raw = (directory / "result.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["result_sha256"]:
        raise ValueError("result changed after download")
    submission = Submission(**manifest["submission"]).validate(policy)
    validate_result(raw, policy)
    return raw, submission


def write_feedback(github, number, policy, *, receipt=None, failure=None, run_url=None):
    # Fetch authenticated metadata again; never post to an arbitrary URL from JSON.
    issue = github.issue(policy.repository, number)
    if (
        not issue.get("title", "").startswith("[Benchmark result] ")
        or "pull_request" in issue
    ):
        return
    if receipt:
        status = (
            "保存しました"
            if receipt["submission_inserted"]
            else "同じ観測はすでに保存されています（再登録・追加の件数消費なし）"
        )
        body = (
            f"{status}。\n\n観測ID: `{receipt['run_id']}`\n\n"
            f"記録された提出者: @{receipt['submitter_login']}\n\n"
            f"提出区分: `{receipt['tier']}` / 信頼ポイント: {receipt['trust_points']}\n\n"
            "検証はJSONの形式・内部整合性の確認です。GPU実行の真正性やkernelの採用を証明するものではありません。"
        )
    else:
        body = (
            "受付処理が完了しませんでした。同じ観測を再実行しても重複登録されません。"
            + (
                f"\n\n{failure}"
                if failure
                else "\n\n受付Actionsのログで、添付形式・計測結果・提出上限を確認してください。"
            )
        )
    if run_url:
        body += f"\n\n[受付Actions]({run_url})"
    github.comment(policy.repository, number, body)
    if receipt:
        github.close(policy.repository, number)


def main():
    parser = argparse.ArgumentParser(description="GitHub issue benchmark submissions")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check", help="validate a result locally without GitHub or database access"
    )
    check.add_argument("result", type=Path)
    get = commands.add_parser(
        "fetch", help="download and validate issue JSON before accessing DB credentials"
    )
    get.add_argument("--issue-number", default="")
    get.add_argument("--output", type=Path, required=True)
    ingest = commands.add_parser(
        "ingest", help="append validated data and attribution with atomic daily quota"
    )
    ingest.add_argument("--input", type=Path, required=True)
    ingest.add_argument("--receipt", type=Path, required=True)
    feedback = commands.add_parser(
        "feedback", help="report admission outcome to the issue"
    )
    feedback.add_argument("--issue-number", required=True)
    feedback.add_argument("--receipt", type=Path, required=True)
    feedback.add_argument("--run-url", required=True)
    args = parser.parse_args()
    try:
        policy = load_policy(args.policy)
        if (
            args.command != "check"
            and os.environ.get("GITHUB_REPOSITORY") != policy.repository
        ):
            raise ValueError("intake must run in the policy repository")
        if args.command == "check":
            validate_result(args.result.read_bytes(), policy)
            result = {
                "validated": True,
                "validation": "consistency_checked",
                "measurement_authenticity": "self_reported",
            }
        elif args.command == "fetch":
            result = fetch(
                GitHub(os.environ.get("GH_TOKEN")),
                issue_number(args, policy),
                policy,
                args.output,
            )
            result = {
                "issue_number": result["submission"]["issue_number"],
                "validated": True,
            }
        elif args.command == "ingest":
            raw, submission = load_fetched(args.input, policy)
            import psycopg

            from benchmarks.database.postgres import (
                Database,
                SubmissionQuotaExceeded,
                connect_from_env,
            )

            try:
                with connect_from_env() as connection:
                    result = Database(connection).import_submission(
                        raw, submission, policy
                    )
            except SubmissionQuotaExceeded:
                args.receipt.parent.mkdir(parents=True, exist_ok=True)
                args.receipt.write_text(
                    encode_json(
                        {
                            "failure": "一般提出の1日あたりの上限に達しました。UTCの日付が変わってから再提出できます。"
                        }
                    )
                )
                raise
            except psycopg.Error as error:
                print(
                    f"database operation failed ({type(error).__name__}); inspect privately",
                    file=sys.stderr,
                )
                return 1
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(encode_json({"receipt": result}))
        else:
            number = int(args.issue_number)
            from benchmarks.submissions.policy import positive_id

            positive_id(number)
            outcome = (
                decode_json(args.receipt.read_bytes()) if args.receipt.exists() else {}
            )
            if not re.fullmatch(
                r"https://github\.com/"
                + re.escape(policy.repository)
                + r"/actions/runs/[1-9][0-9]*",
                args.run_url,
            ):
                raise ValueError("invalid intake Actions URL")
            write_feedback(
                GitHub(os.environ.get("GH_TOKEN")),
                number,
                policy,
                receipt=outcome.get("receipt"),
                failure=outcome.get("failure"),
                run_url=args.run_url,
            )
            result = {"issue_number": number, "feedback": True}
        print(encode_json(result), end="")
        return 0
    except (urllib.error.URLError, urllib.error.HTTPError):
        print(
            "GitHub download/API failed; inspect the workflow privately",
            file=sys.stderr,
        )
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as error:
        # Never echo arbitrary submission values or exception text from parsing.
        print(
            f"submission rejected ({type(error).__name__}); check JSON, issue form and submission rules",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
