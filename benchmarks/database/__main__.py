"""python -m benchmarks.database --help"""

import argparse
import sys
from pathlib import Path

import psycopg

from benchmarks.database.postgres import Database, connect_from_env
from torchcst._backends.cuda.serialization import encode_json


def main():
    parser = argparse.ArgumentParser(
        description="PostgreSQL benchmark evidence storage"
    )
    parser.add_argument(
        "--database-env",
        default="DATABASE_URL",
        help="connection environment variable NAME",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "migrate", help="initialize/apply schema migrations using owner role"
    )
    commands.add_parser("status", help="count stored evidence")
    ingest = commands.add_parser(
        "import-linear", help="validate/store a completed CUDA linear run"
    )
    ingest.add_argument("artifact", type=Path)
    export = commands.add_parser("export-run", help="restore the original JSON bytes")
    export.add_argument("run_id")
    export.add_argument("output", type=Path)
    query = commands.add_parser(
        "records", help="query observations; PASS is not certification"
    )
    for flag in ("plan-id", "case-id", "gpu", "kind"):
        query.add_argument("--" + flag)
    query.add_argument("--limit", type=int, default=100)
    query.add_argument("--adapter-revision", type=int)
    query.add_argument("--after", nargs=2, metavar=("PROJECTION_ID", "ORDINAL"))
    submitted = commands.add_parser(
        "submissions", help="list submission accounts, policy and trust snapshots"
    )
    submitted.add_argument("--submitter-id", type=int)
    submitted.add_argument("--after")
    submitted.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    try:
        with connect_from_env(args.database_env) as connection:
            database = Database(connection)
            if args.command == "migrate":
                result = {"schema_version": database.migrate()}
            elif args.command == "status":
                result = database.counts()
            elif args.command == "import-linear":
                result = database.import_linear(args.artifact.read_bytes())
            elif args.command == "export-run":
                raw = database.export_run(args.run_id)
                # Avoid silently overwriting an existing result or source file.
                with args.output.open("xb") as output:
                    output.write(raw)
                result = {"run_id": args.run_id, "output": str(args.output)}
            elif args.command == "submissions":
                rows = database.list_submissions(
                    submitter_id=args.submitter_id, after=args.after, limit=args.limit
                )
                result = {
                    "submissions": rows,
                    "next_after": rows[-1]["id"] if len(rows) == args.limit else None,
                }
            else:
                after = (args.after[0], int(args.after[1])) if args.after else None
                rows = database.list_records(
                    plan_id=args.plan_id,
                    case_id=args.case_id,
                    gpu=args.gpu,
                    kind=args.kind,
                    adapter_revision=args.adapter_revision,
                    after=after,
                    limit=args.limit,
                )
                result = {
                    "records": rows,
                    "next_after": [rows[-1]["projection_id"], rows[-1]["ordinal"]]
                    if len(rows) == args.limit
                    else None,
                }
        print(encode_json(result), end="")
        return 0
    except psycopg.Error as error:
        # Connection diagnostics can contain credentials/hosts; do not echo them.
        print(
            f"database operation failed ({type(error).__name__}); inspect the database privately",
            file=sys.stderr,
        )
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"invalid input: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
