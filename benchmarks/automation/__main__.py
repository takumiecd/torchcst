"""python -m benchmarks.automation --help"""

import argparse
import sys
from pathlib import Path

from benchmarks.automation.contract import encode, load_request, prepare


def main():
    parser = argparse.ArgumentParser(
        description="Official measurement request / Colab / ingestion tooling"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--source", type=Path, default=Path.cwd())
    create.add_argument(
        "--request", type=Path, required=True, help="committed measurement JSON file"
    )
    create.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("validate")
    check.add_argument("--request", type=Path, required=True)
    check.add_argument("--results", type=Path)
    gpu = sub.add_parser("colab")
    gpu.add_argument("--request", type=Path, required=True)
    gpu.add_argument("--source", type=Path, default=Path.cwd())
    gpu.add_argument("--output", type=Path, required=True)
    gpu.add_argument(
        "--pool",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tools/colab/scripts/pool.py",
    )
    direct = sub.add_parser(
        "run", help="freeze a committed measurement JSON and execute it on Colab"
    )
    direct.add_argument("--request", type=Path, required=True)
    direct.add_argument("--source", type=Path, default=Path.cwd())
    direct.add_argument("--output", type=Path, required=True)
    direct.add_argument("--pool", type=Path, default=gpu.get_default("pool"))
    collect = sub.add_parser(
        "collect", help="retrieve a submitted batch after an interrupted local wait"
    )
    collect.add_argument("--request", type=Path, required=True)
    collect.add_argument("--pending", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--pool", type=Path, default=gpu.get_default("pool"))
    store = sub.add_parser("ingest")
    store.add_argument("--request", type=Path, required=True)
    store.add_argument("--results", type=Path, required=True)
    store.add_argument(
        "--provenance",
        type=Path,
        required=True,
        help="metadata written by the trusted caller",
    )
    store.add_argument("--database-env", default="DATABASE_URL")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.source, args.request)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as file:
                file.write(encode(result))
            result = {
                "request_id": result["request_id"],
                "source_commit": result["request"]["source_commit"],
                "configuration": result["request"]["configuration"],
                "jobs": [
                    {
                        "key": j["key"],
                        "gpu": j["accelerator"],
                        "plans": [p["id"] for p in j["run"]["plans"]],
                        "baseline": j["run"]["baseline"],
                    }
                    for j in result["request"]["jobs"]
                ],
            }
        elif args.command == "validate":
            result = load_request(args.request)
            if args.results:
                from benchmarks.automation.ingest import validate_bundle

                _, entries = validate_bundle(args.request, args.results)
                result = {
                    "request_id": result["request_id"],
                    "observations": sum(raw is not None for _, raw in entries),
                }
        elif args.command in ("colab", "collect", "run"):
            from benchmarks.automation.colab import collect, run

            if args.command == "run":
                request = prepare(args.source, args.request)
                args.output.mkdir(parents=True, exist_ok=False)
                frozen = args.output / "request.json"
                frozen.write_bytes(encode(request))
                print(
                    encode(
                        {
                            "request_id": request["request_id"],
                            "jobs": [j["key"] for j in request["request"]["jobs"]],
                        }
                    ).decode(),
                    flush=True,
                )
                result = run(
                    frozen, args.source, args.output / "measurement", args.pool
                )
            elif args.command == "colab":
                result = run(args.request, args.source, args.output, args.pool)
            else:
                result = collect(args.request, args.pending, args.output, args.pool)
            print(encode(result).decode(), end="")
            return 0 if all(e["status"] == "succeeded" for e in result["jobs"]) else 1
        else:
            import psycopg

            from benchmarks.automation.contract import decode
            from benchmarks.automation.ingest import ingest, validate_bundle
            from benchmarks.database.postgres import Database, connect_from_env

            # Reject mismatched submissions before opening a DB connection.
            validate_bundle(args.request, args.results)
            provenance = decode(args.provenance.read_bytes())
            try:
                with connect_from_env(args.database_env) as connection:
                    result = ingest(
                        args.request, args.results, Database(connection), provenance
                    )
            except psycopg.Error as error:
                print(
                    f"database operation failed ({type(error).__name__}); inspect privately",
                    file=sys.stderr,
                )
                return 1
        print(encode(result).decode(), end="")
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(f"measurement operation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
