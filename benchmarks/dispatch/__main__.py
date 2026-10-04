"""Generate from PostgreSQL or replay a frozen dataset, never inside forward."""

import argparse
import sys
from pathlib import Path

from benchmarks.cuda.linear.manifest import REGISTRY
from torchcst._backends.cuda.serialization import decode_json, encode_json

from .generate import generate
from .request import validate_request


def main():
    parser = argparse.ArgumentParser(
        description="Rank benchmark observations and export an exact dispatcher"
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory for dispatch.json, leaderboard.json and dataset.json",
    )
    parser.add_argument(
        "--dataset", type=Path, help="replay a frozen dataset without DB access"
    )
    parser.add_argument("--database-env", default="DATABASE_URL")
    args = parser.parse_args()
    try:
        request = validate_request(decode_json(args.request.read_bytes()))
        if args.output.exists():
            raise ValueError("output directory already exists")
        if args.dataset:
            dataset = decode_json(args.dataset.read_bytes())
        else:
            from benchmarks.database.postgres import Database, connect_from_env

            from .database import read_dataset

            with connect_from_env(args.database_env) as connection:
                dataset = read_dataset(Database(connection), request)
        result = generate(dataset, request, registry=REGISTRY)
        args.output.mkdir(parents=True, exist_ok=False)
        for name, value in (
            ("dispatch.json", result.artifact),
            ("leaderboard.json", result.leaderboard),
            ("dataset.json", dataset),
        ):
            (args.output / name).write_text(encode_json(value), encoding="utf-8")
        print(
            encode_json(
                {
                    "output": str(args.output),
                    "entries": len(result.artifact["entries"]),
                    "dataset_snapshot": dataset["id"],
                }
            ),
            end="",
        )
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"generation failed: {error}", file=sys.stderr)
    except Exception as error:  # noqa: BLE001 -- keep private database diagnostics out of output
        # Database diagnostics can contain private connection information.
        print(
            f"generation failed ({type(error).__name__}); inspect privately",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
