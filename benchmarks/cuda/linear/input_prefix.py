"""Select a verified frozen input prefix, then call the existing Linear worker.

This orchestration contains no training/timing loop. It keeps all initialized
Parameters and the nested input/target rows fixed in a batch-size comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__, add_help=False)
    parser.add_argument("--input-payload", type=Path, required=True)
    parser.add_argument("--input-payload-sha256", required=True)
    args, rest = parser.parse_known_args(argv)
    began = time.monotonic()
    digest = hashlib.sha256(args.input_payload.read_bytes()).hexdigest()
    if digest != args.input_payload_sha256:
        raise ValueError("frozen input payload changed")
    frozen = torch.load(args.input_payload, map_location="cpu", weights_only=True)
    from benchmarks.cuda.linear import radius_cell_mass as radius
    from benchmarks.cuda.linear import run

    original_fixture, original_metadata = radius.fixture, run._metadata
    fixed = frozen["snapshot"]["case"]
    driver_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def fixture(case):
        declaration = asdict(case)
        if (
            case.fixture != "polar_radius_cell_mass_multirow"
            or not 1 <= case.rows <= fixed["rows"]
        ):
            raise ValueError("input prefix requires an admitted multirow fixture")
        if any(
            declaration[k] != v for k, v in fixed.items() if k not in ("id", "rows")
        ):
            raise ValueError("input prefix may change only Case id and row count")
        model, *_ = original_fixture(replace(case, rows=fixed["rows"]))
        if radius.tensor_hash(model.atoms.p) != radius.tensor_hash(frozen["p"]):
            raise ValueError("initial Parameters differ from the frozen cohort")
        if model.initialization_report != frozen["initialization_report"]:
            raise ValueError("initialization calibration differs")
        return model, *(
            frozen[k][: case.rows].contiguous() for k in ("x", "target", "dy")
        )

    def metadata(args_, cohort):
        value = original_metadata(args_, cohort)
        value["input_prefix_policy"] = {
            "driver_sha256": driver_hash,
            "input_payload_sha256": digest,
            "frozen_rows": fixed["rows"],
            "selected_rows": cohort.case.rows,
            "load_setup_seconds_before_worker": time.monotonic() - began,
            "semantics": "same initialized P, matching leading rows of frozen x/target/dY",
        }
        return value

    radius.fixture, run._metadata = fixture, metadata
    try:
        run.main(rest)
    finally:
        radius.fixture, run._metadata = original_fixture, original_metadata


if __name__ == "__main__":
    main()
