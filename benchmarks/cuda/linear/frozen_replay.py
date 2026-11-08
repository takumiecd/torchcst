"""Replay frozen Parameters across hosts, preserving the existing worker loops.

Record fresh host calibration separately. The actual measured P and data are
restored from the verified original payload, not a new CPU initialization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import torch


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__, add_help=False)
    parser.add_argument("--input-payload", type=Path, required=True)
    parser.add_argument("--input-payload-sha256", required=True)
    args, _ = parser.parse_known_args(argv)
    digest = hashlib.sha256(args.input_payload.read_bytes()).hexdigest()
    if digest != args.input_payload_sha256:
        raise ValueError("frozen input payload changed")
    frozen = torch.load(args.input_payload, map_location="cpu", weights_only=True)
    from benchmarks.cuda.linear import input_prefix, run
    from benchmarks.cuda.linear import radius_cell_mass as radius

    original_fixture, original_metadata = radius.fixture, run._metadata
    observations = []

    def fixture(case):
        model, *data = original_fixture(case)
        p = model.atoms.p
        if (
            p.device.type != "cpu"
            or p.shape != frozen["p"].shape
            or p.dtype != frozen["p"].dtype
        ):
            raise ValueError("frozen Parameter metadata differs from the fixture")
        observations.append(
            {
                "regenerated_p_sha256": radius.tensor_hash(p),
                "regenerated_initialization_report": copy.deepcopy(
                    model.initialization_report
                ),
            }
        )
        with torch.no_grad():
            p.copy_(frozen["p"])
        model.initialization_report = copy.deepcopy(frozen["initialization_report"])
        return model, *data

    def metadata(args_, cohort):
        value = original_metadata(args_, cohort)
        value["frozen_replay_policy"] = {
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_payload_sha256": digest,
            "restored_parameters_sha256": radius.tensor_hash(frozen["p"]),
            "calibration_report_source": "original frozen payload, not current host",
            "current_host_regeneration": observations,
        }
        return value

    radius.fixture, run._metadata = fixture, metadata
    try:
        input_prefix.main(argv)
    finally:
        radius.fixture, run._metadata = original_fixture, original_metadata


if __name__ == "__main__":
    main()
