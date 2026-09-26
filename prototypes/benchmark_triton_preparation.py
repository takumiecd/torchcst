"""Compare Torch and fused preparation through the same production GEMM."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

from prototypes import benchmark_triton_split as paired
from torchcst.nn._backends._preparation import prepare


def execution_mode(name):
    return patch(
        "torchcst.nn._backends._triton.prepare",
        partial(prepare, use_triton=name == "optimized"),
    )


def main():
    import torch
    import triton

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = Path("manifest.json")
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "repeats": args.repeats,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest.exists()
        else None,
        "cases": [],
    }
    with patch.object(paired, "execution_mode", execution_mode):
        for case in (
            (16, 64, 64, 128),
            (128, 64, 64, 128),
            (16, 256, 64, 128),
            (128, 256, 64, 128),
            (32, 512, 256, 512),
            (128, 512, 256, 512),
        ):
            measured = paired.probe(*case, args.repeats)
            result["cases"].append(measured)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(measured), flush=True)


if __name__ == "__main__":
    main()
