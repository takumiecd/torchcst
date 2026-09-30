"""Tiny real-GPU smoke test for the pool; this is not a CST benchmark."""

import argparse
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=["success", "failure", "timeout"], default="success"
)
args = parser.parse_args()
output = Path(os.environ["CST_JOB_OUTPUT"])
if args.mode == "timeout":
    (output / "started.txt").write_text("timeout driver started")
    time.sleep(30)
elif args.mode == "failure":
    raise RuntimeError("Intentional pool smoke-test failure")
else:
    import torch

    assert torch.cuda.get_device_name(0) == "NVIDIA L4"
    torch.manual_seed(21)
    a = torch.randn(128, 128, device="cuda", requires_grad=True)
    b = torch.randn(128, 128, device="cuda", requires_grad=True)
    y = a @ b
    torch.testing.assert_close(
        y.cpu(), a.detach().cpu() @ b.detach().cpu(), atol=1e-4, rtol=1e-4
    )
    y.square().mean().backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    torch.cuda.synchronize()
    (output / "smoke.json").write_text(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "forward_and_backward": "passed",
                "job_id": os.environ["CST_JOB_ID"],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
            indent=2,
        )
        + "\n"
    )
    print("GPU forward/backward smoke passed")
