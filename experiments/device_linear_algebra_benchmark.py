"""GPU Jacobi oracle, replay-lifetime and synchronization measurements."""

import json
import time
from pathlib import Path

import torch

from torchcst._derivatives._jacobi import device_pinv_solve


def main():
    torch.set_num_threads(1)
    out = Path("output/device_linalg")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dtype in (torch.float32, torch.float64):
        torch.manual_seed(19)
        x = torch.randn(256, 100, device="cuda", dtype=dtype)
        a = x @ x.T
        rhs = torch.randn(256, device="cuda", dtype=dtype)
        expected = torch.linalg.pinv(a) @ rhs
        start = time.perf_counter()
        actual, valid = device_pinv_solve(a, rhs)
        torch.cuda.synchronize()
        cold = time.perf_counter() - start
        saved = actual.clone()
        # A changed right-hand side/matrix must not reuse captured coefficients.
        changed, valid2 = device_pinv_solve(2 * a, 3 * rhs)
        torch.testing.assert_close(
            changed,
            1.5 * expected,
            atol=2e-5 if dtype == torch.float32 else 1e-11,
            rtol=2e-3 if dtype == torch.float32 else 1e-9,
        )
        torch.testing.assert_close(actual, saved)
        assert bool(valid & valid2)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(3):
            device_pinv_solve(a, rhs)
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / 3
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            device_pinv_solve(a, rhs)
        events = {e.key: e.count for e in prof.key_averages()}
        forbidden = {
            k: v
            for k, v in events.items()
            if k
            in (
                "aten::_local_scalar_dense",
                "cudaStreamSynchronize",
                "cudaEventSynchronize",
                "cudaDeviceSynchronize",
            )
        }
        # The profiler itself may synchronize at exit; inspect the table and
        # a separately guarded call as well instead of attributing it to solve.
        torch.cuda.set_sync_debug_mode("error")
        try:
            device_pinv_solve(a, rhs)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        row = {
            "dtype": str(dtype),
            "cold": cold,
            "seconds": seconds,
            "relative_error": float((actual - expected).norm() / expected.norm()),
            "events": forbidden,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        (out / (str(dtype) + ".txt")).write_text(
            prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=50)
        )
        (out / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
