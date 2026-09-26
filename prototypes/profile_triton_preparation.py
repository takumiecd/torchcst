"""Attribute warmed forward launches to CST preparation phases on CUDA."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.benchmark_triton_linear import model
from torchcst.profiling import CSTProfiler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layer = model(256)
    x = torch.randn(16, 128, device="cuda")
    with torch.no_grad():
        for _ in range(3):
            layer(x)
        torch.cuda.synchronize()
        with CSTProfiler() as profiler:
            layer(x)
            torch.cuda.synchronize()
    result = {"device": torch.cuda.get_device_name(), "phases": [], "gpu_events": []}
    for event in profiler.key_averages():
        if event.key.startswith("cst."):
            result["phases"].append(
                {
                    "name": event.key,
                    "cpu_us": event.cpu_time_total,
                    "device_us": event.device_time_total,
                }
            )
    for event in profiler.profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            result["gpu_events"].append(
                {"name": event.name, "duration_us": event.device_time_total}
            )
    profiler.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
