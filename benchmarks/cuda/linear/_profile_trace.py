"""Summarize real CUDA kernels, excluding overlapping GPU annotation ranges."""

import json
from collections import defaultdict


def kernel_summary(trace_path, steps):
    events = json.loads(trace_path.read_text())["traceEvents"]
    kernels = defaultdict(lambda: {"calls": 0, "total_us": 0.0})
    for event in events:
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            kernels[event["name"]]["calls"] += 1
            kernels[event["name"]]["total_us"] += event["dur"]
    entries = [{"name": key, **value} for key, value in kernels.items()]
    entries.sort(key=lambda entry: entry["total_us"], reverse=True)
    total = sum(entry["total_us"] for entry in entries)
    assert entries and total > 0 and steps > 0
    for entry in entries:
        entry["share_of_kernel_time"] = entry["total_us"] / total
        entry["us_per_step"] = entry["total_us"] / steps
    return {"kernel_us_per_step": total / steps, "kernels": entries}
