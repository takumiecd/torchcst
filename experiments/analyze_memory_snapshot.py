"""Reconstruct the active-allocation peak from bounded allocator history."""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    # Only read locally generated diagnostic snapshots, never untrusted pickle.
    with (args.directory / "audit_start.pickle").open("rb") as f:
        start = pickle.load(f)
    with (args.directory / "audit_end.pickle").open("rb") as f:
        end = pickle.load(f)
    pools = {tuple(g["pool"]): g["name"] for g in summary["graphs"]}
    static = {}
    for g in summary["graphs"]:
        for t in g["static"]:
            static[t["address"]] = g["name"] + ":static"
    live = {}
    ranges = []
    for snapshot in (start, end):
        for s in snapshot["segments"]:
            ranges.append(
                (
                    s["address"],
                    s["address"] + s["total_size"],
                    tuple(s["segment_pool_id"]),
                )
            )
    for s in start["segments"]:
        for b in s["blocks"]:
            if b["state"] == "active_allocated":
                live[b["address"]] = dict(
                    b, pool=tuple(s["segment_pool_id"]), stream=s["stream"]
                )
    size = sum(b["size"] for b in live.values())
    peak, peak_live, peak_event = size, live.copy(), None
    initial_size = size
    a, b = start["device_traces"][0], end["device_traces"][0]

    def identity(event):
        return tuple(event.get(k) for k in ("action", "addr", "size", "stream"))

    if len(b) < len(a) or [identity(e) for e in b[: len(a)]] != [
        identity(e) for e in a
    ]:
        raise RuntimeError("trace prefix changed/truncated; increase max_entries")
    for event in b[len(a) :]:
        addr = event.get("addr")
        if event["action"] == "alloc":
            pool = next((p for lo, hi, p in ranges if lo <= addr < hi), (0, 0))
            row = {
                "address": addr,
                "size": ((event["size"] + 511) // 512) * 512,
                "frames": event.get("frames", []),
                "pool": pool,
                "stream": event["stream"],
            }
            if addr in live:
                raise RuntimeError("allocation address already active")
            live[addr] = row
            size += row["size"]
        elif event["action"] == "free_requested":
            if addr not in live:
                raise RuntimeError("free of unknown active allocation")
            size -= live.pop(addr)["size"]
        if size > peak:
            peak, peak_live, peak_event = size, live.copy(), event
    groups = defaultdict(int)
    rows = []
    for addr, row in peak_live.items():
        frames = row.get("frames", [])
        names = " ".join(f.get("name", "") for f in frames).lower()
        if "cublas" in names and (
            "workspace" in names or "getcurrentcudablas" in names
        ):
            group = "cuBLAS workspace"
        elif addr in static:
            group = static[addr]
        elif row["pool"] in pools:
            group = pools[row["pool"]] + ":private_pool"
        else:
            group = "other"
        groups[group] += row["size"]
        rows.append(
            {
                "address": addr,
                "bytes": row["size"],
                "group": group,
                "pool": row["pool"],
                "stream": row.get("stream"),
                "frames": frames,
            }
        )
    rows.sort(key=lambda r: r["bytes"], reverse=True)
    result = {
        "initial_allocated": initial_size,
        "reconstructed_peak": peak,
        "reported_peak": summary["audit_peak"],
        "final_allocated": size,
        "groups": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
        "allocations": rows,
        "peak_event": peak_event,
    }
    (args.directory / "attribution.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("allocations", "peak_event")},
            indent=2,
        )
    )
    print("largest allocations:")
    for row in rows[:15]:
        print(row["bytes"], row["group"], [f.get("name") for f in row["frames"][:6]])
    if peak != summary["audit_peak"]:
        raise RuntimeError("reconstructed peak does not match allocator counter")


if __name__ == "__main__":
    main()
