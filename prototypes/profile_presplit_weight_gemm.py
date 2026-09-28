"""Probe an FP16x3 GEMM whose local W is already split into two FP16 parts."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton as tr
import triton.language as tl

from prototypes.bounded_gemm_fp16x3 import bounded_gemm_fp16x3


@tr.jit
def _presplit_weight_gemm(
    A,
    BH,
    BL,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ASM: tl.constexpr,
    ASK: tl.constexpr,
    BSK: tl.constexpr,
    BSN: tl.constexpr,
    CSM: tl.constexpr,
    CSN: tl.constexpr,
    BM: tl.constexpr = 64,
    BN: tl.constexpr = 128,
    BK: tl.constexpr = 32,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    main = tl.full((BM, BN), 0, tl.float32)
    correction = tl.full((BM, BN), 0, tl.float32)
    for offset in range(tr.cdiv(K, BK)):
        ks = offset * BK + inner
        a = tl.load(
            A + rows[:, None] * ASM + ks[None, :] * ASK,
            (rows[:, None] < M) & (ks[None, :] < K),
            0,
        )
        bh = tl.load(
            BH + ks[:, None] * BSK + cols[None, :] * BSN,
            (ks[:, None] < K) & (cols[None, :] < N),
            0,
        )
        bl = tl.load(
            BL + ks[:, None] * BSK + cols[None, :] * BSN,
            (ks[:, None] < K) & (cols[None, :] < N),
            0,
        )
        ah = a.to(tl.float16)
        al = ((a - ah.to(tl.float32)) * 4096.0).to(tl.float16)
        main = tl.dot(ah, bh, main)
        correction = tl.dot(ah, bl, correction)
        correction = tl.dot(al, bh, correction)
    main += correction * (1.0 / 4096.0)
    tl.store(
        C + rows[:, None] * CSM + cols[None, :] * CSN,
        main,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def presplit_weight_gemm(a, bh, bl, out):
    bm, bn, bk = 64, 128, 32
    _presplit_weight_gemm[(tr.cdiv(a.shape[0], bm), tr.cdiv(bh.shape[1], bn))](
        a,
        bh,
        bl,
        out,
        a.shape[0],
        bh.shape[1],
        a.shape[1],
        *a.stride(),
        *bh.stride(),
        *out.stride(),
        bm,
        bn,
        bk,
        num_warps=4,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n, m = args.size, args.rows
    w = torch.randn((1024, n), device="cuda") * 0.01
    x = torch.randn((m, n), device="cuda")
    dy = torch.randn((m, 1024), device="cuda")
    bh = w.to(torch.float16)
    bl = ((w - bh.float()) * 4096.0).to(torch.float16)
    cases = {}
    for name, a, b, high, low, shape in (
        ("forward", x, w.T, bh.T, bl.T, (m, 1024)),
        ("dx", dy, w, bh, bl, (m, n)),
    ):
        reference = torch.empty(shape, device="cuda")
        actual = torch.empty_like(reference)

        def baseline(a=a, b=b, reference=reference):
            bounded_gemm_fp16x3(a, b, reference)

        def presplit(a=a, high=high, low=low, actual=actual):
            presplit_weight_gemm(a, high, low, actual)

        baseline()
        presplit()
        torch.cuda.synchronize()
        difference = (actual - reference).abs()
        checks = {
            "max_abs": difference.max().item(),
            "violations_3e_5": (difference > 3e-5 + 3e-5 * reference.abs())
            .sum()
            .item(),
        }
        samples = {"baseline": [], "presplit": []}
        for repeat in range(args.rounds):
            order = (
                ("baseline", "presplit")
                if repeat % 2 == 0
                else ("presplit", "baseline")
            )
            for mode in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                (baseline if mode == "baseline" else presplit)()
                end.record()
                end.synchronize()
                samples[mode].append(start.elapsed_time(end))
        cases[name] = {
            "checks": checks,
            "median_ms": {
                mode: statistics.median(values) for mode, values in samples.items()
            },
            "samples_ms": samples,
        }
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "rows": m,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in result.items() if key != "cases"}),
        flush=True,
    )
    for name, case in cases.items():
        print(
            json.dumps(
                {"name": name, "checks": case["checks"], "median_ms": case["median_ms"]}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
