"""Diagnostic ablations only: altered operands/checksums are not CST candidates."""

import json
import re
import statistics
from pathlib import Path

import torch
import triton as tr
import triton.language as tl

from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.h_kernels import (
    _geometry,
)
from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.query_kernels import (
    _query,
)


@tr.jit
def probe(
    Bank,
    Meta,
    Cuts,
    Coeff,
    Count,
    Origin,
    Spacing,
    G,
    GP,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    MODE: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        b = tl.arange(0, BB)
        mask = (a[:, None] < A) & (b[None, :] < B)
        if MODE == "factored" or MODE == "pieces" or MODE == "fma":
            value, prime = _query(
                Bank,
                Meta,
                Cuts,
                Coeff,
                Origin,
                Spacing,
                a,
                b,
                A,
                B,
                N,
                0,
                AB,
                BB,
                True,
                True,
                True,
                True,
                True,
                False,
                MODE,
            )
        else:
            value = tl.full((AB, BB), 0, tl.float64)
            prime = tl.full((AB, BB), 0, tl.float64)
            bits = tl.full((AB, BB), 0, tl.int64)
            # Uniform addresses alter operands but retain FP64 arithmetic. This
            # tests address diversity; it does not measure physical DRAM bytes.
            ca = (
                tl.full((AB,), tl.load(Count), tl.int32)
                if (MODE == "uniform_coeff" or MODE == "uniform_both")
                else a
            )
            for knot in tl.static_range(4):
                cut, c0, c1, c2, d0, d1 = _geometry(
                    Meta,
                    Cuts,
                    Coeff,
                    Origin,
                    Spacing,
                    ca,
                    A,
                    N,
                    0,
                    knot,
                    True,
                    True,
                    True,
                )
                cut = tl.load(Cuts + knot * A + a, a < A, 0)
                if MODE == "uniform_bank" or MODE == "uniform_both":
                    cut = tl.full(
                        (AB,), tl.load(Cuts + knot * A + tl.load(Count)), tl.int32
                    )
                addr = cut[:, None] * B + b[None, :]
                p0 = tl.load(Bank + addr, mask, 0)
                p1 = tl.load(Bank + B * (N + 1) + addr, mask, 0)
                p2 = tl.load(Bank + 2 * B * (N + 1) + addr, mask, 0)
                if MODE == "checksum":
                    bits ^= (
                        p0.to(tl.int64, bitcast=True)
                        ^ p1.to(tl.int64, bitcast=True)
                        ^ p2.to(tl.int64, bitcast=True)
                    )
                    for c in (c0, c1, c2, d0, d1):
                        bits ^= c[:, None].to(tl.int64, bitcast=True)
                else:
                    value -= p0 * c0[:, None] + p1 * c1[:, None] + p2 * c2[:, None]
                    prime -= p0 * d0[:, None] + p1 * d1[:, None]
            if MODE == "checksum":
                tl.store(
                    G + a[:, None] * B + b[None, :],
                    bits.to(tl.int32).to(tl.float32),
                    mask,
                )
                tl.store(
                    GP + a[:, None] * B + b[None, :],
                    (bits >> 32).to(tl.int32).to(tl.float32),
                    mask,
                )
            else:
                left = tl.load(Cuts + a, a < A, 0)
                right = tl.load(Cuts + 3 * A + a, a < A, 0)
                value = tl.where((right > left)[:, None], value, 0)
                prime = tl.where((right > left)[:, None], prime, 0)
        if MODE != "checksum":
            tl.store(G + a[:, None] * B + b[None, :], value.to(tl.float32), mask)
            tl.store(GP + a[:, None] * B + b[None, :], prime.to(tl.float32), mask)


def measure(fn, repeats=21, calls=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    samples = []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / calls)
    return {
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "calls_per_replay": calls,
    }


def run(meta, cuts, coeff, count, origin, spacing, bank, output, batch):
    output = Path(output)
    a, n = len(meta), bank.shape[1] - 1
    g = torch.zeros((a, batch), device="cuda")
    gp = torch.zeros_like(g)
    rows = []
    reference = None
    for mode in (
        "full",
        "checksum",
        "uniform_coeff",
        "uniform_bank",
        "uniform_both",
        "factored",
        "pieces",
        "fma",
    ):

        def launch(mode=mode):
            return probe[(tr.cdiv(a, 8),)](
                bank,
                meta,
                cuts,
                coeff,
                count,
                origin,
                spacing,
                g,
                gp,
                a,
                batch,
                n,
                8,
                tr.next_power_of_2(batch),
                mode,
                num_warps=4,
                enable_fp_fusion=False,
            )

        compiled = launch()
        if mode == "full":
            reference = (g.clone(), gp.clone())
        err = None
        if mode in ("factored", "pieces", "fma"):
            start = int(count.item())
            err = {}
            for name, x, y in zip(("g", "gp"), (g, gp), reference):
                dx, dy = x[start:].double(), y[start:].double()
                err[name] = {
                    "max": (dx - dy).abs().max().item(),
                    "rel_l2": ((dx - dy).norm() / dy.norm()).item(),
                }
                assert err[name]["max"] < 4e-4 and err[name]["rel_l2"] < 4e-4
        ptx = compiled.asm["ptx"]
        (output / f"probe-b{batch}-{mode}.ptx").write_text(ptx)
        rows.append(
            {
                "mode": mode,
                "timing": measure(launch),
                "query_difference": err,
                "registers": compiled.n_regs,
                "spill": compiled.n_spills,
                "shared_bytes": compiled.metadata.shared,
                "static_ptx_f64_ops": len(
                    re.findall(r"(?:add|sub|mul|div|fma|rcp)\.[^;\n]*f64", ptx)
                ),
                "static_ptx_global_loads": len(re.findall(r"ld\.global", ptx)),
                "static_ptx_barriers": len(re.findall(r"bar\.sync", ptx)),
            }
        )
        (output / f"probe-progress-b{batch}.json").write_text(
            json.dumps(rows, indent=2)
        )
        print(
            f"probe B{batch} {mode}: {rows[-1]['timing']['median_ms']:.6f} ms",
            flush=True,
        )
    return {
        "rows": rows,
        "scope": "G/GP only; actual mixed geometry and prefix values; altered operands and checksum are diagnostic, never adopted; no atomic, no scans inside query; static PTX is not a hardware counter",
    }
