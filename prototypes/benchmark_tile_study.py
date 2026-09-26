"""Large-forward diagnosis and a bounded-block Strip comparison on A100."""

import argparse
import gc
import json
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench_cudagraph

from prototypes.benchmark_large_forward import check, dense_control, peak_extra
from prototypes.benchmark_triton_linear import model
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.direct_diagnostics import count_rows, diagnostic_forward
from prototypes.local_atom_linear import forward as direct
from prototypes.profile_current_paths import capture
from torchcst.nn._backends._preparation import prepare
from torchcst.nn._backends._triton import forward as fused
from torchcst.nn._backends._triton_kernels import materialize_weights


def log(**data):
    print(json.dumps(data), flush=True)


def timing(functions):
    samples = {key: [] for key in functions}
    for r in range(3):
        names = list(functions) if r % 2 == 0 else list(reversed(functions))
        for name in names:
            ms = do_bench_cudagraph(functions[name], rep=20)
            samples[name].append(ms)
            log(stage="time", path=name, round=r, ms=ms)
    return {
        "median_ms": {key: median(v) for key, v in samples.items()},
        "samples_ms": samples,
    }


def diagnose(size, batch, output_dir):
    torch.manual_seed(21)
    layer = model(size // 16 * 32, rows=size, columns=size)
    p, circle, section, offsets = prepare(layer, layer.atoms.p, support_layout=True)
    bounds = torch.stack((section.amin(0), section.amax(0)))
    x = torch.randn(batch, size, device="cuda")
    y = torch.empty_like(x)
    w, canonical = dense_control(layer, (p, circle, section, offsets))
    assert canonical["passed"], canonical
    opts = {
        "M": batch,
        "N": size,
        "K": size,
        "S": 16,
        "G": size // 16,
        "D": 4,
        "PROFILE": 1,
        "BM": 16,
        "BK": 128,
    }
    resources = {}

    def run(mode):
        compiled = diagnostic_forward[(triton.cdiv(batch, 16), size)](
            x,
            p,
            circle,
            section,
            offsets,
            bounds,
            y,
            **opts,
            MODE=mode,
            num_warps=4,
            enable_fp_fusion=False,
        )
        resources[str(mode)] = {
            "registers_per_thread": compiled.n_regs,
            "compiler_spills": compiled.n_spills,
            "shared_bytes": compiled.metadata.shared,
        }
        return y

    expected = F.linear(x, w)
    checks = {}
    for mode in (0, 1, 2, 3):
        wanted = F.linear(torch.ones_like(x), w) if mode == 1 else expected
        checks[str(mode)] = check(run(mode), wanted)
        assert checks[str(mode)]["passed"], checks
    counts = torch.empty((size, 4), device="cuda", dtype=torch.int32)
    count_opts = {key: value for key, value in opts.items() if key not in ("M", "BM")}
    count_rows[(size,)](
        p,
        circle,
        section,
        offsets,
        bounds,
        counts,
        **count_opts,
        num_warps=4,
        enable_fp_fusion=False,
    )
    candidates, kept, active, active_blocks = counts.sum(0).cpu().tolist()
    counters = {
        "candidate_row_atom_pairs": candidates,
        "kept_row_atom_pairs": kept,
        "evaluated_sites_per_batch_group": kept * size,
        "active_sites_per_batch_group": active,
        "profile_evaluations_total": kept * size * triton.cdiv(batch, 16),
        "active_fraction_of_evaluated_sites": active / (kept * size),
        "active_column_blocks_per_batch_group": active_blocks,
        "input_float_load_requests_original": active * batch,
        "input_float_load_requests_loop_interchange": size * size * batch,
        "note": "Logical requests, not physical HBM bytes or cache hit counters",
    }
    functions = {
        "dense": lambda: F.linear(x, w),
        "fused_bm64_full": lambda: fused(layer, x, layer.atoms.p, batch_tile=64),
        "direct_bm16_full": lambda: direct(layer, x, layer.atoms.p, batch_tile=16),
        "original_prepared": lambda: run(0),
        "profile_only_not_linear": lambda: run(1),
        "no_row_cull": lambda: run(2),
        "reuse_x_loop_interchange": lambda: run(3),
    }
    measured = timing(functions)
    profiles = []
    for name in ("dense", "fused_bm64_full", "direct_bm16_full"):
        profiles.append(capture(layer, x, name, functions[name], False, output_dir))
    return {
        "shape": [size, size],
        "batch": batch,
        "canonical": canonical,
        "checks": checks,
        "resources": resources,
        "counters": counters,
        "profiles": profiles,
        **measured,
    }


def mapped_control(layer, prepared):
    p, circle, section, offsets = prepared
    s, t = layer.tile_shape
    g = layer.strip.chart.tile_count
    virtual = torch.empty((g * s, t), device="cuda")
    materialize_weights[(g * triton.cdiv(s, 16), triton.cdiv(t, 16))](
        p,
        circle,
        section,
        offsets,
        virtual,
        N=g * s,
        K=t,
        D=4,
        G=g,
        STATION_ROWS=s,
        PROFILE=1,
        BN=16,
        BK=16,
        BA=8,
        SUPPORT_LAYOUT=True,
        num_warps=4,
        enable_fp_fusion=False,
    )
    w = layer.unroll_weight(virtual).contiguous()
    n, k = layer.shape
    rows = torch.linspace(0, n - 1, 128, device="cuda").long()
    chosen = torch.cat(
        (
            torch.randint(n * k, (1024,), device="cuda"),
            rows * k + w[rows].abs().argmax(-1),
        )
    )
    virtual_indices = layer.logical_to_virtual(chosen)
    center, amp, prec = layer.strip.kernel.tile_parameters(
        layer.strip.chart, layer.strip.atoms.p
    )
    canonical = []
    for selection in virtual_indices.split(32):
        values = layer.strip.kernel.profile.evaluate_with_precision_slice(
            layer.strip.chart, center, prec, selection
        )
        canonical.append((values * amp).sum(-1))
    checked = check(w.flatten()[chosen], torch.cat(canonical))
    assert checked["passed"], checked
    return w, checked


def block_case(size, batch, tile, atoms, policy):
    torch.manual_seed(21)
    log(stage="block_start", shape=[size, size], tile=tile, atoms=atoms, policy=policy)
    layer = BlockStripLinear((size, size), tile, atoms, device="cuda")
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    w, canonical = mapped_control(layer, prepared)
    x = torch.randn(batch, size, device="cuda")
    functions = {
        "dense": lambda: F.linear(x, w),
        "fused_full": lambda: layer(x, backend="triton_fused"),
        "direct_full": lambda: layer(x, backend="triton_direct"),
        "fused_prepared": lambda: layer(x, backend="triton_fused", prepared=prepared),
        "direct_prepared": lambda: layer(x, backend="triton_direct", prepared=prepared),
    }
    expected = functions["dense"]()
    checks = {name: check(fn(), expected) for name, fn in functions.items()}
    assert all(v["passed"] for v in checks.values()), checks
    peaks = {name: peak_extra(fn) for name, fn in functions.items()}
    counts = torch.diff(prepared[3]).cpu().tolist()
    result = {
        "shape": [size, size],
        "batch": batch,
        "tile_shape": tile,
        "tile_count": layer.strip.chart.tile_count,
        "atoms": atoms,
        "policy": policy,
        "canonical": canonical,
        "checks": checks,
        "bucket_counts": counts,
        "weight_nonzero_fraction": torch.count_nonzero(w).item() / w.numel(),
        "weight_bytes": w.numel() * 4,
        "parameter_bytes": layer.strip.atoms.p.numel() * 4,
        "prepared_bytes": sum(t.numel() * t.element_size() for t in prepared),
        "extra_peak_bytes": peaks,
        **timing(functions),
    }
    log(stage="block_complete", tile=tile, atoms=atoms, medians=result["median_ms"])
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--phase", choices=["all", "diagnose", "tiles"], default="all")
    parser.add_argument("--tiles", nargs="+", default=["64x64", "64x128", "128x128"])
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["fixed_total", "fixed_per_tile"],
        default=["fixed_total", "fixed_per_tile"],
    )
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    result = {
        "device": props.name,
        "multiprocessors": props.multi_processor_count,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "source_commit": args.source_commit,
        "precision": "FP32, TF32 off",
        "blocks": [],
        "timing": "CUDA Graph median of 3 rounds, rep=20ms; dense W precomputed",
    }

    def save():
        (args.output_dir / "results.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

    save()
    if args.phase in ("all", "diagnose"):
        result["diagnose"] = diagnose(args.size, args.batch, args.output_dir)
        save()
        gc.collect()
        torch.cuda.empty_cache()
    if args.phase in ("all", "tiles"):
        for text in args.tiles:
            tile = tuple(map(int, text.split("x")))
            assert len(tile) == 2 and all(v > 0 and args.size % v == 0 for v in tile)
            for policy in args.policies:
                atoms = (
                    args.size // 16 * 32
                    if policy == "fixed_total"
                    else (args.size // tile[0]) * (args.size // tile[1]) * 32
                )
                result["blocks"].append(
                    block_case(args.size, args.batch, tile, atoms, policy)
                )
                save()
                gc.collect()
                torch.cuda.empty_cache()
    result["completed"] = True
    save()
    log(stage="completed", blocks=len(result["blocks"]))


if __name__ == "__main__":
    main()
