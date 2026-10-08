"""Untimed compiler audit. Never infer H/G placement from allocator peaks."""

import copy
import hashlib
import json
from pathlib import Path

import torch


def audit(output):
    import triton

    from torchcst import CSTLinear
    from torchcst._backends.cuda.algorithms.linear.torus_profile_product.onchip.kernels import (
        backward,
        forward,
    )
    from torchcst._backends.torch.algorithms.linear.torus_profile_product.executor import (
        _queries,
        _Scalars,
    )
    from torchcst._backends.torch.parameterizations import polar_amp_width as polar

    from .manifest import load_run
    from .torus_profile_product import fixture_operator, initialize

    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    reports = []
    for n in (1024, 2048):
        case = load_run(
            f"benchmarks/cuda/linear/cases/torus-profile-product-onchip-{n}-sigma3.json",
            "benchmarks/cuda/linear/plans-torus-profile-product-onchip.json",
        ).case
        op = fixture_operator(case)
        layer = CSTLinear(
            chart=op.charts[0],
            kernel=op.kernel,
            atoms=initialize(case)[:23],
            device="cuda",
        )
        p = layer.atoms.p.detach().clone()
        scalars = _Scalars(layer.kernel, p)
        amp, alpha = polar._amplitude_and_alpha(scalars, p[:, :2])
        sigma, _, _ = polar._sigma_bounds(scalars, amp, alpha, side="input")
        precision = sigma.reciprocal().square().detach()
        major, minor, circle, sites, _ = _queries(layer.chart, p)
        maximum = scalars.scalar("amplitude_max")
        for b in (32, 64):
            x = torch.randn(b, n, device="cuda")
            dy = torch.randn_like(x)
            y, dx, dp = torch.zeros_like(x), torch.zeros_like(x), torch.empty_like(p)
            common = {
                "B": b,
                "NI": n,
                "NO": n,
                "PB": triton.next_power_of_2(b),
                "T": 64,
                "FLOOR": 1e-6,
                "num_warps": 4,
                "enable_fp_fusion": False,
            }
            compiled = forward[(len(p),)](
                x, p, precision, maximum, major, minor, circle, sites, y, **common
            )
            variants = [("forward", compiled)]
            for gx, gp in ((True, True), (True, False), (False, True)):
                compiled = backward[(len(p),)](
                    x,
                    dy,
                    p,
                    precision,
                    maximum,
                    major,
                    minor,
                    circle,
                    sites,
                    dx,
                    dp,
                    NEED_X=gx,
                    NEED_P=gp,
                    **common,
                )
                variants.append((f"backward-x{int(gx)}-p{int(gp)}", compiled))
            torch.cuda.synchronize()
            for name, compiled in variants:
                tag = f"{n}-b{b}-{name}"
                ptx = compiled.asm["ptx"]
                (output / f"{tag}.ptx").write_text(ptx)
                report = {
                    "variant": tag,
                    "registers": getattr(compiled, "n_regs", None),
                    "spill_slots": getattr(compiled, "n_spills", None),
                    "shared_bytes": getattr(compiled.metadata, "shared", None),
                    "local_loads": "ld.local" in ptx,
                    "local_stores": "st.local" in ptx,
                    "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                }
                reports.append(copy.deepcopy(report))
                # Preserve all compiled reports before a failing audit exits.
                (output / "compiler.json").write_text(
                    json.dumps(reports, indent=2) + "\n"
                )
                assert report["spill_slots"] == 0, report
                assert not report["local_loads"] and not report["local_stores"], report
    return reports
