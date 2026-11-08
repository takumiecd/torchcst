"""Fixed-state engineering probe using the existing radius train loop.

Not the central submission schema, an LM quality study or a speed adoption.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from benchmarks.cuda.linear.radius_cell_mass import train
from torchcst import (
    ChebyshevSpec,
    CSTAdamW,
    CSTLinear,
    FixedSelector,
    chart_presets,
    init,
    kernels,
    make_algorithm_registry,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_multirow import (
    LM_PHYSICAL_LEVELS,
    RadiusLMPhysicalLevelsAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.radius_envelope import (
    RADIUS_ENVELOPE_LM,
    RadiusEnvelopeLMAlgorithm,
)
from torchcst._backends.torch.parameterizations.radius_envelope import from_state


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--arm",
        choices=["dense", "physical", "envelope", "envelope_cst"],
        required=True,
    )
    p.add_argument("--ni", type=int, required=True)
    p.add_argument("--no", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    charts = tuple(
        chart_presets.regular_grid(n, geometry="flat_torus", spacing=1.0)
        for n in (args.no, args.ni)
    )
    base = CSTLinear(
        *charts,
        atoms=8192,
        dtype=torch.float32,
        backend="factored",
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8)),
            coordinates=kernels.radius_envelope(
                radius_min=0.5, radius_max=50.0, tau=0.03, radius_fraction_init=0.0
            ),
        ),
    )
    report = init.xavier_normal_(base, radius_fraction=0.0)
    pp = from_state(base.kernel, base.atoms.p).detach()
    physical = CSTLinear(
        *charts,
        atoms=pp,
        dtype=torch.float32,
        backend="factored",
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8)),
            coordinates=kernels.physical(radius_init=(0.5, 50.0), radius_floor=0.5),
        ),
    )
    weight = physical.dense_weight().detach()
    data = torch.Generator().manual_seed(71)
    xcpu = torch.randn(256, args.ni, generator=data)
    target_cpu = torch.randn(256, args.no, generator=data)
    hashes = {
        "physical": digest(pp),
        "weight": digest(weight),
        "input": digest(xcpu),
        "target": digest(target_cpu),
    }
    if args.arm == "dense":
        model = torch.nn.Linear(args.ni, args.no, bias=False)
        with torch.no_grad():
            model.weight.copy_(weight)
    elif args.arm == "physical":
        model = physical
    else:
        model = base
    del base, physical
    registry = make_algorithm_registry()
    registry.register(RadiusLMPhysicalLevelsAlgorithm())
    registry.register(RadiusEnvelopeLMAlgorithm())
    if args.arm != "dense":
        model.selector = FixedSelector(
            LM_PHYSICAL_LEVELS if args.arm == "physical" else RADIUS_ENVELOPE_LM,
            registry=registry,
        )
    model = model.cuda()
    x = xcpu.cuda().requires_grad_()
    target = target_cpu.cuda()
    # Core controls use LR0; the public CST policy uses a positive sub-ULP LR
    # to execute projection/transport while preserving the physical workload.
    rate = 1e-20 if args.arm == "envelope_cst" else 0.0
    opt = (
        CSTAdamW(
            model.parameters(),
            model=model,
            lr=rate,
            weight_decay=0,
            foreach=False,
            field_lrs={"radius_fraction": rate},
        )
        if args.arm == "envelope_cst"
        else torch.optim.AdamW(
            model.parameters(), lr=rate, weight_decay=0, foreach=False
        )
    )
    actual = model(x)
    torch.testing.assert_close(
        actual.detach().cpu().double(),
        xcpu.double() @ weight.double().T,
        atol=1e-4,
        rtol=4e-3,
    )
    actual.square().mean().backward()
    assert all(
        t.grad is None or torch.isfinite(t.grad).all() for t in model.parameters()
    )
    del actual, xcpu, target_cpu, weight, pp
    opt.zero_grad(set_to_none=True)
    x.grad = None
    before = digest(
        model.atoms.p
        if args.arm == "physical"
        else from_state(model.kernel, model.atoms.p)
        if args.arm != "dense"
        else model.weight
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        train(model, opt, x, target)
    samples = []
    allocated = []
    for _ in range(21):
        torch.cuda.synchronize()
        start = time.perf_counter()
        train(model, opt, x, target)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        allocated.append(torch.cuda.memory_allocated())
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_res = torch.cuda.max_memory_reserved()
    after = digest(
        model.atoms.p
        if args.arm == "physical"
        else from_state(model.kernel, model.atoms.p)
        if args.arm != "dense"
        else model.weight
    )
    assert before == after, "physical workload drifted"
    result = {
        "arm": args.arm,
        "ni": args.ni,
        "no": args.no,
        "rows": 256,
        "atoms": None if args.arm == "dense" else 8192,
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_res,
        "allocated_spread_bytes": max(allocated) - min(allocated),
        "initial_hashes": hashes,
        "physical_state_unchanged": True,
        "optimizer_lr": rate,
        "initialization_error": report.relative_error,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "scope": "Eager FP32 TF32-off fixed physical state. Existing train: zero_grad/x.grad clear, MSE forward/backward, AdamW, optional public CST projection/transport, sync. Warmup included in peaks; setup excluded; process usage/Graph unmeasured. LR0 controls and positive sub-ULP CST LR; not learning convergence.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("samples_ms", "initial_hashes")
            }
        )
    )


if __name__ == "__main__":
    main()
