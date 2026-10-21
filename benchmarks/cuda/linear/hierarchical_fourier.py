"""Existing Linear runner extension for the new, explicit I-driven semantics.

This fixture uses direct amp/centers/width-logit AdamW, not Polar updates.
Results identify the distinct contract and are not a production dispatch claim.
"""

import gc
import hashlib
import json
import math
import os
import statistics
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
    FourierLinearSpec,
)


def fixture_operator(case):
    return FourierLinearSpec(case.size, int(case.profile[1]))


def initialize(case):
    gen = torch.Generator().manual_seed(case.seed)
    p = torch.randn(case.atoms, 4, generator=gen) * 0.25
    p[:, 1:3] = torch.rand(case.atoms, 2, generator=gen) * (case.size - 1) * 0.1
    kind = case.profile.split("-")[1]
    if kind == "small":
        width = 0.101 + 0.25 * torch.rand(case.atoms, generator=gen)
    elif kind == "medium":
        width = 1.2 + 2.0 * torch.rand(case.atoms, generator=gen)
    elif kind == "large":
        width = 5.0 + 4.9 * torch.rand(case.atoms, generator=gen)
    else:
        width = torch.exp(
            torch.rand(case.atoms, generator=gen) * math.log(98.9) + math.log(0.101)
        )
    if case.profile.endswith("-cluster"):
        # Centres remain independent continuous values, concentrated in four
        # input/output regions to vary sharing without coordinate tying.
        group = torch.arange(case.atoms) % 4
        base = (group.float() + 0.5) * ((case.size - 1) * 0.1 / 4)
        p[:, 1:3] = (
            base[:, None] + (torch.rand(case.atoms, 2, generator=gen) - 0.5) * 1.6
        )
    p[:, 3] = torch.logit((width - 0.1) / 9.9)
    # Domain truncation, zero amp with nonzero amp derivative, floor-active
    # nonempty and empty profiles. No oracle atoms are sampled or omitted.
    p[0, 0] = 0.0
    p[1, 1:3] = -0.09
    # I=.105 is in the base level and before its next period blend in BOTH
    # declared M4/M8 policies. Put the sole supported site near that endpoint.
    period = fixture_operator(case).period
    width = 0.105
    length = (
        math.ceil((width + period.minimum_tile_width / 2) / period.minimum_half_period)
        * period.minimum_half_period
    )
    p[2, 1:3] = -0.99 * length
    p[2, 3] = math.log((width - 0.1) / (10 - width))
    p[3, 1:3] = case.size * 0.1 + 200
    return p


def _axis(p, spec, axis):
    """Independent Python routing, numerical integral, full KKT and sites."""
    width = 0.1 + 9.9 * p[:, 3].sigmoid()
    per = spec.period
    a0 = (per.minimum_tile_width / 2 + 1.25 * per.minimum_half_period) / (
        1 / per.relative_width_floor - 1
    )
    coarse = [
        max(j for j in range(12) if float(w) >= a0 * 2**j) for w in width.detach().cpu()
    ]
    coarse = torch.tensor(coarse, device=p.device)
    fine = (coarse - 1).clamp_min(0)
    th = a0 * 2.0 ** coarse.double()
    interval = torch.where(coarse > 0, 0.15 * th, 1.0)
    z = ((width - th) / interval).clamp(0, 1)
    h = 6 * z**5 - 15 * z**4 + 10 * z**3
    h = torch.where(coarse > 0, h, 1.0)
    nodes, weights = np.polynomial.legendre.leggauss(192)
    nodes = torch.tensor(nodes, device=p.device, dtype=torch.float64)
    weights = (
        torch.tensor(weights, device=p.device, dtype=torch.float64)
        * (1 - nodes.square()) ** 4
    )
    n = torch.arange(spec.modes + 1, device=p.device, dtype=torch.float64)
    rows = 2 if spec.modes == 4 else 3
    constraint = torch.stack(
        [(-1.0) ** n * (n / spec.modes) ** (2 * j) for j in range(rows)]
        + [torch.ones_like(n)]
    )
    metric = torch.diag(torch.where(n == 0, 2.0, 1.0).double())
    system = torch.cat(
        (
            torch.cat((metric, constraint.T), 1),
            torch.cat(
                (
                    constraint,
                    torch.zeros(
                        len(constraint), len(constraint), device=p.device, dtype=p.dtype
                    ),
                ),
                1,
            ),
        ),
        0,
    )
    target = torch.zeros(len(constraint), device=p.device, dtype=p.dtype)
    target[-1] = 1
    sites = torch.arange(spec.size, device=p.device, dtype=torch.float64) * 0.1
    r = sites[:, None] - p[None, :, axis + 1]
    total = torch.zeros_like(r)
    for lev, outer in ((fine, 1 - h), (coarse, h)):
        tile = per.minimum_tile_width * 2.0 ** lev.double()
        step = per.minimum_half_period * 2.0 ** lev.double()
        low = torch.ceil((width.detach() + tile / 2) / step) * step
        z = ((width - low + tile / 2 + 0.25 * step) / (0.25 * step)).clamp(0, 1)
        w = 6 * z**5 - 15 * z**4 + 10 * z**3
        for length, inner in ((low, 1 - w), (low + step, w)):
            q = width / length
            integ = (
                torch.cos(math.pi * q[:, None, None] * n[None, :, None] * nodes)
                @ weights
            )
            raw = q[:, None] / 2 * integ * torch.where(n == 0, 1.0, 2.0)
            rhs = torch.cat((raw @ metric.T, target.expand(len(p), -1)), 1)
            coeff = torch.linalg.solve(system, rhs.T).T[:, : spec.modes + 1]
            f = (
                torch.cos(r[:, :, None] * math.pi * n / length[None, :, None])
                * coeff[None, :, :]
            ).sum(-1)
            total += torch.where(
                r.abs() < length[None, :], f * outer[None, :] * inner[None, :], 0.0
            )
    return total


def _compute_independent_gate(x, p, dy, spec, chunk=32):
    yy = torch.zeros_like(x, dtype=torch.float64)
    dx = torch.zeros_like(yy)
    grads = []
    for start in range(0, len(p), chunk):
        xx = x.detach().double().requires_grad_()
        pp = p[start : start + chunk].detach().double().requires_grad_()
        u, v = _axis(pp, spec, 0), _axis(pp, spec, 1)
        d = (u.norm(dim=0) * v.norm(dim=0)).clamp_min(spec.floor)
        y = ((xx @ v) * (pp[:, 0] / d)) @ u.T
        gx, gp = torch.autograd.grad(y, (xx, pp), dy.double())
        yy += y.detach()
        dx += gx
        grads.append(gp)
    return yy, dx, torch.cat(grads)


_ORACLE_CACHE_EVENTS = []


def independent_gate(x, p, dy, spec, chunk=32):
    """Optional job-local reuse of independent truth, keyed by exact live inputs.

    No candidate result enters this cache. It is never consulted by a timing
    worker. Full FP64 Y/dX/all-P are computed independently on the first miss.
    """
    directory = os.environ.get("CST_FOURIER_ORACLE_CACHE")
    if not directory:
        return _compute_independent_gate(x, p, dy, spec, chunk)
    identity = {
        "oracle_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "spec": asdict(spec),
        "period": asdict(spec.period),
        "chunk": chunk,
        "torch": torch.__version__,
        "inputs": [
            {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "sha256": hashlib.sha256(
                    t.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
            }
            for t in (x, p, dy)
        ],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path, receipt = root / (key + ".pt"), root / (key + ".json")
    if path.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if record.get("identity") != identity or record.get("sha256") != actual:
            raise RuntimeError("independent oracle cache integrity mismatch")
        values = torch.load(path, map_location="cpu", weights_only=True)
        expected = (x.shape, x.shape, p.shape)
        if len(values) != 3 or any(
            t.dtype != torch.float64 or t.shape != shape
            for t, shape in zip(values, expected)
        ):
            raise RuntimeError("independent oracle cache shape/dtype mismatch")
        _ORACLE_CACHE_EVENTS.append({"key": key, "hit": True, "sha256": actual})
        return tuple(t.to(x.device) for t in values)
    values = _compute_independent_gate(x, p, dy, spec, chunk)
    # One correctness worker at a time in the existing runner. The receipt is
    # written last; a partial cache is not admitted as an independent oracle.
    torch.save(tuple(t.detach().cpu() for t in values), path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt.write_text(json.dumps({"identity": identity, "sha256": actual}, indent=2))
    _ORACLE_CACHE_EVENTS.append({"key": key, "hit": False, "sha256": actual})
    return values


def correctness(args, run, model_type):
    from .check_normalized import check

    _ORACLE_CACHE_EVENTS.clear()
    case = run.case
    spec = fixture_operator(case)
    model = model_type(initialize(case).cuda(), spec, run.entry(args.plan_id).plan)
    gen = torch.Generator().manual_seed(case.seed + 1)
    x = torch.randn(case.rows, case.size, generator=gen).cuda().requires_grad_()
    dy = torch.randn(case.rows, case.size, generator=gen).cuda()
    reports = []
    oracle_chunk = 256 if spec.size >= 1024 else 32
    for step in range(3):
        if step:
            with torch.no_grad():
                model.p[:, 3].add_(0.007)
                model.p[:, 1:3].add_(0.0137)
                x.add_(0.0017)
        y = model(x)
        dx, dp = torch.autograd.grad(y, (x, model.p), dy, retain_graph=True)
        repeated = torch.autograd.grad(y, (x, model.p), dy)
        truth, tx, tp = independent_gate(x, model.p, dy, spec, chunk=oracle_chunk)
        reports.append(
            {
                "step": step,
                "Y": check(y, truth, 4e-4),
                "dX": check(dx, tx, 4e-4),
                "dP": check(dp, tp, 4e-4),
            }
        )
        for a, b in zip((dx, dp), repeated):
            check(a, b, 4e-4)
    # Capture and replay with explicit live changes crossing the routing range.
    # Retained eager graphs own AccumulateGrad nodes on the default stream.
    # Release them before creating those nodes on the capture stream.
    del y, dx, dp, repeated, truth, tx, tp
    model.zero_grad(set_to_none=True)
    x.grad = None
    gc.collect()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            x.grad = None
            (model(x) * dy).sum().backward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        model.zero_grad(set_to_none=True)
        x.grad = None
        captured = model(x)
        (captured * dy).sum().backward()
    with torch.no_grad():
        model.p[:, 3].add_(0.31)
        model.p[:, 1:3].add_(0.071)
        x.add_(0.02)
    graph.replay()
    torch.cuda.synchronize()
    truth, tx, tp = independent_gate(x, model.p, dy, spec, chunk=oracle_chunk)
    reports.append(
        {
            "graph_dynamic": True,
            "Y": check(captured, truth, 4e-4),
            "dX": check(x.grad, tx, 4e-4),
            "dP": check(model.p.grad, tp, 4e-4),
        }
    )
    return {
        "status": "PASS",
        "scope": "independent quadrature/KKT FP64 full sites/all atoms, live logit-I; retained backward and dynamic graph",
        "checks": reports,
        "independent_oracle_cache": list(_ORACLE_CACHE_EVENTS),
    }


def measure_case(args, run, model_type):
    from .run import generate_inputs

    case = run.case
    spec = fixture_operator(case)
    p = initialize(case)
    p_hash = hashlib.sha256(p.numpy().tobytes()).hexdigest()
    cx, ct = generate_inputs(case.seed, case.rows, case.size)
    inputs = {
        "x_sha256": hashlib.sha256(cx.numpy().tobytes()).hexdigest(),
        "target_sha256": hashlib.sha256(ct.numpy().tobytes()).hexdigest(),
    }
    x = cx.cuda().requires_grad_()
    target = ct.cuda()
    if args.worker == "dense":
        torch.manual_seed(case.seed)
        model = torch.nn.Linear(case.size, case.size, bias=False, device="cuda")
    else:
        model = model_type(p.cuda(), spec, run.entry(args.plan_id).plan)
    del p, cx, ct
    initial = None if args.worker == "dense" else model.p.detach().clone()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=case.optimizer.lr,
        weight_decay=case.optimizer.weight_decay,
        fused=True,
        capturable=True,
    )

    def step():
        opt.zero_grad(set_to_none=True)
        x.grad = None
        ((model(x) * target).sum() / target.numel()).backward()
        opt.step()

    def timed(fn):
        times = []
        for _ in range(case.rounds):
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            fn()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b))
        return {"median_ms": statistics.median(times), "samples_ms": times}

    for _ in range(case.warmup):
        step()
    eager = timed(step)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(case.warmup):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()
    graph_times = timed(graph.replay)
    torch.cuda.synchronize()
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    change = None if initial is None else model.p.detach() - initial
    return {
        "status": "PASS",
        "initial_p_sha256": p_hash,
        "initial_inputs": inputs,
        "eager": eager,
        "graph": graph_times,
        "allocated_before_capture_bytes": before,
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": reserved,
        "memory_scope": "complete worker including static bases, parameters, optimizer, graph capture/replay; no oracle tensors",
        "width_changed_atoms": None
        if change is None
        else int((change[:, 3] != 0).sum()),
        "operator_contract": "centered support; live I=.1+9.9*sigmoid(logit); complete product L2 floor once; direct-coordinate AdamW, no Polar update",
        "timing_scope": "preparation, sorting, norm, forward, backward, optimizer included; GPU event elapsed",
    }
