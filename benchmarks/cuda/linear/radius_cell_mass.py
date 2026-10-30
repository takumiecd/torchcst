"""Eager radius fixture for the existing Linear runner (no standalone runner)."""

import copy
import math
import statistics
import time
from dataclasses import asdict, replace

import torch

from torchcst import (
    ChebyshevRadiusSpec,
    CSTAdamW,
    CSTLinear,
    CSTOptimizer,
    RadiusBounds,
    chart_presets,
    presets,
)

from .sphere_baseline import error, tensor_hash


def fixture(case):
    torch.manual_seed(case.seed)
    chart = chart_presets.regular_grid(
        ((case.size,),), spacing=1.0, geometry="flat_torus"
    )
    kernel = presets.polar_radius_cell_mass(
        amplitude_max=1.0,
        w_c=0.3,
        input_radius_bounds=RadiusBounds(
            minimum=0.5, birth=4.0, maximum=50.0, upper_floor=1.0
        ),
        profile=ChebyshevRadiusSpec(modes=8),
        merge_factor=2,
    )
    model = CSTLinear(chart, chart, atoms=case.atoms, kernel=kernel, backend="factored")
    # Public initialization, including its decoded radius distribution, is retained.
    x, target, dy = (torch.randn(case.rows, case.size) for _ in range(3))
    return model, x, target, dy


def oracle_factors(model, p):
    """Independent FP64 polynomial quadrature; fixed routing, detached Polar radii."""
    import numpy as np

    q = p[:, :2].square().sum(-1).clamp_min(torch.finfo(p.dtype).tiny)
    amplitude = p[:, 0] / q.sqrt()
    alpha = ((q - 1) / 3).clamp(0, 1)
    z = (amplitude / 0.3).square()
    lower = 0.5 + 3.5 / (1 + float(model.kernel.scalar("lower_kappa")) * z)
    upper = (0.5 + 49.5 * 3 / (3 + z)).clamp_min(1)
    upper = torch.maximum(upper, lower)
    radius = torch.exp((1 - alpha) * lower.log() + alpha * upper.log()).detach()
    nodes, weights = np.polynomial.legendre.leggauss(48)
    nodes, weights = p.new_tensor(nodes), p.new_tensor(weights)
    factors = []
    for column in (2, 3):
        rows = []
        for r, center in zip(radius, p[:, column]):
            tile = 1
            while float(r) > tile:
                tile *= 2
            count = math.ceil(2 * float(r) / tile) + 1
            period = count * tile
            c = center.remainder(model.in_features)
            first = (
                math.floor((float(c.detach()) + 0.5) / tile - (count - 1) / 2) * tile
            )
            sites = torch.arange(first, first + count * tile, dtype=p.dtype)

            def integrate(left, right, r=r, period=period):
                u = (right - left)[..., None] / 2 * nodes + (right + left)[
                    ..., None
                ] / 2
                beta = 1 / torch.cos(torch.pi * r / period)
                v = beta * torch.cos(torch.pi * u / period)
                a, b = torch.ones_like(v), v
                for _ in range(2, 9):
                    a, b = b, 2 * v * b - a
                value = b / torch.cosh(8 * torch.acosh(beta))
                return (right - left) / 2 * (value.square() * weights).sum(-1)

            left = sites - 0.5 - c
            raw = integrate(left, left + 1)
            norm = integrate(p.new_tensor(-period / 2), p.new_tensor(period / 2))
            row = p.new_zeros(model.in_features).scatter_add(
                0, sites.long().remainder(model.in_features), raw / norm
            )
            rows.append(row)
        factors.append(torch.stack(rows).T)
    return factors[0], factors[1] * amplitude[None]


def oracle(model, x, p=None):
    p = model.atoms.p if p is None else p
    vi, uo = oracle_factors(model, p)
    return (x @ vi) @ uo.T


REFERENCE_CHUNK_ATOMS = 256


def _reference_model(model, parameters):
    """Fresh public CPU site with current fixed charts/kernel and selected P."""
    return CSTLinear(
        *model.cst_charts(),
        atoms=parameters.detach().clone(),
        kernel=model.kernel.declaration(),
        backend="factored",
    )


def chunked_reference(model, x, dy=None, *, chunk_atoms=REFERENCE_CHUNK_ATOMS):
    """Public CPU Fourier Y and optional VJP with at most one chunk graph live.

    All current atom rows are evaluated, in original order. Fixed chart/kernel
    values come from the live model, not the initializer. Chunking bounds the
    site-by-atom factors and frees each graph before processing the next chunk.
    Returned values are detached; this is benchmark validation, not a backend.
    """
    if type(chunk_atoms) is not int or chunk_atoms < 1:
        raise ValueError("reference chunk_atoms must be a positive integer")
    if x.device.type != "cpu" or model.atoms.p.device.type != "cpu":
        raise ValueError("chunked reference requires a CPU model and inputs")
    if dy is not None and (dy.shape != x.shape or dy.dtype != x.dtype):
        raise ValueError("reference dy must have the input shape and dtype")
    tx = x.detach().requires_grad_(dy is not None)
    y = x.new_zeros((x.shape[0], model.out_features))
    dx = torch.zeros_like(x) if dy is not None else None
    dp = []
    part = None
    for start in range(0, model.atoms.count, chunk_atoms):
        parameters = model.atoms.p[start : start + chunk_atoms]
        if part is None or part.atoms.count != len(parameters):
            part = _reference_model(model, parameters)
        else:
            with torch.no_grad():
                part.atoms.p.copy_(parameters)
        with torch.set_grad_enabled(dy is not None):
            value = part(tx)
            if dy is not None:
                gx, gp = torch.autograd.grad(value, (tx, part.atoms.p), dy.detach())
                dx.add_(gx.detach())
                dp.append(gp.detach())
        y.add_(value.detach())
        # Explicitly drop SavedVariables before constructing the next chunk.
        del value
    return y if dy is None else (y, dx, torch.cat(dp))


def _chunked_loss_backward(model, x, target, *, chunk_atoms=REFERENCE_CHUNK_ATOMS):
    # Loss couples atoms: derive one global dY before the per-chunk VJPs.
    y = chunked_reference(model, x, chunk_atoms=chunk_atoms)
    dy = 2 * (y - target) / y.numel()
    _, x.grad, model.atoms.p.grad = chunked_reference(
        model, x, dy, chunk_atoms=chunk_atoms
    )


def stratified_subset(model, count=8):
    """Current atom rows spanning the decoded input/output radius range."""
    from torchcst._backends.torch.parameterizations.radius_cell_mass import (
        decoded_values,
    )

    _, ri, ro = decoded_values(model.kernel, model.atoms.p.detach())
    order = torch.maximum(ri, ro).argsort(stable=True)
    indices = order[torch.linspace(0, len(order) - 1, min(count, len(order))).long()]
    return _reference_model(model, model.atoms.p[indices]), indices.tolist()


def check(model, x, dy, *, independent=False):
    cpu = copy.deepcopy(model).cpu().double()
    tx = x.detach().cpu().double().requires_grad_()
    # CPU reference must use its public Torch route after copying a CUDA selector.
    cpu.selector = None
    cpu.backend = "factored"
    chunked = not independent and cpu.in_features > 128
    if chunked:
        expected, *reference = chunked_reference(cpu, tx, dy.detach().cpu().double())
    else:
        expected = oracle(cpu, tx) if independent else cpu(tx)
        reference = torch.autograd.grad(expected, (tx, cpu.atoms.p), dy.cpu().double())
    xx = x.detach().clone().requires_grad_()
    actual = model(xx)
    gradients = torch.autograd.grad(actual, (xx, model.atoms.p), dy)
    errors = {
        name: error(a.cpu(), b)
        for name, a, b in zip(
            ("Y", "dX", "all_dP"), (actual, *gradients), (expected, *reference)
        )
    }
    if any(e["max_abs"] > 5e-4 or e["relative_l2"] > 5e-4 for e in errors.values()):
        raise AssertionError(errors)
    return {
        "status": "PASS",
        "scope": (
            "full sites/all atoms; independent FP64 polynomial quadrature"
            if independent
            else "full sites/all atoms; FP64 public Torch Fourier reference (separately quadrature-qualified)"
        ),
        "errors": errors,
        **({"reference_chunk_atoms": REFERENCE_CHUNK_ATOMS} if chunked else {}),
    }


def configure(model, plan):
    from torchcst._backends.dispatch import FixedSelector

    from .manifest import REGISTRY

    model.selector = FixedSelector(plan, registry=REGISTRY)
    return model


def optimizer(model, case, *, gpu=True, compatibility=False):
    kwargs = {}
    if gpu:
        from torchcst._backends.catalog import get_registry
        from torchcst._backends.cuda.algorithms.polar_update.plans import (
            RADIUS_FLAT_TORUS_FUSED,
        )
        from torchcst._backends.dispatch import FixedSelector

        kwargs["update_selector"] = FixedSelector(
            RADIUS_FLAT_TORUS_FUSED, registry=get_registry()
        )
        # Use the public native optimizer measured by the complete-step study.
        # The CPU oracle below retains the independent compatibility wrapper.
        if not compatibility:
            return CSTAdamW(
                model.parameters(),
                model=model,
                lr=case.optimizer.lr,
                amplitude_weight_decay=case.optimizer.weight_decay,
                weight_decay=0.0,
                fused=case.optimizer.fused,
                capturable=False,
                check_finite=False,
                **kwargs,
            )
    return CSTOptimizer(
        torch.optim.AdamW(
            [
                {
                    "params": model.parameters(),
                    "weight_decay": 0.0,
                    "amplitude_weight_decay": case.optimizer.weight_decay,
                }
            ],
            lr=case.optimizer.lr,
            weight_decay=0.0,
            fused=case.optimizer.fused,
            capturable=False,
        ),
        model=model,
        # Match the user's PyTorch-style finite-value policy. Live configuration
        # validation remains active; GPU finite-error checks are tested separately.
        check_finite=False,
        **kwargs,
    )


def train(model, opt, x, target, *, reference=False):
    opt.zero_grad(set_to_none=True)
    x.grad = None
    y = oracle(model, x) if reference else model(x)
    (y - target).square().mean().backward()
    opt.step()


def parameter_quantization_gate(actual, expected, *, large):
    """Keep the old floor; large centers also allow one adjacent FP32 value."""
    actual = actual.detach().cpu().double()
    expected32 = expected.detach().cpu().float()
    expected = expected32.double()
    delta = (actual - expected).abs()
    up = torch.nextafter(expected32, torch.full_like(expected32, math.inf)).double()
    down = torch.nextafter(expected32, torch.full_like(expected32, -math.inf)).double()
    # Direction matters at power-of-two boundaries: two smaller downward steps
    # must not pass using the larger upward spacing.
    ulp = torch.where(actual >= expected, up - expected, expected - down)
    allowed = torch.full_like(delta, 2e-5)
    if large:
        allowed[:, 2:] = torch.maximum(allowed[:, 2:], ulp[:, 2:])
    worst = tuple(int(v) for v in torch.unravel_index(delta.argmax(), delta.shape))
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    failed = ~finite | (delta > allowed)
    return {
        "status": "FAIL" if bool(failed.any()) else "PASS",
        "absolute_floor": 2e-5,
        "center_adjacent_fp32_allowance": int(large),
        "failed_elements": int(failed.sum()),
        "worst_abs_index": list(worst),
        "worst_abs_delta": float(delta[worst]),
        "worst_abs_allowed": float(allowed[worst]),
        "worst_abs_ulp_up": float((up - expected)[worst]),
        "worst_abs_ulp_down": float((expected - down)[worst]),
        "worst_abs_delta_in_directed_ulp": float((delta / ulp)[worst]),
        "per_column_max_directed_ulp": (delta / ulp).max(dim=0).values.tolist(),
    }


def _copy_optimizer_inputs(actual, source, compatibility, target):
    """Exact same-device point/gradient/Adam inputs before either wrapper runs."""
    with torch.no_grad():
        target.copy_(source)
    target.grad = source.grad.detach().clone()
    compatibility.state[target].clear()
    compatibility.state[target].update(
        {
            name: value.detach().clone()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
            for name, value in actual.state[source].items()
        }
    )


def optimizer_bitgate(actual, source, compatibility, target):
    """Byte equality includes signed zero; report each owned Adam state field."""
    pairs = {"parameters": (source, target), "gradients": (source.grad, target.grad)}
    if actual.state[source].keys() != compatibility.state[target].keys():
        raise AssertionError("optimizer state field names differ")
    pairs.update(
        {
            name: (value, compatibility.state[target][name])
            for name, value in actual.state[source].items()
        }
    )
    fields = {}
    for name, (a, b) in pairs.items():
        same = a.dtype == b.dtype and a.shape == b.shape and a.device == b.device
        same = same and torch.equal(
            a.detach().reshape(-1).view(torch.uint8),
            b.detach().reshape(-1).view(torch.uint8),
        )
        fields[name] = bool(same)
    if not all(fields.values()):
        raise AssertionError(
            {"gate": "GPU_optimizer_wrapper_bitwise", "fields": fields}
        )
    return fields


def matched_steps(base, case, plan):
    """Three local update checks from identical FP32 states along the GPU path.

    Piecewise radius/tile routing makes an autonomous FP64 optimizer path a
    different experiment. The CPU reference is reset before each native step;
    this gate does not claim free-running trajectory equivalence.
    """
    from torchcst._backends.torch.parameterizations.radius_cell_mass import (
        decoded_values,
    )

    a = configure(copy.deepcopy(base).cuda(), plan)
    b = copy.deepcopy(base).float()
    b.selector = None
    b.backend = "factored"
    oa, ob = optimizer(a, case), optimizer(b, case, gpu=False)
    # Large-only same-device wrapper proof is separate from CPU FP32 rounding.
    c = copy.deepcopy(base).cuda() if case.size > 128 else None
    oc = optimizer(c, case, compatibility=True) if c is not None else None
    wrapper_records = []
    torch.manual_seed(case.seed + 1)
    x = torch.randn(case.rows, case.size)
    target = torch.randn_like(x)
    xa, xb = x.cuda().requires_grad_(), x.clone().requires_grad_()
    maxima, diagnostics = {}, []
    for step_index in range(3):
        # Copy actual Parameter and AdamW moments/step, not a CPU proposal from
        # the previous iteration. The native GPU state alone advances the path.
        with torch.no_grad():
            b.atoms.p.copy_(a.atoms.p.detach().cpu())
        ob.state[b.atoms.p].clear()
        ob.state[b.atoms.p].update(
            {
                name: value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else copy.deepcopy(value)
                for name, value in oa.state[a.atoms.p].items()
            }
        )
        _, before_ri, before_ro = decoded_values(b.kernel, b.atoms.p.detach())
        before_a, before_b = (
            a.atoms.p.detach().cpu().double().clone(),
            b.atoms.p.detach().clone(),
        )
        for model, opt, inputs, targets in (
            (a, oa, xa, target.cuda()),
            (b, ob, xb, target),
        ):
            opt.zero_grad(set_to_none=True)
            inputs.grad = None
            if model is b and case.size > 128:
                _chunked_loss_backward(model, inputs, targets)
            else:
                (model(inputs) - targets).square().mean().backward()
        grad_a = a.atoms.p.grad.detach().cpu().double()
        grad_b = b.atoms.p.grad.detach().clone()
        pre_grad_error = error(grad_a, grad_b)
        if oc is not None:
            _copy_optimizer_inputs(oa, a.atoms.p, oc, c.atoms.p)
            optimizer_bitgate(oa, a.atoms.p, oc, c.atoms.p)
        oa.step()
        ob.step()
        if oc is not None:
            oc.step()
            wrapper_records.append(optimizer_bitgate(oa, a.atoms.p, oc, c.atoms.p))
        pairs = {"parameters": (a.atoms.p, b.atoms.p), "dX": (xa.grad, xb.grad)}
        for key in ("exp_avg", "exp_avg_sq", "step"):
            pairs[key] = (oa.state[a.atoms.p][key], ob.state[b.atoms.p][key])
        report = {
            "step": step_index + 1,
            "reference_dtype": "float32",
            "actual_dtype": "float32",
            "preupdate_gradient_error": pre_grad_error,
            "fields": {},
        }
        failures = ["preupdate_all_dP"] if pre_grad_error["max_abs"] > 5e-4 else []
        for name, (actual, expected) in pairs.items():
            quantization = (
                parameter_quantization_gate(actual, expected, large=case.size > 128)
                if name == "parameters"
                else None
            )
            actual, expected = (
                actual.detach().cpu().double(),
                expected.detach().cpu().double(),
            )
            delta = (actual - expected).abs()
            flat_index = int(delta.reshape(-1).argmax())
            index = list(torch.unravel_index(torch.tensor(flat_index), delta.shape))
            index = [int(i) for i in index]
            loc = tuple(index)
            value = float(delta[loc])
            maxima[name] = max(maxima.get(name, 0), value)
            detail = {
                **error(actual, expected),
                "worst_index": index,
                "actual": float(actual[loc]),
                "expected": float(expected[loc]),
            }
            if delta.ndim == 2:
                detail["per_column_max_abs"] = delta.max(dim=0).values.tolist()
            if name == "parameters":
                detail.update(
                    {
                        "float_quantization": quantization,
                        "before_actual": float(before_a[loc]),
                        "before_expected": float(before_b[loc]),
                        "gradient_actual": float(grad_a[loc]),
                        "gradient_expected": float(grad_b[loc]),
                        "gradient_per_column_max_abs": (grad_a - grad_b)
                        .abs()
                        .max(dim=0)
                        .values.tolist(),
                        "gradient_actual_row": grad_a[index[0]].tolist(),
                        "gradient_expected_row": grad_b[index[0]].tolist(),
                        "parameter_actual_row": actual[index[0]].tolist(),
                        "parameter_expected_row": expected[index[0]].tolist(),
                    }
                )
            report["fields"][name] = detail
            if (
                quantization["status"] != "PASS"
                if name == "parameters"
                else value > 5e-4
            ):
                failures.append(name)
        gradient_delta = (grad_a - grad_b).abs()
        gradient_flat = int(gradient_delta.reshape(-1).argmax())
        row, column = divmod(gradient_flat, grad_a.shape[1])
        report["worst_gradient_before_update"] = {
            "index": [row, column],
            "actual": float(grad_a[row, column]),
            "expected": float(grad_b[row, column]),
            "radius_input": float(before_ri[row]),
            "radius_output": float(before_ro[row]),
            "selected_level_input": max(0, math.ceil(math.log2(float(before_ri[row])))),
            "selected_level_output": max(
                0, math.ceil(math.log2(float(before_ro[row])))
            ),
        }
        diagnostics.append(report)
        if failures:
            raise AssertionError(
                {
                    "gate": "matched_state_optimizer_steps",
                    "failed_fields": failures,
                    "max_abs": maxima,
                    "step_diagnostics": diagnostics,
                }
            )
    return {
        "status": "PASS",
        "steps": 3,
        "max_abs": maxima,
        "scope": "3 matched-state CPU FP32 reference steps along native GPU trajectory",
        "reference": "public CPU FP32 Fourier/CSTOptimizer; Parameter and AdamW state reset from GPU before each step",
        "step_diagnostics": diagnostics,
        **(
            {
                "gpu_compatibility_optimizer_bitgate": {
                    "status": "PASS",
                    "steps": len(wrapper_records),
                    "scope": "same GPU P/gradients/Adam states; CSTAdamW versus CSTOptimizer(AdamW), same explicit native Polar Plan; byte-exact outputs",
                    "fused": case.optimizer.fused,
                    "step_fields": wrapper_records,
                }
            }
            if wrapper_records
            else {}
        ),
    }


def correctness(args, run):
    plan = run.entry(args.plan_id).plan
    if run.case.size > 128:
        base, x, _, dy = fixture(run.case)
        small, indices = stratified_subset(base)
        sx, sdy = x, dy
    else:
        small, sx, _, sdy = fixture(replace(run.case, atoms=8))
        indices = None
    independent = check(
        configure(small.cuda(), plan), sx.cuda(), sdy.cuda(), independent=True
    )
    if indices is not None:
        independent["current_fixture_atom_indices"] = indices
    else:
        base, x, _, dy = fixture(run.case)
    model = configure(copy.deepcopy(base).cuda(), plan)
    proof = check(model, x.cuda(), dy.cuda())
    matched = matched_steps(base, run.case, plan)
    wrapper = matched.pop("gpu_compatibility_optimizer_bitgate", None)
    return {
        **proof,
        "independent_quadrature_small_atoms": independent,
        "matched_state_optimizer_steps": matched,
        **({"gpu_compatibility_optimizer_bitgate": wrapper} if wrapper else {}),
    }


def distribution(model):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )
    from torchcst._backends.torch.parameterizations.radius_cell_mass import (
        decoded_values,
    )

    _, ri, ro = decoded_values(model.kernel, model.atoms.p)
    plan = hierarchy_from_operator(model.execution_declaration())
    values = []
    for r, axis in zip((ri, ro), (plan.input, plan.output)):
        r = r.detach()
        levels = torch.ceil(torch.log2(r.detach().clamp_min(1))).long()
        values.append(
            {
                "min": float(r.min()),
                "max": float(r.max()),
                "mean": float(r.mean()),
                "radii": r.detach().cpu().tolist(),
                "level_counts": torch.bincount(
                    levels, minlength=max(b.level for b in axis.banks) + 1
                ).tolist(),
                "bank_counts": [
                    int(
                        (
                            (levels == b.level)
                            & ((2 * r / b.tile_sites).ceil() + 1 == b.k)
                        ).sum()
                    )
                    for b in axis.banks
                ],
            }
        )
    return values


def measure_case(args, run):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    case = run.case
    base, xx, target, dy = fixture(case)
    hashes = {
        "p": tensor_hash(base.atoms.p),
        "x": tensor_hash(xx),
        "target": tensor_hash(target),
    }
    initial_distribution = distribution(base)
    dense = args.worker == "dense"
    if dense:
        model = torch.nn.Linear(case.size, case.size, bias=False)
        with torch.no_grad():
            identity = torch.eye(case.size)
            weight = (
                chunked_reference(base, identity) if case.size > 128 else base(identity)
            )
            model.weight.copy_(weight.T)
        model = model.cuda()
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=case.optimizer.lr,
            weight_decay=case.optimizer.weight_decay,
            fused=case.optimizer.fused,
            capturable=False,
        )
    else:
        model = configure(base.cuda(), run.entry(args.plan_id).plan)
        opt = optimizer(model, case)
    x, target = xx.cuda().requires_grad_(), target.cuda()
    proof = None if dense else check(model, x, dy.cuda())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(case.warmup):
        train(model, opt, x, target)
    samples = []
    for _ in range(case.rounds):
        torch.cuda.synchronize()
        start = time.perf_counter()
        train(model, opt, x, target)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    peak = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
    post = None if dense else check(model, x, dy.cuda())
    updated_distribution = None if dense else distribution(model)
    phases = None
    if args.phase_diagnostics:
        phases = {"forward_loss": [], "backward": [], "optimizer": []}
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            x.grad = None
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss = (model(x) - target).square().mean()
            torch.cuda.synchronize()
            phases["forward_loss"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            phases["backward"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            phases["optimizer"].append((time.perf_counter() - start) * 1000)
        phases = {
            name: {"median_ms": statistics.median(values), "samples_ms": values}
            for name, values in phases.items()
        }
    return {
        "status": "PASS",
        "reference": "dense_linear" if dense else args.plan_id,
        "eager": {"median_ms": statistics.median(samples), "samples_ms": samples},
        "graph": None,
        "initial_p_sha256": hashes["p"],
        "initial_inputs": {"x_sha256": hashes["x"], "target_sha256": hashes["target"]},
        "peak_allocated_training_bytes": peak[0],
        "peak_reserved_training_bytes": peak[1],
        "total_gpu_process_bytes": None,
        "optimizer": asdict(case.optimizer),
        "optimizer_implementation": f"{type(opt).__module__}.{type(opt).__qualname__}",
        "optimizer_effective_groups": [
            {
                "lr": group["lr"],
                "weight_decay": group.get("weight_decay", 0.0),
                "amplitude_weight_decay": group.get("amplitude_weight_decay", 0.0),
            }
            for group in opt.param_groups
        ],
        "optimizer_policy": "case weight_decay is the requested coefficient: CST uses raw weight_decay=0 and amplitude_weight_decay; dense uses ordinary AdamW weight_decay; check_finite=False; live configuration validation; explicit radius fused update",
        "scope": "complete eager forward/loss/backward/optimizer; dX included; FP32 IEEE",
        "parameter_payload_bytes": sum(
            p.numel() * p.element_size() for p in model.parameters()
        ),
        "optimizer_payload_bytes": sum(
            v.numel() * v.element_size()
            for state in opt.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        ),
        "memory_scope": "resident model, gradients, AdamW state and eager temporaries; no Graph; process usage unmeasured",
        "phase_ms_separate_instrumented_steps": phases,
        "phase_diagnostics_scope": "separate synchronized eager steps; never summed or subtracted from complete-step timing",
        "radius_diagnostics": {
            "initial": initial_distribution,
            "updated": updated_distribution,
            "correctness_initial": proof,
            "correctness_updated": post,
        },
    }
