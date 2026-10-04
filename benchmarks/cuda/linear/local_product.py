"""Research plans for the existing Linear runner; no production registration."""

from dataclasses import asdict, dataclass
from functools import lru_cache

import torch

from torchcst._backends.cuda.algorithm import Algorithm
from torchcst._backends.cuda.algorithms.local_product.contract import Domain
from torchcst._backends.cuda.algorithms.local_product.recipe import Recipe
from torchcst._backends.cuda.schema import SupportResult
from torchcst.operators.spec import ChartPairSpec, OperatorSpec

SEMANTICS = "local_polar_product.normalized_triweight.shared_width.v1"


@dataclass(frozen=True)
class LocalRecipe(Recipe):
    route: str = "fused"

    def __post_init__(self):
        if type(self.rho_upper) not in (tuple, list) or any(
            type(x) not in (int, float) for x in self.rho_upper
        ):
            raise ValueError("requires a numeric boundary array")
        object.__setattr__(self, "rho_upper", tuple(self.rho_upper))
        super().__post_init__()
        if self.route not in (
            "fused",
            "saved",
            "torch",
            "support",
            "polar",
            "polar_saved",
            "hybrid",
            "hybrid_support",
            "hybrid_three",
            "hybrid_singletons",
            "hybrid_packed",
            "hybrid_persistent",
            "polar_support",
            "polar_support_saved",
        ):
            raise ValueError("unknown local H route")
        if (
            self.route
            in (
                "polar",
                "polar_saved",
                "hybrid",
                "hybrid_support",
                "hybrid_three",
                "hybrid_singletons",
                "hybrid_packed",
                "hybrid_persistent",
                "polar_support",
                "polar_support_saved",
            )
            and self.pack
        ):
            raise ValueError("fused polar routes require canonical order")
        if self.route in (
            "hybrid_three",
            "hybrid_singletons",
            "hybrid_packed",
            "hybrid_persistent",
        ) and (len(self.rho_upper) < 3 or self.rho_upper[0] != 1.0):
            raise ValueError("three-band route requires boundaries [1, mid, ...]")
        if (
            type(self.pack) is not bool
            or type(self.atom_block) is not int
            or type(self.batch_block) is not int
        ):
            raise ValueError("invalid local execution settings")


@lru_cache(maxsize=8)
def operator_spec(size, minimum=1.0, birth=1.0, maximum=16.0):
    from benchmarks.cuda.linear.fixtures import local_product_state

    domain = Domain(size, size)
    charts = domain.charts()
    return OperatorSpec(
        layout=ChartPairSpec(
            input_chart=charts[0].declaration(), output_chart=charts[1].declaration()
        ),
        kernel=local_product_state(minimum=minimum, birth=birth, maximum=maximum).spec,
    )


def fixture_state(case):
    from benchmarks.cuda.linear.fixtures import local_product_state

    if case.widths is None:
        return local_product_state(birth=1)
    w = case.widths
    return local_product_state(minimum=w.minimum, birth=w.birth, maximum=w.maximum)


def fixture_operator(case, size=None):
    n = case.size if size is None else size
    if case.widths is None:
        return operator_spec(n)
    w = case.widths
    return operator_spec(n, w.minimum, w.birth, w.maximum)


def _matches(operator):
    bounds = getattr(operator.kernel.parameterization, "input_bounds", None)
    if bounds is None:
        return False
    return operator == operator_spec(
        operator.in_features, bounds.minimum, bounds.birth, bounds.maximum
    )


@lru_cache(maxsize=32)
def runtime(operator, device):
    """Only fixed mathematical state is cached; inputs/support/H never are.

    Initialize at the model configuration boundary before CUDA Graph capture.
    """
    from torchcst._backends.cuda.algorithms.local_product.preparation import (
        validate_state,
    )
    from torchcst.kernels.state import KernelState

    if not _matches(operator):
        raise ValueError("unrecognized local product operator")
    value = KernelState(operator.kernel).to(device=device)
    validate_state(value)
    return value, Domain(operator.in_features, operator.out_features)


@dataclass(frozen=True)
class LocalAlgorithm(Algorithm[LocalRecipe]):
    id: str = "research_local_product"
    revision: str = "v2"
    operation_id: str = "linear"
    semantics_id: str = SEMANTICS
    recipe_type: type = LocalRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not LocalRecipe:
            raise TypeError("requires LocalRecipe")
        LocalRecipe(**asdict(recipe))

    def supports(self, context, recipe):
        reasons = []
        if context.operator.in_features not in (16, 32, 64, 128):
            reasons.append("small research sizes are 16/32/64/128")
        elif not _matches(context.operator):
            reasons.append("requires the fixed shared-width normalized polar contract")
        if context.dtype != torch.float32 or context.device.type != "cuda":
            reasons.append("requires CUDA FP32")
        if not 1 <= context.m <= 64 or context.parameter_dim != 4:
            reasons.append("requires batch 1..64 and polar [A,4]")
        if context.precision.autocast or context.precision.allow_tf32:
            reasons.append("requires IEEE FP32 without autocast")
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        # Torch baseline and autograd decode/sort allocations are not bounded here.
        # Measured full-step CUDA peaks are reported by the runner.
        return None

    def execute(self, *, x, parameters, operator, recipe, persistent_layout=None):
        from benchmarks.cuda.linear.fixtures import local_product_dense_factors
        from torchcst._backends.cuda.algorithms.local_product.executor import local_h

        value, domain = runtime(operator, x.device)
        if recipe.route == "torch":
            v, u = local_product_dense_factors(parameters, value, domain)
            return (x @ v) @ u.T
        if recipe.route == "hybrid_persistent" and persistent_layout is None:
            raise ValueError("persistent route requires model-owned layout state")
        return local_h(
            x,
            parameters,
            value,
            domain,
            saved=recipe.route in ("saved", "polar_saved", "polar_support_saved"),
            sparse=recipe.route
            in (
                "support",
                "hybrid_support",
                "hybrid_three",
                "hybrid_singletons",
                "hybrid_packed",
                "hybrid_persistent",
                "polar_support",
                "polar_support_saved",
            ),
            fused_polar=recipe.route
            in (
                "polar",
                "polar_saved",
                "hybrid",
                "hybrid_support",
                "hybrid_three",
                "hybrid_singletons",
                "hybrid_packed",
                "hybrid_persistent",
                "polar_support",
                "polar_support_saved",
            ),
            hybrid=recipe.route
            in (
                "hybrid",
                "hybrid_support",
                "hybrid_three",
                "hybrid_singletons",
                "hybrid_packed",
                "hybrid_persistent",
            ),
            three_band=recipe.route
            in (
                "hybrid_three",
                "hybrid_singletons",
                "hybrid_packed",
                "hybrid_persistent",
            ),
            singletons=recipe.route
            in ("hybrid_singletons", "hybrid_packed", "hybrid_persistent"),
            tile_packed=recipe.route in ("hybrid_packed", "hybrid_persistent"),
            persistent_layout=persistent_layout,
            support_only=recipe.route in ("polar_support", "polar_support_saved"),
            recipe=recipe,
        )


def initialize(case):
    """Same CPU atoms for every candidate; width follows the production map."""
    import math

    gen = torch.Generator().manual_seed(case.seed)
    direction = torch.rand(case.atoms, generator=gen) * 0.4 - 0.2
    initial_rho = {
        "rho1": 1,
        "rho1_5": 1.5,
        "rho2": 2,
        "rho3": 3,
        "rho4": 4,
        "rho8": 8,
        "rho16": 16,
    }.get(case.profile)
    alpha = (
        math.log(initial_rho) / math.log(16)
        if initial_rho
        else {
            "sharp": 0.0,
            "few": math.log(2) / math.log(16),
            "broad": math.log(3) / math.log(16),
            "wide": 1.0,
        }.get(case.profile)
    )
    radius = (
        (1 + 3 * torch.rand(case.atoms, generator=gen)).sqrt()
        if alpha is None
        else torch.full((case.atoms,), math.sqrt(1 + 3 * alpha))
    )
    polar = (
        torch.stack((direction, (1 - direction.square()).sqrt()), 1) * radius[:, None]
    )
    center = torch.rand(case.atoms, 2, generator=gen) * (case.size - 1)
    if case.widths is not None:
        w = case.widths
        # Same centers/amplitudes across mixtures. Only the initial radii differ.
        if w.center_jitter is not None:
            center = (
                center.round()
                + (torch.rand(case.atoms, 2, generator=gen) * 2 - 1) * w.center_jitter
            )
            center.clamp_(0, case.size - 1)
        ends = [
            round(case.atoms * sum(w.fractions[: k + 1])) for k in range(len(w.rho))
        ]
        desired = torch.empty(case.atoms)
        begin = 0
        for rho, end in zip(w.rho, ends):
            desired[begin:end] = rho
            begin = end
        desired = desired[torch.randperm(case.atoms, generator=gen)]
        # Account for the production amplitude-dependent upper bound.
        direction = direction.sign() * 0.6
        upper = w.minimum + (w.maximum - w.minimum) / (1 + (direction / 1e6).square())
        upper.clamp_(min=w.birth)
        alpha = (desired / w.birth).log() / (upper / w.birth).log()
        radius = (1 + 3 * alpha.clamp(0, 1)).sqrt()
        polar = (
            torch.stack((direction, torch.full_like(direction, 0.8)), 1)
            * radius[:, None]
        )
    if initial_rho is not None:
        # Matched amplitude/direction and centers across widths. Only radius
        # differs. These labels set initialization; training does not freeze sigma.
        polar[:, 0] = direction.sign() * 0.6 * radius
        polar[:, 1] = 0.8 * radius
    if case.profile == "sharp":
        # FP32 unit-radius pair: random unit-vector rounding can raise alpha
        # above zero and create tiny positive neighbors at sigma > spacing.
        polar[:, 0] = direction.sign() * 0.6
        polar[:, 1] = 0.8
        center = center.round()
    return torch.cat((polar, center), 1)


def correctness(args, run, model_type):
    """Independent FP64 scalar oracle, including live/floor singleton gradients."""
    from benchmarks.cuda.linear.check_normalized import check
    from torchcst._backends.cuda.algorithms.local_product.polar import graph_update
    from torchcst._backends.torch.kernels import execution

    domain = Domain(16, 16)
    value = fixture_state(run.case).cuda()
    p = initialize(run.case)[: min(run.case.atoms, 13)].cuda()
    # Include exact singleton, two sites, empty input, and norm-floor support.
    p[:4, :2] = p.new_tensor([[0.1, (0.99) ** 0.5]] * 4)
    p[:4, 2:] = p.new_tensor([[4, 5], [4.5, 5.5], [-50, 5], [-0.999, -0.999]])
    if run.case.widths is not None:
        p[3, 2:] = -0.999 * run.case.widths.birth
    model = model_type(p, fixture_operator(run.case, 16), run.entry(args.plan_id).plan)
    gen = torch.Generator().manual_seed(run.case.seed)
    x = torch.randn(7, 16, generator=gen).cuda().requires_grad_()
    dy = torch.randn(7, 16, generator=gen).cuda()
    y = model(x)
    tp, tx = p.double().requires_grad_(), x.detach().double().requires_grad_()
    from torchcst._backends.cuda.algorithms.local_product.preparation import decode

    q = decode(fixture_state(run.case).double().cuda(), tp)
    j = torch.arange(16, device="cuda", dtype=torch.float64)
    truth = torch.zeros_like(tx)
    for amp, inv, ci, co in q:
        v = (1 - (j - ci).square() * inv).clamp_min(0).pow(3)
        u = (1 - (j - co).square() * inv).clamp_min(0).pow(3)
        v, u = v / v.norm().clamp_min(1e-6), u / u.norm().clamp_min(1e-6)
        truth = truth + amp * (tx * v).sum(1)[:, None] * u[None]
    ga = torch.autograd.grad(y, (x, model.p), dy)
    gt = torch.autograd.grad(truth, (tx, tp), dy.double())
    actual_update = graph_update(
        value, p, -run.case.optimizer.lr * ga[1], step_size=run.case.optimizer.lr
    )
    expected_update = execution.apply_parameter_update(
        value,
        *domain.charts(device="cuda"),
        p,
        -run.case.optimizer.lr * ga[1],
        step_size=run.case.optimizer.lr,
    )
    return {
        "status": "PASS",
        "scope": "independent FP64 scalar Y/dX/all atom gradients; production polar update",
        "y": check(y, truth, tol=4e-4),
        "dx": check(ga[0], gt[0], tol=4e-4),
        "dp": check(ga[1], gt[1], tol=4e-4),
        "polar_update": check(actual_update, expected_update, tol=2e-6),
    }


def optimizer_step(model, optimizer, *, step_size):
    """Graph-safe specialization, checked against CSTOptimizer for this fixture."""
    from torchcst._backends.cuda.algorithms.local_product.polar import graph_update

    previous = model.p.detach().clone()
    optimizer.step()
    with torch.no_grad():
        model.p.copy_(
            graph_update(
                model.local_state, previous, model.p - previous, step_size=step_size
            )
        )
