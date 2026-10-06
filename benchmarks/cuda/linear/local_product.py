"""Research plans for the existing Linear runner; no production registration."""

from dataclasses import asdict, dataclass
from functools import lru_cache

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.cuda.algorithms.linear.local_product.contract import Domain
from torchcst._backends.cuda.algorithms.linear.local_product.recipe import Recipe
from torchcst._backends.schema import SupportResult
from torchcst.operators.execution import LinearInputs, linear_execution
from torchcst.operators.spec import ChartPairSpec, OperatorSpec

REUSE_SUFFIXES = (
    "_ordered_reuse_tile32",
    "_ordered_reuse_tile64",
    "_ordered_reuse_tile32_split2",
    "_ordered_reuse_tile64_split2",
    "_ordered_reuse_param2",
    "_ordered_reuse_tile32_param2",
    "_ordered_reuse_paramatom16",
    "_ordered_reuse_paramatom16_param4",
    "_ordered_reuse_paramatom32_param4",
    "_ordered_reuse_paramatom16_param2",
    "_ordered_reuse_paramatom16_param4_param2",
    "_ordered_reuse_paramatom16_param4_param2_split2",
)

CACHE_SUFFIXES = (
    "_ordered_cache_copy4",
    "_ordered_cache_copy8",
    "_ordered_cache_range4",
    "_ordered_cache_range8",
    "_ordered_cache_compact_copy8",
    "_ordered_cache_compact_range8",
    "_ordered_cache_validated_copy8",
    "_ordered_cache_validated_range8",
    "_ordered_cache_gather_copy8",
    "_ordered_cache_gather_range8",
    "_ordered_cache_repair4_copy8",
    "_ordered_cache_repair8_copy8",
    "_ordered_cache_gather_param2_copy8",
    "_ordered_cache_repair4_param2_copy8",
    "_ordered_cache_repair8_param2_copy8",
    "_ordered_cache_gather_paramatom16_param4_param2_copy8",
    "_ordered_cache_repair4_paramatom16_param4_param2_copy8",
    "_ordered_cache_repair8_paramatom16_param4_param2_copy8",
)

SEMANTICS = "local_polar_product.normalized_triweight.shared_width.v1"


@dataclass(frozen=True)
class LocalRecipe(Recipe):
    route: str = "fused"

    @property
    def reuse_layout(self):
        return self.route.endswith(REUSE_SUFFIXES)

    @property
    def cached_order(self):
        return self.route.endswith(CACHE_SUFFIXES)

    @property
    def cache_repair_rounds(self):
        if "_ordered_cache_repair4" in self.route:
            return 4
        if "_ordered_cache_repair8" in self.route:
            return 8
        return 0

    @property
    def gather_validation_key(self):
        return self.cache_repair_rounds > 0 or self.route.endswith(
            (
                "_ordered_cache_gather_copy8",
                "_ordered_cache_gather_range8",
                "_ordered_cache_gather_param2_copy8",
                "_ordered_cache_gather_paramatom16_param4_param2_copy8",
            )
        )

    @property
    def validate_cached_order(self):
        return self.gather_validation_key or self.route.endswith(
            ("_ordered_cache_validated_copy8", "_ordered_cache_validated_range8")
        )

    @property
    def compact_cached_order(self):
        return self.gather_validation_key or self.route.endswith(
            (
                "_ordered_cache_compact_copy8",
                "_ordered_cache_compact_range8",
                "_ordered_cache_validated_copy8",
                "_ordered_cache_validated_range8",
            )
        )

    @property
    def base_route(self):
        for suffix in (
            REUSE_SUFFIXES
            + CACHE_SUFFIXES
            + (
                "_ordered_prep_vector8",
                "_ordered_prep_vector",
                "_ordered_prep_copy8",
                "_ordered_prep_copy",
                "_ordered_prep_parallel8",
                "_ordered_prep_parallel",
                "_ordered_prep_warp8",
                "_ordered_prep_range",
                "_ordered_split4_i32",
                "_ordered_band",
                "_ordered_index",
                "_ordered_cached",
                "_ordered_split2",
                "_ordered_split4",
                "_ordered",
                "_tile32",
                "_tile64",
                "_contract4",
                "_contract8",
                "_param4",
                "_vector",
                "_vector4",
                "_unroll",
                "_index16_release",
                "_index16_release_h",
                "_index16_fused",
                "_index16_local",
                "_index16",
                "_index",
            )
        ):
            if self.route.endswith(suffix):
                return self.route[: -len(suffix)]
        return self.route

    @property
    def output_block(self):
        if self.reuse_layout:
            return (
                64 if "_tile64" in self.route else 32 if "_tile32" in self.route else 16
            )
        if self.route.endswith(("_tile32", "_h32")):
            return 32
        if self.route.endswith(("_tile64", "_h64")):
            return 64
        return 16

    @property
    def recompute_h(self):
        return (
            (self.ordered_layout and not self.route.endswith("_ordered_cached"))
            or self.route.endswith("_index16_local")
            or self.base_route
            in (
                "persistent_onchip_h32",
                "persistent_onchip_h64",
            )
        )

    @property
    def recompute_param_h(self):
        return (
            self.recompute_h
            or self.base_route == "persistent_supportprep_band_recompute_vjp"
        )

    @property
    def owner_index(self):
        return self.route.endswith(
            (
                "_index",
                "_index16",
                "_index16_release",
                "_index16_release_h",
                "_index16_fused",
                "_index16_local",
                "_ordered_band",
                "_ordered_index",
            )
        )

    @property
    def index_bits(self):
        return 16 if "_index16" in self.route or self.ordered_layout else 32

    @property
    def release_forward_index(self):
        return self.route.endswith(
            (
                "_index16_release",
                "_index16_release_h",
                "_index16_fused",
                "_index16_local",
                "_ordered_band",
                "_ordered_index",
            )
        )

    @property
    def release_forward_h(self):
        return self.route.endswith(("_index16_release_h", "_ordered_cached"))

    @property
    def ordered_layout(self):
        return (
            self.reuse_layout
            or self.cached_order
            or self.route.endswith(
                (
                    "_ordered",
                    "_ordered_band",
                    "_ordered_index",
                    "_ordered_cached",
                    "_ordered_split2",
                    "_ordered_split4",
                    "_ordered_split4_i32",
                    "_ordered_prep_vector8",
                    "_ordered_prep_vector",
                    "_ordered_prep_copy8",
                    "_ordered_prep_copy",
                    "_ordered_prep_parallel8",
                    "_ordered_prep_parallel",
                    "_ordered_prep_warp8",
                    "_ordered_prep_range",
                )
            )
        )

    @property
    def order_by_position(self):
        return self.ordered_layout and not self.route.endswith("_ordered_band")

    @property
    def compact_order_key(self):
        return (
            self.reuse_layout
            or self.cached_order
            or self.route.endswith(
                (
                    "_ordered_split4_i32",
                    "_ordered_prep_vector8",
                    "_ordered_prep_vector",
                    "_ordered_prep_copy8",
                    "_ordered_prep_copy",
                    "_ordered_prep_parallel8",
                    "_ordered_prep_parallel",
                    "_ordered_prep_warp8",
                    "_ordered_prep_range",
                )
            )
        )

    @property
    def parallel_owner_ranges(self):
        return (
            self.reuse_layout
            or self.cached_order
            or self.route.endswith(
                (
                    "_ordered_prep_parallel",
                    "_ordered_prep_parallel8",
                    "_ordered_prep_range",
                    "_ordered_prep_copy",
                    "_ordered_prep_copy8",
                )
            )
        )

    @property
    def parallel_order_copy(self):
        if self.reuse_layout:
            return True
        if self.cached_order:
            return self.route.endswith(("_copy4", "_copy8"))
        return self.route.endswith(("_ordered_prep_copy", "_ordered_prep_copy8"))

    @property
    def vector_owner_ranges(self):
        return self.route.endswith(("_ordered_prep_vector", "_ordered_prep_vector8"))

    @property
    def preparation_warps(self):
        if self.reuse_layout:
            return 8
        if self.cached_order:
            return 8 if self.route.endswith(("_copy8", "_range8")) else 4
        return (
            8
            if self.route.endswith(
                (
                    "_ordered_prep_warp8",
                    "_ordered_prep_parallel8",
                    "_ordered_prep_vector8",
                    "_ordered_prep_copy8",
                )
            )
            else 4
        )

    @property
    def owner_splits(self):
        if self.reuse_layout:
            return 2 if self.route.endswith("_split2") else 4
        if self.cached_order:
            return 4
        return (
            2
            if self.route.endswith("_ordered_split2")
            else 4
            if self.route.endswith(
                (
                    "_ordered_split4",
                    "_ordered_split4_i32",
                    "_ordered_prep_parallel",
                    "_ordered_prep_vector8",
                    "_ordered_prep_vector",
                    "_ordered_prep_copy8",
                    "_ordered_prep_copy",
                    "_ordered_prep_parallel8",
                    "_ordered_prep_warp8",
                )
            )
            else 1
        )

    @property
    def parameter_splits(self):
        return (
            2
            if (self.reuse_layout or self.cached_order) and "_param2" in self.route
            else 1
        )

    @property
    def fuse_owner_index(self):
        return self.route.endswith("_index16_fused")

    @property
    def unroll_support(self):
        return self.route.endswith("_unroll")

    @property
    def vector_support(self):
        return self.route.endswith(("_vector", "_vector4"))

    @property
    def contraction_warps(self):
        if self.base_route == "persistent_supportprep_band_recompute_vjp":
            return 4
        if self.output_block != 16:
            return 4
        if self.route.endswith(("_contract4", "_vector4")):
            return 4
        if self.route.endswith("_contract8"):
            return 8
        return 0

    @property
    def parameter_warps(self):
        if self.cached_order and "_param4_" in self.route:
            return 4
        return (
            4
            if self.route.endswith(
                ("_param4", "_param4_param2", "_param4_param2_split2")
            )
            else 0
        )

    @property
    def parameter_atom_block(self):
        if (self.reuse_layout or self.cached_order) and "_paramatom16" in self.route:
            return 16
        return self.atom_block

    @property
    def execution_route(self):
        if self.ordered_layout:
            return "hybrid_packed"
        if self.base_route in (
            "persistent_supportprep",
            "persistent_saved_g",
            "persistent_supportprep_g",
            "persistent_band_dispatch",
            "persistent_supportprep_band",
            "persistent_supportprep_band_recompute_vjp",
            "persistent_onchip_h32",
            "persistent_onchip_h64",
        ):
            return "hybrid_persistent"
        return self.base_route

    @property
    def support_prepare(self):
        return self.base_route in (
            "persistent_supportprep",
            "persistent_supportprep_g",
            "persistent_supportprep_band",
            "persistent_supportprep_band_recompute_vjp",
            "persistent_onchip_h32",
            "persistent_onchip_h64",
        )

    @property
    def save_g(self):
        return self.base_route in ("persistent_saved_g", "persistent_supportprep_g")

    @property
    def band_dispatch(self):
        return self.save_g or self.base_route in (
            "persistent_band_dispatch",
            "persistent_supportprep_band",
            "persistent_supportprep_band_recompute_vjp",
            "persistent_onchip_h32",
            "persistent_onchip_h64",
        )

    def __post_init__(self):
        if type(self.route) is not str:
            raise ValueError("local execution route must be a string")
        if type(self.rho_upper) not in (tuple, list) or any(
            type(x) not in (int, float) for x in self.rho_upper
        ):
            raise ValueError("requires a numeric boundary array")
        object.__setattr__(self, "rho_upper", tuple(self.rho_upper))
        super().__post_init__()
        if (
            self.base_route == "persistent_supportprep_band_recompute_vjp"
            and self.route
            not in (
                self.base_route,
                self.base_route + "_unroll",
                self.base_route + "_index",
                self.base_route + "_index16",
                self.base_route + "_index16_release",
                self.base_route + "_index16_release_h",
                self.base_route + "_index16_fused",
                self.base_route + "_index16_local",
                self.base_route + "_ordered",
                self.base_route + "_ordered_band",
                self.base_route + "_ordered_index",
                self.base_route + "_ordered_cached",
                self.base_route + "_ordered_split2",
                self.base_route + "_ordered_split4",
                self.base_route + "_ordered_split4_i32",
                self.base_route + "_ordered_prep_vector",
                self.base_route + "_ordered_prep_vector8",
                self.base_route + "_ordered_prep_copy",
                self.base_route + "_ordered_prep_copy8",
                self.base_route + "_ordered_prep_parallel8",
                self.base_route + "_ordered_prep_parallel",
                self.base_route + "_ordered_prep_warp8",
                self.base_route + "_ordered_prep_range",
                *(self.base_route + suffix for suffix in REUSE_SUFFIXES),
                *(self.base_route + suffix for suffix in CACHE_SUFFIXES),
            )
        ):
            raise ValueError("unsupported recomputed parameter VJP variant")
        if (
            self.owner_index
            and self.base_route != "persistent_supportprep_band_recompute_vjp"
        ):
            raise ValueError("owner indexing requires recomputed parameter VJP")
        if self.route != self.base_route and self.base_route not in (
            "persistent_supportprep_band",
            "persistent_supportprep_g",
            "persistent_supportprep_band_recompute_vjp",
        ):
            raise ValueError("launch variants require prepared persistent bands")
        if self.base_route not in (
            "persistent_onchip_h32",
            "persistent_onchip_h64",
            "persistent_band_dispatch",
            "persistent_supportprep_band",
            "persistent_supportprep_band_recompute_vjp",
            "persistent_supportprep",
            "persistent_saved_g",
            "persistent_supportprep_g",
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
            self.execution_route
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
        if self.execution_route in (
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
    from torchcst._backends.cuda.algorithms.linear.local_product.preparation import (
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

    input_type: type = LinearInputs

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

    def execute(self, state, inputs):
        x, parameters, operator, binding = linear_execution(state, inputs)
        recipe = state.recipe
        persistent_layout = getattr(binding, "persistent_layout", None)
        from benchmarks.cuda.linear.fixtures import local_product_dense_factors
        from torchcst._backends.cuda.algorithms.linear.local_product.executor import (
            local_h,
        )

        value, domain = runtime(operator, x.device)
        if recipe.execution_route == "torch":
            v, u = local_product_dense_factors(parameters, value, domain)
            return (x @ v) @ u.T
        if recipe.execution_route == "hybrid_persistent" and persistent_layout is None:
            raise ValueError("persistent route requires model-owned layout state")
        return local_h(
            x,
            parameters,
            value,
            domain,
            saved=recipe.execution_route
            in ("saved", "polar_saved", "polar_support_saved"),
            sparse=recipe.execution_route
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
            fused_polar=recipe.execution_route
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
            hybrid=recipe.execution_route
            in (
                "hybrid",
                "hybrid_support",
                "hybrid_three",
                "hybrid_singletons",
                "hybrid_packed",
                "hybrid_persistent",
            ),
            three_band=recipe.execution_route
            in (
                "hybrid_three",
                "hybrid_singletons",
                "hybrid_packed",
                "hybrid_persistent",
            ),
            singletons=recipe.execution_route
            in ("hybrid_singletons", "hybrid_packed", "hybrid_persistent"),
            tile_packed=recipe.execution_route
            in ("hybrid_packed", "hybrid_persistent"),
            persistent_layout=persistent_layout,
            support_only=recipe.execution_route
            in ("polar_support", "polar_support_saved"),
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
    from torchcst._backends.cuda.algorithms.polar_update.executor import fused_update_
    from torchcst._backends.torch.algorithms.polar_update.executor import graph_update
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
    from torchcst._backends.cuda.algorithms.linear.local_product.preparation import (
        decode,
    )

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
    displacement = -run.case.optimizer.lr * ga[1]
    if args.polar_update == "fused":
        actual_update = p + displacement
        # Optimizer integration receives the already-rounded proposal.
        displacement = actual_update - p
        fused_update_(value, p, actual_update, step_size=run.case.optimizer.lr)
    else:
        actual_update = graph_update(
            value, p, displacement, step_size=run.case.optimizer.lr
        )
    expected_update = execution.apply_parameter_update(
        value,
        *domain.charts(device="cuda"),
        p,
        displacement,
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


def measure_prepared_forward(case, recipe=None):
    """Fixed initial-state H+Y diagnosis; excludes preparation and training.

    Reuse the backend's kernels without a replacement mathematical algorithm.
    Capture100 repeated forwards in one graph to avoid Python replay gaps.
    Dense uses the same initial CST operator, materialized only as a reference.
    """
    import hashlib
    import statistics

    import triton as tr

    from benchmarks.cuda.linear.fixtures import local_product_dense_factors
    from benchmarks.cuda.linear.run import generate_inputs
    from torchcst._backends.cuda.algorithms.linear.local_product import kernels
    from torchcst._backends.cuda.algorithms.linear.local_product.executor import (
        _packed_fused,
        prepare_metadata,
        tile_layout,
    )
    from torchcst._backends.cuda.algorithms.linear.local_product.persistent import (
        PersistentLayout,
    )
    from torchcst._backends.cuda.algorithms.linear.local_product.preparation import (
        polar_scalars,
    )

    if recipe is not None and recipe.execution_route not in (
        "polar_support_saved",
        "hybrid_packed",
        "hybrid_persistent",
    ):
        raise ValueError("prepared diagnostic supports saved/packed/persistent routes")
    source = initialize(case).cuda()
    state, domain = fixture_state(case).cuda(), Domain(case.size, case.size)
    x = generate_inputs(case.seed, case.rows, case.size)[0].cuda()
    v, u = local_product_dense_factors(
        source.double(), fixture_state(case).double().cuda(), domain
    )
    weight = u @ v.T
    expected = x.double() @ weight.T
    initial_hash = hashlib.sha256(source.cpu().numpy().tobytes()).hexdigest()
    del v, u
    route = recipe.route if recipe is not None else "dense_same_operator"
    if recipe is None:
        dense_weight = weight.float().contiguous()

        def forward():
            return x @ dense_weight.T
    else:
        a, b, n = len(source), len(x), case.size
        hybrid = recipe.execution_route != "polar_support_saved"
        packed = prepare_metadata(
            source,
            domain,
            sparse=True,
            scalars=polar_scalars(state),
            support_bounded=recipe.support_prepare,
        )
        h = source.new_empty((0,)) if recipe.recompute_h else source.new_empty((b, a))
        ends = None
        if recipe.execution_route == "hybrid_persistent":
            layout = PersistentLayout(source, state, domain, recipe)
            views, orders, offsets, ends = layout.refresh(packed)
        elif hybrid:
            views, orders, offsets = tile_layout(packed, domain, recipe)

        def forward():
            if not recipe.recompute_h:
                kernels.save_h[
                    (tr.cdiv(b, recipe.batch_block), tr.cdiv(a, recipe.atom_block))
                ](
                    x,
                    packed,
                    h,
                    b,
                    n,
                    a,
                    domain.spacing,
                    domain.input_origin,
                    domain.input_start,
                    max(16, tr.next_power_of_2(n)),
                    recipe.batch_block,
                    recipe.atom_block,
                    hybrid,
                    recipe.rho_upper[1],
                    not hybrid,
                    THREE_BAND=hybrid,
                    SINGLETON_FAST=hybrid,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            if hybrid:
                return _packed_fused(
                    x,
                    views[0],
                    h,
                    orders[0],
                    offsets[0],
                    domain,
                    recipe,
                    ends=ends[0] if ends is not None else None,
                    canonical_atoms=a if ends is not None else 0,
                )
            y = x.new_empty((b, n))
            kernels.from_h[(tr.cdiv(b, recipe.batch_block),)](
                h,
                packed,
                y,
                b,
                n,
                a,
                domain.spacing,
                domain.output_origin,
                domain.output_start,
                max(16, tr.next_power_of_2(n)),
                recipe.batch_block,
                recipe.atom_block,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return y

    with torch.no_grad():
        for _ in range(3):
            result = forward()
        torch.testing.assert_close(result.double(), expected, atol=4e-4, rtol=4e-4)
        graph = torch.cuda.CUDAGraph()
        repetitions = 100
        with torch.cuda.graph(graph):
            for _ in range(repetitions):
                result = forward()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        samples = []
        for _ in range(case.rounds):
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / repetitions)
        torch.testing.assert_close(result.double(), expected, atol=4e-4, rtol=4e-4)
    return {
        "status": "PASS",
        "route": route,
        "scope": "fixed initial-state prepared forward; includes H producer and Y; excludes normalization/layout/backward/loss/optimizer",
        "timing_method": "GPU events; 100 repeated forwards within one CUDA Graph; warm reused inputs; not complete-step latency",
        "reference_scope": "FP64 Torch normalized factors; independent full-shape scalar gradient gates are separate",
        "initial_p_sha256": initial_hash,
        "size": case.size,
        "batch": case.rows,
        "atoms": case.atoms,
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "max_abs_y": float((result.double() - expected).abs().max()),
    }
