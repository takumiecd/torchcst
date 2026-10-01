"""Normalized Strip registration metadata; GPU modules remain lazily imported."""

from ...registry import AlgorithmEntry, Registry
from ...schema import OPERATION, SEMANTICS, FullRecipe, SupportResult, WindowRecipe
from .constraints import routing_reasons


def _validate_full(recipe):
    # Initial registration exposes precisely the existing production configuration.
    if type(recipe) is not FullRecipe or (
        recipe != FullRecipe()
        or type(recipe.atom_num_warps) is not int
        or any(
            type(getattr(recipe, field)) is not bool
            for field in (
                "sorted_forward",
                "sorted_backward",
                "enable_fp_fusion",
                "saved_support_flags",
                "tuple_grads",
            )
        )
    ):
        raise ValueError(
            "unvalidated full recipe; only the legacy default is registered"
        )


def _validate_window(recipe):
    if type(recipe) is not WindowRecipe or (
        recipe != WindowRecipe()
        or type(recipe.window_rows) is not int
        or type(recipe.enable_fp_fusion) is not bool
    ):
        raise ValueError("unvalidated window recipe; only rows512 is registered")


def _supports(context, recipe):
    op = context.operator
    reasons = []
    if context.device.type != "cuda":
        reasons.append("normalized Strip GPU algorithms require CUDA")
    if str(context.dtype) != "torch.float32":
        reasons.append("normalized Strip CUDA requires float32")
    if context.deterministic:
        reasons.append(
            "normalized Strip CUDA requires nondeterministic atomic accumulation"
        )
    if context.precision.autocast:
        reasons.append("normalized Strip CUDA does not support autocast")
    if context.precision.allow_tf32:
        reasons.append("normalized Strip CUDA requires TF32 to be disabled")
    if len(context.input_shape) != 2 or context.input_strides != (op.k, 1):
        reasons.append("executor requires a contiguous flattened [M, K] input")
    if context.m == 0 or context.atom_count == 0:
        reasons.append("empty inputs/atoms use the module's zero-operator path")
    reasons.extend(routing_reasons(op))
    if type(recipe) is WindowRecipe:
        if op.n < 32 or op.n % 32:
            reasons.append("window requires output rows divisible by 32")
        if any(o * 4 != round(o * 4) for o in op.origin):
            reasons.append("window requires quarter-grid origins")
        if any(
            abs(o) + n * s > 16384 for o, n, s in zip(op.origin, op.sizes, op.spacing)
        ):
            reasons.append("window requires bounded chart coordinates")
    return SupportResult(tuple(reasons))


def _workspace_unknown(context, recipe):
    # Weight/halo formulas exclude norm/routing/saved state and vendor workspace.
    # Do not call them a bound on all algorithm-managed allocation.
    return None


def _load_full():
    from .impl import execute_full

    return execute_full


def _load_window():
    from .impl import execute_window

    return execute_window


def make_registry():
    registry = Registry()
    for name, recipe_type, validate, loader in (
        ("normalized_full", FullRecipe, _validate_full, _load_full),
        ("normalized_window", WindowRecipe, _validate_window, _load_window),
    ):
        registry.register(
            AlgorithmEntry(
                id=name,
                revision="v1",
                operation_id=OPERATION,
                semantics_id=SEMANTICS,
                recipe_type=recipe_type,
                validate_recipe=validate,
                supports=_supports,
                workspace_bound=_workspace_unknown,
                load_executor=loader,
            )
        )
    return registry


REGISTRY = make_registry()
