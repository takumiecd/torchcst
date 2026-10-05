"""Common metadata guards for the full and window implementations."""

from torchcst._backends.algorithm import Algorithm, RecipeT
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.window.recipe import (
    WindowRecipe,
)
from torchcst._backends.schema import SupportResult
from torchcst._backends.torch.algorithms.normalized_radial.layout import geometry
from torchcst.operators.context import LinearContext as DispatchContext

from ..constraints import routing_reasons


def _supports(context, recipe):
    reasons = []
    if not isinstance(context, DispatchContext):
        return SupportResult(("requires a LinearContext",))
    try:
        op = geometry(context.operator)
    except (TypeError, ValueError) as error:
        return SupportResult((str(error),))
    if context.parameter_dim != 5:
        reasons.append("normalized Strip requires parameters [atoms, 5]")
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


class _NormalizedAlgorithm(Algorithm[RecipeT]):
    def matches(self, context, recipe):
        try:
            op = geometry(context.operator)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            context.device.type == "cuda"
            and str(context.dtype) == "torch.float32"
            and not routing_reasons(op)
        )

    def supports(self, context: DispatchContext, recipe: RecipeT) -> SupportResult:
        return _supports(context, recipe)

    def workspace_bound(self, context: DispatchContext, recipe: RecipeT) -> int | None:
        # Weight/halo formulas exclude norm/routing/saved state and vendor workspace.
        return None
