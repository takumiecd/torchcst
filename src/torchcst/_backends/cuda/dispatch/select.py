"""Initial policy reproduces legacy full/window selection, with explicit diagnostics.

This is a compatibility policy, not a learned performance tree or certification.
"""

from torchcst._backends.cuda.algorithms.normalized_euclidean_strip import REGISTRY
from torchcst._backends.cuda.schema import (
    DispatchDecision,
    ExecutionPlan,
    FullRecipe,
    WindowRecipe,
)

FULL = ExecutionPlan("normalized_full", "v1", FullRecipe())
WINDOW = ExecutionPlan("normalized_window", "v1", WindowRecipe())
POLICY_REVISION = "normalized-legacy-v1"


def select_normalized(context, *, memory="full", registry=REGISTRY):
    if memory not in ("full", "window"):
        raise ValueError("memory must be 'full' or 'window'")
    preferred = WINDOW if memory == "window" else FULL
    fallback_reason = None
    try:
        algorithm = registry.validate(preferred, context)
        plan = preferred
    except ValueError as error:
        if memory != "window":
            raise
        fallback_reason = str(error)
        plan = FULL
        algorithm = registry.validate(plan, context)
    return DispatchDecision(
        plan=plan,
        tree_revision=POLICY_REVISION,
        matched_path=(
            context.operator.operation_id,
            f"memory={memory}",
            plan.algorithm_id,
        ),
        reason=f"fallback to full: {fallback_reason}"
        if fallback_reason
        else f"requested {memory} route",
        workspace_upper_bound_bytes=algorithm.workspace_bound(context, plan.recipe),
    )
