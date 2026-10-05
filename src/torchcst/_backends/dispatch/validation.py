"""One metadata-only eligibility boundary for selection and forced execution."""

from torchcst._backends.schema import Context, ExecutionPlan


class UnsupportedPlan(ValueError):
    """A valid declaration cannot execute under these valid runtime conditions."""


def validate_plan_context(registry, plan: ExecutionPlan, context: Context):
    algorithm = registry.validate_plan(plan)
    if algorithm.operation_id != context.operation_id:
        raise ValueError("plan and binding mathematical contract differ")
    limit = context.workspace_limit_bytes
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("workspace limit must be nonnegative or None")
    support = algorithm.supports(context, plan.recipe)
    if not support.supported:
        raise UnsupportedPlan(
            "unsupported execution plan: " + "; ".join(support.reasons)
        )
    bound = algorithm.workspace_bound(context, plan.recipe)
    if bound is not None and (type(bound) is not int or bound < 0):
        raise ValueError("invalid workspace upper bound")
    if limit is not None:
        if bound is None:
            raise UnsupportedPlan(
                "workspace upper bound is unknown; cannot enforce limit"
            )
        if bound > limit:
            raise UnsupportedPlan("plan exceeds workspace limit")
    return algorithm, bound
