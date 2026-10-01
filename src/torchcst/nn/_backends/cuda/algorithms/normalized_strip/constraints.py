"""Shared metadata guards used by selection and the actual GPU executors."""


def routing_reasons(operator):
    reasons = []
    if operator.spacing != (1.0, 0.5, 0.5):
        reasons.append("normfast supports common spacing(1,.5,.5)")
    ends = tuple(
        o + (n - 1) * s
        for o, n, s in zip(operator.origin, operator.sizes, operator.spacing)
    )
    if abs(operator.origin[0]) + abs(ends[0]) + operator.sizes[0] > 20000:
        reasons.append("row coordinate range exceeds supported FP32 routing regime")
    if any(
        abs(operator.origin[d]) + abs(ends[d]) + operator.sizes[d] * operator.spacing[d]
        > 1024
        for d in (1, 2)
    ):
        reasons.append("column coordinate range exceeds supported FP32 routing regime")
    return tuple(reasons)
