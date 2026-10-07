"""CPU reconstruction of the two benchmark fixtures' mathematical contracts."""

from benchmarks.cuda.linear.fixtures import operator_spec
from benchmarks.cuda.linear.local_product import fixture_operator
from benchmarks.cuda.linear.manifest import _case

LOCAL_ORACLE_SCOPE = (
    "independent FP64 scalar Y/dX/all atom gradients; production polar update"
)
PRODUCT_ORACLE_SCOPE = (
    "independent FP64 full-site Y/dX/all atom gradients; production polar update"
)

STRIP_PRODUCT_ORACLE_SCOPE = "independent FP64 complete Strip sites Y/dX/all atom gradients; global norm; production polar update"

LOCAL_OPTIMIZER_POLICY = (
    "euclidean polar finite_chord; AdamW proposal + activity/radial update"
)


def measurement_operator(case):
    """Rebuild from a validated Case, never from arbitrary serialized types."""
    case = _case({"id": "stored-case", **case})
    if case.fixture == "polar_profile_product_strip":
        from .strip_profile_product import fixture_operator as strip_operator

        return strip_operator(case)
    if case.fixture == "polar_profile_product":
        from benchmarks.cuda.linear.profile_product import (
            fixture_operator as product_operator,
        )

        return product_operator(case)
    if case.fixture == "local_polar_product":
        return fixture_operator(case)
    n = case.size
    h, j = (32, 32) if n == 1024 else (64, 128)
    return operator_spec(
        sizes=(n, h, j),
        origin=(-(n - 1) / 2, -(h - 1) / 4, -(j - 1) / 4),
        spacing=(1.0, 0.5, 0.5),
    )
