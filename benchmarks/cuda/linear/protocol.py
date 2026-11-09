"""CPU reconstruction of the benchmark fixtures' mathematical contracts."""

from benchmarks.cuda.linear.fixtures import operator_spec
from benchmarks.cuda.linear.manifest import RADIUS_FIXTURES, _case

LOCAL_ORACLE_SCOPE = (
    "independent FP64 scalar Y/dX/all atom gradients; production polar update"
)
PRODUCT_ORACLE_SCOPE = (
    "independent FP64 full-site Y/dX/all atom gradients; production polar update"
)
STRIP_PRODUCT_ORACLE_SCOPE = "independent FP64 complete Strip sites Y/dX/all atom gradients; global norm; production polar update"
SQUARE_STRIP_PRODUCT_ORACLE_SCOPE = "independent FP64 complete square Strip factor sites Y/dX/all atom gradients; factored global norm; production polar update"
GLOBAL_PRODUCT_ORACLE_SCOPE = "independent FP64 complete Product sites Y/dX/all atom gradients; factored global norm; production polar update"
TORUS_PRODUCT_ORACLE_SCOPE = "independent FP64 complete S1xS2 Strip embedded-fibre sites Y/dX/all atom gradients; factored global norm; intrinsic Torus polar update"
TORUS_OPTIMIZER_POLICY = "intrinsic torus polar finite_chord; AdamW proposal + activity/radial update + retraction"
LOCAL_OPTIMIZER_POLICY = (
    "euclidean polar finite_chord; AdamW proposal + activity/radial update"
)


def measurement_operator(case):
    """Rebuild from a validated Case, never from arbitrary serialized types."""
    case = _case({"id": "stored-case", **case})
    if case.fixture == "connected_radius_multirow":
        from .connected_cell_mass import operator
        return operator(case.size, 16, True)
    if case.fixture in RADIUS_FIXTURES:
        from .radius_cell_mass import fixture

        model, *_ = fixture(case)
        return model.execution_declaration()
    n = case.size
    h, j = (32, 32) if n == 1024 else (64, 128)
    return operator_spec(
        sizes=(n, h, j),
        origin=(-(n - 1) / 2, -(h - 1) / 4, -(j - 1) / 4),
        spacing=(1.0, 0.5, 0.5),
    )
