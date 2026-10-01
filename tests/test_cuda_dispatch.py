"""Control-plane contract and rejection tests run without a GPU or Triton."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from torchcst.nn._backends.cuda.algorithm import Algorithm
from torchcst.nn._backends.cuda.algorithms.normalized_strip import REGISTRY
from torchcst.nn._backends.cuda.context import context_from_tensors
from torchcst.nn._backends.cuda.dispatch.select import FULL, WINDOW, select_normalized
from torchcst.nn._backends.cuda.registry import Registry
from torchcst.nn._backends.cuda.schema import (
    DeviceInfo,
    DispatchContext,
    ExecutionPlan,
    FullRecipe,
    OperatorSpec,
    PrecisionPolicy,
    RequiredGrads,
    SupportResult,
    WindowRecipe,
)


def context(**kwargs):
    defaults = {
        "operator": OperatorSpec((1024, 4, 4), (0.0, 0.0, 0.0), (1.0, 0.5, 0.5)),
        "input_shape": (6, 16),
        "input_strides": (16, 1),
        "dtype": torch.float32,
        "atom_count": 8,
        "device": DeviceInfo("cuda", 1, "NVIDIA L4", (8, 9), 58),
        "required_grads": RequiredGrads(True, True),
    }
    return DispatchContext(**(defaults | kwargs))


def test_selection_preserves_legacy_routes_and_explains_fallback():
    assert select_normalized(context()).plan == FULL
    assert select_normalized(context(), memory="window").plan == WINDOW
    op = replace(context().operator, origin=(0.1, 0.0, 0.0))
    decision = select_normalized(context(operator=op), memory="window")
    assert decision.plan == FULL
    assert "quarter-grid" in decision.reason
    assert decision.evidence_ids == ()  # compatibility is not new certification
    assert decision.workspace_upper_bound_bytes is None


@pytest.mark.parametrize("rows", [1, 31, 33, 65])
def test_window_row_guard_falls_back(rows):
    op = replace(context().operator, sizes=(rows, 4, 4))
    assert select_normalized(context(operator=op), memory="window").plan == FULL
    with pytest.raises(ValueError, match="divisible by 32"):
        REGISTRY.validate(WINDOW, context(operator=op))


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"dtype": torch.float64}, "float32"),
        ({"device": DeviceInfo("cpu")}, "CUDA"),
        ({"precision": PrecisionPolicy(allow_tf32=True)}, "TF32"),
        ({"precision": PrecisionPolicy(autocast=True)}, "autocast"),
        ({"deterministic": True}, "atomic"),
        ({"atom_count": 0}, "empty"),
        ({"input_strides": (1, 6)}, "contiguous"),
        ({"workspace_limit_bytes": 100000000}, "unknown"),
    ],
)
def test_common_constraints_cannot_be_bypassed_by_fallback(kwargs, reason):
    for memory in ("full", "window"):
        with pytest.raises(ValueError, match=reason):
            select_normalized(context(**kwargs), memory=memory)


def test_registry_rejects_revision_recipe_and_duplicate_registration():
    with pytest.raises(ValueError, match="unknown"):
        REGISTRY.get("normalized_full", revision="v0")
    with pytest.raises(TypeError, match="recipe type"):
        REGISTRY.validate(replace(FULL, recipe=WindowRecipe()), context())
    with pytest.raises(ValueError, match="unvalidated"):
        REGISTRY.validate(
            replace(WINDOW, recipe=WindowRecipe(window_rows=256)), context()
        )
    registry = Registry()
    entry = REGISTRY.get("normalized_full", revision="v1")
    registry.register(entry)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(entry)
    with pytest.raises(ValueError, match="schema"):
        ExecutionPlan("normalized_full", "v1", FullRecipe(), schema_version=2)


def test_context_is_metadata_and_immutable():
    x = torch.randn(2, 3, 16).requires_grad_()
    p = torch.randn(8, 5).requires_grad_()
    with torch.no_grad():
        ctx = context_from_tensors(context().operator, x, p)
    assert ctx.m == 6 and ctx.operator.n == 1024 and ctx.operator.k == 16
    assert ctx.required_grads == RequiredGrads()
    assert ctx.device == DeviceInfo("cpu")
    with pytest.raises(FrozenInstanceError):
        ctx.atom_count = 9
    with pytest.raises(ValueError, match="mathematical contract"):
        replace(ctx.operator, semantics_id="approximate")


def test_import_does_not_load_triton_or_gpu_implementations():
    code = (
        "import sys; from torchcst.nn._backends.cuda.dispatch.select import FULL; "
        "assert 'triton' not in sys.modules; "
        "assert 'torchcst.nn._backends.normalized_strip.full' not in sys.modules; "
        "assert 'torchcst.nn._backends.cuda.algorithms.normalized_strip.impl' not in sys.modules"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_direct_execution_checks_metadata_and_preserves_autograd_per_invocation():
    # A CPU executor verifies the generic connection, independently of GPU availability.
    registry = Registry()

    class CpuConnectionAlgorithm(Algorithm[FullRecipe]):
        def validate_recipe(self, recipe):
            pass

        def supports(self, context, recipe):
            return SupportResult()

        def workspace_bound(self, context, recipe):
            return 0

        def execute(self, *, x, parameters, operator, recipe):
            return x * parameters[0, 0]

    registry.register(
        CpuConnectionAlgorithm(
            id="cpu_connection_test",
            revision="v1",
            operation_id="normalized_strip_linear",
            semantics_id="normalized-strip-triweight-l2-v1",
            recipe_type=FullRecipe,
        )
    )
    plan = ExecutionPlan("cpu_connection_test", "v1", FullRecipe())
    op = context().operator
    p = torch.ones(1, 5, requires_grad=True)
    x1 = torch.ones(2, 16, requires_grad=True)
    x2 = torch.full((2, 16), 2.0, requires_grad=True)
    c1, c2 = [context_from_tensors(op, x, p) for x in (x1, x2)]
    y1 = registry.execute(plan, c1, x=x1, parameters=p, operator=op)
    y2 = registry.execute(plan, c2, x=x2, parameters=p, operator=op)
    (y1.sum() + y2.sum()).backward()
    assert torch.equal(x1.grad, torch.ones_like(x1))
    assert torch.equal(x2.grad, torch.ones_like(x2))
    assert p.grad[0, 0] == 96
    with pytest.raises(ValueError, match="input metadata"):
        registry.execute(
            plan, replace(c1, input_shape=(3, 16)), x=x1, parameters=p, operator=op
        )
    with pytest.raises(ValueError, match="dtype"):
        registry.execute(plan, c1, x=x1.double(), parameters=p, operator=op)
    with pytest.raises(ValueError, match="gradient requirements"):
        registry.execute(
            plan,
            replace(c1, required_grads=RequiredGrads()),
            x=x1,
            parameters=p,
            operator=op,
        )


@pytest.mark.parametrize(
    "recipe", [FullRecipe(atom_num_warps=True), WindowRecipe(enable_fp_fusion=1)]
)
def test_recipe_field_types_cannot_alias_validated_values(recipe):
    plan = replace(FULL if type(recipe) is FullRecipe else WINDOW, recipe=recipe)
    with pytest.raises(ValueError, match="unvalidated"):
        REGISTRY.validate(plan, context())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("plan", [FULL, WINDOW])
@pytest.mark.parametrize(
    "input_grad,parameter_grad",
    [(True, True), (False, True), (True, False), (False, False)],
)
def test_cuda_direct_plan_grad_subsets_and_two_live_forwards(
    plan, input_grad, parameter_grad
):
    import runpy
    from pathlib import Path

    helpers = runpy.run_path(
        str(Path(__file__).with_name("test_normalized_strip_public.py"))
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        sizes = (1024, 4, 4)
        p = helpers["mixed"](torch.float32, "cuda").requires_grad_(parameter_grad)
        op = OperatorSpec(sizes, (0.0, 0.0, 0.0), (1.0, 0.5, 0.5))
        x1 = torch.randn(2, 16, device="cuda", requires_grad=input_grad)
        x2 = torch.randn(3, 16, device="cuda", requires_grad=input_grad)
        y1, y2 = [
            REGISTRY.execute(
                plan, context_from_tensors(op, x, p), x=x, parameters=p, operator=op
            )
            for x in (x1, x2)
        ]
        tp = p.detach().double().requires_grad_(parameter_grad)
        tx1, tx2 = [x.detach().double().requires_grad_(input_grad) for x in (x1, x2)]
        w = helpers["oracle"](tp, sizes, stored_dtype=torch.float32)
        t1, t2 = tx1 @ w.T, tx2 @ w.T
        torch.testing.assert_close(y1.double(), t1, atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(y2.double(), t2, atol=3e-4, rtol=3e-4)
        if input_grad or parameter_grad:
            (y1.sum() + y2.sum()).backward()
            (t1.sum() + t2.sum()).backward()
            for actual, truth in ((x1, tx1), (x2, tx2), (p, tp)):
                if actual.requires_grad:
                    torch.testing.assert_close(
                        actual.grad.double(), truth.grad, atol=3e-4, rtol=3e-4
                    )
                else:
                    assert actual.grad is None
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_algorithm_requires_all_four_operations():
    with pytest.raises(TypeError, match="abstract"):
        Algorithm("incomplete", "v1", "op", "semantics", FullRecipe)

    class MissingExecute(Algorithm[FullRecipe]):
        def validate_recipe(self, recipe):
            pass

        def supports(self, context, recipe):
            return SupportResult()

        def workspace_bound(self, context, recipe):
            return None

    with pytest.raises(TypeError, match="execute"):
        MissingExecute("incomplete", "v1", "op", "semantics", FullRecipe)


def test_registry_accepts_only_algorithm_contract_and_identity_is_immutable():
    registry = Registry()
    with pytest.raises(TypeError, match="Algorithm instance"):
        registry.register(object())
    algorithm = REGISTRY.get("normalized_full", revision="v1")
    assert isinstance(algorithm, Algorithm)
    with pytest.raises(FrozenInstanceError):
        algorithm.revision = "v2"


def test_algorithm_identity_requires_valid_metadata():
    from torchcst.nn._backends.cuda.algorithms.normalized_strip import (
        NormalizedFullAlgorithm,
    )

    with pytest.raises(ValueError, match="revision"):
        NormalizedFullAlgorithm(revision="")
    with pytest.raises(TypeError, match="recipe_type"):
        NormalizedFullAlgorithm(recipe_type="FullRecipe")
