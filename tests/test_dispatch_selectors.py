"""Selector interchange, exact artifacts and registry guards without a GPU."""

import copy
from dataclasses import dataclass, replace

import pytest
import torch

from benchmarks.cuda.linear.fixtures import operator_spec
from torchcst._backends.cuda.algorithm import Algorithm
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip import REGISTRY
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.plans import (
    FULL,
    WINDOW,
)
from torchcst._backends.cuda.dispatch import (
    ExactEntry,
    ExactSelector,
    FixedSelector,
    Match,
    Selector,
    load_selector,
)
from torchcst._backends.cuda.dispatch.conditions import (
    condition_key,
    context_key,
    dump_condition,
)
from torchcst._backends.cuda.registry import Registry
from torchcst._backends.cuda.schema import (
    DeviceInfo,
    DispatchContext,
    ExecutionPlan,
    PrecisionPolicy,
    RequiredGrads,
    SupportResult,
)


def context(**changes):
    return DispatchContext(
        **(
            {
                "operator": operator_spec(
                    sizes=(64, 4, 4), origin=(0.0, 0.0, 0.0), spacing=(1.0, 0.5, 0.5)
                ),
                "input_shape": (2, 16),
                "input_strides": (16, 1),
                "dtype": torch.float32,
                "atom_count": 8,
                "parameter_dim": 5,
                "device": DeviceInfo("cuda", 0, "NVIDIA L4", (8, 9), 58),
                "required_grads": RequiredGrads(True, True),
            }
            | changes
        )
    )


def table(entries=None, **kwargs):
    return ExactSelector.from_entries(
        entries
        if entries is not None
        else [ExactEntry(context(), WINDOW, ("fixture-observation",))],
        **(
            {
                "registry": REGISTRY,
                "fallback_plan": FULL,
                "revision": "test-policy-v1",
                "score_policy": {
                    "id": "test-score",
                    "revision": "v1",
                    "parameters": {"nested": [1, 2]},
                },
                "dataset_snapshot": "test-dataset",
            }
            | kwargs
        ),
    )


def test_json_roundtrip_compiles_once_and_does_not_read_json_during_select(monkeypatch):
    selector = table()
    decoded = load_selector(selector.dumps().encode(), registry=REGISTRY)
    import torchcst._backends.cuda.dispatch.exact as exact

    monkeypatch.setattr(
        exact, "decode_json", lambda *_: pytest.fail("per-selection JSON parsing")
    )
    for _ in range(3):
        decision = decoded.select(context())
        assert decision.plan == WINDOW
        assert decision.selector_revision == "test-policy-v1"
        assert decision.evidence_ids == ("fixture-observation",)
        assert decision.matched_path[0] == "exact_table"
    assert decoded.select(context(atom_count=9)).plan == FULL


def test_artifact_does_not_alias_input_or_dumped_data():
    data = table().dump()
    selector = ExactSelector.load(data, registry=REGISTRY)
    data["entries"].clear()
    exported = selector.dump()
    exported["entries"][0]["evidence_ids"][0] = "changed"
    exported["score_policy"]["parameters"]["nested"].append(3)
    assert selector.select(context()).evidence_ids == ("fixture-observation",)
    assert selector.dump()["score_policy"]["parameters"]["nested"] == [1, 2]


@pytest.mark.parametrize(
    "change",
    [
        {"input_shape": (3, 16), "input_strides": (16, 1)},
        {"input_strides": (1, 2)},
        {"atom_count": 9},
        {"parameter_dim": 6},
        {"dtype": torch.float64},
        {"required_grads": RequiredGrads()},
        {"execution_mode": "cuda_graph"},
        {"deterministic": True},
        {"precision": PrecisionPolicy(allow_tf32=True)},
        {"workspace_limit_bytes": 100},
        {"device": DeviceInfo("cuda", 0, "NVIDIA RTX 4090", (8, 9), 128)},
        {"device": DeviceInfo("cuda", 0, "NVIDIA L4", (9, 0), 58)},
        {"device": DeviceInfo("cuda", 0, "NVIDIA L4", (8, 9), 59)},
    ],
)
def test_every_execution_dimension_changes_exact_identity(change):
    original = context()
    altered = context(**change)
    assert context_key(original) != context_key(altered)
    assert condition_key(dump_condition(altered)) == context_key(altered)


def test_typed_nested_declarations_and_revisions_are_part_of_key():
    original = context()
    chart = original.operator.layout.chart
    altered = replace(
        original,
        operator=replace(
            original.operator,
            layout=replace(
                original.operator.layout,
                chart=replace(
                    chart,
                    axes=(replace(chart.axes[0], start=(1.0,)), chart.axes[1]),
                ),
            ),
        ),
    )
    assert context_key(original) != context_key(altered)
    kernel_revision = replace(
        original,
        operator=replace(
            original.operator, kernel=replace(original.operator.kernel, revision=2)
        ),
    )
    assert context_key(original) != context_key(kernel_revision)
    assert table().select(altered).plan == FULL
    with pytest.raises(ValueError, match="contract"):
        table().select(kernel_revision)


def test_gpu_ordinal_is_local_and_unobserved_context_has_no_claimed_evidence():
    selector = table()
    assert (
        selector.select(context(device=replace(context().device, index=3))).plan
        == WINDOW
    )
    decision = selector.select(context(atom_count=9))
    assert decision.plan == FULL and "unobserved" in decision.reason
    assert decision.evidence_ids == ()


def test_no_family_specific_fallback_is_implicit():
    ctx = context()
    op = ctx.operator
    chart = op.layout.chart
    op = replace(
        op,
        layout=replace(
            op.layout,
            chart=replace(
                chart,
                axes=(
                    replace(chart.axes[0], start=(0.1,)),
                    chart.axes[1],
                ),
            ),
        ),
    )
    ctx = replace(ctx, operator=op)
    with pytest.raises(ValueError, match="quarter-grid"):
        FixedSelector(WINDOW, registry=REGISTRY).select(ctx)
    # Either Plan can be chosen as fallback; an invalid fallback still raises.
    with pytest.raises(ValueError, match="quarter-grid"):
        table([], fallback_plan=WINDOW).select(ctx)
    decision = table([ExactEntry(ctx, WINDOW, ("incompatible-fixture",))]).select(ctx)
    assert decision.plan == FULL and "quarter-grid" in decision.reason
    assert not decision.evidence_ids
    with pytest.raises(ValueError, match="unknown"):
        table().select(context(workspace_limit_bytes=100000000))


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda d: d.update(schema_version=True), "schema"),
        (lambda d: d["selector"].update(kind="mlp"), "kind"),
        (lambda d: d["selector"].update(kind="rules"), "kind"),
        (lambda d: d["runtime_versions"].update(torch="0.0"), "runtime"),
        (
            lambda d: d["entries"].append(copy.deepcopy(d["entries"][0])),
            "duplicate exact",
        ),
        (lambda d: d["entries"][0].update(plan_id="missing"), "unknown entry"),
        (lambda d: d.update(fallback_plan_id="missing"), "unknown fallback"),
        (lambda d: d["entries"][0].update(evidence_ids=[]), "evidence"),
        (lambda d: d["entries"][0]["condition"].update(atom_count=True), "integer"),
        (
            lambda d: d["entries"][0]["condition"]["device"].update(sm_count=True),
            "integer",
        ),
        (
            lambda d: d["entries"][0]["condition"]["required_grads"].update(inputs=1),
            "bool",
        ),
        (lambda d: d["entries"][0]["condition"].update(extra=0), "exactly"),
        (
            lambda d: d["plans"][d["fallback_plan_id"]].update(
                algorithm_revision="missing"
            ),
            "unknown",
        ),
        (
            lambda d: d["score_policy"].update(parameters={"weight": float("nan")}),
            "finite",
        ),
    ],
)
def test_artifact_rejects_ambiguous_or_unimplemented_data(mutate, reason):
    data = table().dump()
    mutate(data)
    with pytest.raises((ValueError, TypeError), match=reason):
        ExactSelector.load(data, registry=REGISTRY)


def test_json_duplicate_keys_and_nonfinite_values_rejected():
    for text in (
        '{"schema_version":1,"schema_version":1}',
        '{"score":NaN}',
        '{"score":1e999}',
    ):
        with pytest.raises(ValueError):
            load_selector(text, registry=REGISTRY)


@dataclass(frozen=True)
class Recipe:
    scratch: int = 0


class ToyAlgorithm(Algorithm):
    def __init__(self, name):
        super().__init__(name, "v1", "linear", "toy-sum-v1", Recipe)

    def validate_recipe(self, recipe):
        if type(recipe.scratch) is not int or recipe.scratch < 0:
            raise ValueError("invalid scratch")

    def supports(self, context, recipe):
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return recipe.scratch

    def execute(self, *, x, parameters, operator, recipe):
        return (
            x.sum(-1, keepdim=True).expand(*x.shape[:-1], operator.out_features)
            * parameters[0, 0]
        )


def test_other_algorithm_ids_use_same_selector_and_budget_guards():
    registry = Registry()
    for name in ("toy_fast", "toy_small"):
        registry.register(ToyAlgorithm(name))
    fast = ExecutionPlan("toy_fast", "v1", Recipe(256))
    small = ExecutionPlan("toy_small", "v1", Recipe())
    ctx = context(device=DeviceInfo("cpu"), workspace_limit_bytes=128)
    selector = table(
        [ExactEntry(ctx, fast, ("toy-fixture",))],
        registry=registry,
        fallback_plan=small,
    )
    decision = selector.select(ctx)
    assert decision.plan == small
    assert "workspace limit" in decision.reason
    assert decision.workspace_upper_bound_bytes == 0
    assert not decision.evidence_ids
    x = torch.randn(2, 16, requires_grad=True)
    p = torch.randn(8, 5, requires_grad=True)
    first = registry.execute(
        decision.plan, ctx, x=x, parameters=p, operator=ctx.operator
    )
    second = registry.execute(
        decision.plan, ctx, x=x * 2, parameters=p, operator=ctx.operator
    )
    (first.sum() + second.sum()).backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 3 * 64) * p.detach()[0, 0])
    expected = torch.zeros_like(p)
    expected[0, 0] = 3 * 64 * x.detach().sum()
    torch.testing.assert_close(p.grad, expected)


def test_custom_rule_reuses_same_validation_and_decision_interface():
    class Rule(Selector):
        def _match(self, ctx):
            if ctx.atom_count < 10:
                return Match(WINDOW, ("rule", "atoms<10"), "test rule")
            return None

    selector = Rule(revision="rule-fixture-v1", registry=REGISTRY, fallback_plan=FULL)
    assert selector.select(context()).plan == WINDOW
    assert selector.select(context(atom_count=10)).plan == FULL
    with pytest.raises(ValueError, match="TF32"):
        selector.select(context(precision=PrecisionPolicy(allow_tf32=True)))
