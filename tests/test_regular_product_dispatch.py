"""Regular Product eligibility, fallback and live configuration without a GPU."""

from dataclasses import replace

import pytest
import torch

from benchmarks.cuda.linear.manifest import REGISTRY
from torchcst import (
    BandwidthBounds,
    CSTLinear,
    Dispatcher,
    FixedSelector,
    LinearInputs,
    TriweightSpec,
    chart_presets,
    pattern_presets,
    presets,
)
from torchcst._backends.dispatch.conditions import context_key
from torchcst._backends.schema import DefaultRecipe, DeviceInfo, ExecutionPlan

PLANS = (
    ExecutionPlan(
        "research_profile_product_global",
        "v1",
        REGISTRY.get("research_profile_product_global", revision="v1").recipe_type(),
    ),
    ExecutionPlan(
        "research_profile_product_large_prepared",
        "v1",
        REGISTRY.get(
            "research_profile_product_large_prepared", revision="v1"
        ).recipe_type(),
    ),
    ExecutionPlan(
        "research_profile_product_large_matrix",
        "v1",
        REGISTRY.get(
            "research_profile_product_large_matrix", revision="v1"
        ).recipe_type(),
    ),
)
FALLBACK = ExecutionPlan("torch_factored", "v1", DefaultRecipe())


def layer_for(axes=None, *, explicit=False, n=33):
    if axes is None:
        axes = tuple(pattern_presets.line(n, spacing=1) for _ in range(2))
    chart = (
        chart_presets.grid((n, n), spacing=1)
        if explicit
        else chart_presets.product((n, n), axes)
    )
    return CSTLinear(
        chart=chart,
        atoms=torch.tensor([[0.3, 1.5, 2.4, 3.7]]),
        kernel=presets.polar_profile_product(
            profiles=(TriweightSpec(), TriweightSpec()),
            amplitude_max=1,
            bounds=BandwidthBounds(minimum=0.25, birth=1, maximum=16, upper_floor=1),
            w_c=1e6,
        ),
    )


def context_for(layer):
    # Only eligibility/selection is simulated; no CUDA execution is claimed.
    context = layer.build_context(LinearInputs(torch.zeros(2, layer.in_features)))
    return replace(context, device=DeviceInfo("cuda", 0))


def selector_for(plan):
    return FixedSelector(plan, registry=REGISTRY, fallback_plan=FALLBACK)


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("kind", ["zero", "unequal", "grid_axis"])
def test_nonregular_or_degenerate_chart_falls_back_before_execution(plan, kind):
    regular = pattern_presets.line(33, spacing=1)
    other = {
        "zero": pattern_presets.line(33, low=2, high=2),
        "unequal": pattern_presets.line(33, spacing=2),
        "grid_axis": pattern_presets.grid((33,), spacing=1),
    }[kind]
    axes = (other, other) if kind == "zero" else (regular, other)
    context = context_for(layer_for(axes))
    decision = selector_for(plan).select(context)
    assert decision.plan == FALLBACK
    assert decision.matched_path[0] == "fallback"
    with pytest.raises(ValueError, match="regular Product"):
        Dispatcher(registry=REGISTRY).select(context, plan=plan)


@pytest.mark.parametrize("explicit", [False, True])
def test_arbitrary_points_are_rejected_by_profile_product_declaration(explicit):
    points = pattern_presets.points(tuple((float(i),) for i in range(33)))
    reason = "single chart" if explicit else "Product/Strip grid chart"
    with pytest.raises(ValueError, match=reason):
        layer_for((points, points), explicit=explicit)


@pytest.mark.parametrize("plan", PLANS)
def test_positive_line_axes_are_eligible_and_atom_updates_do_not_change_key(plan):
    layer = layer_for()
    before = context_for(layer)
    assert selector_for(plan).select(before).plan == plan
    with torch.no_grad():
        layer.atoms.p[:, 1:].add_(0.2)
    after = context_for(layer)
    assert context_key(after) == context_key(before)
    assert selector_for(plan).select(after).plan == plan


@pytest.mark.parametrize("plan", PLANS[1:])
def test_large_regular_boundary_is_eligible(plan):
    context = context_for(layer_for(n=8192))
    assert selector_for(plan).select(context).plan == plan


def test_live_spacing_and_origin_refresh_declaration_and_exact_condition():
    layer = layer_for()
    plan = PLANS[1]
    original = context_for(layer)
    layer.chart.axes[0].start.add_(0.25)
    shifted = context_for(layer)
    assert context_key(shifted) != context_key(original)
    assert selector_for(plan).select(shifted).plan == plan
    layer.chart.axes[1].spacing.fill_(2)
    unequal = context_for(layer)
    assert context_key(unequal) != context_key(shifted)
    assert selector_for(plan).select(unequal).plan == FALLBACK
    layer.chart.axes[0].spacing.fill_(2)
    regular = context_for(layer)
    assert context_key(regular) != context_key(unequal)
    assert selector_for(plan).select(regular).plan == plan
