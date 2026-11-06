"""An explicit, narrow FFN cohort must not broaden old radius Plan support."""

from dataclasses import replace

import pytest
import torch

from torchcst import ChebyshevSpec, CSTLinear, RadiusBounds, chart_presets, kernels
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_ACTIVE_GRAD4_NATIVE,
    GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048 as CANDIDATE,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo
from torchcst.operators.execution import LinearInputs


def context(ni=512, no=2048, *, count=8192, spacing=1, modes=8, maximum=50):
    model = CSTLinear(
        chart_presets.regular_grid(no, spacing=spacing, geometry="flat_torus"),
        chart_presets.regular_grid(ni, spacing=spacing, geometry="flat_torus"),
        atoms=1,
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=modes), merge_factor=2),
            coordinates=kernels.polar(
                amplitude_max=1,
                w_c=0.3,
                input_bounds=RadiusBounds(
                    minimum=0.5, birth=4, maximum=maximum, upper_floor=1
                ),
            ),
        ),
    )
    ctx = model.build_context(LinearInputs(torch.zeros(1, ni)))
    return replace(
        ctx,
        input_shape=(256, ni),
        input_strides=(ni, 1),
        atom_count=count,
        device=DeviceInfo("cuda", 0, "metadata-only"),
    )


@pytest.mark.parametrize(
    "ni,no,count",
    [(512, 2048, 8192), (2048, 512, 8192), (128, 2048, 4096), (2048, 128, 4096)],
)
def test_explicit_cohort_support_and_old_plan_rejection(ni, no, count):
    algorithm = RadiusCellMassAlgorithm()
    ctx = context(ni, no, count=count)
    assert algorithm.supports(ctx, CANDIDATE.recipe).supported
    for plan in (
        GLOBAL_ACTIVE_GRAD4_NATIVE,
        GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C,
    ):
        assert not algorithm.supports(ctx, plan.recipe).supported
    registry = Registry()
    registry.register(algorithm)
    assert registry.loads_plan(registry.dumps_plan(CANDIDATE)) == CANDIDATE


@pytest.mark.parametrize(
    "change",
    [
        {"atom_count": 8191},
        {"atom_count": 8193},
        {"input_shape": (257, 512)},
        {"dtype": torch.float64},
        {"device": DeviceInfo("cpu", None, "metadata-only")},
    ],
)
def test_cohort_rejects_unqualified_atom_batch_precision_and_device(change):
    assert (
        not RadiusCellMassAlgorithm()
        .supports(replace(context(), **change), CANDIDATE.recipe)
        .supported
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ni": 1024, "no": 2048},
        {"ni": 2048, "no": 2048},
        {"spacing": 0.5},
        {"modes": 3},
        {"maximum": 51},
    ],
)
def test_cohort_rejects_other_shapes_and_geometry(kwargs):
    assert (
        not RadiusCellMassAlgorithm()
        .supports(context(**kwargs), CANDIDATE.recipe)
        .supported
    )


@pytest.mark.parametrize(
    "change",
    [
        {"native_entry": False},
        {"moment_method": "direct"},
        {"input_direct_levels": 0},
        {"output_tile_sites": 16},
        {"schedule": 1},
    ],
)
def test_cohort_cannot_be_attached_to_another_execution_recipe(change):
    with pytest.raises(ValueError):
        RadiusCellMassAlgorithm().validate_recipe(replace(CANDIDATE.recipe, **change))
