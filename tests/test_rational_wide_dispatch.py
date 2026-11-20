"""The bundled training dispatch table only selects the observed cohort."""

import json
from dataclasses import replace
from importlib.resources import files

import pytest
import torch

from torchcst import ChebyshevSpec, CSTLinear, chart_presets, kernels
from torchcst._backends.catalog import get_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.rational_wide import (
    RATIONAL_WIDE_AXIS_GRADIENT,
    RATIONAL_WIDE_FOURIER_WEIGHT,
)
from torchcst._backends.dispatch import UnsupportedPlan, exact
from torchcst._backends.schema import DeviceInfo, RequiredGrads
from torchcst.nn.linear import _default_selector
from torchcst.operators.context import LinearContext


def artifact():
    resource = files(exact.__package__).joinpath("rational-wide-rtx4090.json")
    return json.loads(resource.read_bytes())


@pytest.mark.parametrize(
    ("width", "coefficient", "expected"),
    (
        (2048, 32, RATIONAL_WIDE_FOURIER_WEIGHT),
        (4096, 32, RATIONAL_WIDE_AXIS_GRADIENT),
        (8192, 16, RATIONAL_WIDE_AXIS_GRADIENT),
        (8192, 32, RATIONAL_WIDE_AXIS_GRADIENT),
        (16384, 32, RATIONAL_WIDE_AXIS_GRADIENT),
    ),
)
def test_rational_wide_dispatch_matches_observed_cuda_training(
    monkeypatch, width, coefficient, expected
):
    document = artifact()
    versions = document["runtime_versions"]
    monkeypatch.setattr(torch, "__version__", versions["torch"])
    monkeypatch.setattr(exact, "version", versions.__getitem__)
    grid = chart_presets.regular_grid(width, geometry="flat_torus")
    layer = CSTLinear(
        charts=(grid, grid),
        atoms=coefficient * width,
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8)),
            coordinates=kernels.rational_radius(radius_max=512),
        ),
    )
    context = LinearContext(
        operator=layer.execution_declaration(),
        input_shape=(128, width),
        input_strides=(width, 1),
        dtype=torch.float32,
        atom_count=coefficient * width,
        parameter_dim=4,
        device=DeviceInfo("cuda", 0, "NVIDIA GeForce RTX 4090", (8, 9), 128),
        required_grads=RequiredGrads(True, True),
    )
    _default_selector.cache_clear()
    try:
        selector = _default_selector()
        picked = selector.select(context)
        assert picked.plan == expected
        assert len(picked.evidence_ids) == 2
        with pytest.raises(UnsupportedPlan):
            selector.select(replace(context, input_shape=(256, width)))
        with pytest.raises(UnsupportedPlan):
            selector.select(replace(context, required_grads=RequiredGrads(False, True)))
        assert (
            selector.select(
                replace(context, device=DeviceInfo("cpu"))
            ).plan.algorithm_id
            == "torch_materialized"
        )
    finally:
        _default_selector.cache_clear()


def test_rational_wide_dispatch_ignores_a_runtime_mismatch(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "0.0.0")
    assert exact.bundled_rational_wide_selector(get_registry()) is None
