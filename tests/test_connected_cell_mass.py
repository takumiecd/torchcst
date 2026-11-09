from dataclasses import replace

import pytest
import torch

from benchmarks.cuda.linear.connected_cell_mass import kernel, scalar_axis
from torchcst import CSTLinear, chart_presets
from torchcst.kernels.spec import RadiusConnectionSpec


def model():
    chart = chart_presets.regular_grid(128, spacing=1.0, geometry="flat_torus")
    return CSTLinear(
        chart,
        chart,
        atoms=2,
        kernel=kernel(16, True),
        dtype=torch.float64,
        backend="factored",
    )


def test_declaration_and_checkpoint_guard():
    m = model()
    assert m.kernel.spec.revision == 9
    old = CSTLinear(
        *m.cst_charts(),
        atoms=2,
        kernel=kernel(16, False),
        dtype=torch.float64,
        backend="factored",
    )
    with pytest.raises(RuntimeError):
        old.load_state_dict(m.state_dict())
    with pytest.raises(ValueError):
        RadiusConnectionSpec(scale_upper=0.95)
    with pytest.raises(ValueError):
        replace(m.kernel.spec, revision=7)


@pytest.mark.parametrize(
    "radius,center", [(0.8, 7.5), (1.6, 13.5), (31.9, -0.1), (50.0, 127.5)]
)
def test_cpu_matches_independent_quadrature_and_gradient(radius, center):
    m = model()
    p = torch.tensor(
        [[0.3, radius, center, 13.5], [-0.2, 1.5, 1.0, 2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    vi, _vo = m.operator.factors(p)
    expected = torch.stack([scalar_axis(row, 128, 16, True, 3) for row in p]).T
    torch.testing.assert_close(vi, expected, atol=2e-12, rtol=1e-10)
    torch.testing.assert_close(vi.sum(0), torch.ones(2, dtype=torch.float64))
    dy = torch.randn_like(vi)
    a = torch.autograd.grad(vi, p, dy, retain_graph=True)[0]
    b = torch.autograd.grad(expected, p, dy)[0]
    torch.testing.assert_close(a, b, atol=2e-11, rtol=1e-9)


def test_registry_roundtrip():
    from benchmarks.cuda.linear.manifest import REGISTRY, load_run

    run = load_run(
        "benchmarks/cuda/linear/cases/connected-radius-2048-r256-c8.json",
        "benchmarks/cuda/linear/plans-connected-radius.json",
    )
    for entry in run.plans:
        assert REGISTRY.loads_plan(REGISTRY.dumps_plan(entry.plan)) == entry.plan


def test_admission_uses_declarations_and_rejects_unimplemented_contracts():
    from torchcst._backends.cuda.algorithms.linear.connected_cell_mass.algorithm import (
        ConnectedAlgorithm,
        ConnectedRecipe,
    )
    from torchcst._backends.schema import DeviceInfo
    from torchcst.operators.execution import LinearInputs

    m = model()
    c = m.build_context(LinearInputs(torch.zeros(2, 128, dtype=torch.float64)))
    c = replace(c, dtype=torch.float32, device=DeviceInfo("cuda", 0, "metadata-only"))
    algorithm, recipe = ConnectedAlgorithm(), ConnectedRecipe()
    assert algorithm.supports(c, recipe).supported
    for fields in (
        {"dtype": torch.float64},
        {"execution_mode": "cuda_graph"},
        {"deterministic": True},
        {"input_shape": (257, 128)},
    ):
        assert not algorithm.supports(replace(c, **fields), recipe).supported
    spec = c.operator.kernel
    other = replace(
        spec.profiles[1], connection=RadiusConnectionSpec(position_half_width=0.05)
    )
    mismatch = replace(
        c.operator, kernel=replace(spec, profiles=(spec.profiles[0], other))
    )
    assert not algorithm.supports(replace(c, operator=mismatch), recipe).supported

    # Small lower scale ratios can require banks beyond the finite chart
    # hierarchy; only the measured default connection is admitted initially.
    unqualified = RadiusConnectionSpec(scale_ratio=1.9)
    changed = replace(
        spec, profiles=tuple(replace(p, connection=unqualified) for p in spec.profiles)
    )
    assert not algorithm.supports(
        replace(c, operator=replace(c.operator, kernel=changed)), recipe
    ).supported
