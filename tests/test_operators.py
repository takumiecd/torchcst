"""Operator boundaries, live state and independent reference gradients."""

import copy
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from torchcst import (
    Amplitude,
    Chart,
    ChartPairSpec,
    CSTLinear,
    DirectAmpWidth,
    Gaussian,
    LinePattern,
    Operator,
    ProductChart,
    Separable,
    SingleChartSpec,
    Triweight,
)


def make_single():
    chart = ProductChart(
        shape=(3, 4),
        axes=(LinePattern(3, spacing=0.5), LinePattern(4, spacing=0.4)),
    )
    kernel = DirectAmpWidth(
        amplitude_max=1.0,
        sigma_min=0.5,
        sigma_birth=1.0,
        sigma_max=1.5,
        w_c=0.05,
        kappa=3.0,
        profile=Triweight(0.5, normalize_columns=False),
    )
    return CSTLinear(chart=chart, atoms=2, kernel=kernel, dtype=torch.float64)


def make_pair(*, trainable=False, backend="materialized"):
    return CSTLinear(
        Chart.linspace(4, low=-1, high=1, trainable=trainable),
        Chart.linspace(3, low=-0.8, high=0.8, trainable=trainable),
        atoms=2,
        kernel=Amplitude(
            Separable(input_profile=Gaussian(1.4), output_profile=Gaussian(1.2))
        ),
        backend=backend,
        dtype=torch.float64,
    )


def test_single_operator_has_one_chart_and_radial_kernel():
    model = make_single()
    spec = model.declaration()
    assert isinstance(spec.layout, SingleChartSpec)
    assert spec.charts == (model.chart.declaration(),)
    assert spec.shape == (3, 4)
    assert spec.kernel.composition == "radial"
    assert model.operator.p is model.atoms.p
    assert model.operator is model.operator
    with pytest.raises(FrozenInstanceError):
        spec.revision = 2
    with pytest.raises(FrozenInstanceError):
        model.operator.charts = ()


def test_chart_pair_preserves_input_output_roles():
    model = make_pair()
    spec = model.declaration()
    assert isinstance(spec.layout, ChartPairSpec)
    assert spec.charts == tuple(c.declaration() for c in model.cst_charts())
    assert spec.shape == (3, 4)
    assert (spec.in_features, spec.out_features) == (4, 3)
    assert spec.kernel.composition == "amplitude"
    assert spec.kernel.inner.composition == "separable"
    reversed_layout = ChartPairSpec(
        input_chart=spec.charts[1], output_chart=spec.charts[0]
    )
    assert reversed_layout.shape == (4, 3)


def test_incompatible_layouts_fail_instead_of_reinterpreting_kernel():
    single, pair = make_single().declaration(), make_pair().declaration()
    for spec, layout in ((single, pair.layout), (pair, single.layout)):
        with pytest.raises(ValueError, match="composition"):
            replace(spec, layout=layout)
    with pytest.raises(ValueError, match="revision"):
        replace(single, revision=2)
    with pytest.raises(TypeError, match="layout"):
        replace(single, layout=(single.charts[0],))
    with pytest.raises(ValueError, match="out, in"):
        SingleChartSpec(chart=pair.charts[0])


@pytest.mark.parametrize("maker", [make_single, make_pair])
def test_binding_checks_settings_without_cloning_state(maker):
    model = maker()
    spec = model.declaration()
    bound = spec.bind(charts=model.cst_charts(), kernel=model.kernel, atoms=model.atoms)
    assert bound.p is model.atoms.p
    assert bound.kernel is model.kernel
    assert all(a is b for a, b in zip(bound.charts, model.cst_charts()))
    altered = replace(spec, kernel=replace(spec.kernel, revision=2))
    with pytest.raises(ValueError, match="settings differ"):
        altered.bind(charts=model.cst_charts(), kernel=model.kernel, atoms=model.atoms)


@pytest.mark.parametrize("maker", [make_single, make_pair])
def test_bound_view_observes_updates_conversion_and_parameter_replacement(maker):
    model = maker()
    bound = model.operator
    before = bound.weight().detach().clone()
    with torch.no_grad():
        model.atoms.p[:, 0].mul_(0.5)
    assert not torch.equal(before, bound.weight())
    model.float()
    assert bound.p is model.atoms.p and bound.p.dtype == torch.float32
    model.atoms.p = torch.nn.Parameter(model.atoms.p.detach().clone() * 0.7)
    assert bound.p is model.atoms.p
    torch.testing.assert_close(bound.weight(), model.dense_weight())


def test_operator_does_not_add_checkpoint_or_parameter_ownership():
    model = make_pair()
    keys = set(model.state_dict())
    parameters = tuple(model.named_parameters())
    assert not isinstance(model.operator, torch.nn.Module)
    assert not any("operator" in key for key in keys)
    assert tuple(model.named_parameters()) == parameters
    restored = copy.deepcopy(model)
    assert restored.operator.atoms is restored.atoms
    assert restored.operator.kernel is restored.kernel
    assert restored.operator.atoms is not model.atoms
    restored.load_state_dict(model.state_dict())
    assert set(restored.state_dict()) == keys
    torch.testing.assert_close(restored.operator.weight(), model.operator.weight())


@pytest.mark.parametrize("backend", ["materialized", "factored"])
def test_pair_execution_matches_independent_oracle_with_live_chart_gradients(backend):
    model = make_pair(trainable=True, backend=backend)
    p = model.atoms.p.detach().clone().requires_grad_()
    ci = model.input_chart.coordinates.detach().clone().requires_grad_()
    co = model.output_chart.coordinates.detach().clone().requires_grad_()
    # Gaussian profile coordinates are centers; each chart has its own L2 norm.
    u = torch.exp(-0.5 * (ci[:, None, 0] - p[None, :, 1]).square() / 1.4**2)
    v = torch.exp(-0.5 * (co[:, None, 0] - p[None, :, 2]).square() / 1.2**2)
    u = u / torch.linalg.vector_norm(u, dim=0)
    v = v / torch.linalg.vector_norm(v, dim=0)
    w = (v * p[:, 0]) @ u.T
    x = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    xo = x.detach().clone().requires_grad_()
    actual, expected = model(x), torch.nn.functional.linear(xo, w)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a, b in (
        (x.grad, xo.grad),
        (model.atoms.p.grad, p.grad),
        (model.input_chart.coordinates.grad, ci.grad),
        (model.output_chart.coordinates.grad, co.grad),
    ):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("maker", [make_single, make_pair])
def test_execution_never_takes_declaration_snapshots(maker, monkeypatch):
    model = maker()

    def forbidden(*args, **kwargs):
        raise AssertionError("snapshot in execution")

    for obj in (model, model.kernel, *model.cst_charts()):
        monkeypatch.setattr(obj, "declaration", forbidden)
    model(torch.randn(2, model.in_features, dtype=torch.float64)).sum().backward()
    model.operator.weight()
    model.operator.materialize_atoms(model.atoms.p[:1])


def test_bound_operator_rejects_wrong_shapes_and_algorithms():
    model = make_single()
    with pytest.raises(ValueError, match="parameter width"):
        model.operator.weight(model.atoms.p[:, :1])
    with pytest.raises(ValueError, match="input shape"):
        model.operator.apply(torch.randn(3, 5))
    x = torch.randn(2, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="factors"):
        model.operator.apply(x, algorithm="factored")
    with pytest.raises(ValueError, match="unknown"):
        model.operator.apply(x, algorithm="unregistered")
    with pytest.raises(TypeError, match="one or two"):
        Operator(charts=(), kernel=model.kernel, atoms=model.atoms)
    with pytest.raises(ValueError, match="parameter width"):
        Operator(
            charts=model.cst_charts(), kernel=model.kernel, atoms=make_pair().atoms
        )


def test_existing_custom_kernel_requires_declaration_only_when_requested():
    from test_cst_linear import NonFactorizedKernel

    model = CSTLinear(
        Chart.linspace(3, spacing=1),
        Chart.linspace(2, spacing=1),
        atoms=2,
        kernel=NonFactorizedKernel(),
    )
    assert model(torch.randn(4, 3)).shape == (4, 2)
    with pytest.raises(NotImplementedError, match="custom kernels"):
        model.declaration()


def test_operator_rebinds_after_replacing_module_state():
    model = make_single()
    previous = model.operator
    model.atoms = copy.deepcopy(model.atoms)
    assert model.operator is not previous
    assert model.operator.atoms is model.atoms
