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


def test_normalized_strip_is_a_distinct_single_chart_contract():
    from torchcst import GridPattern, NormalizedStripLinear, StripChart
    from torchcst.nn._backends.cuda.schema import OperatorSpec as CudaOperatorSpec

    chart = StripChart(
        shape=(5, 6),
        tile_shape=(2, 6),
        axis=0,
        tile_pitch=2.0,
        axes=(LinePattern(5, spacing=1.0), GridPattern((2, 3), spacing=0.5)),
    )
    model = NormalizedStripLinear(chart, torch.tensor([[0.2, 0.0, 1.0, 0.2, 0.3]]))
    spec = model.declaration()
    assert isinstance(spec.layout, SingleChartSpec)
    assert spec.shape == (5, 6)
    assert spec.kernel.parameterization.id == "signed_amplitude_log_width"
    assert spec.kernel.profiles[0].normalization.domain == "operator_sites"
    adapted = CudaOperatorSpec.from_declaration(spec)
    assert adapted.sizes == (5, 2, 3)
    assert adapted.declaration() == spec
    assert model._cuda_operator == adapted
    # Tile storage is absent from the canonical contiguous observation sites.
    physical = replace(spec, layout=SingleChartSpec(chart=chart.declaration()))
    assert CudaOperatorSpec.from_declaration(physical) == adapted
    discontinuous = replace(physical.layout.chart, tile_pitch=3.0)
    with pytest.raises(ValueError, match="contiguous"):
        CudaOperatorSpec.from_declaration(
            replace(physical, layout=SingleChartSpec(chart=discontinuous))
        )
    before = spec
    with torch.no_grad():
        model.origin.add_(0.25)
    after = model.declaration()
    assert before != after
    assert model._cuda_operator.declaration() == after
    model.double()
    assert model.declaration() == after


def test_specialized_cuda_bridge_rejects_different_mathematical_meanings():
    from torchcst.kernels import BiweightSpec, NormalizationSpec
    from torchcst.nn._backends.cuda.schema import OperatorSpec as CudaOperatorSpec

    original = CudaOperatorSpec((5, 2, 3), (0.0, 0.0, 0.0), (1.0, 0.5, 0.5))
    spec = original.declaration()
    binding = spec.kernel.profiles[0]
    variants = [
        make_single().declaration(),
        make_pair().declaration(),
        replace(
            spec,
            kernel=replace(
                spec.kernel, profiles=(replace(binding, profile=BiweightSpec()),)
            ),
        ),
        replace(
            spec,
            kernel=replace(
                spec.kernel,
                profiles=(replace(binding, normalization=NormalizationSpec()),),
            ),
        ),
        replace(
            spec,
            kernel=replace(
                spec.kernel,
                profiles=(
                    replace(
                        binding,
                        normalization=replace(
                            binding.normalization, domain="chart_sites"
                        ),
                    ),
                ),
            ),
        ),
        replace(
            spec,
            kernel=replace(
                spec.kernel,
                profiles=(
                    replace(
                        binding,
                        normalization=replace(binding.normalization, floor=1e-5),
                    ),
                ),
            ),
        ),
        replace(
            spec,
            kernel=replace(
                spec.kernel,
                parameterization=replace(spec.kernel.parameterization, sigma_max=4.0),
            ),
        ),
    ]
    for candidate in variants:
        with pytest.raises(ValueError, match="contract"):
            CudaOperatorSpec.from_declaration(candidate)


def test_log_width_declaration_validates_finite_positive_bounds():
    from torchcst.kernels import LogWidthSpec

    for lo, hi in ((0, 1), (-1, 1), (1, float("inf")), (1, float("nan")), (2, 1)):
        with pytest.raises(ValueError):
            LogWidthSpec(sigma_min=lo, sigma_max=hi)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "maker,algorithm",
    [
        (make_single, "materialized"),
        (make_pair, "materialized"),
        (make_pair, "factored"),
    ],
)
def test_operator_live_cuda_state_matches_cpu_values_and_gradients(maker, algorithm):
    cpu = maker()
    gpu = copy.deepcopy(cpu).cuda()
    x = torch.randn(2, cpu.in_features, dtype=torch.float64, requires_grad=True)
    gx = x.detach().cuda().requires_grad_()
    expected = cpu.operator.apply(x, algorithm=algorithm)
    actual = gpu.operator.apply(gx, algorithm=algorithm)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(gx.grad.cpu(), x.grad, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(
        gpu.atoms.p.grad.cpu(), cpu.atoms.p.grad, rtol=1e-6, atol=1e-8
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pair_graph_reuses_operator_without_configuration_reads(monkeypatch):
    model = make_pair(backend="factored").float().cuda()
    bound = model.operator
    x = torch.randn(2, model.in_features, device="cuda", requires_grad=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            x.grad = None
            model(x).square().sum().backward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    expected = model(x).detach().clone()
    expected_p = model.atoms.p.grad.detach().clone()
    expected_x = x.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    x.grad = None

    def forbidden(*args, **kwargs):
        raise AssertionError("configuration read during graph")

    for obj in (model, model.kernel, *model.cst_charts()):
        monkeypatch.setattr(obj, "declaration", forbidden)
    monkeypatch.setattr(model.kernel, "parameter_dim", forbidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = model(x)
        actual.square().sum().backward()
    graph.replay()
    torch.cuda.synchronize()
    assert model.operator is bound
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.atoms.p.grad, expected_p)
    torch.testing.assert_close(x.grad, expected_x)
