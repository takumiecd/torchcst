"""Radius declarations preserve legacy checkpoints, proposals and dispatch limits."""

import copy
import io
from dataclasses import asdict, replace

import pytest
import torch

from torchcst import (
    BandwidthBounds,
    ChebyshevSpec,
    CSTLinear,
    CSTOptimizer,
    GaussianSpec,
    RadiusBounds,
    chart_presets,
    kernels,
)
from torchcst._backends.schema import DefaultRecipe, DeviceInfo
from torchcst._backends.torch.algorithms.atom_update.algorithm import (
    AtomUpdateAlgorithm,
)
from torchcst._backends.torch.algorithms.linear.factored.algorithm import (
    FactoredAlgorithm,
)
from torchcst._backends.torch.algorithms.linear.materialized.algorithm import (
    MaterializedAlgorithm,
)
from torchcst._backends.torch.algorithms.polar_update.algorithm import (
    PolarUpdateAlgorithm,
)
from torchcst.kernels.state import KernelState
from torchcst.operators.atom_update import AtomUpdateBinding, AtomUpdateInputs
from torchcst.operators.execution import LinearInputs


def bounds(kind):
    return kind(minimum=0.2, birth=0.7, maximum=2.0, upper_floor=0.5)


def polar_kernel(*, radius, activity_mode="finite_chord"):
    settings = {
        "amplitude_max": 3.0,
        "w_c": 0.7,
        "kappa": 2.5,
        "lower_kappa": 1.7,
        "upper_decay_power": 0.8,
        "alpha_init": 0.4,
        "radial_regularization": 0.12,
        "activity_gain": 0.8,
        "activity_mode": activity_mode,
        "dormant_expansion_rate": 0.2,
    }
    if radius:
        return kernels.product(
            kernels.cell_mass(),
            coordinates=kernels.polar(input_bounds=bounds(RadiusBounds), **settings),
        )
    return kernels.product(
        kernels.l2(GaussianSpec()),
        coordinates=kernels.polar(input_bounds=bounds(BandwidthBounds), **settings),
    )


def model(kernel, *, atom_count=5, backend="factored"):
    chart = chart_presets.regular_grid(16, spacing=1.0, geometry="flat_torus")
    angle = torch.linspace(0.3, 2.7, atom_count, dtype=torch.float64)
    magnitude = torch.linspace(1.0, 2.0, atom_count, dtype=torch.float64)
    atoms = torch.stack(
        (
            angle.cos() * magnitude,
            angle.sin() * magnitude,
            torch.linspace(0.2, 12.1, atom_count, dtype=torch.float64),
            torch.linspace(1.1, 15.2, atom_count, dtype=torch.float64),
        ),
        dim=1,
    )
    return CSTLinear(
        chart, chart, atoms=atoms, kernel=kernel, backend=backend, dtype=torch.float64
    )


@pytest.mark.parametrize("radius", [False, True])
def test_polar_module_and_optimizer_checkpoint_roundtrip_and_cross_rejection(radius):
    source = model(polar_kernel(radius=radius))
    optimizer = CSTOptimizer(
        torch.optim.AdamW(source.parameters(), lr=0.03), model=source
    )
    source.atoms.p.grad = torch.ones_like(source.atoms.p) * 0.2
    optimizer.step()
    stream = io.BytesIO()
    torch.save((source.state_dict(), optimizer.state_dict()), stream)
    stream.seek(0)
    saved_model, saved_optimizer = torch.load(stream, weights_only=True)
    target = model(polar_kernel(radius=radius))
    target.load_state_dict(saved_model)
    restored = CSTOptimizer(
        torch.optim.AdamW(target.parameters(), lr=0.03), model=target
    )
    restored.load_state_dict(saved_optimizer)
    torch.testing.assert_close(source.atoms.p, target.atoms.p, rtol=0, atol=0)
    assert optimizer._manifest() == restored._manifest()
    for key in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(
            optimizer.state[source.atoms.p][key],
            restored.state[target.atoms.p][key],
            rtol=0,
            atol=0,
        )
    other = model(polar_kernel(radius=not radius))
    with pytest.raises(RuntimeError, match="checkpoint contract differs"):
        other.load_state_dict(copy.deepcopy(saved_model))
    other_optimizer = CSTOptimizer(
        torch.optim.AdamW(other.parameters(), lr=0.03), model=other
    )
    with pytest.raises(ValueError, match="checkpoint contract differs"):
        other_optimizer.load_state_dict(saved_optimizer)


@pytest.mark.parametrize("radius", [False, True])
def test_state_snapshot_preserves_distinct_bounds_and_legacy_buffer_names(radius):
    declaration = polar_kernel(radius=radius)
    state = KernelState(declaration).double()
    snapshot = state.declaration()
    assert asdict(KernelState(snapshot).double().declaration()) == asdict(snapshot)
    assert type(state.declaration().parameterization.input_bounds) is (
        RadiusBounds if radius else BandwidthBounds
    )
    names = set(state._buffers)
    if radius:
        assert "radius_min_input" in names
        assert "radius_upper_floor_output" in names
        assert not any("sigma" in name for name in names)
    else:
        assert names == {
            "amplitude_max",
            "w_c",
            "kappa",
            "lower_kappa",
            "upper_decay_power",
            "sigma_min_input",
            "sigma_birth_input",
            "sigma_max_input",
            "upper_floor_input",
            "sigma_min_output",
            "sigma_birth_output",
            "sigma_max_output",
            "upper_floor_output",
            "alpha_init",
            "radial_regularization",
            "activity_gain",
            "dormant_expansion_rate",
        }
        assert state.get_extra_state() == {
            "version": 1,
            "composition": "separable",
            "center_order": "charts",
            "revision": 1,
            "parameterization_type": "PolarAmpWidthSpec",
            "parameterization_flags": (("id", "polar_activity_width"),),
            "initialization": ("polar_centers", ("alpha_init",)),
            "update": (
                "polar_activity_width",
                (
                    ("radial_regularization", None),
                    ("activity_gain", None),
                    ("activity_mode", "finite_chord"),
                    ("dormant_expansion_rate", None),
                ),
            ),
        }
        assert state.profiles[0].get_extra_state() == {
            "version": 1,
            "profile": {
                "id": "gaussian",
                "revision": 1,
            },
            "normalization": {
                "kind": "discrete_l2",
                "domain": "chart_sites",
                "floor": None,
            },
            "fixed_width": False,
        }
    state.scalar("w_c").fill_(0.9)
    restored = KernelState(declaration).double()
    restored.load_state_dict(state.state_dict())
    assert asdict(restored.declaration()) == asdict(state.declaration())


def test_radius_and_sigma_bounds_and_keyword_names_cannot_alias():
    assert not isinstance(bounds(RadiusBounds), BandwidthBounds)
    assert not isinstance(bounds(BandwidthBounds), RadiusBounds)
    radius_rule = kernels.polar(amplitude_max=3.0, w_c=0.7, bounds=bounds(RadiusBounds))
    sigma_rule = kernels.polar(
        amplitude_max=3.0, w_c=0.7, bounds=bounds(BandwidthBounds)
    )
    with pytest.raises(ValueError, match="radius cell masses"):
        kernels.product(kernels.l2(GaussianSpec()), coordinates=radius_rule)
    with pytest.raises(ValueError, match="radius cell masses"):
        kernels.product(kernels.cell_mass(), coordinates=sigma_rule)
    for input_kind, output_kind in [
        (RadiusBounds, BandwidthBounds),
        (BandwidthBounds, RadiusBounds),
    ]:
        with pytest.raises(TypeError, match="bounds"):
            kernels.polar(
                amplitude_max=3.0,
                w_c=0.7,
                input_bounds=bounds(input_kind),
                output_bounds=bounds(output_kind),
            )
    with pytest.raises(ValueError, match="shared bounds"):
        kernels.polar(
            amplitude_max=3.0,
            w_c=0.7,
            bounds=bounds(RadiusBounds),
            input_bounds=bounds(RadiusBounds),
        )
    with pytest.raises(TypeError, match="input_radius_bounds"):
        kernels.polar(
            amplitude_max=3.0, w_c=0.7, input_radius_bounds=bounds(RadiusBounds)
        )
    with pytest.raises(TypeError, match="sigma_init"):
        kernels.physical(sigma_init=(0.3, 1.0))


@pytest.mark.parametrize("activity_mode", ["finite_chord", "time_energy"])
@pytest.mark.parametrize(
    "optimizer_type", [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW]
)
def test_fixed_gradient_updates_and_moments_equal_legacy_polar(
    activity_mode, optimizer_type
):
    old = model(polar_kernel(radius=False, activity_mode=activity_mode))
    new = model(polar_kernel(radius=True, activity_mode=activity_mode))
    options = (
        {"momentum": 0.8} if optimizer_type is torch.optim.SGD else {"amsgrad": True}
    )
    left = CSTOptimizer(
        optimizer_type(old.parameters(), lr=0.025, **options), model=old
    )
    right = CSTOptimizer(
        optimizer_type(new.parameters(), lr=0.025, **options), model=new
    )
    generator = torch.Generator().manual_seed(733)
    for _ in range(5):
        gradient = torch.randn(
            old.atoms.p.shape, generator=generator, dtype=torch.float64
        )
        old.atoms.p.grad = gradient.clone()
        new.atoms.p.grad = gradient.clone()
        left.step()
        right.step()
        torch.testing.assert_close(old.atoms.p, new.atoms.p, rtol=0, atol=0)
        assert left.state[old.atoms.p].keys() == right.state[new.atoms.p].keys()
        for key, value in left.state[old.atoms.p].items():
            torch.testing.assert_close(
                value, right.state[new.atoms.p][key], rtol=0, atol=0
            )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("polar_coordinates", [False, True])
def test_linear_algorithms_reject_radius_cuda_metadata_including_amplitude_wrapper(
    wrapped,
    polar_coordinates,
):
    layer = model(
        polar_kernel(radius=True)
        if polar_coordinates
        else kernels.product(kernels.cell_mass(), coordinates=kernels.physical())
    )
    context = layer.build_context(LinearInputs(torch.ones(2, 16, dtype=torch.float64)))
    if wrapped:
        context = replace(
            context,
            operator=replace(
                context.operator, kernel=kernels.amplitude(context.operator.kernel)
            ),
            parameter_dim=5,
        )
    recipe = DefaultRecipe()
    for algorithm in (FactoredAlgorithm(), MaterializedAlgorithm()):
        assert algorithm.supports(context, recipe).supported
        result = algorithm.supports(
            replace(context, device=DeviceInfo("cuda", 0)), recipe
        )
        assert not result.supported
        assert "CPU" in " ".join(result.reasons)


@pytest.mark.parametrize("polar_coordinates", [False, True])
def test_atom_update_algorithms_reject_radius_cuda_metadata(polar_coordinates):
    layer = model(
        polar_kernel(radius=True)
        if polar_coordinates
        else kernels.product(kernels.cell_mass(), coordinates=kernels.physical())
    )
    binding = AtomUpdateBinding(layer.operator)
    context = binding.build_context(
        AtomUpdateInputs(layer.atoms.p.detach().clone(), 0.01)
    )
    recipe = DefaultRecipe()
    algorithms = (
        (AtomUpdateAlgorithm(), PolarUpdateAlgorithm())
        if polar_coordinates
        else (AtomUpdateAlgorithm(),)
    )
    for algorithm in algorithms:
        assert algorithm.supports(context, recipe).supported
        assert not algorithm.supports(
            replace(context, device=DeviceInfo("cuda", 0)), recipe
        ).supported
    wrapped = replace(
        context, kernel=kernels.amplitude(context.kernel), parameter_shape=(5, 5)
    )
    assert (
        not AtomUpdateAlgorithm()
        .supports(replace(wrapped, device=DeviceInfo("cuda", 0)), recipe)
        .supported
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("polar_coordinates", [False, True])
def test_radius_linear_support_rejects_cpu_low_precision_metadata(
    dtype, wrapped, polar_coordinates
):
    layer = model(
        polar_kernel(radius=True)
        if polar_coordinates
        else kernels.product(kernels.cell_mass(), coordinates=kernels.physical())
    )
    context = layer.build_context(LinearInputs(torch.ones(2, 16, dtype=torch.float64)))
    if wrapped:
        context = replace(
            context,
            operator=replace(
                context.operator, kernel=kernels.amplitude(context.operator.kernel)
            ),
            parameter_dim=5,
        )
    unsupported = replace(context, dtype=dtype)
    for algorithm in (FactoredAlgorithm(), MaterializedAlgorithm()):
        result = algorithm.supports(unsupported, DefaultRecipe())
        assert not result.supported
        assert "float32 or float64" in " ".join(result.reasons)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("polar_coordinates", [False, True])
def test_radius_update_support_rejects_cpu_low_precision_metadata(
    dtype, polar_coordinates
):
    layer = model(
        polar_kernel(radius=True)
        if polar_coordinates
        else kernels.product(kernels.cell_mass(), coordinates=kernels.physical())
    )
    context = AtomUpdateBinding(layer.operator).build_context(
        AtomUpdateInputs(layer.atoms.p.detach().clone(), 0.01)
    )
    unsupported = replace(context, dtype=dtype)
    algorithms = (
        (AtomUpdateAlgorithm(), PolarUpdateAlgorithm())
        if polar_coordinates
        else (AtomUpdateAlgorithm(),)
    )
    for algorithm in algorithms:
        result = algorithm.supports(unsupported, DefaultRecipe())
        assert not result.supported
        assert "float32 or float64" in " ".join(result.reasons)
    wrapped = replace(
        unsupported,
        kernel=kernels.amplitude(unsupported.kernel),
        parameter_shape=(5, 5),
    )
    assert not AtomUpdateAlgorithm().supports(wrapped, DefaultRecipe()).supported


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_legacy_sigma_support_keeps_all_four_floating_point_dtypes(dtype):
    layer = model(polar_kernel(radius=False))
    linear = replace(
        layer.build_context(LinearInputs(torch.ones(2, 16, dtype=torch.float64))),
        dtype=dtype,
    )
    for wrapped in (False, True):
        context = (
            replace(
                linear,
                operator=replace(
                    linear.operator, kernel=kernels.amplitude(linear.operator.kernel)
                ),
                parameter_dim=5,
            )
            if wrapped
            else linear
        )
        for algorithm in (FactoredAlgorithm(), MaterializedAlgorithm()):
            assert algorithm.supports(context, DefaultRecipe()).supported
    update = replace(
        AtomUpdateBinding(layer.operator).build_context(
            AtomUpdateInputs(layer.atoms.p.detach().clone(), 0.01)
        ),
        dtype=dtype,
    )
    for algorithm in (AtomUpdateAlgorithm(), PolarUpdateAlgorithm()):
        assert algorithm.supports(update, DefaultRecipe()).supported


@pytest.mark.parametrize("polar_coordinates", [False, True])
def test_gaussian_cannot_enter_radius_integrated_factors(polar_coordinates):
    with pytest.raises(ValueError, match="Chebyshev profile"):
        kernels.cell_mass(GaussianSpec())
    with pytest.raises(ValueError, match="Chebyshev profile"):
        kernels.product(
            kernels.cell_mass(GaussianSpec()), coordinates=kernels.physical()
        )
    if polar_coordinates:
        with pytest.raises(ValueError, match="Chebyshev profile"):
            kernels.product(
                kernels.cell_mass(GaussianSpec()),
                coordinates=kernels.polar(
                    amplitude_max=3.0, w_c=0.7, input_bounds=bounds(RadiusBounds)
                ),
            )
    assert (
        kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=3)), coordinates=kernels.physical()
        ).revision
        == 7
    )


@pytest.mark.parametrize("polar_coordinates", [False, True])
def test_radius_dense_export_uses_chunked_weight_without_atom_operator_tensor(
    monkeypatch, polar_coordinates
):
    from torchcst._backends.torch.kernels import execution

    kernel = (
        polar_kernel(radius=True)
        if polar_coordinates
        else kernels.product(kernels.cell_mass(), coordinates=kernels.physical())
    )
    layer = model(kernel, atom_count=131, backend="materialized")
    if not polar_coordinates:
        with torch.no_grad():
            layer.atoms.p[:, 1].fill_(0.7)
    expected = layer.operator.factors()
    expected = expected[1] @ expected[0].T

    def forbidden(*args, **kwargs):
        raise AssertionError("dense export must not allocate [atoms,output,input]")

    monkeypatch.setattr(execution, "materialize_atoms", forbidden)
    torch.testing.assert_close(layer.dense_weight(), expected, rtol=2e-14, atol=2e-14)
    inputs = torch.ones(2, 16, dtype=torch.float64)
    torch.testing.assert_close(
        layer(inputs), inputs @ expected.T, rtol=2e-14, atol=2e-14
    )
