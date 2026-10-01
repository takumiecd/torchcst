from __future__ import annotations

import torch
from kernel_cases import gaussian_state, polar_state, separable_state, triweight_state

from torchcst import BandwidthBounds, CSTLinear, CSTOptimizer
from torchcst._backends.torch.charts import construction as _construction
from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.geometry import execution as _geometry
from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.profiles import execution as _profile


def test_sphere_chart_separates_intrinsic_and_embedding_dimensions() -> None:
    torch.manual_seed(17)
    chart = _construction.sphere_chart(32, intrinsic_dim=3, radius=2.0)
    assert chart.features == 32
    assert chart.intrinsic_dim == 3
    assert chart.embedding_dim == 4
    assert chart.embedding_dim == 4
    torch.testing.assert_close(
        torch.linalg.vector_norm(chart.coordinates, dim=-1), torch.full((32,), 2.0)
    )


def test_intrinsic_sphere_centers_store_exactly_d_coordinates() -> None:
    torch.manual_seed(19)
    chart = _construction.sphere_chart(
        32, intrinsic_dim=3, radius=2.0, representation="intrinsic"
    )
    profile = triweight_state(1.8)
    centers = _profile.initialize(profile, chart, 7, mode="balanced")
    decoded = _geometry.decode_centers(chart.geometry, centers)
    assert chart.coordinates.shape == (32, 4)
    assert chart.center_parameter_dim == 3
    assert _profile.parameter_dim(profile, chart) == 3
    assert centers.shape == (7, 3)
    torch.testing.assert_close(
        torch.linalg.vector_norm(decoded, dim=-1), torch.full((7,), 2.0)
    )


def test_intrinsic_and_ambient_sphere_centers_have_matching_distances() -> None:
    coordinates = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    intrinsic_geometry = _construction.sphere(2, representation="intrinsic").double()
    intrinsic_chart = _construction.points(coordinates, geometry=intrinsic_geometry)
    ambient_chart = _construction.points(
        coordinates, geometry=_construction.sphere(2, representation="ambient").double()
    )
    intrinsic_centers = torch.tensor([[0.2, -0.4], [0.7, 0.3]], dtype=torch.float64)
    ambient_centers = _geometry.decode_centers(intrinsic_geometry, intrinsic_centers)
    torch.testing.assert_close(
        _charts.squared_distance(intrinsic_chart, intrinsic_centers),
        _charts.squared_distance(ambient_chart, ambient_centers),
    )


def test_sphere_retraction_and_transport_remain_tangent() -> None:
    geometry = _construction.sphere(2, radius=1.5).double()
    old = torch.tensor([[1.5, 0.0, 0.0]], dtype=torch.float64)
    displacement = torch.tensor([[2.0, 3.0, -4.0]], dtype=torch.float64)
    new = _geometry.retract(geometry, old, displacement)
    tangent = _geometry.transport(geometry, old, new, displacement)
    torch.testing.assert_close(
        torch.linalg.vector_norm(new, dim=-1), torch.tensor([1.5], dtype=torch.float64)
    )
    torch.testing.assert_close(
        (new * tangent).sum(dim=-1),
        torch.zeros(1, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_spherical_triweight_tangent_matches_autograd() -> None:
    coordinates = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    chart = _construction.points(coordinates, geometry=_construction.sphere(2).double())
    profile = triweight_state(1.8).double()
    centers = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    precision = torch.tensor([1 / 1.8**2], dtype=torch.float64)
    values, tangent, _ = _profile.tangent_with_precision(
        profile, chart, centers, precision
    )
    jacobian = torch.autograd.functional.jacobian(
        lambda p: _profile.evaluate_with_precision(profile, chart, p, precision),
        centers,
    )
    torch.testing.assert_close(values, _profile.evaluate(profile, chart, centers))
    torch.testing.assert_close(tangent[:, 0], jacobian[:, 0, 0])


def test_intrinsic_spherical_triweight_tangent_matches_autograd() -> None:
    coordinates = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    chart = _construction.points(
        coordinates,
        geometry=_construction.sphere(2, representation="intrinsic").double(),
    )
    profile = triweight_state(1.8).double()
    centers = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    precision = torch.tensor([1 / 1.8**2], dtype=torch.float64)
    _, tangent, _ = _profile.tangent_with_precision(profile, chart, centers, precision)
    jacobian = torch.autograd.functional.jacobian(
        lambda p: _profile.evaluate_with_precision(profile, chart, p, precision),
        centers,
    )
    torch.testing.assert_close(tangent[:, 0], jacobian[:, 0, 0])


def test_kernel_storage_width_and_intrinsic_dof_are_distinct() -> None:
    input_chart = _construction.sphere_chart(12, intrinsic_dim=2)
    output_chart = _construction.sphere_chart(8, intrinsic_dim=3)
    separable = separable_state(
        input_profile=triweight_state(1.0), output_profile=triweight_state(1.0)
    )
    polar = polar_state(
        amplitude_max=1.0,
        w_c=0.1,
        profile=triweight_state(1.0),
        input_bounds=BandwidthBounds(
            minimum=1.0, maximum=2.0, birth=2.0, upper_floor=1.0
        ),
    )
    assert _kernel.parameter_dim(separable, input_chart, output_chart) == 7
    assert _kernel.parameter_dof(separable, input_chart, output_chart) == 5
    assert _kernel.parameter_dim(polar, input_chart, output_chart) == 9
    assert _kernel.parameter_dof(polar, input_chart, output_chart) == 7


def test_intrinsic_sphere_kernel_storage_width_equals_its_dof() -> None:
    input_chart = _construction.sphere_chart(
        12, intrinsic_dim=2, representation="intrinsic"
    )
    output_chart = _construction.sphere_chart(
        8, intrinsic_dim=3, representation="intrinsic"
    )
    separable = separable_state(
        input_profile=triweight_state(1.0), output_profile=triweight_state(1.0)
    )
    polar = polar_state(
        amplitude_max=1.0,
        w_c=0.1,
        profile=triweight_state(1.0),
        input_bounds=BandwidthBounds(
            minimum=1.0, maximum=2.0, birth=2.0, upper_floor=1.0
        ),
    )
    assert _kernel.parameter_dim(separable, input_chart, output_chart) == 5
    assert _kernel.parameter_dof(separable, input_chart, output_chart) == 5
    assert _kernel.parameter_dim(polar, input_chart, output_chart) == 7
    assert _kernel.parameter_dof(polar, input_chart, output_chart) == 7


def test_intrinsic_sphere_update_remains_inside_fixed_chart() -> None:
    geometry = _construction.sphere(2, representation="intrinsic").double()
    old = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    displacement = torch.tensor([[100.0, -200.0]], dtype=torch.float64)
    new = _geometry.retract(geometry, old, displacement)
    assert torch.linalg.vector_norm(new) <= _geometry.max_parameter_radius(geometry)
    torch.testing.assert_close(
        torch.linalg.vector_norm(_geometry.decode_centers(geometry, new), dim=-1),
        torch.ones(1, dtype=torch.float64),
    )


def test_polar_update_retracts_both_center_blocks_to_their_spheres() -> None:
    torch.manual_seed(23)
    input_chart = _construction.sphere_chart(12, intrinsic_dim=2).double()
    output_chart = _construction.sphere_chart(8, intrinsic_dim=3).double()
    kernel = polar_state(
        amplitude_max=1.0,
        w_c=0.1,
        profile=triweight_state(1.0),
        input_bounds=BandwidthBounds(
            minimum=1.0, maximum=2.0, birth=2.0, upper_floor=1.0
        ),
    ).double()
    point = _kernel.initialize(kernel, input_chart, output_chart, 4, mode="uniform")
    displacement = torch.randn_like(point)
    updated = _kernel.apply_parameter_update(
        kernel, input_chart, output_chart, point, displacement, step_size=0.1
    )
    _, input_centers, output_centers = _kernel.coordinate(
        kernel, "_split", input_chart, output_chart, updated
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(input_centers, dim=-1),
        torch.ones(4, dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(output_centers, dim=-1),
        torch.ones(4, dtype=torch.float64),
    )


def test_parameter_adam_retracts_spherical_centers() -> None:
    torch.manual_seed(29)
    input_chart = _construction.sphere_chart(7, intrinsic_dim=2).double()
    output_chart = _construction.sphere_chart(5, intrinsic_dim=2).double()
    kernel = separable_state(
        input_profile=gaussian_state(1.0), output_profile=gaussian_state(1.0)
    ).double()
    model = CSTLinear(
        input_chart,
        output_chart,
        atoms=3,
        kernel=kernel.declaration(),
        backend="factored",
        dtype=torch.float64,
    )
    optimizer = CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(),
            lr=0.1,
            betas=(0.5, 0.99),
            eps=1e-08,
            weight_decay=0.0,
            foreach=False,
        ),
        model=model,
    )
    inputs = torch.randn(4, input_chart.features, dtype=torch.float64)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    input_dim = input_chart.embedding_dim
    input_centers = model.atoms.p[:, :input_dim]
    output_centers = model.atoms.p[:, input_dim:]
    torch.testing.assert_close(
        torch.linalg.vector_norm(input_centers, dim=-1),
        torch.ones(model.atom_count, dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(output_centers, dim=-1),
        torch.ones(model.atom_count, dtype=torch.float64),
    )
    first_moment = optimizer.state[model.atoms.p]["exp_avg"]
    input_moment = first_moment[:, :input_dim]
    output_moment = first_moment[:, input_dim:]
    torch.testing.assert_close(
        (input_centers * input_moment).sum(dim=-1),
        torch.zeros(model.atom_count, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        (output_centers * output_moment).sum(dim=-1),
        torch.zeros(model.atom_count, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    assert model.atom_parameter_dof == 4
    assert model.cst_degrees_of_freedom == 12


def test_parameter_adam_uses_d_coordinate_spherical_centers() -> None:
    torch.manual_seed(31)
    input_chart = _construction.sphere_chart(
        7, intrinsic_dim=2, representation="intrinsic"
    ).double()
    output_chart = _construction.sphere_chart(
        5, intrinsic_dim=2, representation="intrinsic"
    ).double()
    kernel = polar_state(
        amplitude_max=1.0,
        w_c=0.1,
        profile=triweight_state(1.5),
        input_bounds=BandwidthBounds(
            minimum=1.5, maximum=2.0, birth=2.0, upper_floor=1.5
        ),
    ).double()
    model = CSTLinear(
        input_chart,
        output_chart,
        atoms=3,
        kernel=kernel.declaration(),
        backend="factored",
        dtype=torch.float64,
    )
    optimizer = CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(),
            lr=0.1,
            betas=(0.5, 0.99),
            eps=1e-08,
            weight_decay=0.0,
            foreach=False,
        ),
        model=model,
    )
    inputs = torch.randn(4, input_chart.features, dtype=torch.float64)
    model(inputs).square().mean().backward()
    optimizer.step()
    assert model.atoms.p.shape == (3, 6)
    input_centers = model.atoms.p[:, 2:4]
    output_centers = model.atoms.p[:, 4:]
    assert torch.all(
        torch.linalg.vector_norm(input_centers, dim=-1)
        <= _geometry.max_parameter_radius(input_chart.geometry)
    )
    assert torch.all(
        torch.linalg.vector_norm(output_centers, dim=-1)
        <= _geometry.max_parameter_radius(output_chart.geometry)
    )
    assert optimizer.state[model.atoms.p]["exp_avg"].shape == (3, 6)
