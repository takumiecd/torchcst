import torch

from torchcst import (
    Chart,
    CSTLinear,
    CSTParameterAdam,
    Gaussian,
    ParameterAdamConfig,
    PolarAmpWidth,
    Separable,
    SphereGeometry,
    Triweight,
)


def test_sphere_chart_separates_intrinsic_and_embedding_dimensions() -> None:
    torch.manual_seed(17)
    chart = Chart.sphere(32, intrinsic_dim=3, radius=2.0)

    assert chart.features == 32
    assert chart.intrinsic_dim == 3
    assert chart.embedding_dim == 4
    assert chart.dim == 4
    torch.testing.assert_close(
        torch.linalg.vector_norm(chart.coordinates, dim=-1),
        torch.full((32,), 2.0),
    )


def test_intrinsic_sphere_centers_store_exactly_d_coordinates() -> None:
    torch.manual_seed(19)
    chart = Chart.sphere(
        32,
        intrinsic_dim=3,
        radius=2.0,
        representation="intrinsic",
    )
    profile = Triweight(1.8)
    centers = profile.initialize(chart, 7, mode="balanced")
    decoded = chart.geometry.decode_centers(centers)

    assert chart.coordinates.shape == (32, 4)
    assert chart.center_parameter_dim == 3
    assert profile.parameter_dim(chart) == 3
    assert centers.shape == (7, 3)
    torch.testing.assert_close(
        torch.linalg.vector_norm(decoded, dim=-1),
        torch.full((7,), 2.0),
    )


def test_intrinsic_and_ambient_sphere_centers_have_matching_distances() -> None:
    coordinates = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    intrinsic_geometry = SphereGeometry(2, representation="intrinsic").double()
    intrinsic_chart = Chart.points(coordinates, geometry=intrinsic_geometry)
    ambient_chart = Chart.points(
        coordinates,
        geometry=SphereGeometry(2, representation="ambient").double(),
    )
    intrinsic_centers = torch.tensor([[0.2, -0.4], [0.7, 0.3]], dtype=torch.float64)
    ambient_centers = intrinsic_geometry.decode_centers(intrinsic_centers)

    torch.testing.assert_close(
        intrinsic_chart.squared_distance(intrinsic_centers),
        ambient_chart.squared_distance(ambient_centers),
    )


def test_sphere_retraction_and_transport_remain_tangent() -> None:
    geometry = SphereGeometry(2, radius=1.5).double()
    old = torch.tensor([[1.5, 0.0, 0.0]], dtype=torch.float64)
    displacement = torch.tensor([[2.0, 3.0, -4.0]], dtype=torch.float64)

    new = geometry.retract(old, displacement)
    tangent = geometry.transport(old, new, displacement)

    torch.testing.assert_close(
        torch.linalg.vector_norm(new, dim=-1),
        torch.tensor([1.5], dtype=torch.float64),
    )
    torch.testing.assert_close(
        (new * tangent).sum(dim=-1),
        torch.zeros(1, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_spherical_triweight_tangent_matches_autograd() -> None:
    coordinates = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    chart = Chart.points(coordinates, geometry=SphereGeometry(2).double())
    profile = Triweight(1.8).double()
    centers = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    precision = torch.tensor([1 / 1.8**2], dtype=torch.float64)

    values, tangent, _ = profile.tangent_with_precision(
        chart,
        centers,
        precision,
    )
    jacobian = torch.autograd.functional.jacobian(
        lambda p: profile.evaluate_with_precision(chart, p, precision),
        centers,
    )

    torch.testing.assert_close(values, profile.evaluate(chart, centers))
    torch.testing.assert_close(tangent[:, 0], jacobian[:, 0, 0])


def test_intrinsic_spherical_triweight_tangent_matches_autograd() -> None:
    coordinates = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    chart = Chart.points(
        coordinates,
        geometry=SphereGeometry(2, representation="intrinsic").double(),
    )
    profile = Triweight(1.8).double()
    centers = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    precision = torch.tensor([1 / 1.8**2], dtype=torch.float64)

    _, tangent, _ = profile.tangent_with_precision(chart, centers, precision)
    jacobian = torch.autograd.functional.jacobian(
        lambda p: profile.evaluate_with_precision(chart, p, precision),
        centers,
    )

    torch.testing.assert_close(tangent[:, 0], jacobian[:, 0, 0])


def test_kernel_storage_width_and_intrinsic_dof_are_distinct() -> None:
    input_chart = Chart.sphere(12, intrinsic_dim=2)
    output_chart = Chart.sphere(8, intrinsic_dim=3)
    separable = Separable(
        input_profile=Triweight(1.0),
        output_profile=Triweight(1.0),
    )
    polar = PolarAmpWidth(
        amplitude_max=1.0,
        sigma_min=1.0,
        sigma_max=2.0,
        w_c=0.1,
        profile=Triweight(1.0),
    )

    assert separable.parameter_dim(input_chart, output_chart) == 7
    assert separable.parameter_dof(input_chart, output_chart) == 5
    assert polar.parameter_dim(input_chart, output_chart) == 9
    assert polar.parameter_dof(input_chart, output_chart) == 7


def test_intrinsic_sphere_kernel_storage_width_equals_its_dof() -> None:
    input_chart = Chart.sphere(12, intrinsic_dim=2, representation="intrinsic")
    output_chart = Chart.sphere(8, intrinsic_dim=3, representation="intrinsic")
    separable = Separable(
        input_profile=Triweight(1.0),
        output_profile=Triweight(1.0),
    )
    polar = PolarAmpWidth(
        amplitude_max=1.0,
        sigma_min=1.0,
        sigma_max=2.0,
        w_c=0.1,
        profile=Triweight(1.0),
    )

    assert separable.parameter_dim(input_chart, output_chart) == 5
    assert separable.parameter_dof(input_chart, output_chart) == 5
    assert polar.parameter_dim(input_chart, output_chart) == 7
    assert polar.parameter_dof(input_chart, output_chart) == 7


def test_intrinsic_sphere_update_remains_inside_fixed_chart() -> None:
    geometry = SphereGeometry(2, representation="intrinsic").double()
    old = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    displacement = torch.tensor([[100.0, -200.0]], dtype=torch.float64)

    new = geometry.retract(old, displacement)

    assert torch.linalg.vector_norm(new) <= geometry.max_parameter_radius
    torch.testing.assert_close(
        torch.linalg.vector_norm(geometry.decode_centers(new), dim=-1),
        torch.ones(1, dtype=torch.float64),
    )


def test_polar_update_retracts_both_center_blocks_to_their_spheres() -> None:
    torch.manual_seed(23)
    input_chart = Chart.sphere(12, intrinsic_dim=2).double()
    output_chart = Chart.sphere(8, intrinsic_dim=3).double()
    kernel = PolarAmpWidth(
        amplitude_max=1.0,
        sigma_min=1.0,
        sigma_max=2.0,
        w_c=0.1,
        profile=Triweight(1.0),
    ).double()
    point = kernel.initialize(input_chart, output_chart, 4, mode="uniform")
    displacement = torch.randn_like(point)

    updated = kernel.apply_parameter_update(
        input_chart,
        output_chart,
        point,
        displacement,
        step_size=0.1,
    )
    _, input_centers, output_centers = kernel._split(
        input_chart,
        output_chart,
        updated,
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
    input_chart = Chart.sphere(7, intrinsic_dim=2).double()
    output_chart = Chart.sphere(5, intrinsic_dim=2).double()
    kernel = Separable(
        input_profile=Gaussian(1.0),
        output_profile=Gaussian(1.0),
    ).double()
    model = CSTLinear(
        input_chart,
        output_chart,
        atoms=3,
        kernel=kernel,
        backend="factored",
        dtype=torch.float64,
    )
    optimizer = CSTParameterAdam(
        model,
        cst=ParameterAdamConfig(lr=0.1, decay_steps=None),
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
    input_chart = Chart.sphere(7, intrinsic_dim=2, representation="intrinsic").double()
    output_chart = Chart.sphere(5, intrinsic_dim=2, representation="intrinsic").double()
    kernel = PolarAmpWidth(
        amplitude_max=1.0,
        sigma_min=1.5,
        sigma_max=2.0,
        w_c=0.1,
        profile=Triweight(1.5),
    ).double()
    model = CSTLinear(
        input_chart,
        output_chart,
        atoms=3,
        kernel=kernel,
        backend="factored",
        dtype=torch.float64,
    )
    optimizer = CSTParameterAdam(
        model,
        cst=ParameterAdamConfig(lr=0.1, decay_steps=None),
    )

    inputs = torch.randn(4, input_chart.features, dtype=torch.float64)
    model(inputs).square().mean().backward()
    optimizer.step()

    assert model.atoms.p.shape == (3, 6)
    input_centers = model.atoms.p[:, 2:4]
    output_centers = model.atoms.p[:, 4:]
    assert torch.all(
        torch.linalg.vector_norm(input_centers, dim=-1)
        <= input_chart.geometry.max_parameter_radius
    )
    assert torch.all(
        torch.linalg.vector_norm(output_centers, dim=-1)
        <= output_chart.geometry.max_parameter_radius
    )
    assert optimizer.state[model.atoms.p]["exp_avg"].shape == (3, 6)
