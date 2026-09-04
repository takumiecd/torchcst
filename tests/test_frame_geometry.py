import torch

from torchcst import Chart, CSTLinear, Gaussian, Separable
from torchcst._derivatives import AutogradFrameGeometry


def make_geometry() -> tuple[CSTLinear, AutogradFrameGeometry]:
    torch.manual_seed(23)
    site = CSTLinear(
        Chart.linspace(4),
        Chart.linspace(3),
        atoms=2,
        kernel=Separable(
            input_profile=Gaussian(0.8),
            output_profile=Gaussian(0.6),
        ),
        dtype=torch.float64,
    )
    return site, site.cst_frame_geometry()


def test_cross_frame_pullback_matches_explicit_visible_expansion() -> None:
    site, geometry = make_geometry()
    derivatives = site.cst_derivatives()
    current_point = geometry.current_point()
    source_point = current_point + 0.03 * torch.randn_like(current_point)
    source_displacement = 0.02 * torch.randn_like(current_point)
    source_coefficients = torch.randn_like(current_point)
    target_displacement = 0.04 * torch.randn_like(current_point)
    source_frame = geometry.frame(source_point, source_displacement)

    affine = geometry.pullback_from_frame(
        current_point=current_point,
        source_frame=source_frame,
        source_coefficients=source_coefficients,
    )

    visible = derivatives.pushforward(
        source_coefficients,
        at=source_displacement,
        parameter_point=source_point,
    )
    expected = derivatives.pullback(
        visible,
        at=target_displacement,
        parameter_point=current_point,
    )
    torch.testing.assert_close(affine.at(target_displacement), expected)


def test_gram_system_matches_visible_frame_inner_products() -> None:
    _, geometry = make_geometry()
    point = geometry.current_point()
    frame = geometry.frame(point, 0.05 * torch.randn_like(point))
    left = torch.randn_like(point)
    right = torch.randn_like(point)
    gram = geometry.gram(frame)

    local_inner = (left * gram.matvec(right)).sum()
    visible_inner = (
        geometry.visible_pushforward(frame, left)
        * geometry.visible_pushforward(frame, right)
    ).sum()

    torch.testing.assert_close(local_inner, visible_inner)
    torch.testing.assert_close(gram.matrix, gram.matrix.T)


def test_compression_solves_the_accepted_frame_normal_equation() -> None:
    site, geometry = make_geometry()
    derivatives = site.cst_derivatives()
    point = geometry.current_point()
    frame = geometry.frame(point, 0.03 * torch.randn_like(point))
    visible_target = torch.randn(
        site.out_features,
        site.in_features,
        dtype=point.dtype,
    )
    numerator = derivatives.pullback(
        visible_target,
        at=frame.displacement,
        parameter_point=frame.point,
    )

    coefficients = geometry.compress(
        frame=frame,
        pullback_numerator=numerator,
        rtol=1e-12,
    )

    torch.testing.assert_close(
        geometry.gram(frame).matvec(coefficients),
        numerator,
        rtol=1e-8,
        atol=1e-9,
    )


def test_frame_snapshots_do_not_alias_caller_tensors() -> None:
    _, geometry = make_geometry()
    point = geometry.current_point()
    displacement = torch.randn_like(point)
    expected_point = point.clone()
    expected_displacement = displacement.clone()

    frame = geometry.frame(point, displacement)
    point.zero_()
    displacement.zero_()

    torch.testing.assert_close(frame.point, expected_point)
    torch.testing.assert_close(frame.displacement, expected_displacement)

