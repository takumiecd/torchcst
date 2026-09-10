import pytest
import torch
from test_training_smoke import amplitude_bandwidth

from torchcst import Chart, CSTLinear
from torchcst._derivatives._captured import call, frame_transport, local_derivatives
from torchcst._runtime.validation import device_checks


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_captured_geometry_matches_original_derivative_contract(device):
    torch.manual_seed(71)
    model = CSTLinear(
        Chart.linspace(3),
        Chart.linspace(2),
        atoms=3,
        kernel=amplitude_bandwidth(),
        dtype=torch.float64,
    ).to(device)
    geometry = model.cst_frame_geometry()
    point = geometry.current_point()
    source = geometry.frame(point + 0.01, torch.randn_like(point) * 0.02)
    alpha = torch.randn_like(point)
    expected = geometry.pullback_from_frame(
        current_point=point, source_frame=source, source_coefficients=alpha
    )
    materialize = geometry.derivatives._materialize_atoms
    actual = frame_transport(
        materialize, point, source.point, source.displacement, alpha
    )
    torch.testing.assert_close(actual, (expected.constant, expected.linear))
    reference = geometry.derivatives.materialized_local_derivatives(
        parameter_point=point
    )
    torch.testing.assert_close(local_derivatives(materialize, point), reference)
    if device == "cuda":
        for offset in (0.0, 0.01, 0.03):
            p = point + offset
            result = call("local_derivatives", materialize, p)
            torch.testing.assert_close(result, local_derivatives(materialize, p))
            result = call(
                "frame_transport",
                materialize,
                p,
                source.point,
                source.displacement,
                alpha,
            )
            torch.testing.assert_close(
                result,
                frame_transport(
                    materialize, p, source.point, source.displacement, alpha
                ),
            )
        with device_checks():
            actual = geometry.pullback_from_frame(
                current_point=point, source_frame=source, source_coefficients=alpha
            )
        torch.testing.assert_close(
            (actual.constant, actual.linear), (expected.constant, expected.linear)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_capture_invalidates_when_kernel_buffer_changes():
    model = CSTLinear(
        Chart.linspace(3),
        Chart.linspace(2),
        atoms=3,
        kernel=amplitude_bandwidth(),
        dtype=torch.float64,
    ).cuda()
    materialize = model.cst_derivatives()._materialize_atoms
    point = model.atoms.p.detach().clone()
    before = call("local_derivatives", materialize, point)
    saved = tuple(x.clone() for x in before)
    model.kernel.profile.sigma.mul_(1.5)
    after = call("local_derivatives", materialize, point)
    torch.testing.assert_close(after, local_derivatives(materialize, point))
    torch.testing.assert_close(before, saved)
    assert not torch.allclose(before[0], after[0])
