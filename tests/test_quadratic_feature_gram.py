import pytest
import torch

from torchcst import (
    Amplitude,
    AmplitudeBandwidthSeparable,
    Chart,
    CSTLinear,
    FullQuartic,
    Gaussian,
    SecondOrderAdamConfig,
    Separable,
)
from torchcst._derivatives import AffinePullback
from torchcst.optim import MomentContext, QuarticProblem
from torchcst.optim.moments import (
    ExpandedFirstMoment,
    ExpandedMoments,
    ExpandedSecondMoment,
)
from torchcst.optim.moments.second import SeparableDiagonalMetric


def make_pair(
    dtype,
    gated,
    *,
    eps=1e-3,
    zero_metric=False,
    input_size=5,
    output_size=4,
    device="cpu",
):
    torch.manual_seed(402)
    kernel = (
        AmplitudeBandwidthSeparable(
            sigma_min=0.4, sigma_max=0.6, tau=0.15, temperature=0.5
        )
        if gated
        else Amplitude(
            Separable(input_profile=Gaussian(0.6), output_profile=Gaussian(0.4))
        )
    )
    site = CSTLinear(
        Chart.linspace(input_size, low=-1.0, high=1.0),
        Chart.linspace(output_size, low=-1.0, high=1.0),
        atoms=3,
        kernel=kernel,
        dtype=dtype,
    ).to(device)
    with torch.no_grad():
        site.atoms.p[:, 0] = torch.tensor([-0.15, 0.0, 0.2], dtype=dtype, device=device)
    geometry = site.cst_frame_geometry()
    context = MomentContext(geometry, geometry.current_point())
    first = AffinePullback(
        torch.randn_like(context.current_point),
        torch.randn(
            3,
            site.atoms.parameter_dim,
            site.atoms.parameter_dim,
            dtype=dtype,
            device=device,
        ),
    )
    row = torch.rand(output_size, dtype=dtype, device=device)
    if zero_metric:
        row.zero_()
    metric = SeparableDiagonalMetric(
        row, torch.rand(input_size, dtype=dtype, device=device), eps=eps
    )
    moments = ExpandedMoments(
        previous_step=0,
        next_step=1,
        owner_token=object(),
        first=ExpandedFirstMoment(first, first, None),
        second=ExpandedSecondMoment(metric, None),
    )
    return (
        QuarticProblem(context, moments, learning_rate=0.2, evaluation="visible"),
        QuarticProblem(context, moments, learning_rate=0.2, evaluation="gram"),
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("zero_metric", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_factored_gram_matches_visible_autograd(dtype, gated, zero_metric, device):
    visible, gram = make_pair(dtype, gated, zero_metric=zero_metric, device=device)
    assert gram.evaluation_backend == "gram"
    for scale in (0.0, 0.01, 0.25, 1.0):
        d = (scale * torch.randn_like(visible.context.current_point)).requires_grad_()
        expected = visible.value(d)
        expected_gradient = torch.autograd.grad(expected, d)[0]
        actual, gradient = gram.value_and_gradient(d.detach())
        tolerance = (
            {"rtol": 3e-5, "atol": 3e-5}
            if dtype == torch.float32
            else {"rtol": 1e-10, "atol": 1e-10}
        )
        torch.testing.assert_close(actual, expected, **tolerance)
        torch.testing.assert_close(gradient, expected_gradient, **tolerance)
        torch.testing.assert_close(gram.value(d), expected, **tolerance)
        torch.testing.assert_close(
            torch.autograd.grad(gram.value(d), d)[0], gradient, **tolerance
        )


def test_gram_retains_cross_atom_terms_and_eps():
    visible, gram = make_pair(torch.float64, True, eps=0.7)
    d = 0.1 * torch.randn_like(visible.context.current_point)
    full_value, full_gradient = gram.value_and_gradient(d)
    torch.testing.assert_close(full_value, visible.value(d))
    torch.testing.assert_close(full_gradient, visible.gradient(d))
    separate_sum = d.new_zeros(())
    for atom in range(d.shape[0]):
        single = torch.zeros_like(d)
        single[atom] = d[atom]
        separate_sum += gram.value(single)
    assert (full_value - separate_sum).abs() > 1e-5


def test_full_solver_agrees_on_both_evaluation_paths():
    visible, gram = make_pair(torch.float64, False)
    solver = FullQuartic(
        starts=2, max_iter=35, tolerance_grad=1e-9, tolerance_change=1e-12
    )
    reference = solver.solve(visible, trust_radius=0.15)
    actual = solver.solve(gram, trust_radius=0.15)
    torch.testing.assert_close(
        visible.value(actual.displacement), reference.objective, rtol=1e-7, atol=1e-9
    )
    torch.testing.assert_close(
        actual.displacement, reference.displacement, rtol=1e-5, atol=1e-6
    )


def test_auto_uses_visible_for_small_problems_and_validates_selection():
    visible, _ = make_pair(torch.float64, False)
    problem = QuarticProblem(visible.context, visible.moments, learning_rate=0.2)
    assert problem.evaluation_backend == "visible"
    with pytest.raises(ValueError, match="evaluation must"):
        QuarticProblem(
            visible.context, visible.moments, learning_rate=0.2, evaluation="bad"
        )
    with pytest.raises(ValueError, match="quartic_evaluation must"):
        SecondOrderAdamConfig(quartic_evaluation="bad")


def test_auto_gram_does_not_materialize_visible_derivatives(monkeypatch):
    visible, _ = make_pair(torch.float64, True, input_size=16, output_size=16)
    geometry = visible.context.geometry
    d = 0.1 * torch.randn_like(visible.context.current_point)
    expected_value, expected_gradient = visible.value_and_gradient(d)

    def forbidden(*args, **kwargs):
        raise AssertionError("Gram construction and evaluation must use only factors")

    monkeypatch.setattr(geometry.derivatives, "_materialize_atoms", forbidden)
    monkeypatch.setattr(geometry, "displacement", forbidden)
    monkeypatch.setattr(geometry, "pullback", forbidden)
    problem = QuarticProblem(visible.context, visible.moments, learning_rate=0.2)
    assert problem.evaluation_backend == "gram"
    value, gradient = problem.value_and_gradient(d)
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(gradient, expected_gradient)


def test_unsupported_kernel_keeps_visible_backend():
    visible, _ = make_pair(torch.float64, False)
    visible.context.geometry.derivatives.factor_atoms = None
    problem = QuarticProblem(visible.context, visible.moments, learning_rate=0.2)
    assert problem.evaluation_backend == "visible"
    with pytest.raises(ValueError, match="gram evaluation requires"):
        QuarticProblem(
            visible.context, visible.moments, learning_rate=0.2, evaluation="gram"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_auto_keeps_visible_but_explicit_gram_is_supported():
    visible, gram = make_pair(
        torch.float64, True, input_size=16, output_size=16, device="cuda"
    )
    automatic = QuarticProblem(visible.context, visible.moments, learning_rate=0.2)
    assert automatic.evaluation_backend == "visible"
    assert gram.evaluation_backend == "gram"
