"""Identity coordinate maps retain validation without rebuilding their tensors."""

import copy

import pytest
import torch
from kernel_cases import polar_state
from test_cst_optimizer import site
from test_optimizer_amplitude_decay import layer, optimizer
from test_radius_polar_cuda_update import radius_model

from torchcst import BandwidthBounds, OptimizerStateAdapter
from torchcst._backends.torch.algorithms.polar_update import executor as polar


def euclidean_model(*, sphere=False):
    return site(
        sphere=sphere,
        kernel=polar_state(
            amplitude_max=1.0,
            w_c=0.1,
            input_bounds=BandwidthBounds(
                minimum=0.1, maximum=2.0, birth=1.0, upper_floor=0.1
            ),
        ),
    )


@pytest.mark.parametrize("geometry", ["flat_torus", "euclidean"])
def test_identity_maps_return_original_tensor_after_profile_validation(
    geometry, monkeypatch
):
    model = radius_model() if geometry == "flat_torus" else euclidean_model()
    p = model.atoms.p.detach()
    gradient = torch.ones_like(p)
    vector = torch.randn_like(p)
    calls = []
    original_project = polar._profile.project_gradient
    original_transport = polar._profile.transport_state

    def project(*args):
        calls.append("project")
        return original_project(*args)

    def transport(*args):
        calls.append("transport")
        return original_transport(*args)

    monkeypatch.setattr(polar._profile, "project_gradient", project)
    monkeypatch.setattr(polar._profile, "transport_state", transport)
    assert (
        polar.project_parameter_gradient(model.kernel, *model.cst_charts(), p, gradient)
        is gradient
    )
    assert (
        polar.transport_parameter_state(
            model.kernel, *model.cst_charts(), p, p.clone(), vector
        )
        is vector
    )
    assert calls == ["project", "project", "transport", "transport"]


@pytest.mark.parametrize("operation", ["project", "transport"])
def test_nonidentity_sphere_geometry_still_projects_vectors(operation):
    model = euclidean_model(sphere=True)
    p = model.atoms.p.detach()
    # A center's own ambient radial vector must disappear from its tangent plane.
    vector = p.clone()
    if operation == "project":
        result = polar.project_parameter_gradient(
            model.kernel, *model.cst_charts(), p, vector
        )
    else:
        result = polar.transport_parameter_state(
            model.kernel, *model.cst_charts(), p, p, vector
        )
    assert result is not vector
    torch.testing.assert_close(result[:, :2], vector[:, :2], atol=0, rtol=0)
    torch.testing.assert_close(
        result[:, 2:], torch.zeros_like(result[:, 2:]), atol=2e-14, rtol=0
    )


@pytest.mark.parametrize("operation", ["project", "transport"])
def test_custom_profile_nonidentity_result_is_not_discarded(monkeypatch, operation):
    model = radius_model()
    p = model.atoms.p.detach()
    vector = torch.ones_like(p)
    callback = "project_gradient" if operation == "project" else "transport_state"
    calls = []

    def change_center(*args):
        calls.append(1)
        return args[-1] * 2

    monkeypatch.setattr(polar._profile, callback, change_center)
    if operation == "project":
        result = polar.project_parameter_gradient(
            model.kernel, *model.cst_charts(), p, vector
        )
    else:
        result = polar.transport_parameter_state(
            model.kernel, *model.cst_charts(), p, p, vector
        )
    assert calls == [1, 1]
    assert result is not vector
    torch.testing.assert_close(result[:, :2], vector[:, :2], atol=0, rtol=0)
    torch.testing.assert_close(result[:, 2:], 2 * vector[:, 2:], atol=0, rtol=0)


@pytest.mark.parametrize(
    "fault", ["integer", "width", "rows", "old_nan", "new_nan", "vector_nan"]
)
def test_identity_transport_keeps_structural_and_finite_errors(fault):
    model = radius_model()
    old = model.atoms.p.detach().clone()
    new, vector = old.clone(), torch.ones_like(old)
    if fault == "integer":
        vector = vector.long()
    elif fault == "width":
        vector = vector[:, :-1]
    elif fault == "rows":
        new = new[:-1]
    else:
        {"old_nan": old, "new_nan": new, "vector_nan": vector}[fault][0, 2] = float(
            "nan"
        )
    with pytest.raises((ValueError, TypeError)):
        polar.transport_parameter_state(
            model.kernel, *model.cst_charts(), old, new, vector
        )


def test_identity_projection_still_rejects_invalid_layout_and_integer_dtype():
    model = radius_model()
    p = model.atoms.p.detach()
    with pytest.raises(TypeError):
        polar.project_parameter_gradient(
            model.kernel, *model.cst_charts(), p, torch.ones_like(p).long()
        )
    model.kernel.scalar("radius_min_input").fill_(-1)
    with pytest.raises(ValueError):
        polar.project_parameter_gradient(
            model.kernel, *model.cst_charts(), p, torch.ones_like(p)
        )


def test_identity_state_transport_does_not_mutate_vector_version():
    model = radius_model()
    p = model.atoms.p.detach()
    vector = torch.ones_like(p)
    version = vector._version
    OptimizerStateAdapter(("exp_avg",)).transport(
        model, p, p.clone(), {"exp_avg": vector}
    )
    assert vector._version == version


def test_amplitude_optimizer_matches_forced_materialized_identity_baseline(monkeypatch):
    actual = layer()
    baseline = copy.deepcopy(actual)
    oa, ob = optimizer(actual, 0.3), optimizer(baseline, 0.3)
    original_project = polar._profile.project_gradient
    original_transport = polar._profile.transport_state
    for step in range(3):
        grad = torch.sin(
            torch.arange(actual.atoms.p.numel(), dtype=actual.atoms.p.dtype).reshape_as(
                actual.atoms.p
            )
            + step
        )
        actual.atoms.p.grad, baseline.atoms.p.grad = grad.clone(), grad.clone()
        gradient_version = actual.atoms.p.grad._version
        oa.step()
        # Clone the otherwise identical profile results to force the old cat /
        # copy path without replacing any update or optimizer mathematics.
        with monkeypatch.context() as patch:
            patch.setattr(
                polar._profile,
                "project_gradient",
                lambda *args: original_project(*args).clone(),
            )
            patch.setattr(
                polar._profile,
                "transport_state",
                lambda *args: original_transport(*args).clone(),
            )
            ob.step()
        assert actual.atoms.p.grad._version == gradient_version
        torch.testing.assert_close(actual.atoms.p, baseline.atoms.p, atol=0, rtol=0)
        actual_state, baseline_state = (
            oa.state[actual.atoms.p],
            ob.state[baseline.atoms.p],
        )
        assert actual_state.keys() == baseline_state.keys()
        for key in actual_state:
            torch.testing.assert_close(
                actual_state[key], baseline_state[key], atol=0, rtol=0
            )


def test_identity_projection_preserves_existing_nonfinite_passthrough():
    model = radius_model()
    p = model.atoms.p.detach()
    gradient = torch.ones_like(p)
    gradient[0, 2] = float("nan")
    result = polar.project_parameter_gradient(
        model.kernel, *model.cst_charts(), p, gradient
    )
    assert result is gradient
    assert torch.isnan(result[0, 2])


def test_identity_transport_respects_optional_finite_check_scope():
    from torchcst._backends.torch import finite

    model = radius_model()
    p = model.atoms.p.detach()
    vector = torch.ones_like(p)
    vector[0, 2] = float("nan")
    with finite.checking(False):
        assert (
            polar.transport_parameter_state(
                model.kernel, *model.cst_charts(), p, p, vector
            )
            is vector
        )
    with pytest.raises(ValueError):
        polar.transport_parameter_state(model.kernel, *model.cst_charts(), p, p, vector)
