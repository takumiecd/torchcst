"""Independent proposal math, scheduler/checkpoint and semantic-field contracts."""

import copy
import io

import pytest
import torch
from test_native_cst_adamw import equal, gradients, model

from torchcst import CSTAdamW, CSTOptimizer
from torchcst._backends.torch.kernels import execution
from torchcst.kernels.parameterizations import FixedWidthSpec, ParameterFieldSpec


def make(m, wrapped=False, field_lrs=None, **overrides):
    groups = [
        {"params": list(m["cst"].parameters()), "lr": 0.0048},
        {"params": list(m["ordinary"].parameters()), "lr": 0.0003},
    ]
    options = dict(
        lr=0.0003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0, foreach=False
    )
    options.update(overrides)
    if wrapped:
        return CSTOptimizer(
            torch.optim.AdamW(groups, **options), model=m, field_lrs=field_lrs
        )
    return CSTAdamW(groups, model=m, field_lrs=field_lrs, **options)


@pytest.mark.parametrize("wrapped", [False, True])
def test_normalized_proposals_match_independent_adam_and_declared_retraction(wrapped):
    actual = model()
    expected = copy.deepcopy(actual)
    fields = {"amplitude": 0.0024, "output_center": 0.0096, "input_center": 0.0012}
    opt = make(actual, wrapped, fields)
    states = {
        p: [torch.zeros_like(p), torch.zeros_like(p)] for p in expected.parameters()
    }
    rng = torch.Generator().manual_seed(711)
    for step, factor in enumerate((0.5, 1.0, 0.25, 0.0, 0.8), 1):
        for group, peak in zip(opt.param_groups, (0.0048, 0.0003)):
            group["lr"] = peak * factor
        for p, q in zip(actual.parameters(), expected.parameters()):
            g = torch.randn(p.shape, dtype=p.dtype, generator=rng) * 0.01
            p.grad = g.clone()
            old = q.detach().clone()
            mean, square = states[q]
            mean.mul_(0.9).add_(g, alpha=0.1)
            square.mul_(0.95).addcmul_(g, g, value=0.05)
            atom = q is expected["cst"].atoms.p
            rate = (0.0048 if atom else 0.0003) * factor
            delta = (
                -rate
                * (mean / (1 - 0.9**step))
                / ((square / (1 - 0.95**step)).sqrt() + 1e-8)
            )
            with torch.no_grad():
                proposal = old + delta
                if atom:
                    for first, last, ratio in ((0, 2, 0.5), (2, 3, 2.0), (3, 4, 0.25)):
                        proposal[:, first:last] = (
                            old[:, first:last]
                            + (proposal[:, first:last] - old[:, first:last]) * ratio
                        )
                    if rate:
                        site = expected["cst"]
                        proposal = execution.apply_parameter_update(
                            site.kernel,
                            *site.cst_charts(),
                            old,
                            proposal - old,
                            step_size=rate,
                        )
                q.copy_(proposal)
        opt.step()
        for p, q in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(p, q, atol=3e-13, rtol=1e-12)
            for key, ref in zip(("exp_avg", "exp_avg_sq"), states[q]):
                torch.testing.assert_close(
                    opt.state[p][key], ref, atol=3e-14, rtol=1e-12
                )
            assert float(opt.state[p]["step"]) == step
        assert opt.get_field_lrs()[1] == {}
        assert opt.get_field_lrs()[0] == {
            name: peak * factor for name, peak in fields.items()
        }


@pytest.mark.parametrize("wrapped", [False, True])
def test_unit_overrides_are_bitwise_default_and_default_checkpoint_stays_legacy(
    wrapped,
):
    a = model()
    b = copy.deepcopy(a)
    oa = make(a, wrapped)
    ob = make(b, wrapped, {"amplitude": 0.0048, "input_center": 0.0048})
    for step in range(3):
        gradients([a, b], step)
        oa.step()
        ob.step()
        equal(a, b, oa, ob)
    assert "cst_fields" not in oa.state_dict()
    assert all("field_lrs" not in group for group in oa.state_dict()["param_groups"])
    assert "cst_fields" in ob.state_dict()


@pytest.mark.parametrize("wrapped", [False, True])
def test_scheduler_and_serialized_resume_restore_rates_into_default_constructor(
    wrapped,
):
    a = model()
    oa = make(a, wrapped, {"amplitude": 0.0024, "input_center": 0.0048})
    schedule = torch.optim.lr_scheduler.ExponentialLR(oa, gamma=0.5)
    for step in range(2):
        gradients([a], step)
        oa.step()
        schedule.step()
    storage = io.BytesIO()
    torch.save(
        {
            "model": a.state_dict(),
            "optimizer": oa.state_dict(),
            "scheduler": schedule.state_dict(),
        },
        storage,
    )
    storage.seek(0)
    saved = torch.load(storage, weights_only=False)
    b = model()
    b.load_state_dict(saved["model"])
    ob = make(b, wrapped)
    other_schedule = torch.optim.lr_scheduler.ExponentialLR(ob, gamma=0.5)
    ob.load_state_dict(saved["optimizer"])
    other_schedule.load_state_dict(saved["scheduler"])
    assert (
        oa.get_field_lrs()
        == ob.get_field_lrs()
        == [{"amplitude": 0.0006, "input_center": 0.0012}, {}]
    )
    gradients([a, b], 9)
    oa.step()
    ob.step()
    schedule.step()
    other_schedule.step()
    equal(a, b, oa, ob)
    assert oa.get_field_lrs() == ob.get_field_lrs()


@pytest.mark.parametrize("wrapped", [False, True])
def test_bad_checkpoint_layout_or_rates_fail_before_state_or_parameter_mutation(
    wrapped,
):
    a = model()
    opt = make(a, wrapped, {"amplitude": 0.0024})
    gradients([a], 0)
    opt.step()
    before = [p.detach().clone() for p in a.parameters()]
    saved = copy.deepcopy(opt.state_dict())
    for key, value in (("field_lrs", {"q": 0.001}), ("_cst_field_lr_reference", 0.0)):
        broken = copy.deepcopy(saved)
        broken["param_groups"][0][key] = value
        with pytest.raises(ValueError):
            opt.load_state_dict(broken)
        for p, old in zip(a.parameters(), before):
            torch.testing.assert_close(p, old, atol=0, rtol=0)
        for index, state in saved["state"].items():
            for name, val in state.items():
                torch.testing.assert_close(
                    opt.state_dict()["state"][index][name], val, atol=0, rtol=0
                )
    broken = copy.deepcopy(saved)
    broken["cst_fields"]["version"] = 2
    with pytest.raises(ValueError, match="field declarations"):
        opt.load_state_dict(broken)


@pytest.mark.parametrize("wrapped", [False, True])
def test_unknown_or_nonfinite_fields_rejected_without_binding_failed_model(wrapped):
    for fields in (
        {"amp": 0.001},
        {"q": 0.001},
        {"amplitude": float("nan")},
        {"amplitude": -1},
        {"amplitude": True},
    ):
        a = model()
        with pytest.raises(ValueError):
            make(a, wrapped, fields)
        # A failed configuration must not leave a conflicting optimizer owner.
        make(a, wrapped, {"amplitude": 0.0024})


def test_parameterization_fields_use_stored_geometry_dimensions():
    fixed = FixedWidthSpec(sigma=1.0)
    assert fixed.optimizer_fields((3, 2)) == (
        ParameterFieldSpec("output_center", 0, 3),
        ParameterFieldSpec("input_center", 3, 5),
    )
    polar = model()["cst"].kernel.spec.parameterization
    assert polar.optimizer_fields((3, 2)) == (
        ParameterFieldSpec("amplitude", 0, 2),
        ParameterFieldSpec("output_center", 2, 5),
        ParameterFieldSpec("input_center", 5, 7),
    )


def test_wrapper_base_posthook_changes_recipe_only_for_next_update():
    a = model()
    b = copy.deepcopy(a)
    fields = {"amplitude": 0.0024}
    oa, ob = make(a, True, fields), make(b, True, fields)

    def mutate(base, args, kwargs):
        base.param_groups[0]["field_lrs"]["amplitude"] = 0.0048
        base.param_groups[0]["lr"] = 0.0024

    handle = oa.base_optimizer.register_step_post_hook(mutate)
    gradients([a, b], 16)
    oa.step()
    ob.step()
    equal(a, b, oa, ob)
    handle.remove()
    ob.param_groups[0]["field_lrs"]["amplitude"] = 0.0048
    ob.param_groups[0]["lr"] = 0.0024
    gradients([a, b], 17)
    oa.step()
    ob.step()
    equal(a, b, oa, ob)


@pytest.mark.parametrize("wrapped", [False, True])
def test_loading_default_checkpoint_disables_constructor_field_overrides(wrapped):
    a = model()
    oa = make(a, wrapped)
    gradients([a], 20)
    oa.step()
    saved_model, saved_optimizer = (
        copy.deepcopy(a.state_dict()),
        copy.deepcopy(oa.state_dict()),
    )
    b = model()
    b.load_state_dict(saved_model)
    ob = make(b, wrapped, {"amplitude": 0.0024})
    ob.load_state_dict(saved_optimizer)
    assert ob.get_field_lrs() == [{}, {}]
    gradients([a, b], 21)
    oa.step()
    ob.step()
    equal(a, b, oa, ob)


@pytest.mark.parametrize("wrapped", [False, True])
def test_zero_amplitude_rate_preserves_direction_and_none_gradient_skips_clock(wrapped):
    a = model()
    opt = make(a, wrapped, {"amplitude": 0.0})
    point = a["cst"].atoms.p
    old = point.detach().clone()
    gradients([a], 22)
    opt.step()
    torch.testing.assert_close(
        point[:, 0] / point[:, :2].norm(dim=1),
        old[:, 0] / old[:, :2].norm(dim=1),
        atol=2e-15,
        rtol=2e-14,
    )
    # The field freezes task direction; the declared activity law still advances.
    assert not torch.equal(point[:, :2].norm(dim=1), old[:, :2].norm(dim=1))
    before, step = point.detach().clone(), opt.state[point]["step"].clone()
    gradients([a], 23, skip_atoms=True)
    opt.step()
    torch.testing.assert_close(point, before, atol=0, rtol=0)
    torch.testing.assert_close(opt.state[point]["step"], step, atol=0, rtol=0)


def test_center_field_spans_multidimensional_chart_coordinates():
    from torchcst import (
        BandwidthBounds,
        CSTLinear,
        GaussianSpec,
        chart_presets,
        kernels,
    )

    m = CSTLinear(
        charts=(
            chart_presets.regular_grid((3, 4), geometry="flat_torus"),
            chart_presets.regular_grid((2, 3), geometry="flat_torus"),
        ),
        atoms=5,
        kernel=kernels.product(
            kernels.l2(GaussianSpec()),
            coordinates=kernels.polar(
                amplitude_max=1.0,
                w_c=0.3,
                bounds=BandwidthBounds(
                    minimum=0.1, birth=0.4, maximum=1.0, upper_floor=0.1
                ),
            ),
        ),
        dtype=torch.float64,
        backend="factored",
    )
    assert m.atoms.p.shape == (5, 6)
    with torch.no_grad():
        m.atoms.p[:, 2:] = m.atoms.p.new_tensor([1.0, 1.0, 0.5, 0.5])
    opt = CSTAdamW(
        m.parameters(),
        model=m,
        lr=0.01,
        field_lrs={"output_center": 0.02, "input_center": 0.0},
        foreach=False,
    )
    before = m.atoms.p.detach().clone()
    m.atoms.p.grad = torch.full_like(m.atoms.p, 0.03)
    opt.step()
    torch.testing.assert_close(m.atoms.p[:, 4:], before[:, 4:], atol=0, rtol=0)
    torch.testing.assert_close(
        m.atoms.p[:, 2:4],
        before[:, 2:4] - 0.02 * 0.03 / (0.03 + 1e-8),
        atol=1e-14,
        rtol=1e-13,
    )


def test_per_group_override_and_added_group_copy_field_settings():
    from test_optimizer_amplitude_decay import layer

    m = torch.nn.ModuleDict(
        {
            "first": layer(),
            "second": layer(),
            "ordinary": torch.nn.Linear(2, 2).double(),
        }
    )
    settings = {"amplitude": 0.0024}
    opt = CSTAdamW(
        [
            {"params": list(m["first"].parameters())},
            {"params": list(m["second"].parameters()), "field_lrs": {}},
        ],
        model=m,
        lr=0.0048,
        field_lrs=settings,
        weight_decay=0,
    )
    settings["amplitude"] = 1.0
    assert opt.get_field_lrs() == [{"amplitude": 0.0024}, {}]
    opt.add_param_group({"params": list(m["ordinary"].parameters())})
    assert opt.get_field_lrs() == [{"amplitude": 0.0024}, {}, {}]
    count = len(opt.param_groups)
    with pytest.raises(ValueError):
        opt.add_param_group(
            {"params": list(m["first"].parameters()), "field_lrs": {"q": 0.1}}
        )
    assert len(opt.param_groups) == count


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("amsgrad,maximize", [(False, False), (True, True)])
def test_native_and_wrapped_field_rates_match_with_adam_options(
    foreach, amsgrad, maximize
):
    a = model()
    b = copy.deepcopy(a)
    fields = {"amplitude": 0.0024, "input_center": 0.0012}
    oa = make(a, False, fields, foreach=foreach, amsgrad=amsgrad, maximize=maximize)
    ob = make(b, True, fields, foreach=foreach, amsgrad=amsgrad, maximize=maximize)
    for step in range(3):
        gradients([a, b], 30 + step)
        oa.step()
        ob.step()
        equal(a, b, oa, ob)


@pytest.mark.parametrize("wrapped", [False, True])
def test_logical_amplitude_decay_uses_outer_time_after_scaled_task(wrapped):
    from test_optimizer_amplitude_decay import independent_decay

    a = model()
    b = copy.deepcopy(a)
    fields = {"amplitude": 0.0024}
    oa, ob = make(a, wrapped, fields), make(b, wrapped, fields)
    oa.param_groups[0]["amplitude_weight_decay"] = 0.3
    old = a["cst"].atoms.p.detach().clone()
    gradients([a, b], 40)
    oa.step()
    ob.step()
    expected = independent_decay(old, b["cst"].atoms.p.detach(), 0.0048, 0.3)
    torch.testing.assert_close(a["cst"].atoms.p, expected, atol=3e-14, rtol=3e-14)


def test_sgd_wrapper_scales_proposal_without_scaling_momentum():
    a = model()["cst"]
    b = copy.deepcopy(a)
    opt = CSTOptimizer(
        torch.optim.SGD(a.parameters(), lr=0.01, momentum=0.9),
        model=a,
        field_lrs={"amplitude": 0.005},
    )
    momentum = torch.zeros_like(b.atoms.p)
    for step in range(3):
        g = torch.randn(
            momentum.shape,
            dtype=momentum.dtype,
            generator=torch.Generator().manual_seed(50 + step),
        )
        a.atoms.p.grad = g.clone()
        momentum.mul_(0.9).add_(g)
        old = b.atoms.p.detach().clone()
        delta = -0.01 * momentum
        # Match finite-precision base proposal subtraction before field scaling.
        proposal = old + delta
        proposal[:, :2] = old[:, :2] + (proposal[:, :2] - old[:, :2]) * 0.5
        with torch.no_grad():
            b.atoms.p.copy_(
                execution.apply_parameter_update(
                    b.kernel, *b.cst_charts(), old, proposal - old, step_size=0.01
                )
            )
        opt.step()
        torch.testing.assert_close(a.atoms.p, b.atoms.p, atol=2e-14, rtol=2e-14)
        torch.testing.assert_close(
            opt.state[a.atoms.p]["momentum_buffer"], momentum, atol=0, rtol=0
        )
