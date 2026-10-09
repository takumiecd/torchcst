"""Independent FP64 oracle reads actual stored Sphere Polar scalar values."""

import copy

import torch

from benchmarks.cuda.linear.sphere_baseline import error, fixture, oracle_vjp


def gate(actual, truth):
    for a, b in zip(actual, truth):
        if a.numel() == 0:
            assert a.shape == b.shape
            continue
        e = error(a, b)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert e["max_abs"] <= 4e-4 and e["relative_l2"] <= 4e-4, e


def test_fp64_oracle_reads_live_scalars_without_changing_initial_spec():
    from benchmarks.cuda.linear.sphere_baseline import oracle_factors
    from torchcst.kernels.state import KernelState

    model, x, _, dy = fixture(17, 3.0, batch=3, atoms=19)
    initial = model.kernel.spec
    before = oracle_factors(model, model.atoms.p.detach().double())
    values = {
        "amplitude_max": 0.8,
        "w_c": 0.5,
        "kappa": 2.7,
        "lower_kappa": 1.3,
        "upper_decay_power": 0.7,
        "sigma_min_input": 0.8,
        "sigma_birth_input": 1.7,
        "sigma_max_input": 12.0,
        "upper_floor_input": 1.1,
        "sigma_min_output": 0.9,
        "sigma_birth_output": 1.8,
        "sigma_max_output": 10.0,
        "upper_floor_output": 1.2,
    }
    with torch.no_grad():
        for name, value in values.items():
            model.kernel.scalar(name).fill_(value)
    assert model.kernel.spec is initial
    assert initial.parameterization.amplitude_max == 1.0
    # A second physical state initialized from the live declaration embeds the
    # actual stored values in its immutable spec. A stale spec oracle cannot
    # agree with both models after these scalar changes.
    reference = copy.deepcopy(model)
    reference.kernel = KernelState(model.kernel.declaration())
    assert (
        reference.kernel.spec.parameterization.amplitude_max
        != initial.parameterization.amplitude_max
    )
    actual = oracle_factors(model, model.atoms.p.detach().double())
    expected = oracle_factors(reference, reference.atoms.p.detach().double())
    assert not torch.allclose(before[1], actual[1], atol=1e-10, rtol=1e-10)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    gate(oracle_vjp(model, x, dy), oracle_vjp(reference, x, dy))
