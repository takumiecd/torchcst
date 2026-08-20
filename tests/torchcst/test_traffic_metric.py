"""The preconditioner's metric, weighted by the traffic the site actually carries.

`J^2` measures how much moving an atom moves `W`, and it has always measured
that in the Frobenius norm -- the metric of a customer base arriving equally
from every direction, which no layer has.  E-func measured the gap on a trained
conv: rescoring a census under the real traffic took conv1's error from 0.36 to
0.16, so half of what looked like a representation failure was a scoring
artefact.  The same identity was sitting inside the optimizer's metric.

`traffic_rows=0` must reproduce the frozen lm1d rule exactly: an opt-in that
silently changed the default would invalidate every number the arc has.
"""

from __future__ import annotations

import pytest
import torch

from torchcst import CoordPreconditioner
from torchcst.compute import CSTLinear
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

SIGMA = 0.25


def _site(atoms=6, d=2, n_in=9, n_out=7):
    store = SynapseStore(
        "traffic", d, d, atoms,
        spec=RepresentationSpec.continuous(d, d, bounds=(-1.0, 1.0)),
        dtype=torch.float64,
    )
    rng = torch.Generator().manual_seed(5)
    store.apply([SynapseBirth(
        store.site,
        torch.rand(atoms, d, generator=rng, dtype=torch.float64) - 0.5,
        torch.rand(atoms, d, generator=rng, dtype=torch.float64) - 0.5,
        0.5 + torch.rand(atoms, generator=rng, dtype=torch.float64),
        torch.arange(atoms, dtype=torch.int64),
    )])
    charts = torch.Generator().manual_seed(11)
    module = CSTLinear(
        NeuronStore("in", n_in, mu=torch.rand(n_in, d, generator=charts,
                                              dtype=torch.float64) - 0.5,
                    initial_live=n_in, dtype=torch.float64),
        NeuronStore("out", n_out, mu=torch.rand(n_out, d, generator=charts,
                                                dtype=torch.float64) - 0.5,
                    initial_live=n_out, dtype=torch.float64),
        store, GaussianKernel(SIGMA, learnable=False).double(),
    )
    return module, store


def _drive(module, rows=32, scale=None):
    """One forward/backward so the hooks see something."""
    x = torch.randn(rows, module.in_features, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(3))
    if scale is not None:
        x = x * scale
    module(x).square().sum().backward()
    return x


def test_traffic_zero_is_the_frozen_rule_to_the_bit():
    """The default must be byte-identical to the metric lm1d froze."""
    module, _ = _site()
    plain = CoordPreconditioner(module, cap_sigma=0.1, subscribe=False)
    weighted = CoordPreconditioner(module, cap_sigma=0.1, subscribe=False,
                                   traffic_rows=0)
    _drive(module)
    for a, b in zip(plain._jacobian_sq(), weighted._jacobian_sq()):
        assert torch.equal(a, b)


def test_identity_traffic_is_the_frobenius_metric_times_a_constant():
    """The claim stated where it is exactly true, rather than approximately.

    Frobenius *is* the traffic-weighted metric of traffic whose covariance is
    the identity, so feeding one unit row per feature must return the frozen
    metric up to a single global factor -- and a global factor changes no step,
    because the step calibration divides it back out.  Sampled random rows do
    not test this: with a few dozen rows the empirical covariance is nowhere
    near the identity, and the output gradients are never isotropic at all.
    """
    module, _ = _site()
    metric = CoordPreconditioner(module, cap_sigma=0.1, subscribe=False,
                                 traffic_rows=64)
    plain = CoordPreconditioner(module, cap_sigma=0.1, subscribe=False)
    metric._x = torch.eye(module.in_features, dtype=torch.float64)
    metric._g = torch.eye(module.out_features, dtype=torch.float64)

    j_weighted, _ = metric._jacobian_sq()
    j_frobenius, _ = plain._jacobian_sq()
    ratio = j_weighted / j_frobenius
    torch.testing.assert_close(ratio, torch.full_like(ratio, float(ratio[0])))


def test_anisotropic_traffic_reorders_the_atoms():
    """And the point of doing it: which atom counts as sensitive changes."""
    module, _ = _site()
    metric = CoordPreconditioner(module, cap_sigma=0.1, subscribe=False,
                                 traffic_rows=64)
    plain = CoordPreconditioner(module, cap_sigma=0.1, subscribe=False)
    tilt = torch.logspace(1.5, -1.5, module.in_features, dtype=torch.float64)
    metric._x = torch.eye(module.in_features, dtype=torch.float64) * tilt
    metric._g = torch.eye(module.out_features, dtype=torch.float64)

    j_weighted, _ = metric._jacobian_sq()
    j_frobenius, _ = plain._jacobian_sq()
    ratio = j_weighted / j_frobenius
    assert float(ratio.max() / ratio.min()) > 3.0


def test_the_hooks_capture_the_sites_own_traffic():
    """Opt-in means the hooks exist only when asked, and then they fire."""
    module, _ = _site()
    off = CoordPreconditioner(module, cap_sigma=0.1, subscribe=False)
    on = CoordPreconditioner(module, cap_sigma=0.1, subscribe=False,
                             traffic_rows=16)
    assert off._x is None and on._x is None

    _drive(module, rows=40)
    assert off._x is None and off._g is None
    assert on._x is not None and on._x.shape == (16, module.in_features)
    assert on._g is not None and on._g.shape == (16, module.out_features)


def test_traffic_rows_must_be_a_non_negative_count():
    module, _ = _site()
    with pytest.raises((ValueError, TypeError)):
        CoordPreconditioner(module, cap_sigma=0.1, subscribe=False,
                            traffic_rows=-1)
