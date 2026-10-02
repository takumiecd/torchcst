"""Signed-amplitude radial atoms with clamped log-width coordinates."""

import torch

from torchcst._backends.torch.profiles import execution as _profile


def initialize(state, chart, atoms, *, mode):
    raise ValueError("provided_atoms initialization requires explicit atom parameters")


def materialize_atoms(state, chart, p):
    precision = torch.exp(
        -2
        * p[:, 1].clamp(
            min=state.scalar("sigma_min").to(p).log(),
            max=state.scalar("sigma_max").to(p).log(),
        )
    )
    values = _profile.evaluate_with_precision(
        state.profiles[0], chart, p[:, 2:], precision
    )
    return (values * p[:, 0]).T.reshape(p.shape[0], *chart.shape)


def weight(state, chart, p):
    return materialize_atoms(state, chart, p).sum(0)
