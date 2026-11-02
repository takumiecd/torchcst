"""Fixed-width radial profiles on one chart or a pair of charts."""

import torch

from torchcst._backends.torch.profiles import execution as _profile


def initialize(state, *charts_and_atoms, mode):
    *charts, atoms = charts_and_atoms
    return torch.cat(
        [
            _profile.initialize(
                state.profiles[0], chart, atoms, mode=mode if i == 0 else "uniform"
            )
            for i, chart in enumerate(charts)
        ],
        dim=-1,
    )


def materialize_atoms(state, chart, p):
    return _profile.evaluate(state.profiles[0], chart, p).T.reshape(
        p.shape[0], *chart.shape
    )


def weight(state, chart, p):
    return materialize_atoms(state, chart, p).sum(0)
