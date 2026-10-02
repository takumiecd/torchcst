"""Single-chart evaluation of a fixed-width radial profile."""

from torchcst._backends.torch.profiles import execution as _profile


def initialize(state, chart, atoms, *, mode):
    return _profile.initialize(state.profiles[0], chart, atoms, mode=mode)


def materialize_atoms(state, chart, p):
    return _profile.evaluate(state.profiles[0], chart, p).T.reshape(
        p.shape[0], *chart.shape
    )


def weight(state, chart, p):
    return materialize_atoms(state, chart, p).sum(0)
