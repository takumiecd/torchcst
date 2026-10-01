"""Signed-amplitude radial atoms with clamped log-width coordinates."""

from torchcst._backends.torch.profiles import execution as _profile


def initialize(state, chart, atoms, *, mode):
    raise ValueError("provided_atoms initialization requires explicit atom parameters")


def materialize_atoms(state, chart, p):
    sigma = p[:, 1].exp().clamp(state.scalar("sigma_min"), state.scalar("sigma_max"))
    values = _profile.evaluate_with_precision(
        state.profiles[0], chart, p[:, 2:], sigma.reciprocal().square()
    )
    return (values * p[:, 0]).T.reshape(p.shape[0], *chart.shape)


def weight(state, chart, p):
    return materialize_atoms(state, chart, p).sum(0)
