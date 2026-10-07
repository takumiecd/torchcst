"""Single-chart interpretation of the existing shared Polar coordinate law."""

from torchcst._backends.torch.kernels import execution as _kernel

from .polar_amp_width import _amplitude_and_alpha, _sigma_bounds


def _split(state, chart, p):
    width = _kernel.parameter_dim(state, chart)
    if p.ndim != 2 or p.shape[1] != width:
        raise ValueError(f"p must have shape [atoms, {width}]")
    return p[:, :2], p[:, 2:]


def amplitude(state, chart, p):
    polar, _ = _split(state, chart, p)
    return _amplitude_and_alpha(state, polar)[0]


def bandwidth_alpha(state, chart, p):
    polar, _ = _split(state, chart, p)
    return _amplitude_and_alpha(state, polar)[1]


def bandwidth_bounds(state, chart, p):
    amp = amplitude(state, chart, p)
    _, lower, upper = _sigma_bounds(state, amp)
    return lower, upper


def bandwidth_sigma(state, chart, p):
    polar, _ = _split(state, chart, p)
    amp, alpha = _amplitude_and_alpha(state, polar)
    return _sigma_bounds(state, amp, alpha)[0]


def bandwidth_precision(state, chart, p):
    return bandwidth_sigma(state, chart, p).reciprocal().square()
