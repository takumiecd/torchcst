"""Opt-in Dense-moment-compatible CST initialization; no runtime output gain.

Call on a CPU CSTLinear before creating its optimizer or moving it to CUDA.
The returned report describes an applied initialization, not a quality guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch


def _fans(module):
    from .nn import CSTLinear

    if not isinstance(module, CSTLinear):
        raise TypeError("variance initialization requires a CSTLinear")
    return module.in_features, module.out_features


@dataclass(frozen=True)
class InitializationReport:
    """Conditional sign-energy admission and amplitude clipping diagnostics."""

    scheme: str
    q: float
    variance: float
    target_energy: float
    expected_energy: float
    relative_error: float
    amplitude_scale: float
    amplitude_rms: float
    clipped_atoms: int
    atom_count: int
    evaluations: int


def variance_scaling_(module, variance, *, q=0.0, generator=None):
    """Match the average operator second moment of Dense weight variance.

    This is the extension point for new initialization presets. Centers, Atom
    count, Parameter identity and declarations are preserved. A failure leaves
    module values unchanged (the random generator can still advance).

    Initially supported: CPU FP32/FP64 CSTLinear, two-chart Polar/envelope radius cell
    mass. ``q`` selects the logarithmic position between amplitude-dependent
    radius bounds. Independent symmetric amplitude signs define the expected
    energy; no dataset, Dense model or dense W is required. The bounded scale
    search includes amplitude-dependent radii and symmetric clipping. Thus the
    physical amplitude distribution can differ from an unbounded Gaussian.
    """
    _fans(module)
    if (
        isinstance(variance, bool)
        or not isinstance(variance, (int, float))
        or not math.isfinite(variance)
        or variance <= 0
    ):
        raise ValueError("positive finite Dense weight variance required")
    if (
        isinstance(q, bool)
        or not isinstance(q, (int, float))
        or not math.isfinite(q)
        or not 0 <= q <= 1
    ):
        raise ValueError("q must be a finite fraction in [0,1]")
    if module.atoms.p.grad is not None:
        raise ValueError("initialize before backward and optimizer construction")
    from ._backends.torch.initialization import match_variance

    values, measured = match_variance(
        module.kernel,
        module.cst_charts(),
        module.atoms.p,
        variance=float(variance),
        q=float(q),
        generator=generator,
    )
    with torch.no_grad():
        module.atoms.p.copy_(values)
    return InitializationReport(
        scheme="variance_scaling", q=float(q), variance=float(variance), **measured
    )


def kaiming_normal_(
    module, a=0.0, mode="fan_in", nonlinearity="leaky_relu", *, q=0.0, generator=None
):
    """Use PyTorch's He/Kaiming gain and logical fan, with CST moment matching.

    ``normal`` describes the pre-clip amplitude proposal, not independent normal
    dense weights. Returns InitializationReport, rather than a Tensor/Module.
    """
    if mode not in ("fan_in", "fan_out"):
        raise ValueError("mode must be fan_in or fan_out")
    fan_in, fan_out = _fans(module)
    gain = torch.nn.init.calculate_gain(nonlinearity, a)
    fan = fan_in if mode == "fan_in" else fan_out
    report = variance_scaling_(module, gain**2 / fan, q=q, generator=generator)
    return replace(report, scheme="kaiming_normal")


def xavier_normal_(module, gain=1.0, *, q=0.0, generator=None):
    """Use Xavier's logical fan average; returns InitializationReport."""
    if (
        isinstance(gain, bool)
        or not isinstance(gain, (float, int))
        or not math.isfinite(gain)
        or gain <= 0
    ):
        raise ValueError("positive finite gain required")
    fan_in, fan_out = _fans(module)
    variance = gain**2 * 2 / (fan_in + fan_out)
    report = variance_scaling_(module, variance, q=q, generator=generator)
    return replace(report, scheme="xavier_normal")


__all__ = [
    "InitializationReport",
    "kaiming_normal_",
    "variance_scaling_",
    "xavier_normal_",
]
