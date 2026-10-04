"""Production polar decoding and sigma/spacing-based physical ordering."""

import torch

from torchcst._backends.torch.kernels import execution
from torchcst._backends.torch.parameterizations import polar_amp_width as polar
from torchcst.kernels import TriweightSpec


def validate_state(value):
    spec = value.spec
    if execution.family(value) != "polar_amp_width":
        raise ValueError("requires PolarAmpWidth")
    if spec.parameterization.input_bounds != spec.parameterization.output_bounds:
        raise ValueError("local prototype requires shared input/output sigma bounds")
    for binding in spec.profiles:
        if (
            type(binding.profile) is not TriweightSpec
            or binding.normalization.kind != "discrete_l2"
            or binding.normalization.floor != 1e-6
        ):
            raise ValueError("requires production normalized Triweight profiles")
    # Configuration-boundary check. Do not call this host synchronization in a step.
    polar._require_shared_bandwidths(value)


def decode(value, p):
    """Task gradients see amplitude/centers, never the activity-derived width."""
    amp, alpha = polar._amplitude_and_alpha(value, p[:, :2])
    sigma, _ = polar._bandwidth_sigmas(value, amp, alpha)
    return torch.stack((amp, sigma.reciprocal().square().detach(), p[:, 2], p[:, 3]), 1)


def order_atoms(q, domain, recipe):
    """One permutation: rho bucket, output owner, input owner; no duplicated atoms."""
    if not recipe.pack:
        return torch.arange(len(q), device=q.device)
    rho = q[:, 1].rsqrt() / domain.spacing
    scale = torch.zeros(len(q), dtype=torch.long, device=q.device)
    for upper in recipe.rho_upper[:-1]:
        scale.add_((rho > upper).long())
    ni = (domain.input_size + 15) // 16
    no = (domain.output_size + 15) // 16
    ci = ((q[:, 2] - domain.input_origin) / (16 * domain.spacing)).floor().long()
    co = ((q[:, 3] - domain.output_origin) / (16 * domain.spacing)).floor().long()
    key = (scale * no + co.clamp(0, no - 1)) * ni + ci.clamp(0, ni - 1)
    return torch.argsort(key, stable=True)


def polar_scalars(value):
    """Live scalar buffers for the fused decoder; no host reads or frozen sigma."""
    return tuple(
        value.scalar(name)
        for name in (
            "amplitude_max",
            "w_c",
            "kappa",
            "lower_kappa",
            "upper_decay_power",
            "sigma_min_input",
            "sigma_birth_input",
            "sigma_max_input",
            "upper_floor_input",
        )
    )
