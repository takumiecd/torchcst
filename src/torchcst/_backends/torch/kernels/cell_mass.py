"""Differentiable FP64 cell integration used to qualify the CUDA prototype.

This is an intentionally direct reference. Its temporary Fourier/profile
tensors are not the CUDA storage policy. It never materializes the full W.
"""

import torch

from torchcst.kernels.cell_mass import CellMassAxisSpec, CellMassLinearSpec

from ..profiles.fourier import kernel_coefficients


def validate_inputs(spec: CellMassLinearSpec, x, atoms):
    """Validate physical inputs before entering the research numeric route."""
    if not isinstance(spec, CellMassLinearSpec):
        raise TypeError("expected CellMassLinearSpec")
    if x.ndim != 2 or x.shape[1] != spec.input.site_count:
        raise ValueError("X must have shape [batch, input site_count]")
    if atoms.ndim != 2 or atoms.shape[1] != 4:
        raise ValueError("physical atoms must have four columns")
    if x.dtype not in (torch.float32, torch.float64) or atoms.dtype != x.dtype:
        raise TypeError("X and atoms must share float32 or float64 dtype")
    if x.device != atoms.device:
        raise ValueError("X and atoms must share a device")
    if not x.is_contiguous() or not atoms.is_contiguous():
        raise ValueError("prototype requires contiguous X and physical atoms")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(atoms).all()):
        raise ValueError("X and atoms must be finite")
    maximum_sigma = min(spec.input.profile_period, spec.output.profile_period) / 3
    if bool(((atoms[:, 1] <= 0) | (atoms[:, 1] > maximum_sigma)).any()):
        raise ValueError("sigma must lie in (0, min(profile_period)/3]")


def profile_cell_masses(axis: CellMassAxisSpec, centers, sigma):
    """Integrate complete cells in the selected one-period tile window.

    No atom-centred endpoint clamp or per-cell support test exists here. Integer
    routing selects whole tiles; autograd differentiates the smooth branch.
    The scalar quadrature oracle in tests does not use these coefficients.
    """
    if centers.ndim != 1 or sigma.shape != centers.shape:
        raise ValueError("centers and sigma must have equal one-dimensional shapes")
    if centers.device != sigma.device:
        raise ValueError("centers and sigma must share a device")
    if centers.numel() == 0:
        return (centers[:, None] + sigma[:, None]).expand(0, axis.site_count).double()

    period = axis.profile_period
    modes = axis.profile.modes
    coefficients = kernel_coefficients(axis.profile, sigma, period)
    constant = coefficients[:, modes]
    mass = period * constant
    center = (centers.double() - axis.origin).remainder(axis.period)
    owner = ((center + axis.spacing / 2) / axis.tile_width).floor().long()
    first_site = (owner - axis.period_tiles // 2) * axis.tile_sites
    offsets = torch.arange(axis.period_tiles * axis.tile_sites, device=centers.device)
    unwrapped = first_site[:, None] + offsets[None, :]
    left = (unwrapped.double() - 0.5) * axis.spacing - center[:, None]
    right = left + axis.spacing
    omega = (2 * torch.pi / period) * torch.arange(
        1, modes + 1, device=centers.device, dtype=torch.float64
    )
    area = constant[:, None] * axis.spacing
    area = area + (
        (2 * coefficients[:, modes + 1 :] / omega)[:, None, :]
        * ((right[..., None] * omega).sin() - (left[..., None] * omega).sin())
    ).sum(-1)
    result = centers.new_zeros((centers.numel(), axis.site_count), dtype=torch.float64)
    return result.scatter_add(
        1, unwrapped.remainder(axis.site_count), area / mass[:, None]
    )


def physical_linear_reference(spec: CellMassLinearSpec, x, atoms):
    """Direct factored reference with live amplitude, center and sigma gradients."""
    validate_inputs(spec, x, atoms)
    sigma = atoms[:, 1]
    input_profile = profile_cell_masses(spec.input, atoms[:, 3], sigma)
    output_profile = profile_cell_masses(spec.output, atoms[:, 2], sigma)
    h = x.double() @ input_profile.T
    y = (h * atoms[:, 0].double()[None, :]) @ output_profile
    return y.to(x.dtype)
