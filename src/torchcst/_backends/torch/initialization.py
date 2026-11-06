"""CPU initial Atom construction and moment matching for Polar cell masses."""

from __future__ import annotations

import copy
import math

import torch

from torchcst.kernels.parameterizations import PolarAmpRadiusSpec

from .kernels import execution
from .parameterizations.radius_cell_mass import decoded_values


def match_variance(kernel, charts, original, *, variance, q, generator):
    if (
        kernel.spec.revision != 7
        or type(kernel.spec.parameterization) is not PolarAmpRadiusSpec
        or len(charts) != 2
        or original.ndim != 2
        or original.shape[1] != 4
    ):
        raise NotImplementedError(
            "variance initialization currently supports two-chart Polar radius cell-mass Linear only"
        )
    if original.device.type != "cpu" or original.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise ValueError("initialize CPU float32/float64 Atoms before moving to CUDA")
    if not bool(torch.isfinite(original).all()):
        raise ValueError("finite initial Atom coordinates required")
    if any(c.reference.device.type != "cpu" for c in charts):
        raise ValueError("CPU charts required for initialization")
    reference = copy.deepcopy(kernel).double()
    axes = tuple(copy.deepcopy(c).double() for c in charts)
    maximum = float(reference.scalar("amplitude_max"))
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("positive finite amplitude_max required")
    noise = torch.randn(
        len(original), dtype=original.dtype, device="cpu", generator=generator
    )
    target = variance * charts[0].features * charts[1].features
    if not math.isfinite(target) or target <= 0:
        raise ValueError("finite positive operator energy target required")
    rho = math.sqrt(1 + 3 * q)
    tolerance = 0.005
    history = []

    def trial(scale):
        ratio_unclipped = noise * (scale / maximum)
        ratio = ratio_unclipped.clamp(-1 + 1e-6, 1 - 1e-6)
        p = original.detach().clone()
        p[:, :2] = torch.stack((ratio, (1 - ratio.square()).sqrt()), -1) * rho
        energy = 0.0
        for start in range(0, len(p), 128):
            x, y = execution.factors(reference, *axes, p[start : start + 128].double())
            energy += float((x.square().sum(0) * y.square().sum(0)).sum())
        if not math.isfinite(energy) or energy < 0:
            raise ValueError("nonfinite initial Atom energy")
        error = abs(energy / target - 1)
        history.append((scale, energy, error))
        return p, energy, error, int((ratio_unclipped.abs() >= 1 - 1e-6).sum())

    low, high = 0.0, None
    accepted = None
    with torch.no_grad():
        for relative_scale in (
            0.0,
            0.025,
            0.05,
            0.1,
            0.2,
            0.4,
            0.8,
            1.6,
            3.2,
            6.4,
            12.8,
            32.0,
        ):
            scale = relative_scale * maximum
            value = trial(scale)
            if value[2] <= tolerance:
                accepted = (scale, value)
                break
            if value[1] >= target:
                high = scale
                break
            low = scale
        if accepted is None and high is not None:
            for _ in range(32):
                scale = (low + high) / 2
                value = trial(scale)
                if value[2] <= tolerance:
                    accepted = (scale, value)
                    break
                if value[1] < target:
                    low = scale
                else:
                    high = scale
        if accepted is None:
            raise ValueError(
                "variance target not reached in the bounded scale search; "
                "module unchanged. q, Atom count or amplitude bounds may need an "
                f"explicit design change. Search history: {history}"
            )
        scale, (p, energy, error, clipped) = accepted
        amplitude, _, _ = decoded_values(reference, p.double())
    return p, dict(
        target_energy=target,
        expected_energy=energy,
        relative_error=error,
        amplitude_scale=scale,
        amplitude_rms=float(amplitude.square().mean().sqrt()),
        clipped_atoms=clipped,
        atom_count=len(p),
        evaluations=len(history),
    )
