"""Pure declarations for the explicitly bounded I-driven research operator.

P[a] = (amplitude, output center, input center, width logit).
I=.1+9.9*sigmoid(logit) is live in task differentiation. This is deliberately
not the existing Polar parameterization (whose task width is stopped).
"""

from dataclasses import dataclass

import torch

from torchcst import FourierPeakSpec, HierarchicalFourierPeriodSpec


@dataclass(frozen=True)
class FourierLinearSpec:
    size: int
    modes: int = 8
    spacing: float = 0.1
    floor: float = 1e-6

    def __post_init__(self):
        if type(self.size) is not int or self.size not in (64, 128, 256, 1024):
            raise ValueError("research size must be 64,128,256,1024")
        FourierPeakSpec(modes=self.modes)
        if self.spacing != 0.1 or self.floor != 1e-6:
            raise ValueError("revision1 fixes spacing=.1 and global floor=1e-6")

    @property
    def period(self):
        return HierarchicalFourierPeriodSpec(
            minimum_tile_width=0.2 if self.modes == 8 else 0.05,
            minimum_half_period=0.05 if self.modes == 8 else 0.0125,
            relative_width_floor=0.3 if self.modes == 8 else 0.7,
        )

    @property
    def in_features(self):
        return self.size

    @property
    def out_features(self):
        return self.size


@dataclass(frozen=True)
class FourierRecipe:
    strategy: str = "shared"
    atom_chunk: int = 4096
    minimum_sites: int = 64
    width_cutoff: float = 2.0

    def __post_init__(self):
        if self.strategy not in ("shared", "direct"):
            raise ValueError("strategy must be shared or direct")
        if type(self.atom_chunk) is not int or self.atom_chunk not in (
            256,
            1024,
            4096,
            8192,
        ):
            raise ValueError("invalid atom chunk")
        if type(self.minimum_sites) is not int or self.minimum_sites not in (
            32,
            64,
            128,
        ):
            raise ValueError("minimum_sites must be 32,64,128")
        if (
            type(self.width_cutoff) not in (float, int)
            or not 0.1 <= self.width_cutoff <= 10
        ):
            raise ValueError("width_cutoff must be .1..10")


def torch_reference(x, p, spec):
    from torchcst._backends.torch.kernels.hierarchical_fourier import values

    sites = torch.arange(spec.size, device=x.device, dtype=torch.float64) * spec.spacing
    width = 0.1 + 9.9 * p[:, 3].double().sigmoid()
    profile = FourierPeakSpec(modes=spec.modes)
    u = values(profile, spec.period, sites, p[:, 1], width, support="centered")
    v = values(profile, spec.period, sites, p[:, 2], width, support="centered")
    denom = (u.norm(dim=0) * v.norm(dim=0)).clamp_min(spec.floor)
    return ((x.double() @ v) * (p[:, 0].double() / denom)) @ u.T
