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
    sort_atoms: bool = False
    strategy: str = "shared"
    atom_chunk: int = 4096
    minimum_sites: int = 64
    width_cutoff: float = 2.0
    native_cache: bool = False
    table_coefficients: bool = False
    atom_block: int = 1
    batch_block: int = 1
    reuse: str = "none"
    moment_policy: str = "all"
    sample_cache: int = 0
    coefficient_norm_warps: int = 4
    coefficient_norm: bool = False
    boundary_cache: bool = False
    minimum_sharing: int = 0

    def __post_init__(self):
        if type(
            self.coefficient_norm_warps
        ) is not int or self.coefficient_norm_warps not in (1, 4):
            raise ValueError("coefficient_norm_warps must be 1 or 4")
        if type(self.coefficient_norm) is not bool:
            raise ValueError("coefficient_norm must be bool")
        if self.coefficient_norm and not self.boundary_cache:
            raise ValueError("coefficient_norm requires boundary_cache")
        if type(self.boundary_cache) is not bool:
            raise ValueError("boundary_cache must be bool")
        if self.boundary_cache and (not self.sample_cache or self.atom_block == 1):
            raise ValueError("boundary_cache requires streamed cache and grouped atoms")
        if type(self.minimum_sharing) is not int or self.minimum_sharing < 0:
            raise ValueError("minimum_sharing must be a nonnegative integer")
        if self.minimum_sharing and self.strategy != "shared":
            raise ValueError("minimum_sharing requires shared strategy")
        if type(self.sample_cache) is not int or self.sample_cache not in (0, 128, 512):
            raise ValueError("sample_cache must be 0,128,512")
        if self.sample_cache and not self.native_cache:
            raise ValueError("sample_cache requires native_cache")
        if type(self.atom_block) is not int or self.atom_block not in (1, 4):
            raise ValueError("atom_block must be 1 or 4")
        if type(self.batch_block) is not int or self.batch_block not in (1, 4, 8):
            raise ValueError("batch_block must be 1,4,8")
        if self.reuse not in ("none", "h", "all", "chunk"):
            raise ValueError("reuse must be none,h,all,chunk")
        if self.moment_policy not in ("all", "demand", "hierarchical"):
            raise ValueError("invalid moment policy")
        if type(self.native_cache) is not bool:
            raise ValueError("native_cache must be bool")
        if type(self.table_coefficients) is not bool:
            raise ValueError("table_coefficients must be bool")
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


@dataclass(frozen=True)
class LocalFourierRecipe(FourierRecipe):
    """Separate serialized recipe for the bounded local-output prototype."""

    local_blocks: int = 0
    local_overlap: bool = False
    local_width_cutoff: float = 0.35

    def __post_init__(self):
        super().__post_init__()
        if type(self.local_blocks) is not int or not 0 <= self.local_blocks <= 4096:
            raise ValueError("local_blocks must be an integer in 0..4096")
        if type(self.local_overlap) is not bool:
            raise ValueError("local_overlap must be bool")
        if (
            type(self.local_width_cutoff) not in (int, float)
            or not 0.1 <= self.local_width_cutoff <= 0.5
        ):
            raise ValueError("local_width_cutoff must be .1..0.5")
        if self.local_overlap and not self.local_blocks:
            raise ValueError("local_overlap requires local_blocks")
        if self.local_blocks and (
            self.atom_block != 4
            or not self.sample_cache
            or self.reuse != "none"
            or self.sort_atoms
            or self.coefficient_norm
            or self.strategy != "shared"
        ):
            raise ValueError(
                "local prototype requires grouped sampled shared execution, reuse=none, no sort/coefficient norm"
            )


@dataclass(frozen=True)
class PackedFourierRecipe(LocalFourierRecipe):
    """N128/M8 live geometry packing, distinct from the existing recipes."""

    atom_chunk: int = 256
    minimum_sites: int = 32
    width_cutoff: float = 0.1
    native_cache: bool = True
    table_coefficients: bool = True
    atom_block: int = 4
    batch_block: int = 4
    moment_policy: str = "hierarchical"
    sample_cache: int = 128
    boundary_cache: bool = True
    minimum_sharing: int = 8
    local_blocks: int = 8
    group_workers: int = 32

    def __post_init__(self):
        super().__post_init__()
        if type(self.group_workers) is not int or self.group_workers not in (
            0,
            8,
            16,
            32,
        ):
            raise ValueError("group_workers must be 0,8,16,32")
        if not self.local_blocks:
            raise ValueError("packed prototype requires local output")


@dataclass(frozen=True)
class BoundedCacheFourierRecipe(PackedFourierRecipe):
    """Cache only profiles that fit; wider supports remain fully evaluated."""

    cache_sites: int = 32

    def __post_init__(self):
        super().__post_init__()
        if type(self.cache_sites) is not int or self.cache_sites not in (32, 64, 128):
            raise ValueError("cache_sites must be 32,64,128")
        if self.sample_cache != 128 or self.coefficient_norm:
            raise ValueError("bounded cache requires sampled FP64 preparation")


def torch_reference(x, p, spec):
    from torchcst._backends.torch.kernels.hierarchical_fourier import values

    sites = torch.arange(spec.size, device=x.device, dtype=torch.float64) * spec.spacing
    width = 0.1 + 9.9 * p[:, 3].double().sigmoid()
    profile = FourierPeakSpec(modes=spec.modes)
    u = values(profile, spec.period, sites, p[:, 1], width, support="centered")
    v = values(profile, spec.period, sites, p[:, 2], width, support="centered")
    denom = (u.norm(dim=0) * v.norm(dim=0)).clamp_min(spec.floor)
    return ((x.double() @ v) * (p[:, 0].double() / denom)) @ u.T
