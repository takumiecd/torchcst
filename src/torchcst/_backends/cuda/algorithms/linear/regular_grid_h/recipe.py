"""Small explicit atom/batch schedule; no global H/G retention."""

from dataclasses import dataclass

from ..periodic_product.recipe import PeriodicRecipe


@dataclass(frozen=True)
class OnchipHRecipe(PeriodicRecipe):
    batch_tile: int = 8

    def __post_init__(self):
        super().__post_init__()
        if self.gemm != "torch":
            raise ValueError("onchip H has no GEMM; use canonical gemm='torch'")
        if type(self.batch_tile) is not int or self.batch_tile not in (4, 8, 16):
            raise ValueError("batch_tile must be 4, 8 or 16")


@dataclass(frozen=True)
class OutputOwnedHRecipe(OnchipHRecipe):
    output_tile: int = 16

    def __post_init__(self):
        super().__post_init__()
        if type(self.output_tile) is not int or self.output_tile not in (8, 16, 32, 64):
            raise ValueError("output_tile must be 8, 16, 32 or 64")


@dataclass(frozen=True)
class ReusedHRecipe(OutputOwnedHRecipe):
    """One atom × batch_tile scratch buffer, reused across output owners."""


@dataclass(frozen=True)
class ParallelReusedHRecipe(ReusedHRecipe):
    """Independent H batch capacity; preserve batch_tile compute/layout."""

    h_batch: int = 16

    def __post_init__(self):
        super().__post_init__()
        if type(self.h_batch) is not int or self.h_batch not in (16, 32):
            raise ValueError("h_batch must be 16 or 32")
        if self.h_batch < self.batch_tile or self.h_batch % self.batch_tile:
            raise ValueError("h_batch must be a multiple of batch_tile")


@dataclass(frozen=True)
class PreparedReusedHRecipe(ParallelReusedHRecipe):
    """Three sorted output fields; finite output scale divided once per atom."""


@dataclass(frozen=True)
class GroupedOutputHRecipe(PreparedReusedHRecipe):
    """Wider output atom reduction; H generation/backward keep atom_group."""

    output_group: int = 32

    def __post_init__(self):
        super().__post_init__()
        if type(self.output_group) is not int or self.output_group not in (16, 32):
            raise ValueError("output_group must be 16 or 32")


@dataclass(frozen=True)
class InputOwnedHRecipe(GroupedOutputHRecipe):
    """Symmetric input-owner backward with one ephemeral G batch chunk."""

    input_tile: int = 8

    def __post_init__(self):
        super().__post_init__()
        if type(self.input_tile) is not int or self.input_tile not in (8, 16, 32, 64):
            raise ValueError("input_tile must be 8, 16, 32 or 64")


@dataclass(frozen=True)
class StreamingInputHRecipe(InputOwnedHRecipe):
    """Independent G cap and recycled parameter partials; forward H unchanged."""

    g_batch: int = 8

    def __post_init__(self):
        super().__post_init__()
        if type(self.g_batch) is not int or self.g_batch not in (8, 16, 32):
            raise ValueError("g_batch must be 8, 16 or 32")
        if self.g_batch < self.batch_tile or self.g_batch % self.batch_tile:
            raise ValueError("g_batch must be a multiple of batch_tile")


@dataclass(frozen=True)
class SiteRoutedHRecipe(GroupedOutputHRecipe):
    """Centre-site prefix intervals for forward output ownership."""


@dataclass(frozen=True)
class SiteRoutedStreamingHRecipe(StreamingInputHRecipe):
    """Site-routed forward H; streaming G retains coarse input routing."""


@dataclass(frozen=True)
class OwnerBatchHRecipe(SiteRoutedHRecipe):
    """BM16 output owner over BM8 H slabs; backward scheduling unchanged."""

    h_batch: int = 32
    owner_batch_tile: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.batch_tile != 8:
            raise ValueError("owner BM16 requires producer batch_tile=8")
        if type(self.owner_batch_tile) is not int or self.owner_batch_tile != 16:
            raise ValueError("owner_batch_tile must be 16")


@dataclass(frozen=True)
class OwnerBatchStreamingHRecipe(SiteRoutedStreamingHRecipe):
    """BM16 output owner over BM8 H slabs; backward scheduling unchanged."""

    h_batch: int = 32
    owner_batch_tile: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.batch_tile != 8:
            raise ValueError("owner BM16 requires producer batch_tile=8")
        if type(self.owner_batch_tile) is not int or self.owner_batch_tile != 16:
            raise ValueError("owner_batch_tile must be 16")
        if self.g_batch != 8:
            raise ValueError("owner BM16 streaming requires g_batch=8")
