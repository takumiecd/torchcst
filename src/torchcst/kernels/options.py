"""Execution limits, separate from the mathematical KernelSpec."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelOptions:
    site_chunk: int = 2048
    atom_chunk: int = 64
    checkpoint_blocks: bool = True

    def __post_init__(self):
        if (
            type(self.site_chunk) is not int
            or self.site_chunk < 1
            or type(self.atom_chunk) is not int
            or self.atom_chunk < 1
            or type(self.checkpoint_blocks) is not bool
        ):
            raise ValueError("invalid Torch kernel execution options")
