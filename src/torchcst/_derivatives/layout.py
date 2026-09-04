"""Stable flattening for the trainable parameters owned by one CST site."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ParameterSpec:
    """One named slice in a flattened CST parameter vector."""

    name: str
    shape: torch.Size
    start: int
    stop: int


class ParameterLayout:
    """Capture a deterministic, duplicate-free parameter layout."""

    def __init__(self, module: nn.Module) -> None:
        named = tuple(module.named_parameters(remove_duplicate=True))
        if not named:
            raise ValueError("a CST site must own at least one parameter")

        self._parameters = tuple(parameter for _, parameter in named)
        specs: list[ParameterSpec] = []
        offset = 0
        for name, parameter in named:
            size = parameter.numel()
            specs.append(ParameterSpec(name, parameter.shape, offset, offset + size))
            offset += size
        self.specs = tuple(specs)
        self.numel = offset

        devices = {parameter.device for parameter in self._parameters}
        dtypes = {parameter.dtype for parameter in self._parameters}
        if len(devices) != 1 or len(dtypes) != 1:
            raise ValueError("all CST site parameters must share one device and dtype")
        self.device = next(iter(devices))
        self.dtype = next(iter(dtypes))

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        """The uniquely owned parameters in layout order."""

        return self._parameters

    def current(self) -> Tensor:
        """Return a detached copy of the current flattened parameter point."""

        return torch.cat(
            [parameter.detach().reshape(-1) for parameter in self._parameters]
        ).clone()

    def unpack(self, vector: Tensor) -> dict[str, Tensor]:
        """Return named tensor views into a flat parameter vector."""

        self.validate(vector, name="parameter vector")
        return {
            spec.name: vector[spec.start : spec.stop].view(spec.shape)
            for spec in self.specs
        }

    def validate(self, vector: Tensor, *, name: str) -> None:
        if vector.ndim != 1 or vector.numel() != self.numel:
            raise ValueError(f"{name} must have shape [{self.numel}]")
        if vector.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")
        if vector.dtype != self.dtype:
            raise ValueError(f"{name} must have dtype {self.dtype}")
