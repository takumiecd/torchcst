"""Immutable execution contracts; importing this module needs no CUDA or Triton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Context(Protocol):
    """Operation-specific immutable metadata; no required tensor interface."""

    operation_id: str
    workspace_limit_bytes: int | None

    def validate_inputs(self, **inputs) -> None: ...


@dataclass(frozen=True)
class DeviceInfo:
    type: str
    index: int | None = None
    name: str = ""
    compute_capability: tuple[int, int] | None = None
    sm_count: int | None = None

    def __post_init__(self):
        if type(self.type) is not str or not self.type or type(self.name) is not str:
            raise ValueError("device type/name must be strings")
        if self.index is not None and (type(self.index) is not int or self.index < 0):
            raise ValueError("device index must be nonnegative or None")
        if self.compute_capability is not None and (
            type(self.compute_capability) is not tuple
            or len(self.compute_capability) != 2
            or any(type(n) is not int or n < 0 for n in self.compute_capability)
        ):
            raise ValueError("compute capability must contain two nonnegative integers")
        if self.sm_count is not None and (
            type(self.sm_count) is not int or self.sm_count < 1
        ):
            raise ValueError("SM count must be positive or None")


@dataclass(frozen=True)
class PrecisionPolicy:
    autocast: bool = False
    allow_tf32: bool = False

    def __post_init__(self):
        if any(type(v) is not bool for v in (self.autocast, self.allow_tf32)):
            raise ValueError("precision flags must be bool")


@dataclass(frozen=True)
class RequiredGrads:
    inputs: bool = False
    parameters: bool = False

    def __post_init__(self):
        if any(type(v) is not bool for v in (self.inputs, self.parameters)):
            raise ValueError("gradient flags must be bool")


@dataclass(frozen=True)
class ExecutionPlan:
    algorithm_id: str
    algorithm_revision: str
    recipe: object
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported execution plan schema version")
        if not self.algorithm_id or not self.algorithm_revision:
            raise ValueError("algorithm ID and revision are required")


@dataclass(frozen=True)
class SupportResult:
    reasons: tuple[str, ...] = ()

    @property
    def supported(self):
        return not self.reasons


@dataclass(frozen=True)
class DispatchDecision:
    plan: ExecutionPlan
    selector_revision: str
    matched_path: tuple[str, ...]
    reason: str
    evidence_ids: tuple[str, ...] = ()
    workspace_upper_bound_bytes: int | None = None


@dataclass(frozen=True)
class DefaultRecipe:
    """No tunable settings for a registered reference implementation."""
