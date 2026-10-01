"""Immutable execution contracts; importing this module needs no CUDA or Triton."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

OPERATION = "normalized_strip_linear"
SEMANTICS = "normalized-strip-triweight-l2-v1"


@dataclass(frozen=True)
class NormalizedStripSpec:
    sizes: tuple[int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    operation_id: str = OPERATION
    semantics_id: str = SEMANTICS

    def __post_init__(self):
        if self.operation_id != OPERATION or self.semantics_id != SEMANTICS:
            raise ValueError("unknown normalized Strip mathematical contract")
        if (
            not isinstance(self.sizes, tuple)
            or len(self.sizes) != 3
            or any(type(n) is not int or n <= 0 for n in self.sizes)
        ):
            raise ValueError("sizes must contain three positive integers")
        for name, values in (("origin", self.origin), ("spacing", self.spacing)):
            if (
                not isinstance(values, tuple)
                or len(values) != 3
                or not all(math.isfinite(v) for v in values)
            ):
                raise ValueError(f"{name} must contain three finite values")
        if min(self.spacing) <= 0:
            raise ValueError("spacing must be positive")

    @property
    def n(self):
        return self.sizes[0]

    @property
    def k(self):
        return math.prod(self.sizes[1:])

    def declaration(self):
        """Expose this legacy specialized metadata as a common operator contract."""
        from torchcst.operators.normalized_strip import normalized_strip_declaration

        return normalized_strip_declaration(
            sizes=self.sizes, origin=self.origin, spacing=self.spacing
        )

    @classmethod
    def from_declaration(cls, spec):
        """Adapt only the exact normalized Strip meaning; keep launch guards separate."""
        from torchcst.operators.normalized_strip import normalized_strip_metadata

        sizes, origin, spacing = normalized_strip_metadata(spec)
        return cls(sizes=sizes, origin=origin, spacing=spacing)


# Compatibility for the first registry API; common OperatorSpec lives outside CUDA.
OperatorSpec = NormalizedStripSpec

@dataclass(frozen=True)
class DeviceInfo:
    type: str
    index: int | None = None
    name: str = ""
    compute_capability: tuple[int, int] | None = None
    sm_count: int | None = None


@dataclass(frozen=True)
class PrecisionPolicy:
    autocast: bool = False
    allow_tf32: bool = False


@dataclass(frozen=True)
class RequiredGrads:
    inputs: bool = False
    parameters: bool = False


@dataclass(frozen=True)
class DispatchContext:
    operator: NormalizedStripSpec
    input_shape: tuple[int, ...]
    input_strides: tuple[int, ...]
    dtype: torch.dtype
    atom_count: int
    device: DeviceInfo
    required_grads: RequiredGrads = RequiredGrads()
    execution_mode: str = "eager"
    deterministic: bool = False
    precision: PrecisionPolicy = PrecisionPolicy()
    workspace_limit_bytes: int | None = None

    def __post_init__(self):
        if not self.input_shape or self.input_shape[-1] != self.operator.k:
            raise ValueError("input feature dimension differs from operator")
        if any(type(n) is not int or n < 0 for n in self.input_shape):
            raise ValueError("input shape must contain nonnegative integers")
        if len(self.input_strides) != len(self.input_shape) or any(
            type(s) is not int or s < 0 for s in self.input_strides
        ):
            raise ValueError("input strides must match shape")
        if type(self.atom_count) is not int or self.atom_count < 0:
            raise ValueError("atom count must be nonnegative")
        if self.execution_mode not in ("eager", "cuda_graph"):
            raise ValueError("unknown execution mode")
        if self.workspace_limit_bytes is not None and (
            type(self.workspace_limit_bytes) is not int
            or self.workspace_limit_bytes < 0
        ):
            raise ValueError("workspace limit must be nonnegative or None")

    @property
    def m(self):
        return math.prod(self.input_shape[:-1])


@dataclass(frozen=True)
class FullRecipe:
    id: str = "normalized_full.default.v1"
    atom_num_warps: int = 1
    sorted_forward: bool = True
    sorted_backward: bool = True
    support: str = "ball"
    enable_fp_fusion: bool = True
    saved_support_flags: bool = True
    tuple_grads: bool = False


@dataclass(frozen=True)
class WindowRecipe:
    id: str = "normalized_window.rows512.v1"
    window_rows: int = 512
    enable_fp_fusion: bool = True


@dataclass(frozen=True)
class ExecutionPlan:
    algorithm_id: str
    algorithm_revision: str
    recipe: FullRecipe | WindowRecipe
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
    tree_revision: str
    matched_path: tuple[str, ...]
    reason: str
    evidence_ids: tuple[str, ...] = ()
    workspace_upper_bound_bytes: int | None = None
