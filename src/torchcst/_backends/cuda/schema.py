"""Immutable execution contracts; importing this module needs no CUDA or Triton."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from torchcst.operators import spec as operators


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
    operator: operators.OperatorSpec
    input_shape: tuple[int, ...]
    input_strides: tuple[int, ...]
    dtype: torch.dtype
    atom_count: int
    device: DeviceInfo
    parameter_dim: int
    required_grads: RequiredGrads = RequiredGrads()
    execution_mode: str = "eager"
    deterministic: bool = False
    precision: PrecisionPolicy = PrecisionPolicy()
    workspace_limit_bytes: int | None = None

    def __post_init__(self):
        if not isinstance(self.operator, operators.OperatorSpec):
            raise TypeError("operator must be a common OperatorSpec")
        if type(self.parameter_dim) is not int or self.parameter_dim <= 0:
            raise ValueError("parameter dimension must be positive")
        if not self.input_shape or self.input_shape[-1] != self.operator.in_features:
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
    tree_revision: str
    matched_path: tuple[str, ...]
    reason: str
    evidence_ids: tuple[str, ...] = ()
    workspace_upper_bound_bytes: int | None = None
