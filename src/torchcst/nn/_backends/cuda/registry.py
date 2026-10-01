"""Explicit ID-to-implementation registry, with strict direct-plan execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .schema import DispatchContext, ExecutionPlan, OperatorSpec, SupportResult

Executor = Callable[..., Tensor]


@dataclass(frozen=True)
class AlgorithmEntry:
    id: str
    revision: str
    operation_id: str
    semantics_id: str
    recipe_type: type
    validate_recipe: Callable[[Any], None]
    supports: Callable[[DispatchContext, Any], SupportResult]
    workspace_bound: Callable[[DispatchContext, Any], int | None]
    load_executor: Callable[[], Executor]


class Registry:
    def __init__(self):
        self._entries: dict[tuple[str, str], AlgorithmEntry] = {}

    def register(self, entry: AlgorithmEntry):
        key = (entry.id, entry.revision)
        if key in self._entries:
            raise ValueError(f"duplicate algorithm registration: {key}")
        self._entries[key] = entry

    def get(self, algorithm_id: str, *, revision: str) -> AlgorithmEntry:
        try:
            return self._entries[(algorithm_id, revision)]
        except KeyError:
            raise ValueError(
                f"unknown algorithm/revision: {algorithm_id}@{revision}"
            ) from None

    def validate(self, plan: ExecutionPlan, context: DispatchContext) -> AlgorithmEntry:
        if type(plan.schema_version) is not int or plan.schema_version != 1:
            raise ValueError("unsupported execution plan schema version")
        entry = self.get(plan.algorithm_id, revision=plan.algorithm_revision)
        if (
            entry.operation_id != context.operator.operation_id
            or entry.semantics_id != context.operator.semantics_id
        ):
            raise ValueError("plan and operator mathematical contract differ")
        if type(plan.recipe) is not entry.recipe_type:
            raise TypeError("recipe type does not match algorithm")
        entry.validate_recipe(plan.recipe)
        support = entry.supports(context, plan.recipe)
        if not support.supported:
            raise ValueError(
                "unsupported execution plan: " + "; ".join(support.reasons)
            )
        bound = entry.workspace_bound(context, plan.recipe)
        if bound is not None and (type(bound) is not int or bound < 0):
            raise ValueError("invalid workspace upper bound")
        if context.workspace_limit_bytes is not None:
            if bound is None:
                raise ValueError(
                    "workspace upper bound is unknown; cannot enforce limit"
                )
            if bound > context.workspace_limit_bytes:
                raise ValueError("plan exceeds workspace limit")
        return entry

    def execute(
        self,
        plan: ExecutionPlan,
        context: DispatchContext,
        *,
        x: Tensor,
        parameters: Tensor,
        operator: OperatorSpec,
    ) -> Tensor:
        # Check tensor metadata against the context even for directly forced plans.
        if operator != context.operator:
            raise ValueError("operator differs from dispatch context")
        if (
            tuple(x.shape) != context.input_shape
            or tuple(x.stride()) != context.input_strides
        ):
            raise ValueError("input metadata differs from dispatch context")
        if parameters.shape != (context.atom_count, 5):
            raise ValueError("parameters must have shape [atoms, 5]")
        if parameters.stride() != (5, 1):
            raise ValueError("parameters must be contiguous [atoms, 5]")
        if x.dtype != context.dtype or parameters.dtype != context.dtype:
            raise ValueError("tensor dtype differs from dispatch context")
        if x.device != parameters.device or x.device.type != context.device.type:
            raise ValueError("tensor device differs from dispatch context")
        if x.device.index != context.device.index:
            raise ValueError("tensor device index differs from dispatch context")
        actual_grads = (
            torch.is_grad_enabled() and x.requires_grad,
            torch.is_grad_enabled() and parameters.requires_grad,
        )
        if actual_grads != (
            context.required_grads.inputs,
            context.required_grads.parameters,
        ):
            raise ValueError("gradient requirements differ from dispatch context")
        if x.is_cuda:
            from .context import context_from_tensors

            actual = context_from_tensors(operator, x, parameters)
            if (
                actual.precision != context.precision
                or actual.deterministic != context.deterministic
                or actual.execution_mode != context.execution_mode
                or actual.device != context.device
            ):
                raise ValueError("CUDA execution settings differ from dispatch context")
        entry = self.validate(plan, context)
        return entry.load_executor()(
            x=x, parameters=parameters, operator=operator, recipe=plan.recipe
        )
