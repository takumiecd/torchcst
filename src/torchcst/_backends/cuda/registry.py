"""Explicit ID-to-implementation registry, with strict direct-plan execution."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import torch
from torch import Tensor

from torchcst._backends.cuda.algorithm import Algorithm
from torchcst._backends.cuda.schema import (
    DispatchContext,
    ExecutionPlan,
)
from torchcst.operators.spec import OperatorSpec


class Registry:
    def __init__(self):
        self._entries: dict[tuple[str, str], Algorithm] = {}

    def register(self, algorithm: Algorithm):
        if not isinstance(algorithm, Algorithm):
            raise TypeError("registry requires an Algorithm instance")
        key = (algorithm.id, algorithm.revision)
        if key in self._entries:
            raise ValueError(f"duplicate algorithm registration: {key}")
        self._entries[key] = algorithm

    def get(self, algorithm_id: str, *, revision: str) -> Algorithm:
        try:
            return self._entries[(algorithm_id, revision)]
        except KeyError:
            raise ValueError(
                f"unknown algorithm/revision: {algorithm_id}@{revision}"
            ) from None

    def validate_plan(self, plan: ExecutionPlan) -> Algorithm:
        """Validate identity and recipe without tensor metadata or GPU imports."""
        if type(plan) is not ExecutionPlan:
            raise TypeError("expected an ExecutionPlan")
        if type(plan.schema_version) is not int or plan.schema_version != 1:
            raise ValueError("unsupported execution plan schema version")
        algorithm = self.get(plan.algorithm_id, revision=plan.algorithm_revision)
        if type(plan.recipe) is not algorithm.recipe_type:
            raise TypeError("recipe type does not match algorithm")
        algorithm.validate_recipe(plan.recipe)
        return algorithm

    def load_plan(self, value: dict) -> ExecutionPlan:
        """Decode a complete declaration using only registered recipe types."""
        keys = {"schema_version", "algorithm_id", "algorithm_revision", "recipe"}
        if type(value) is not dict or set(value) != keys:
            raise ValueError("execution plan needs exactly: " + ", ".join(sorted(keys)))
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported execution plan schema version")
        if any(
            type(value[k]) is not str or not value[k]
            for k in ("algorithm_id", "algorithm_revision")
        ):
            raise ValueError("algorithm ID and revision must be nonempty strings")
        algorithm = self.get(
            value["algorithm_id"], revision=value["algorithm_revision"]
        )
        if not is_dataclass(algorithm.recipe_type):
            raise TypeError("serialized plans require a dataclass recipe")
        recipe_keys = {
            field.name for field in fields(algorithm.recipe_type) if field.init
        }
        if type(value["recipe"]) is not dict or set(value["recipe"]) != recipe_keys:
            raise ValueError("recipe needs exactly: " + ", ".join(sorted(recipe_keys)))
        plan = ExecutionPlan(
            value["algorithm_id"],
            value["algorithm_revision"],
            algorithm.recipe_type(**value["recipe"]),
            value["schema_version"],
        )
        self.validate_plan(plan)
        return plan

    def validate(self, plan: ExecutionPlan, context: DispatchContext) -> Algorithm:
        algorithm = self.validate_plan(plan)
        if algorithm.operation_id != context.operator.operation_id:
            raise ValueError("plan and operator mathematical contract differ")
        support = algorithm.supports(context, plan.recipe)
        if not support.supported:
            raise ValueError(
                "unsupported execution plan: " + "; ".join(support.reasons)
            )
        bound = algorithm.workspace_bound(context, plan.recipe)
        if bound is not None and (type(bound) is not int or bound < 0):
            raise ValueError("invalid workspace upper bound")
        if context.workspace_limit_bytes is not None:
            if bound is None:
                raise ValueError(
                    "workspace upper bound is unknown; cannot enforce limit"
                )
            if bound > context.workspace_limit_bytes:
                raise ValueError("plan exceeds workspace limit")
        return algorithm

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
        if parameters.shape != (context.atom_count, context.parameter_dim):
            raise ValueError("parameters shape differs from dispatch context")
        if parameters.stride() != (context.parameter_dim, 1):
            raise ValueError("parameters must be contiguous [atoms, D]")
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
            from torchcst._backends.cuda.context import context_from_tensors

            actual = context_from_tensors(operator, x, parameters)
            if (
                actual.precision != context.precision
                or actual.deterministic != context.deterministic
                or actual.execution_mode != context.execution_mode
                or actual.device != context.device
            ):
                raise ValueError("CUDA execution settings differ from dispatch context")
        algorithm = self.validate(plan, context)
        return algorithm.execute(
            x=x, parameters=parameters, operator=operator, recipe=plan.recipe
        )
