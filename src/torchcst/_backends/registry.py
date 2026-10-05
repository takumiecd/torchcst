"""Explicit ID-to-implementation registry and strict Plan declaration codec."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import ExecutionPlan
from torchcst._backends.serialization import decode_json, encode_json


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

    def dump_plan(self, plan: ExecutionPlan) -> dict:
        """Export a validated Plan as detached, losslessly reloadable JSON data."""
        algorithm = self.validate_plan(plan)
        if not is_dataclass(algorithm.recipe_type):
            raise TypeError("serialized plans require a dataclass recipe")
        value = {
            "schema_version": plan.schema_version,
            "algorithm_id": plan.algorithm_id,
            "algorithm_revision": plan.algorithm_revision,
            "recipe": {
                field.name: getattr(plan.recipe, field.name)
                for field in fields(algorithm.recipe_type)
                if field.init
            },
        }
        # JSON-native fields only. Avoid deepcopy/asdict on unknown recipe values,
        # which could copy tensors or invoke arbitrary serialization hooks.
        value = decode_json(encode_json(value))
        if self.load_plan(value) != plan:
            raise ValueError("recipe cannot round-trip through JSON losslessly")
        return value

    def dumps_plan(self, plan: ExecutionPlan) -> str:
        """Export one Plan to deterministic JSON text, without importing GPU code."""
        return encode_json(self.dump_plan(plan))

    def loads_plan(self, value: str | bytes | bytearray) -> ExecutionPlan:
        """Import JSON text using only Algorithm/recipe types in this registry."""
        return self.load_plan(decode_json(value))

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
