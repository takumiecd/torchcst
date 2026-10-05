"""Lossless Plan JSON round trips and strict metadata boundaries, without a GPU."""

import json
from dataclasses import asdict, dataclass, replace

import pytest
import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.catalog import REGISTRY
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.full.recipe import (
    FullRecipe,
)
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.plans import (
    FULL,
    WINDOW,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import (
    ExecutionPlan,
    SupportResult,
)


@pytest.mark.parametrize("plan", [FULL, WINDOW])
def test_registered_plan_round_trip_as_dictionary_json_and_file(plan, tmp_path):
    data = REGISTRY.dump_plan(plan)
    assert data == asdict(plan)
    assert REGISTRY.load_plan(data) == plan
    text = REGISTRY.dumps_plan(plan)
    assert json.loads(text) == data
    assert text == REGISTRY.dumps_plan(plan)
    restored = REGISTRY.loads_plan(text)
    assert restored == plan and type(restored.recipe) is type(plan.recipe)
    path = tmp_path / "plan.json"
    path.write_text(text, encoding="utf-8")
    assert REGISTRY.loads_plan(path.read_bytes()) == plan


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"recipe":NaN}',
        '{"recipe":Infinity}',
        '{"recipe":1e999}',
        '{"recipe":-1e999}',
        "[]",
        "null",
    ],
)
def test_json_import_rejects_ambiguous_nonfinite_or_nonobject_plan(text):
    with pytest.raises(ValueError):
        REGISTRY.loads_plan(text)


def test_import_and_export_both_reject_unknown_or_unvalidated_plans():
    unknown = replace(FULL, algorithm_revision="missing")
    invalid = replace(FULL, recipe=FullRecipe(atom_num_warps=2))
    for plan in (unknown, invalid):
        with pytest.raises(ValueError):
            REGISTRY.dumps_plan(plan)
        with pytest.raises(ValueError):
            REGISTRY.loads_plan(json.dumps(asdict(plan)))


@dataclass(frozen=True)
class MetadataRecipe:
    payload: object


class MetadataAlgorithm(Algorithm[MetadataRecipe]):
    def validate_recipe(self, recipe):
        pass

    def supports(self, context, recipe):
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return None

    def execute(self, **kwargs):
        raise AssertionError("serialization must not execute an Algorithm")


def metadata_registry():
    registry = Registry()
    registry.register(
        MetadataAlgorithm("metadata", "v1", "test", "test-v1", MetadataRecipe)
    )
    return registry


def test_export_is_detached_from_recipe_and_reloadable():
    registry = metadata_registry()
    payload = {"values": [1, 2.5, True, None], "label": "試験"}
    plan = ExecutionPlan("metadata", "v1", MetadataRecipe(payload))
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    data = registry.dump_plan(plan)
    data["recipe"]["payload"]["values"].append(10)
    assert payload["values"] == [1, 2.5, True, None]


class NoCopy:
    def __deepcopy__(self, memo):
        raise AssertionError("metadata export must not copy arbitrary objects")


@pytest.mark.parametrize(
    "payload",
    [(1, 2), {1: "value"}, {1, 2}, float("nan"), float("inf"), torch.ones(2), NoCopy()],
)
def test_export_rejects_non_json_recipe_values_without_coercion_or_copy(payload):
    registry = metadata_registry()
    plan = ExecutionPlan("metadata", "v1", MetadataRecipe(payload))
    with pytest.raises((ValueError, TypeError)):
        registry.dumps_plan(plan)
