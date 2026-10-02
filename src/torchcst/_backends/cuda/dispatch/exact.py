"""Versioned exact-table artifacts, compiled once into an in-memory index."""

import hashlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType

import torch

from torchcst._backends.cuda.schema import DispatchContext, ExecutionPlan
from torchcst._backends.cuda.serialization import decode_json, encode_json

from .base import Match, Selector
from .conditions import condition_key, context_key, dump_condition, exact_fields


def _identifier(value, label):
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")


@dataclass(frozen=True)
class ExactEntry:
    context: DispatchContext
    plan: ExecutionPlan
    evidence_ids: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.context, DispatchContext):
            raise TypeError("exact entry requires a DispatchContext")
        if type(self.plan) is not ExecutionPlan:
            raise TypeError("exact entry requires an ExecutionPlan")
        if type(self.evidence_ids) is not tuple or not self.evidence_ids:
            raise ValueError("exact entry requires an immutable tuple of evidence IDs")
        for item in self.evidence_ids:
            _identifier(item, "evidence ID")


class ExactSelector(Selector):
    """An offline ranking result; loading it does not certify its evidence."""

    def __init__(self, *, revision, registry, fallback_plan, index, document):
        super().__init__(
            revision=revision, registry=registry, fallback_plan=fallback_plan
        )
        self._index = MappingProxyType(dict(index))
        self._document = document

    @classmethod
    def from_entries(
        cls,
        entries,
        *,
        registry,
        fallback_plan,
        revision,
        score_policy,
        dataset_snapshot,
        runtime_versions=None,
    ):
        plans, records = {}, []

        def add_plan(plan):
            value = registry.dump_plan(plan)
            plan_id = hashlib.sha256(encode_json(value).encode()).hexdigest()
            plans[plan_id] = value
            return plan_id

        fallback_id = add_plan(fallback_plan)
        for entry in entries:
            records.append(
                {
                    "condition": dump_condition(entry.context),
                    "plan_id": add_plan(entry.plan),
                    "evidence_ids": list(entry.evidence_ids),
                }
            )
        return cls.load(
            {
                "schema_version": 1,
                "selector": {"kind": "exact_table", "revision": revision},
                "score_policy": score_policy,
                "dataset_snapshot": dataset_snapshot,
                "runtime_versions": (
                    {"torch": str(torch.__version__)}
                    if runtime_versions is None
                    else runtime_versions
                ),
                "plans": plans,
                "entries": records,
                "fallback_plan_id": fallback_id,
            },
            registry=registry,
        )

    @classmethod
    def loads(cls, text, *, registry):
        return cls.load(decode_json(text), registry=registry)

    @classmethod
    def load(cls, data, *, registry):
        text = encode_json(data)
        data = decode_json(text)
        exact_fields(
            data,
            (
                "schema_version",
                "selector",
                "score_policy",
                "dataset_snapshot",
                "runtime_versions",
                "plans",
                "entries",
                "fallback_plan_id",
            ),
            "dispatch artifact",
        )
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported dispatch artifact schema version")
        selector = data["selector"]
        exact_fields(selector, ("kind", "revision"), "selector")
        if selector["kind"] != "exact_table":
            raise ValueError("unsupported selector kind")
        _identifier(selector["revision"], "selector revision")
        score = data["score_policy"]
        exact_fields(score, ("id", "revision", "parameters"), "score_policy")
        _identifier(score["id"], "score policy ID")
        _identifier(score["revision"], "score policy revision")
        if type(score["parameters"]) is not dict:
            raise ValueError("score policy parameters must be an object")
        _identifier(data["dataset_snapshot"], "dataset snapshot")
        runtime = data["runtime_versions"]
        if (
            type(runtime) is not dict
            or "torch" not in runtime
            or set(runtime) - {"torch", "triton"}
        ):
            raise ValueError("runtime_versions requires torch and optionally triton")
        for package, expected in runtime.items():
            _identifier(expected, "runtime version")
            try:
                installed = (
                    str(torch.__version__) if package == "torch" else version(package)
                )
            except PackageNotFoundError:
                raise ValueError(
                    f"required runtime is not installed: {package}"
                ) from None
            if installed != expected:
                raise ValueError(f"dispatch runtime version differs: {package}")
        if type(data["plans"]) is not dict or not data["plans"]:
            raise ValueError("plans must be a nonempty object")
        plans = {}
        for name, declaration in data["plans"].items():
            _identifier(name, "plan ID")
            plans[name] = registry.load_plan(declaration)
        fallback_id = data["fallback_plan_id"]
        _identifier(fallback_id, "fallback plan ID")
        if fallback_id not in plans:
            raise ValueError("unknown fallback plan ID")
        if type(data["entries"]) is not list:
            raise ValueError("entries must be an array")
        index = {}
        for ordinal, entry in enumerate(data["entries"]):
            exact_fields(entry, ("condition", "plan_id", "evidence_ids"), "entry")
            key = condition_key(entry["condition"])
            if key in index:
                raise ValueError("duplicate exact condition")
            name = entry["plan_id"]
            _identifier(name, "entry plan ID")
            if name not in plans:
                raise ValueError("unknown entry plan ID")
            evidence = entry["evidence_ids"]
            if type(evidence) is not list or not evidence:
                raise ValueError("entry requires evidence IDs")
            for item in evidence:
                _identifier(item, "evidence ID")
            if len(set(evidence)) != len(evidence):
                raise ValueError("duplicate evidence ID")
            index[key] = Match(
                plan=plans[name],
                path=("exact_table", f"entry={ordinal}", plans[name].algorithm_id),
                reason="observed exact condition",
                evidence_ids=tuple(evidence),
            )
        result = cls(
            revision=selector["revision"],
            registry=registry,
            fallback_plan=plans[fallback_id],
            index=index,
            document=text,
        )
        return result

    def dump(self):
        """Return detached metadata for inspection/export, outside forward."""
        return decode_json(self._document)

    def dumps(self):
        return self._document

    def _match(self, context):
        return self._index.get(context_key(context))


def load_selector(text, *, registry):
    """Versioned artifact entry point; unimplemented kinds fail explicitly."""
    return ExactSelector.loads(text, registry=registry)
