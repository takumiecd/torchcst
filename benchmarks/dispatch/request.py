"""Data-only generation requests; executable score functions are never imported."""

import re

from torchcst._backends.dispatch.conditions import exact_fields
from torchcst._backends.serialization import decode_json, encode_json


def validate_request(value):
    value = decode_json(encode_json(value))
    exact_fields(
        value,
        (
            "schema_version",
            "revision",
            "dataset",
            "execution_mode",
            "min_runs",
            "excluded_run_ids",
            "score_policy",
            "fallback_plan",
        ),
        "generation request",
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported generation request")
    if type(value["revision"]) is not str or not value["revision"]:
        raise ValueError("generation revision is required")
    if value["execution_mode"] not in ("eager", "cuda_graph"):
        raise ValueError("unknown execution mode")
    if type(value["min_runs"]) is not int or value["min_runs"] < 1:
        raise ValueError("min_runs must be positive")
    dataset = value["dataset"]
    if (
        type(dataset) is not dict
        or not {"adapter", "adapter_revision", "gpu", "source_id"} <= set(dataset)
        or set(dataset)
        - {
            "adapter",
            "adapter_revision",
            "gpu",
            "source_id",
            "case_id",
            "environment_id",
            "protocol_id",
        }
    ):
        raise ValueError("dataset needs adapter, revision, GPU and exact source_id")
    if (
        dataset["adapter"] != "cuda.linear.complete-step"
        or type(dataset["adapter_revision"]) is not int
        or dataset["adapter_revision"] not in (2, 3, 7)
    ):
        raise ValueError("unsupported generation adapter/revision")
    if type(dataset["gpu"]) is not str or not dataset["gpu"]:
        raise ValueError("GPU name is required")
    for key in ("source_id", "case_id", "environment_id", "protocol_id"):
        if key in dataset:
            _hash(dataset[key])
    excluded = value["excluded_run_ids"]
    if type(excluded) is not list or len(set(excluded)) != len(excluded):
        raise ValueError("excluded_run_ids must be unique IDs")
    for item in excluded:
        _hash(item)
    score = value["score_policy"]
    exact_fields(score, ("id", "revision", "parameters"), "score policy")
    if (
        any(type(score[k]) is not str or not score[k] for k in ("id", "revision"))
        or type(score["parameters"]) is not dict
    ):
        raise ValueError("invalid score policy metadata")
    return value


def _hash(value):
    if type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("dataset/exclusion IDs must be SHA256 hex strings")
