"""Public two-tier policy. GitHub supplies identity; uploaded JSON does not."""

import re
from dataclasses import dataclass
from pathlib import Path

from benchmarks.database.model import digest
from torchcst._backends.cuda.serialization import decode_json

DEFAULT_POLICY = (
    Path(__file__).resolve().parents[2] / ".github/benchmark-submissions.json"
)


def positive_id(value):
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError("GitHub ID must be a positive signed bigint")
    return value


def repository_name(value):
    if type(value) is not str or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", value
    ):
        raise ValueError("invalid GitHub repository")
    return value


def account_login(value):
    if type(value) is not str or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", value
    ):
        raise ValueError("invalid GitHub account login")
    return value


@dataclass(frozen=True)
class Policy:
    declaration: dict

    def __post_init__(self):
        value = self.declaration
        if type(value) is not dict or set(value) != {
            "schema_version",
            "repository",
            "general_daily_runs",
            "general_trust_points",
            "recognized_trust_points",
            "max_file_bytes",
            "recognized",
        }:
            raise ValueError("invalid submission policy fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported submission policy")
        repository_name(value["repository"])
        if (
            type(value["general_daily_runs"]) is not int
            or not 1 <= value["general_daily_runs"] <= 10000
        ):
            raise ValueError("general daily run limit must be 1..10000")
        self._points(value["general_trust_points"])
        self._points(value["recognized_trust_points"])
        if (
            type(value["max_file_bytes"]) is not int
            or not 1 <= value["max_file_bytes"] <= 25 * 1024**2
        ):
            raise ValueError("attachment limit must be 1..25 MiB")
        if type(value["recognized"]) is not list:
            raise ValueError("recognized accounts must be a list")
        ids = set()
        for entry in value["recognized"]:
            if (
                type(entry) is not dict
                or not {"github_user_id", "login", "reason"} <= set(entry)
                or not set(entry)
                <= {"github_user_id", "login", "trust_points", "reason"}
            ):
                raise ValueError(
                    "recognized entry needs GitHub ID, display login, points and reason"
                )
            user_id = positive_id(entry["github_user_id"])
            account_login(entry["login"])
            self._points(entry.get("trust_points", value["recognized_trust_points"]))
            if user_id in ids:
                raise ValueError("duplicate recognized account ID")
            ids.add(user_id)
            if type(entry["reason"]) is not str or not 1 <= len(entry["reason"]) <= 500:
                raise ValueError(
                    "recognition needs a public reason of 1..500 characters"
                )

    @staticmethod
    def _points(value):
        if type(value) is not int or not 1 <= value <= 2147483647:
            raise ValueError("trust points must be a positive integer")

    @property
    def id(self):
        return digest(self.declaration)

    @property
    def repository(self):
        return self.declaration["repository"]

    @property
    def max_file_bytes(self):
        return self.declaration["max_file_bytes"]

    def tier(self, user_id):
        positive_id(user_id)
        return (
            "recognized"
            if any(
                e["github_user_id"] == user_id for e in self.declaration["recognized"]
            )
            else "general"
        )

    def daily_limit(self, user_id):
        return (
            None
            if self.tier(user_id) == "recognized"
            else self.declaration["general_daily_runs"]
        )

    def trust_points(self, user_id):
        positive_id(user_id)
        return next(
            (
                e.get("trust_points", self.declaration["recognized_trust_points"])
                for e in self.declaration["recognized"]
                if e["github_user_id"] == user_id
            ),
            self.declaration["general_trust_points"],
        )


def load_policy(path=DEFAULT_POLICY):
    return Policy(decode_json(Path(path).read_bytes()))
