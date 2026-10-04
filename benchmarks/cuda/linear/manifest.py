"""Strict Plan catalog and benchmark Case declarations, loadable without a GPU."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.cuda.linear.local_product import SEMANTICS as LOCAL_SEMANTICS
from benchmarks.cuda.linear.local_product import LocalAlgorithm
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip import (
    NormalizedFullAlgorithm,
    NormalizedWindowAlgorithm,
)
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.contract import (
    OPERATION,
    SEMANTICS,
)
from torchcst._backends.cuda.registry import Registry
from torchcst._backends.cuda.schema import ExecutionPlan
from torchcst._backends.cuda.serialization import decode_json


class BenchmarkRegistry(Registry):
    """Local recipes have immutable boundary tuples, represented as JSON arrays."""

    def dump_plan(self, plan):
        if plan.algorithm_id != "research_local_product":
            return super().dump_plan(plan)
        self.validate_plan(plan)
        recipe = asdict(plan.recipe)
        recipe["rho_upper"] = list(recipe["rho_upper"])
        value = {
            "schema_version": plan.schema_version,
            "algorithm_id": plan.algorithm_id,
            "algorithm_revision": plan.algorithm_revision,
            "recipe": recipe,
        }
        if self.load_plan(value) != plan:
            raise ValueError("local recipe does not round-trip losslessly")
        return value


# This catalog is benchmark-local. Production registration/selection is unchanged.
REGISTRY = BenchmarkRegistry()
REGISTRY.register(NormalizedFullAlgorithm())
REGISTRY.register(NormalizedWindowAlgorithm())
REGISTRY.register(LocalAlgorithm())

DEFAULT_PLANS = Path(__file__).with_name("plans.json")


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"{name} needs exactly: {', '.join(sorted(expected))}")


def _id(value, name):
    if type(value) is not str or not re.fullmatch(r"[a-z][a-z0-9_.-]*", value):
        raise ValueError(f"{name} must be a lowercase identifier")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def read_json(path):
    raw = Path(path).read_bytes()
    value = decode_json(raw)
    return value, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PlanEntry:
    id: str
    plan: ExecutionPlan

    def __post_init__(self):
        _id(self.id, "plan id")
        REGISTRY.validate_plan(self.plan)


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    lr: float
    weight_decay: float
    fused: bool
    capturable: bool

    def __post_init__(self):
        if (
            self.name != "AdamW"
            or self.fused is not True
            or self.capturable is not True
        ):
            raise ValueError("this fixture requires fused capturable AdamW")
        for name, minimum, inclusive in [("lr", 0, False), ("weight_decay", 0, True)]:
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or (value < minimum if inclusive else value <= minimum)
            ):
                raise ValueError(f"invalid optimizer {name}")


@dataclass(frozen=True)
class WidthFixture:
    """Initial width mixture; all widths remain trainable through polar activity."""

    minimum: float
    birth: float
    maximum: float
    rho: tuple[float, ...]
    fractions: tuple[float, ...]
    center_jitter: float | None

    def __post_init__(self):
        for name in ("minimum", "birth", "maximum"):
            x = getattr(self, name)
            if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
                raise ValueError("width bounds must be positive finite numbers")
        if (
            not self.minimum <= self.birth <= self.maximum
            or self.minimum == self.maximum
        ):
            raise ValueError("invalid width bounds")
        for name in ("rho", "fractions"):
            values = getattr(self, name)
            if type(values) not in (tuple, list) or not values:
                raise ValueError("width mixture must have nonempty arrays")
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
                raise ValueError("width mixture must contain finite numbers")
            object.__setattr__(self, name, tuple(values))
        if len(self.rho) != len(self.fractions) or any(x < 0 for x in self.fractions):
            raise ValueError("invalid width mixture fractions")
        if not math.isclose(sum(self.fractions), 1.0, abs_tol=1e-9):
            raise ValueError("width mixture fractions must sum to one")
        # This benchmark's domain spacing is one. Initialize within its shared bounds.
        if any(not self.birth <= x <= self.maximum for x in self.rho):
            raise ValueError("initial rho must be within birth..maximum")
        if self.center_jitter is not None and (
            type(self.center_jitter) not in (int, float)
            or not math.isfinite(self.center_jitter)
            or not 0 <= self.center_jitter <= 0.5
        ):
            raise ValueError("center jitter must be None or in [0,.5]")


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    fixture: str
    size: int
    rows: int
    atoms: int
    profile: str
    dtype: str
    seed: int
    warmup: int
    rounds: int
    optimizer: OptimizerSpec
    widths: WidthFixture | None = None

    def __post_init__(self):
        _id(self.id, "case id")
        if self.fixture not in ("normalized_euclidean_strip", "local_polar_product"):
            raise ValueError("unknown benchmark fixture")
        sizes = (
            (16, 32, 64, 128) if self.fixture == "local_polar_product" else (1024, 8192)
        )
        if type(self.size) is not int or self.size not in sizes:
            raise ValueError(f"fixture size must be one of {sizes}")
        if self.fixture == "local_polar_product" and (
            type(self.rows) is not int or not 1 <= self.rows <= 64 or self.atoms < 4
        ):
            raise ValueError(
                "local fixture requires batch 1..64 and at least four atoms"
            )
        for name in ["rows", "atoms", "warmup", "rounds"]:
            _integer(getattr(self, name), name)
        _integer(self.seed, "seed", 0)
        if self.seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
        profiles = (
            (
                "broad",
                "sharp",
                "few",
                "wide",
                "mixed",
                "rho1",
                "rho1_5",
                "rho2",
                "rho3",
                "rho4",
                "rho8",
                "rho16",
            )
            if self.fixture == "local_polar_product"
            else ("broad", "sharp")
        )
        if self.profile not in profiles or self.dtype != "float32":
            raise ValueError("fixture requires broad/sharp profile and float32")
        if type(self.optimizer) is not OptimizerSpec:
            raise TypeError("case needs an OptimizerSpec")
        if self.widths is not None and (
            type(self.widths) is not WidthFixture
            or self.fixture != "local_polar_product"
            or self.profile != "mixed"
        ):
            raise ValueError("custom widths require the local mixed fixture")


@dataclass(frozen=True)
class BenchmarkRun:
    case: BenchmarkCase
    plans: tuple[PlanEntry, ...]
    baseline: str
    dense: bool
    input_hashes: dict[str, str]

    def __post_init__(self):
        if not self.plans or len({p.id for p in self.plans}) != len(self.plans):
            raise ValueError("run needs unique nonempty plans")
        if self.baseline not in {p.id for p in self.plans}:
            raise ValueError("baseline must be explicitly included in plans")
        if type(self.dense) is not bool:
            raise ValueError("dense must be a bool")
        for entry in self.plans:
            algorithm = REGISTRY.validate_plan(entry.plan)
            if algorithm.operation_id != OPERATION or algorithm.semantics_id != (
                LOCAL_SEMANTICS
                if self.case.fixture == "local_polar_product"
                else SEMANTICS
            ):
                raise ValueError("plan mathematical contract differs from fixture")

    def entry(self, id):
        try:
            return next(p for p in self.plans if p.id == id)
        except StopIteration:
            raise ValueError(f"plan is not selected in this run: {id}") from None

    def snapshot(self):
        return {
            "schema_version": 1,
            "case": asdict(self.case),
            "plans": [
                {"id": p.id, "plan": REGISTRY.dump_plan(p.plan)} for p in self.plans
            ],
            "baseline": self.baseline,
            "dense": self.dense,
            "input_hashes": self.input_hashes,
        }


def _version(value):
    if type(value) is not int or value != 1:
        raise ValueError("unsupported benchmark schema version")


def decode_catalog(value):
    _keys(value, ["schema_version", "plans"], "plan catalog")
    _version(value["schema_version"])
    if type(value["plans"]) is not list or not value["plans"]:
        raise ValueError("catalog needs a nonempty plan list")
    entries = []
    for item in value["plans"]:
        _keys(item, ["id", "plan"], "plan entry")
        entries.append(PlanEntry(item["id"], REGISTRY.load_plan(item["plan"])))
    if len({p.id for p in entries}) != len(entries):
        raise ValueError("duplicate plan id")
    identities = {REGISTRY.dumps_plan(p.plan) for p in entries}
    if len(identities) != len(entries):
        raise ValueError("duplicate execution plan under different ids")
    return tuple(entries)


def _case(value):
    keys = [
        "id",
        "fixture",
        "size",
        "rows",
        "atoms",
        "profile",
        "dtype",
        "seed",
        "warmup",
        "rounds",
        "optimizer",
    ]
    if "widths" in value:
        keys.append("widths")
    _keys(value, keys, "case")
    optimizer = value["optimizer"]
    _keys(optimizer, ["name", "lr", "weight_decay", "fused", "capturable"], "optimizer")
    widths = value.get("widths")
    if widths is not None:
        _keys(
            widths,
            ["minimum", "birth", "maximum", "rho", "fractions", "center_jitter"],
            "width fixture",
        )
        widths = WidthFixture(**widths)
    return BenchmarkCase(
        **(value | {"optimizer": OptimizerSpec(**optimizer), "widths": widths})
    )


def load_run(case_file, plans_file=DEFAULT_PLANS):
    catalog, catalog_hash = read_json(plans_file)
    entries = decode_catalog(catalog)
    value, case_hash = read_json(case_file)
    _keys(
        value, ["schema_version", "case", "plans", "baseline", "dense"], "benchmark run"
    )
    _version(value["schema_version"])
    if (
        type(value["plans"]) is not list
        or not value["plans"]
        or any(type(id) is not str for id in value["plans"])
    ):
        raise ValueError("run needs a list of plan ids")
    by_id = {p.id: p for p in entries}
    if any(id not in by_id for id in value["plans"]):
        raise ValueError("run selects an unknown plan id")
    return BenchmarkRun(
        _case(value["case"]),
        tuple(by_id[id] for id in value["plans"]),
        value["baseline"],
        value["dense"],
        {"case_sha256": case_hash, "plans_sha256": catalog_hash},
    )


def load_snapshot(path, *, expected_hash=None):
    value, actual_hash = read_json(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError("run snapshot hash differs from coordinator")
    return decode_snapshot(value)


def decode_snapshot(value):
    """Decode a frozen run declaration without needing a temporary file."""
    _keys(
        value,
        ["schema_version", "case", "plans", "baseline", "dense", "input_hashes"],
        "run snapshot",
    )
    _version(value["schema_version"])
    _keys(value["input_hashes"], ["case_sha256", "plans_sha256"], "input hashes")
    if any(
        type(v) is not str or not re.fullmatch(r"[0-9a-f]{64}", v)
        for v in value["input_hashes"].values()
    ):
        raise ValueError("input hashes must be SHA256 hex strings")
    entries = decode_catalog({"schema_version": 1, "plans": value["plans"]})
    return BenchmarkRun(
        _case(value["case"]),
        entries,
        value["baseline"],
        value["dense"],
        value["input_hashes"],
    )
