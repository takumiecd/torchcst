"""Canonical, self-verifying structural event ledger."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from torch import Tensor

from cstf.storage import SynapseBirth, SynapseDeath


class Ledger:
    """Record structural operations, counters, and file provenance as JSON."""

    SCHEMA = "cstf-ledger-v1"

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._counters: dict[str, dict[str, int]] = {}
        self._provenance: dict[str, str] = {}
        self._sha256: str | None = None

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._json_copy(event) for event in self._events)

    @property
    def counters(self) -> dict[str, dict[str, int]]:
        return self._json_copy(self._counters)

    @property
    def provenance_hashes(self) -> dict[str, str]:
        return dict(self._provenance)

    @property
    def sha256(self) -> str | None:
        return self._sha256

    def record_event(
        self, event_index: int, ops: Sequence[SynapseBirth | SynapseDeath]
    ) -> dict[str, Any]:
        """Append one event after canonicalizing supported structural ops."""
        if isinstance(event_index, bool) or not isinstance(event_index, int):
            raise TypeError("event_index must be an int")
        if event_index < 0:
            raise ValueError("event_index must be non-negative")
        ops = tuple(ops)
        serialized = [self._serialize_op(op) for op in ops]
        event = {"event_index": event_index, "ops": serialized}
        self._events.append(event)
        for op in ops:
            counts = self._counters.setdefault(op.site, {"birth": 0, "death": 0})
            if isinstance(op, SynapseBirth):
                counts["birth"] += int(op.w.numel())
            else:
                counts["death"] += int(op.ids.numel())
        self._sha256 = None
        return self._json_copy(event)

    def provenance(self, files: Iterable[str | Path]) -> dict[str, str]:
        """Hash arbitrary files and retain their caller-supplied path labels."""
        recorded: dict[str, str] = {}
        for raw_path in files:
            path = Path(raw_path)
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            label = str(raw_path)
            value = digest.hexdigest()
            self._provenance[label] = value
            recorded[label] = value
        self._sha256 = None
        return dict(recorded)

    def payload(self) -> dict[str, Any]:
        """Return the unhashed canonical payload represented by this ledger."""
        return {
            "counters": self.counters,
            "events": [self._json_copy(event) for event in self._events],
            "provenance": dict(self._provenance),
            "schema": self.SCHEMA,
        }

    def to_json(self, path: str | Path) -> str:
        """Write canonical JSON and return its embedded semantic self-hash."""
        payload = self.payload()
        digest = hashlib.sha256(self._canonical_bytes(payload)).hexdigest()
        document = {**payload, "sha256": digest}
        Path(path).write_bytes(self._canonical_bytes(document))
        self._sha256 = digest
        return digest

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        """Load only canonical JSON whose embedded self-hash verifies."""
        raw = Path(path).read_bytes()
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid ledger JSON") from exc
        if not isinstance(document, dict):
            raise ValueError("ledger document must be a JSON object")
        if raw != cls._canonical_bytes(document):
            raise ValueError("ledger JSON is not in canonical form")
        observed = document.get("sha256")
        if not isinstance(observed, str):
            raise ValueError("ledger is missing its sha256 self-hash")
        payload = {key: value for key, value in document.items() if key != "sha256"}
        expected = hashlib.sha256(cls._canonical_bytes(payload)).hexdigest()
        if observed != expected:
            raise ValueError("ledger sha256 self-hash mismatch")
        cls._validate_payload(payload)

        ledger = cls()
        ledger._events = payload["events"]
        ledger._counters = payload["counters"]
        ledger._provenance = payload["provenance"]
        ledger._sha256 = observed
        return ledger

    @classmethod
    def serialize_ops(
        cls, ops: Sequence[SynapseBirth | SynapseDeath]
    ) -> list[dict[str, Any]]:
        """Expose the ledger's canonical op representation for replay checks."""
        return [cls._serialize_op(op) for op in ops]

    @classmethod
    def _serialize_op(cls, op: SynapseBirth | SynapseDeath) -> dict[str, Any]:
        if isinstance(op, SynapseBirth):
            return {
                "lineage": cls._serialize_tensor(op.lineage),
                "op": "SynapseBirth",
                "s": cls._serialize_tensor(op.s),
                "site": op.site,
                "t": cls._serialize_tensor(op.t),
                "w": cls._serialize_tensor(op.w),
            }
        if isinstance(op, SynapseDeath):
            return {
                "ids": cls._serialize_tensor(op.ids),
                "op": "SynapseDeath",
                "site": op.site,
            }
        raise TypeError(f"unsupported ledger op type {type(op)!r}")

    @classmethod
    def _serialize_tensor(cls, tensor: Tensor) -> dict[str, Any]:
        if not isinstance(tensor, Tensor):
            raise TypeError("ledger tensor fields must be Tensors")
        value = tensor.detach().to(device="cpu")
        data = value.tolist()
        result: dict[str, Any] = {
            "data": cls._json_tensor_data(data),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        if value.is_complex():
            result["encoding"] = "complex-pairs"
        return result

    @classmethod
    def _json_tensor_data(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._json_tensor_data(item) for item in value]
        if isinstance(value, complex):
            return [value.real, value.imag]
        return value

    @classmethod
    def _validate_payload(cls, payload: dict[str, Any]) -> None:
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported ledger schema")
        if not isinstance(payload.get("events"), list):
            raise ValueError("ledger events must be a list")
        if not isinstance(payload.get("counters"), dict):
            raise ValueError("ledger counters must be an object")
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict) or not all(
            isinstance(path, str) and isinstance(value, str)
            for path, value in provenance.items()
        ):
            raise ValueError("ledger provenance must map paths to hashes")

    @staticmethod
    def _canonical_bytes(payload: Any) -> bytes:
        return (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def _json_copy(cls, value: Any) -> Any:
        return json.loads(cls._canonical_bytes(value).decode("utf-8"))
