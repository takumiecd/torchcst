"""Independent, named random-number streams."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch


class RngStreams:
    """Own deterministic generators whose seeds depend only on name and root.

    Streams must be registered before use.  This deliberately makes a misspelled
    stream name fail instead of silently coupling two kinds of random work.
    """

    DEFAULT_STREAMS = ("data", "init", "proposal", "diag")
    _SEED_DOMAIN = b"torchcst.RngStreams.seed.v1\0"

    def __init__(self, root_seed: int) -> None:
        self._root_seed = self._validate_root_seed(root_seed)
        self._generators: dict[str, torch.Generator] = {}
        for name in self.DEFAULT_STREAMS:
            self.register(name)

    @property
    def root_seed(self) -> int:
        return self._root_seed

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._generators))

    def seed(self, name: str) -> int:
        """Return the stable seed assigned to a registered stream."""
        self._require_registered(name)
        payload = (
            self._SEED_DOMAIN
            + str(self._root_seed).encode("ascii")
            + b"\0"
            + name.encode("utf-8")
        )
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def register(self, name: str) -> torch.Generator:
        """Explicitly add a stream, returning the existing one when registered."""
        self._validate_name(name)
        existing = self._generators.get(name)
        if existing is not None:
            return existing
        generator = torch.Generator(device="cpu")
        self._generators[name] = generator
        generator.manual_seed(self.seed(name))
        return generator

    def get(self, name: str) -> torch.Generator:
        """Return one stable generator instance; unknown names are errors."""
        self._require_registered(name)
        return self._generators[name]

    def state_dict(self) -> dict[str, Any]:
        """Snapshot every registered generator without sharing state tensors."""
        return {
            "schema": "torchcst-rng-streams-v1",
            "root_seed": self._root_seed,
            "streams": {
                name: generator.get_state().clone()
                for name, generator in sorted(self._generators.items())
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the root, registry, and all generator states from a snapshot."""
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if state.get("schema") != "torchcst-rng-streams-v1":
            raise ValueError("unsupported RNG stream state schema")
        root_seed = self._validate_root_seed(state.get("root_seed"))
        raw_streams = state.get("streams")
        if not isinstance(raw_streams, Mapping):
            raise TypeError("state['streams'] must be a mapping")
        missing = set(self.DEFAULT_STREAMS).difference(raw_streams)
        if missing:
            raise ValueError(
                f"RNG stream state is missing defaults: {sorted(missing)}"
            )

        names = tuple(raw_streams)
        validated_states: dict[str, torch.Tensor] = {}
        for name in names:
            self._validate_name(name)
            value = raw_streams[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"state for stream {name!r} must be a Tensor")
            cloned = value.detach().cpu().clone()
            probe = torch.Generator(device="cpu")
            try:
                probe.set_state(cloned)
            except RuntimeError as exc:
                raise ValueError(f"invalid state for RNG stream {name!r}") from exc
            validated_states[name] = cloned

        self._root_seed = root_seed
        for name in tuple(self._generators):
            if name not in raw_streams:
                del self._generators[name]
        for name in names:
            generator = self._generators.get(name)
            if generator is None:
                generator = torch.Generator(device="cpu")
                self._generators[name] = generator
            generator.set_state(validated_states[name])

    def _require_registered(self, name: str) -> None:
        self._validate_name(name)
        if name not in self._generators:
            raise KeyError(
                f"unknown RNG stream {name!r}; call register(name) explicitly"
            )

    @staticmethod
    def _validate_root_seed(root_seed: Any) -> int:
        if isinstance(root_seed, bool) or not isinstance(root_seed, int):
            raise TypeError("root_seed must be an int")
        return root_seed

    @staticmethod
    def _validate_name(name: Any) -> None:
        if not isinstance(name, str):
            raise TypeError("stream name must be a str")
        if not name:
            raise ValueError("stream name must not be empty")
