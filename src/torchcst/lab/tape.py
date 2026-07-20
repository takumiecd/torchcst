"""Immutable, repeatable minibatch tapes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator

import torch
from torch import Tensor

from .streams import RngStreams


class BatchTape:
    """A deterministic sequence of shuffled ``(x, y)`` batches.

    Epoch permutations are derived from the named data seed, rather than from a
    generator's mutable position.  Iterating a tape therefore never changes it.
    Inputs are snapshotted at construction so later caller mutation cannot alter
    either replay or the hash.
    """

    _EPOCH_DOMAIN = b"torchcst.BatchTape.epoch.v1\0"
    _HASH_SCHEMA = "torchcst-batch-tape-v1"

    def __init__(
        self,
        x: Tensor,
        y: Tensor,
        batch_size: int,
        root_seed: int | RngStreams = 0,
        *,
        epochs: int = 1,
        drop_last: bool = False,
        streams: RngStreams | None = None,
    ) -> None:
        if not isinstance(x, Tensor) or not isinstance(y, Tensor):
            raise TypeError("x and y must be Tensors")
        if x.ndim == 0 or y.ndim == 0:
            raise ValueError("x and y must have a batch dimension")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same leading dimension")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an int")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(epochs, bool) or not isinstance(epochs, int):
            raise TypeError("epochs must be an int")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a bool")

        if streams is not None:
            if not isinstance(streams, RngStreams):
                raise TypeError("streams must be an RngStreams instance")
            source = streams
        elif isinstance(root_seed, RngStreams):
            source = root_seed
        else:
            source = RngStreams(root_seed)

        self._x = x.detach().clone()
        self._y = y.detach().clone()
        self.batch_size = batch_size
        self.epochs = epochs
        self.drop_last = drop_last
        self.root_seed = source.root_seed
        self.data_seed = source.seed("data")
        self._orders = tuple(self._make_order(epoch) for epoch in range(epochs))
        self._tape_hash = self._compute_hash()

    @property
    def tape_hash(self) -> str:
        return self._tape_hash

    def __len__(self) -> int:
        count = self._x.shape[0]
        per_epoch = count // self.batch_size
        if not self.drop_last and count % self.batch_size:
            per_epoch += 1
        return self.epochs * per_epoch

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for epoch in range(self.epochs):
            yield from self.iter_epoch(epoch)

    def iter_epoch(self, epoch: int) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate one zero-based epoch without advancing any shared RNG."""
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("epoch must be an int")
        if not 0 <= epoch < self.epochs:
            raise IndexError("epoch out of range")
        order = self._orders[epoch]
        stop = order.numel()
        if self.drop_last:
            stop -= stop % self.batch_size
        for start in range(0, stop, self.batch_size):
            indices = order[start : start + self.batch_size]
            yield (
                self._x.index_select(0, indices.to(self._x.device)),
                self._y.index_select(0, indices.to(self._y.device)),
            )

    def _make_order(self, epoch: int) -> Tensor:
        payload = (
            self._EPOCH_DOMAIN
            + str(self.data_seed).encode("ascii")
            + b"\0"
            + str(epoch).encode("ascii")
        )
        epoch_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(epoch_seed)
        return torch.randperm(self._x.shape[0], generator=generator)

    def _compute_hash(self) -> str:
        digest = hashlib.sha256()
        convention = {
            "batch_size": self.batch_size,
            "data_seed": self.data_seed,
            "drop_last": self.drop_last,
            "epochs": self.epochs,
            "order": "sha256(data-stream-seed,epoch)-randperm-v1",
            "schema": self._HASH_SCHEMA,
        }
        self._update_frame(
            digest,
            json.dumps(
                convention,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        for batch_x, batch_y in self:
            self._update_tensor(digest, batch_x)
            self._update_tensor(digest, batch_y)
        return digest.hexdigest()

    @classmethod
    def _update_tensor(cls, digest: "hashlib._Hash", tensor: Tensor) -> None:
        value = tensor.detach().to(device="cpu").contiguous()
        metadata = json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        cls._update_frame(digest, metadata)
        raw = bytes(value.reshape(-1).view(torch.uint8).tolist())
        cls._update_frame(digest, raw)

    @staticmethod
    def _update_frame(digest: "hashlib._Hash", value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
