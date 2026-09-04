"""Transient represented-gradient accumulation."""

from __future__ import annotations

from torch import Tensor


class RepresentedGradientAccumulator:
    """Accumulate detached cotangents of one represented operator.

    This object deliberately is not an ``nn.Module`` and owns no registered
    tensor. Its value is step-local observation state, not model state.
    """

    def __init__(self, visible_shape: tuple[int, ...]) -> None:
        if not visible_shape or any(size < 1 for size in visible_shape):
            raise ValueError("visible_shape must contain positive dimensions")
        self._visible_shape = visible_shape
        self._enabled = False
        self._generation = 0
        self._value: Tensor | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def generation(self) -> int:
        """Identify the current capture scope for pending autograd hooks."""

        return self._generation

    def enable(self, *, clear: bool = True) -> None:
        self._generation += 1
        if clear:
            self.clear()
        self._enabled = True

    def disable(self) -> None:
        self._generation += 1
        self._enabled = False

    def clear(self) -> None:
        self._value = None

    def add(self, contribution: Tensor, *, generation: int) -> None:
        """Add one backward contribution when capture is enabled."""

        if not self._enabled or generation != self._generation:
            return
        if contribution.shape != self._visible_shape:
            raise ValueError(
                "represented-gradient contribution must have shape "
                f"{list(self._visible_shape)}"
            )
        contribution = contribution.detach()
        if self._value is None:
            self._value = contribution.clone()
            return
        if (
            contribution.device != self._value.device
            or contribution.dtype != self._value.dtype
        ):
            raise ValueError(
                "represented-gradient contributions must share one device and dtype"
            )
        self._value.add_(contribution)

    def value(self) -> Tensor:
        """Return an isolated snapshot of the accumulated cotangent."""

        if self._value is None:
            raise RuntimeError("no represented gradient has been captured")
        return self._value.clone()
