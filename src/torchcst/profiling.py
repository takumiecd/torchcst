"""Opt-in operator ranges for profiling CST internals with PyTorch."""

from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

_enabled: ContextVar[bool] = ContextVar("torchcst_profiling_enabled", default=False)


def cst_span(name: str):
    """Record a CST range only while a CSTProfiler is active."""

    return record_function(name) if _enabled.get() else nullcontext()


class CSTProfiler:
    """PyTorch profiler with named CST forward, kernel, and optimizer ranges.

    Use it around an ordinary training step. The backward graph remains visible
    through PyTorch's autograd and operator events; callers may add their own
    phase ranges with ``torch.profiler.record_function``. No ranges are created
    outside this context.
    """

    def __init__(
        self,
        *,
        activities: list[ProfilerActivity] | None = None,
        **profiler_options: Any,
    ) -> None:
        if activities is None:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
        self.profiler = profile(activities=activities, **profiler_options)
        self._token = None

    def __enter__(self):
        self.profiler.__enter__()
        self._token = _enabled.set(True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        if self._token is not None:
            _enabled.reset(self._token)
            self._token = None
        return self.profiler.__exit__(exc_type, exc_value, traceback)

    def key_averages(self, **kwargs: Any):
        return self.profiler.key_averages(**kwargs)

    def export_chrome_trace(self, path: str) -> None:
        self.profiler.export_chrome_trace(path)
