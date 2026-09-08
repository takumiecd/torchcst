"""CUDA-event stage timing; no synchronization between measured stages.

Event intervals include stream idle time waiting for host dispatch. Nested
intervals must not be added to their parents. Host time measures enqueue work.
"""

import time
from contextlib import contextmanager
from functools import wraps

import torch


class StageTimer:
    def __init__(self):
        self.events = {}
        self.samples = {}

    @contextmanager
    def stage(self, name):
        if name not in self.events:
            self.events[name] = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        start, end = self.events[name]
        start.record()
        begin = time.perf_counter()
        try:
            yield
        finally:
            host_ms = (time.perf_counter() - begin) * 1000
            end.record()
            self.samples[name] = host_ms

    def wrap(self, obj, method, name):
        original = getattr(obj, method)

        @wraps(original)
        def measured(*args, **kwargs):
            with self.stage(name):
                return original(*args, **kwargs)

        setattr(obj, method, measured)

    def collect(self):
        # Caller already drained the step. No additional synchronization here.
        return {
            name: {
                "cuda_ms": self.events[name][0].elapsed_time(self.events[name][1]),
                "host_ms": host,
            }
            for name, host in self.samples.items()
        }

    def install(self, optimizer):
        from torchcst._derivatives.tangent import TangentGeometry
        from torchcst.optim.moments.second import SeparableDiagonalSecondMoment
        from torchcst.optim.moments.tangent import TangentFirstMoment

        self.wrap(optimizer, "zero_grad", "zero_grad")
        self.wrap(optimizer, "step", "optimizer_total")
        self.wrap(optimizer, "_solve", "update_solve")
        self.wrap(optimizer, "_complete_capture", "complete_observation")
        self.wrap(TangentGeometry, "pullback_from_frame", "transport")
        self.wrap(TangentGeometry, "compress", "recompression")
        self.wrap(TangentFirstMoment, "expand", "first_expand")
        self.wrap(SeparableDiagonalSecondMoment, "expand", "second_expand")
