"""Policyがprepare時に購読するtyped capture point。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Generic, TypeVar

from torch.utils.hooks import RemovableHandle


RecordT = TypeVar("RecordT")


class CapturePoint(Generic[RecordT]):
    """PyTorch hookで生成されたrecordを購読者へ同期配信する薄いfacade。"""

    def __init__(self, site: str, record_type: type[RecordT]):
        self.site = site
        self.record_type = record_type
        self._subscribers: OrderedDict[int, Callable[[RecordT], None]] = OrderedDict()
        self._context: BackwardContext | None = None

    @property
    def active(self) -> bool:
        return bool(self._subscribers)

    def subscribe(self, observer: Callable[[RecordT], None]) -> RemovableHandle:
        handle = RemovableHandle(self._subscribers)
        self._subscribers[handle.id] = observer
        return handle

    def emit(self, record: RecordT) -> None:
        if not isinstance(record, self.record_type):
            raise TypeError(
                f"capture {self.site!r} expects {self.record_type.__name__}, "
                f"got {type(record).__name__}"
            )
        if self._context is not None:
            self._context._queue(self, record)
            return
        self._dispatch(record)

    def _dispatch(self, record: RecordT) -> None:
        for observer in tuple(self._subscribers.values()):
            observer(record)


class BackwardContext:
    """Backward中のcaptureを蓄積し、終了後にPolicy observerへ配信する。"""

    def __init__(self, captures: tuple[CapturePoint, ...]):
        self._captures = captures
        self._records: list[tuple[CapturePoint, object]] = []

    def __enter__(self) -> "BackwardContext":
        for capture in self._captures:
            if capture._context is not None:
                raise RuntimeError("nested BackwardContext is not supported")
            capture._context = self
        return self

    def _queue(self, capture: CapturePoint, record: object) -> None:
        self._records.append((capture, record))

    def finalize(self) -> None:
        for capture in self._captures:
            capture._context = None
        records = self._records
        self._records = []
        for capture, record in records:
            capture._dispatch(record)

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finalize()
        else:
            for capture in self._captures:
                capture._context = None
            self._records.clear()
