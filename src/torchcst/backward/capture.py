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
        for observer in tuple(self._subscribers.values()):
            observer(record)
