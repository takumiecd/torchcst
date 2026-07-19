"""Policy.prepare()にだけ渡すtyped binding surface。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar

import torch

from ..backward import CapturePoint
from ..storage.base import EntityStore, View


ViewT = TypeVar("ViewT", bound=View)
RecordT = TypeVar("RecordT")


class ReadPort(Generic[ViewT]):
    """Store mutation能力を公開せず、現在Viewだけを返すbind済みhandle。"""

    def __init__(self, store: EntityStore, view_type: type[ViewT]):
        self.site = store.site
        self._store = store
        self._view_type = view_type

    def view(self) -> ViewT:
        with torch.no_grad():
            view = self._store.view()
        if not isinstance(view, self._view_type):
            raise TypeError(
                f"site {self.site!r} returned {type(view).__name__}; "
                f"expected {self._view_type.__name__}"
            )
        return view


class PolicyBinding:
    """文字列siteをtyped handleへ解決するprepare-time専用registry。"""

    def __init__(
        self,
        stores: dict[str, EntityStore],
        captures: Iterable[CapturePoint],
    ):
        self._stores = dict(stores)
        self._captures = tuple(captures)

    def read(self, site: str, view_type: type[ViewT]) -> ReadPort[ViewT]:
        try:
            store = self._stores[site]
        except KeyError as exc:
            raise ValueError(f"unknown policy site {site!r}") from exc

        port = ReadPort(store, view_type)
        # prepare時に型不一致を即座に検出する。
        port.view()
        return port

    def capture(
        self, site: str, record_type: type[RecordT]
    ) -> CapturePoint[RecordT]:
        matches = [
            capture
            for capture in self._captures
            if capture.site == site and capture.record_type is record_type
        ]
        if not matches:
            raise ValueError(
                f"site {site!r} does not provide {record_type.__name__} capture"
            )
        if len(matches) != 1:
            raise ValueError(
                f"site {site!r} provides {len(matches)} "
                f"{record_type.__name__} capture points; binding is ambiguous"
            )
        return matches[0]
