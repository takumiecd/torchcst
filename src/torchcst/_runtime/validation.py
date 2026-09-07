"""Scoped, deferred device checks; the optimizer owns the commit decision."""

from contextlib import contextmanager
from contextvars import ContextVar

_CHECKS = ContextVar("torchcst_device_checks", default=None)


def deferred():
    return _CHECKS.get() is not None


@contextmanager
def device_checks():
    checks = []
    token = _CHECKS.set(checks)
    try:
        yield checks
    finally:
        _CHECKS.reset(token)


def require(condition, message, error=ValueError):
    checks = _CHECKS.get()
    if checks is None:
        if not condition:
            raise error(message)
    else:
        checks.append(condition)
