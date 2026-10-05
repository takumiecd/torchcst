"""Immutable site metadata adapter for execution."""

from functools import cache
from types import SimpleNamespace

from torchcst._backends.torch.algorithms.normalized_radial.layout import geometry


@cache
def _geometry(operator, device):
    operator = geometry(operator)
    # Only structural offsets are cached here. Norms, supports, row buckets and
    # gradients remain owned by each forward/backward invocation.
    return SimpleNamespace(
        sizes=operator.sizes, origin=operator.origin, spacing=operator.spacing
    )
