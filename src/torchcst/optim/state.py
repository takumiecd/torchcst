"""Interpret vector state without coupling CST updates to an optimizer family."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import SGD, Adam, Adamax, AdamW, RMSprop


@dataclass(frozen=True)
class OptimizerStateAdapter:
    """State entries to transport as vectors after a coordinate retraction.

    Other entries (including coordinatewise squared accumulators and clocks)
    stay in the optimizer's stored coordinate basis. This is coordinate
    optimization with vector transport, not intrinsic Riemannian Adam.
    Custom optimizers can explicitly declare their vector entries here.
    """

    vector_keys: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.vector_keys, tuple) or any(
            not isinstance(key, str) for key in self.vector_keys
        ):
            raise TypeError("vector_keys must be a tuple of state key strings")
        if len(set(self.vector_keys)) != len(self.vector_keys):
            raise ValueError("vector_keys must be unique")

    def transport(self, site, old: Tensor, new: Tensor, state: dict) -> None:
        for key in self.vector_keys:
            vector = state.get(key)
            if vector is None:
                continue
            if not isinstance(vector, Tensor) or vector.shape != old.shape:
                raise ValueError(f"optimizer vector state {key!r} has the wrong shape")
            transported = site.kernel.transport_parameter_state(
                *site.cst_charts(), old, new, vector
            )
            if transported.shape != vector.shape:
                raise ValueError("transported optimizer state has the wrong shape")
            if not bool(torch.isfinite(transported).all()):
                raise FloatingPointError("non-finite transported optimizer state")
            vector.copy_(transported)


def default_state_adapter(optimizer) -> OptimizerStateAdapter:
    # Exact types: a subclass may change the meaning of the same state keys.
    adapters = {
        SGD: ("momentum_buffer",),
        Adam: ("exp_avg",),
        AdamW: ("exp_avg",),
        Adamax: ("exp_avg",),
        RMSprop: ("momentum_buffer", "grad_avg"),
    }
    if type(optimizer) in adapters:
        return OptimizerStateAdapter(adapters[type(optimizer)])
    from torch.optim import ASGD, Adadelta, Adagrad, Rprop

    if type(optimizer) in (Adadelta, Adagrad):
        return OptimizerStateAdapter()
    if type(optimizer) is Rprop:
        return OptimizerStateAdapter(("prev",))
    if type(optimizer) is ASGD:
        raise ValueError("ASGD averaged parameter state needs a dedicated adapter")
    raise ValueError("custom optimizer requires an explicit OptimizerStateAdapter")
