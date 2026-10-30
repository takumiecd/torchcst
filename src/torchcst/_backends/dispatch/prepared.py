"""Bounded, binding-owned fixed-Plan preparation metadata."""

from dataclasses import dataclass


def placement_signature(binding):
    """Track storage/ownership, excluding ordinary atom numerical updates."""
    from torchcst._backends.state import tensor_signature

    owner = getattr(binding, "atom_state", None)
    return (
        id(owner) if owner is not None else None,
        None if owner is None else owner.layout_version,
        tensor_signature(binding.execution_parameters()),
        # build_context just refreshed this live configuration signature; borrow
        # it without a second module/buffer traversal. Atom values are excluded.
        binding.__dict__.get("_execution_declaration", (None,))[0],
    )


@dataclass
class PreparedDispatch:
    registry: object
    routing: object
    context: object
    placement: object
    decision: object
    algorithm: object
    state: object
    recipe: object
    state_recipe: object


# Optional prepared execution may specialize only these original definitions.
from torchcst._builtin_callbacks import seal_functions as _seal_functions

_seal_functions(globals())
