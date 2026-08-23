"""Parameter groups: which knobs share a rate, and which must not."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from ..storage import SynapseStore


def parameter_groups(
    stores_or_model: nn.Module | Iterable[nn.Module],
    *,
    amplitude_lr: float,
    coordinate_lr: float,
    default_lr: float,
) -> list[dict[str, object]]:
    """Split parameters into amplitude / coordinate / default learning-rate groups.

    Every :class:`SynapseStore` amplitude ``w`` goes into the amplitude group
    and every learnable coordinate ``s``/``t`` into the coordinate group;
    remaining parameters take ``default_lr``. Coordinates with a buffer role
    (e.g. the entry family's ``IntegerGrid``) are not parameters and never
    appear in any group.

    A kernel family's declared per-atom columns
    (:class:`~torchcst.representation.AtomColumn`) land in the default group
    deliberately, not by omission: a Gabor frequency is conjugate to position
    and a per-atom bandwidth is a scale, so neither carries the amplitude's
    nor the coordinate's units and neither may silently inherit their rate.
    A family that wants its own schedule asks for its own group explicitly.
    """
    modules: list[nn.Module] = (
        [stores_or_model]
        if isinstance(stores_or_model, nn.Module)
        else list(stores_or_model)
    )
    if not modules:
        raise ValueError("stores_or_model must contain at least one module")

    amplitude_ids: set[int] = set()
    coordinate_ids: set[int] = set()
    for root in modules:
        for submodule in root.modules():
            if not isinstance(submodule, SynapseStore):
                continue
            weight = submodule.w
            if isinstance(weight, nn.Parameter):
                amplitude_ids.add(id(weight))
            for name in ("s", "t"):
                coordinate = getattr(submodule, name)
                if isinstance(coordinate, nn.Parameter):
                    coordinate_ids.add(id(coordinate))

    amplitude_params: list[nn.Parameter] = []
    coordinate_params: list[nn.Parameter] = []
    default_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for root in modules:
        for parameter in root.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if id(parameter) in amplitude_ids:
                amplitude_params.append(parameter)
            elif id(parameter) in coordinate_ids:
                coordinate_params.append(parameter)
            else:
                default_params.append(parameter)

    groups: list[dict[str, object]] = []
    if amplitude_params:
        groups.append({"params": amplitude_params, "lr": amplitude_lr})
    if coordinate_params:
        groups.append({"params": coordinate_params, "lr": coordinate_lr})
    if default_params:
        groups.append({"params": default_params, "lr": default_lr})
    return groups
