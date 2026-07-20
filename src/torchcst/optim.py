"""Amplitude/coordinate分離learning rateのためのparameter group分割。"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from .storage import SynapseStore


def parameter_groups(
    stores_or_model: nn.Module | Iterable[nn.Module],
    *,
    amplitude_lr: float,
    coordinate_lr: float,
    default_lr: float,
) -> list[dict[str, object]]:
    """SynapseStoreのw/座標/その他をcst repo踏襲の3群learning rateへ分ける。

    cstリポジトリの precedent (``cst.optim.parameter_groups``,
    ``amplitude_lr=1e-3``, ``coordinate_lr=2e-4``) に倣い、振幅 ``w`` と座標
    ``s``/``t`` へ異なるlearning rateを割り当てる。座標が buffer role
    (entry familyのIntegerGridなど) の場合は ``nn.Parameter`` ではないため
    optimizerの対象外であり、このgroup分割にも現れない。
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
