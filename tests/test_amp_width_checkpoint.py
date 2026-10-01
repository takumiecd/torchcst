from __future__ import annotations

import copy

import pytest
import torch
from kernel_cases import direct_state, polar_state, triweight_state

from torchcst import BandwidthBounds, CSTLinear
from torchcst._backends.torch.charts import construction as _construction


def make_model(kernel_type, *, normalize_columns=True):
    return CSTLinear(
        _construction.linspace(4, spacing=1.0),
        _construction.linspace(3, spacing=1.0),
        atoms=2,
        kernel=kernel_type(
            amplitude_max=1.0,
            w_c=0.1,
            profile=triweight_state(0.5, normalize_columns=normalize_columns),
            input_bounds=BandwidthBounds(
                minimum=0.5, maximum=2.0, birth=2.0, upper_floor=0.5
            ),
        ).declaration(),
        backend="factored",
    )


def test_polar_and_direct_checkpoints_cannot_be_reinterpreted() -> None:
    polar = make_model(polar_state)
    direct = make_model(direct_state)
    assert direct.kernel.spec.parameterization.id == "direct_activity_width"
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        direct.load_state_dict(polar.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        polar.load_state_dict(direct.state_dict(), strict=True)


def test_checkpoint_rejects_different_profile_normalization() -> None:
    raw = make_model(polar_state, normalize_columns=False)
    normalized = make_model(polar_state)
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        normalized.load_state_dict(raw.state_dict(), strict=True)


@pytest.mark.parametrize("kernel_type", [polar_state, direct_state])
def test_tagged_checkpoint_roundtrip(kernel_type) -> None:
    source = make_model(kernel_type, normalize_columns=False)
    restored = make_model(kernel_type, normalize_columns=False)
    restored.load_state_dict(source.state_dict(), strict=True)
    torch.testing.assert_close(restored.dense_weight(), source.dense_weight())


def test_untagged_new_format_requires_explicit_identification() -> None:
    source = make_model(direct_state)
    state = copy.deepcopy(source.state_dict())
    del state["kernel._extra_state"]
    with pytest.raises(RuntimeError, match="Missing key"):
        make_model(direct_state).load_state_dict(state, strict=True)
