import copy

import pytest
import torch

from torchcst import Chart, CSTLinear, DirectAmpWidth, PolarAmpWidth, Triweight


def make_model(kernel_type, *, normalize_columns=True):
    return CSTLinear(
        Chart.linspace(4, spacing=1.0),
        Chart.linspace(3, spacing=1.0),
        atoms=2,
        kernel=kernel_type(
            amplitude_max=1.0,
            sigma_min=0.5,
            sigma_max=2.0,
            w_c=0.1,
            profile=Triweight(0.5, normalize_columns=normalize_columns),
        ),
        backend="factored",
    )


def test_polar_and_direct_checkpoints_cannot_be_reinterpreted() -> None:
    polar = make_model(PolarAmpWidth)
    direct = make_model(DirectAmpWidth)

    assert not isinstance(direct.kernel, PolarAmpWidth)

    with pytest.raises(RuntimeError, match="checkpoint contract"):
        direct.load_state_dict(polar.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        polar.load_state_dict(direct.state_dict(), strict=True)


def test_checkpoint_rejects_different_profile_normalization() -> None:
    raw = make_model(PolarAmpWidth, normalize_columns=False)
    normalized = make_model(PolarAmpWidth)

    with pytest.raises(RuntimeError, match="checkpoint contract"):
        normalized.load_state_dict(raw.state_dict(), strict=True)


@pytest.mark.parametrize("kernel_type", [PolarAmpWidth, DirectAmpWidth])
def test_tagged_checkpoint_roundtrip(kernel_type) -> None:
    source = make_model(kernel_type, normalize_columns=False)
    restored = make_model(kernel_type, normalize_columns=False)

    restored.load_state_dict(source.state_dict(), strict=True)

    torch.testing.assert_close(restored.dense_weight(), source.dense_weight())


def test_untagged_new_format_requires_explicit_identification() -> None:
    source = make_model(DirectAmpWidth)
    state = copy.deepcopy(source.state_dict())
    del state["kernel._extra_state"]

    with pytest.raises(RuntimeError, match="ambiguous atom coordinates"):
        make_model(DirectAmpWidth).load_state_dict(state, strict=True)


def test_direct_loads_its_earlier_tagged_checkpoint_without_unused_polar_buffers() -> None:
    source = make_model(DirectAmpWidth)
    older = copy.deepcopy(source.state_dict())
    older["kernel._extra_state"] = {
        **older["kernel._extra_state"],
        "format_version": 1,
        "activity_mode": None,
    }
    older["kernel.activity_gain"] = torch.tensor(1.0)
    older["kernel.dormant_expansion_rate"] = torch.tensor(0.0)

    restored = make_model(DirectAmpWidth)
    restored.load_state_dict(older, strict=True)

    torch.testing.assert_close(restored.dense_weight(), source.dense_weight())


def test_old_shared_bandwidth_polar_checkpoint_loads_without_changing_operator() -> None:
    source = make_model(PolarAmpWidth)
    with torch.no_grad():
        source.atoms.p[0, :2] = torch.tensor([1.2, 1.6])
    expected = source.dense_weight().detach().clone()
    legacy = copy.deepcopy(source.state_dict())
    del legacy["kernel._extra_state"]
    legacy["kernel.sigma_max"] = legacy.pop("kernel.sigma_max_input")
    for name in (
        "sigma_min_input",
        "sigma_min_output",
        "sigma_birth_input",
        "sigma_birth_output",
        "sigma_max_output",
        "lower_kappa",
        "upper_decay_power",
        "upper_floor_input",
        "upper_floor_output",
        "alpha_init",
        "dormant_expansion_rate",
    ):
        del legacy["kernel." + name]

    restored = make_model(PolarAmpWidth)
    restored.load_state_dict(legacy, strict=True)

    torch.testing.assert_close(restored.dense_weight(), expected)


def test_old_polar_checkpoint_rejects_raw_triweight_target() -> None:
    source = make_model(PolarAmpWidth)
    legacy = copy.deepcopy(source.state_dict())
    del legacy["kernel._extra_state"]
    legacy["kernel.sigma_max"] = legacy.pop("kernel.sigma_max_input")
    del legacy["kernel.sigma_max_output"]

    with pytest.raises(RuntimeError, match="ambiguous atom coordinates"):
        make_model(PolarAmpWidth, normalize_columns=False).load_state_dict(
            legacy, strict=True
        )
