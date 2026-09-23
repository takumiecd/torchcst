import copy

import pytest
import torch

from torchcst import (
    Amplitude,
    AmpWidth,
    Chart,
    CSTConv2d,
    CSTLinear,
    Gaussian,
    Separable,
    Triweight,
)


def _spherical_linear(input_representation: str, output_representation: str):
    return CSTLinear(
        Chart.sphere(5, intrinsic_dim=1, representation=input_representation),
        Chart.sphere(4, intrinsic_dim=1, representation=output_representation),
        atoms=2,
        kernel=Amplitude(
            Separable(
                input_profile=Gaussian(0.4),
                output_profile=Gaussian(0.5),
            )
        ),
        backend="factored",
    )


def _bandwidth_linear(profile, *, law="interpolating", couple_bandwidth=True):
    return CSTLinear(
        Chart.linspace(5, spacing=0.5),
        Chart.linspace(4, spacing=0.5),
        atoms=2,
        kernel=AmpWidth(
            sigma_min=1.0,
            sigma_max=2.0,
            law=law,
            couple_bandwidth=couple_bandwidth,
            profile=profile,
        ),
        backend="factored",
    )


def test_swapped_sphere_center_representations_reject_same_shape_checkpoint():
    source = _spherical_linear("ambient", "intrinsic")
    target = _spherical_linear("intrinsic", "ambient")
    assert source.atoms.p.shape == target.atoms.p.shape

    with pytest.raises(RuntimeError, match="geometry checkpoint contract"):
        target.load_state_dict(source.state_dict(), strict=True)


def test_profile_type_and_normalization_are_checkpoint_contracts():
    raw = _bandwidth_linear(Triweight(1.0, normalize_columns=False))
    normalized = _bandwidth_linear(Triweight(1.0))
    gaussian = _bandwidth_linear(Gaussian(1.0))

    with pytest.raises(RuntimeError, match="profile checkpoint contract"):
        normalized.load_state_dict(raw.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="profile checkpoint contract"):
        gaussian.load_state_dict(raw.state_dict(), strict=True)


@pytest.mark.parametrize(
    "settings",
    [{"law": "inverse"}, {"couple_bandwidth": False}],
)
def test_ampwidth_law_and_coupling_are_checkpoint_contracts(settings):
    source = _bandwidth_linear(Gaussian(1.0))
    target = _bandwidth_linear(Gaussian(1.0), **settings)

    with pytest.raises(RuntimeError, match="AmpWidth checkpoint contract"):
        target.load_state_dict(source.state_dict(), strict=True)


def test_conv_layout_is_checkpoint_contract():
    def model(stride):
        return CSTConv2d(
            Chart.linspace(4, spacing=1.0),
            Chart.linspace(2, spacing=1.0),
            in_channels=1,
            out_channels=2,
            kernel_size=2,
            stride=stride,
            atoms=1,
            kernel=Amplitude(
                Separable(
                    input_profile=Gaussian(0.4),
                    output_profile=Gaussian(0.5),
                )
            ),
            backend="factored",
        )

    with pytest.raises(RuntimeError, match="CST site checkpoint contract"):
        model(2).load_state_dict(model(1).state_dict(), strict=True)


def test_untagged_profile_checkpoint_is_rejected():
    source = _bandwidth_linear(Triweight(1.0))
    state = copy.deepcopy(source.state_dict())
    del state["kernel.profile._extra_state"]

    with pytest.raises(RuntimeError, match="Missing key"):
        _bandwidth_linear(Triweight(1.0)).load_state_dict(state, strict=True)


def test_matching_contract_roundtrip_keeps_operator():
    torch.manual_seed(123)
    source = _bandwidth_linear(Triweight(1.0, normalize_columns=False))
    target = _bandwidth_linear(Triweight(1.0, normalize_columns=False))
    target.load_state_dict(source.state_dict(), strict=True)

    torch.testing.assert_close(target.dense_weight(), source.dense_weight())
