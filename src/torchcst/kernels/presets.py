"""Convenient compositions of pure mathematical declarations."""

from dataclasses import replace

from .normalization import NormalizationSpec
from .parameterizations import (
    AmpWidthSpec,
    BandwidthBounds,
    DirectAmpWidthSpec,
    FixedWidthSpec,
    LogWidthSpec,
    PolarAmpWidthSpec,
)
from .profiles import (
    GaussianSpec,
    ProfileSpec,
    TriweightSpec,
)
from .spec import KernelSpec, ProfileBinding, StatePolicySpec


def _profile_binding(shape: ProfileSpec, *, normalize: bool = True) -> ProfileBinding:
    """Bind shape and normalization; a kernel parameterization supplies width."""
    if type(normalize) is not bool:
        raise TypeError("normalize must be a bool")
    return ProfileBinding(
        profile=shape,
        normalization=NormalizationSpec(
            kind="discrete_l2",
            domain="chart_sites",
            floor=None if isinstance(shape, GaussianSpec) else 1e-6,
        )
        if normalize
        else NormalizationSpec(),
    )


def profile(shape: ProfileSpec, *, normalize: bool = True) -> ProfileBinding:
    return _profile_binding(shape, normalize=normalize)


def fixed_profile(
    shape: ProfileSpec, sigma: float, *, normalize: bool = True
) -> ProfileBinding:
    """Add a fixed width to a profile with center-only coordinates."""
    return replace(
        _profile_binding(shape, normalize=normalize),
        parameterization=FixedWidthSpec(sigma=sigma),
    )


def separable(*, input_profile, output_profile):
    return KernelSpec(composition="separable", profiles=(input_profile, output_profile))


def radial(profile):
    return KernelSpec(composition="radial", profiles=(profile,))


def amplitude(kernel):
    return KernelSpec(
        composition="amplitude",
        inner=kernel,
        initialization=StatePolicySpec(id="signed_amplitude"),
        update=StatePolicySpec(id="amplitude_and_inner"),
    )


def amplitude_width(
    *,
    sigma_min,
    sigma_max,
    profile=None,
    law="interpolating",
    couple_bandwidth=True,
    tau=5e-3,
    temperature=0.25,
    gate_eps=1e-12,
):
    binding = profile or _profile_binding(GaussianSpec())
    return KernelSpec(
        composition="separable",
        profiles=(binding,) * 2,
        parameterization=AmpWidthSpec(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            tau=tau,
            temperature=temperature,
            gate_eps=gate_eps,
            law=law,
            couple_bandwidth=couple_bandwidth,
        ),
        initialization=StatePolicySpec(id="signed_amplitude_centers"),
        update=StatePolicySpec(id="amplitude_and_geometry"),
    )


def _activity(
    *,
    coordinate,
    composition,
    amplitude_max,
    input_bounds,
    output_bounds,
    w_c,
    kappa,
    lower_kappa,
    upper_decay_power,
    alpha_init,
    radial_regularization,
    profile,
    activity_gain=1.0,
    activity_mode="finite_chord",
    dormant_expansion_rate=0.0,
):
    if type(alpha_init) not in (float, int) or not 0 <= alpha_init <= 1:
        raise ValueError("alpha_init must be in [0, 1]")
    if type(radial_regularization) not in (float, int) or radial_regularization < 0:
        raise ValueError("radial_regularization must be nonnegative")
    if kappa <= 1 or upper_decay_power > 1:
        raise ValueError("kappa must exceed 1 and upper_decay_power must not exceed 1")
    if composition not in ("radial", "separable") or (
        coordinate == "polar" and composition != "separable"
    ):
        raise ValueError("unsupported activity kernel composition")
    if composition == "radial" and input_bounds != output_bounds:
        raise ValueError("radial kernel requires one set of bandwidth bounds")
    binding = profile or _profile_binding(
        GaussianSpec(), normalize=composition == "separable"
    )
    if composition == "radial" and binding.normalization.kind != "none":
        raise ValueError("radial activity kernel requires an unnormalized profile")
    cls = DirectAmpWidthSpec if coordinate == "direct" else PolarAmpWidthSpec
    parameterization = cls(
        amplitude_max=amplitude_max,
        input_bounds=input_bounds,
        output_bounds=output_bounds,
        w_c=w_c,
        kappa=kappa,
        lower_kappa=kappa if lower_kappa is None else lower_kappa,
        upper_decay_power=upper_decay_power,
    )
    updates = [("radial_regularization", radial_regularization)]
    if coordinate == "polar":
        if (
            activity_gain <= 0
            or dormant_expansion_rate < 0
            or activity_mode not in ("finite_chord", "time_energy")
        ):
            raise ValueError("invalid polar activity update settings")
        updates.extend(
            [
                ("activity_gain", activity_gain),
                ("activity_mode", activity_mode),
                ("dormant_expansion_rate", dormant_expansion_rate),
            ]
        )
    return KernelSpec(
        composition=composition,
        profiles=(binding,) * (1 if composition == "radial" else 2),
        parameterization=parameterization,
        initialization=StatePolicySpec(
            id=coordinate + "_centers", settings=(("alpha_init", alpha_init),)
        ),
        update=StatePolicySpec(id=parameterization.id, settings=tuple(updates)),
    )


def direct_activity(
    *,
    amplitude_max,
    input_bounds: BandwidthBounds,
    output_bounds: BandwidthBounds | None = None,
    w_c,
    profile=None,
    composition="radial",
    kappa=3.0,
    lower_kappa=None,
    upper_decay_power=1.0,
    alpha_init=0.0,
    radial_regularization=0.1,
):
    return _activity(
        coordinate="direct",
        composition=composition,
        amplitude_max=amplitude_max,
        input_bounds=input_bounds,
        output_bounds=output_bounds or input_bounds,
        w_c=w_c,
        kappa=kappa,
        lower_kappa=lower_kappa,
        upper_decay_power=upper_decay_power,
        alpha_init=alpha_init,
        radial_regularization=radial_regularization,
        profile=profile,
    )


def polar_activity(
    *,
    amplitude_max,
    input_bounds: BandwidthBounds,
    output_bounds: BandwidthBounds | None = None,
    w_c,
    profile=None,
    kappa=3.0,
    lower_kappa=None,
    upper_decay_power=1.0,
    alpha_init=0.0,
    radial_regularization=0.1,
    activity_gain=1.0,
    activity_mode="finite_chord",
    dormant_expansion_rate=0.0,
):
    return _activity(
        coordinate="polar",
        composition="separable",
        amplitude_max=amplitude_max,
        input_bounds=input_bounds,
        output_bounds=output_bounds or input_bounds,
        w_c=w_c,
        kappa=kappa,
        lower_kappa=lower_kappa,
        upper_decay_power=upper_decay_power,
        alpha_init=alpha_init,
        radial_regularization=radial_regularization,
        profile=profile,
        activity_gain=activity_gain,
        activity_mode=activity_mode,
        dormant_expansion_rate=dormant_expansion_rate,
    )


NORMALIZED_RADIAL_TRIWEIGHT = KernelSpec(
    composition="radial",
    profiles=(
        ProfileBinding(
            profile=TriweightSpec(),
            normalization=NormalizationSpec(
                kind="discrete_l2", domain="operator_sites", floor=1e-6
            ),
        ),
    ),
    parameterization=LogWidthSpec(sigma_min=0.03, sigma_max=3.25),
    initialization=StatePolicySpec(id="provided_atoms"),
    update=StatePolicySpec(id="euclidean"),
)
