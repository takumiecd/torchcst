"""Tensor-state fixtures; public models receive declaration snapshots."""

from dataclasses import replace

from torchcst import (
    BiweightSpec,
    GaussianSpec,
    TriangleSpec,
    TriweightSpec,
    WendlandC2Spec,
    presets,
)
from torchcst.kernels.state import KernelState, ProfileState


def gaussian_state(sigma, *, normalize_columns=True):
    return ProfileState(
        presets.fixed_profile(GaussianSpec(), sigma, normalize=normalize_columns)
    )


def biweight_state(sigma, *, normalize_columns=False):
    return ProfileState(
        presets.fixed_profile(BiweightSpec(), sigma, normalize=normalize_columns)
    )


def triangle_state(sigma, *, normalize_columns=False):
    return ProfileState(
        presets.fixed_profile(TriangleSpec(), sigma, normalize=normalize_columns)
    )


def triweight_state(sigma, *, normalize_columns=True):
    return ProfileState(
        presets.fixed_profile(TriweightSpec(), sigma, normalize=normalize_columns)
    )


def wendland_state(sigma, *, normalize_columns=True):
    return ProfileState(
        presets.fixed_profile(WendlandC2Spec(), sigma, normalize=normalize_columns)
    )


def separable_state(*, input_profile, output_profile):
    return KernelState(
        presets.separable(
            input_profile=input_profile.declaration(),
            output_profile=output_profile.declaration(),
        )
    )


def amplitude_state(kernel):
    return KernelState(presets.amplitude(kernel.declaration()))


def amp_width_state(*, sigma_min, sigma_max, profile=None, **settings):
    return KernelState(
        presets.amplitude_width(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            profile=replace(profile.declaration(), parameterization=None)
            if profile
            else None,
            **settings,
        )
    )


def direct_state(
    *,
    input_bounds,
    output_bounds=None,
    profile=None,
    composition="separable",
    options=None,
    **settings,
):
    return KernelState(
        presets.direct_activity(
            input_bounds=input_bounds,
            output_bounds=output_bounds,
            profile=replace(profile.declaration(), parameterization=None)
            if profile
            else None,
            composition=composition,
            **settings,
        ),
        options=options,
    )


def polar_state(*, input_bounds, output_bounds=None, profile=None, **settings):
    return KernelState(
        presets.polar_activity(
            input_bounds=input_bounds,
            output_bounds=output_bounds,
            profile=replace(profile.declaration(), parameterization=None)
            if profile
            else None,
            **settings,
        )
    )
