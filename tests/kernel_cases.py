"""Tensor-state fixtures; public models receive declaration snapshots."""

from dataclasses import replace

from torchcst import (
    BiweightSpec,
    GaussianSpec,
    TriangleSpec,
    TriweightSpec,
    WendlandC2Spec,
    kernels,
)
from torchcst.kernels.state import KernelState, ProfileState


def gaussian_state(sigma, *, normalize_columns=True):
    if type(normalize_columns) is not bool:
        raise TypeError("normalize_columns must be a bool")
    return ProfileState(
        (kernels.l2 if normalize_columns else kernels.point)(GaussianSpec(), sigma)
    )


def biweight_state(sigma, *, normalize_columns=False):
    if type(normalize_columns) is not bool:
        raise TypeError("normalize_columns must be a bool")
    return ProfileState(
        (kernels.l2 if normalize_columns else kernels.point)(BiweightSpec(), sigma)
    )


def triangle_state(sigma, *, normalize_columns=False):
    if type(normalize_columns) is not bool:
        raise TypeError("normalize_columns must be a bool")
    return ProfileState(
        (kernels.l2 if normalize_columns else kernels.point)(TriangleSpec(), sigma)
    )


def triweight_state(sigma, *, normalize_columns=True):
    if type(normalize_columns) is not bool:
        raise TypeError("normalize_columns must be a bool")
    return ProfileState(
        (kernels.l2 if normalize_columns else kernels.point)(TriweightSpec(), sigma)
    )


def wendland_state(sigma, *, normalize_columns=True):
    if type(normalize_columns) is not bool:
        raise TypeError("normalize_columns must be a bool")
    return ProfileState(
        (kernels.l2 if normalize_columns else kernels.point)(WendlandC2Spec(), sigma)
    )


def separable_state(*, output_profile, input_profile):
    return KernelState(
        kernels.product(
            output_factor=output_profile.declaration(),
            input_factor=input_profile.declaration(),
        )
    )


def amplitude_state(kernel):
    return KernelState(kernels.amplitude(kernel.declaration()))


def amp_width_state(*, sigma_min, sigma_max, profile=None, **settings):
    return KernelState(
        kernels.amplitude_width(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            factor=replace(profile.declaration(), parameterization=None)
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
        (kernels.product if composition == "separable" else kernels.radial)(
            replace(profile.declaration(), parameterization=None)
            if profile
            else (
                kernels.l2(GaussianSpec())
                if composition == "separable"
                else kernels.point(GaussianSpec())
            ),
            coordinates=kernels.direct(
                input_bounds=input_bounds, output_bounds=output_bounds, **settings
            ),
        ),
        options=options,
    )


def polar_state(*, input_bounds, output_bounds=None, profile=None, **settings):
    return KernelState(
        kernels.product(
            replace(profile.declaration(), parameterization=None)
            if profile
            else kernels.l2(GaussianSpec()),
            coordinates=kernels.polar(
                input_bounds=input_bounds, output_bounds=output_bounds, **settings
            ),
        )
    )
