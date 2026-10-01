"""Configuration snapshots for legacy public façades.

Call at configuration boundaries, never from a forward, backward or CUDA
Graph capture. Scalar buffer reads may synchronize the device. This module
imports declarations only; execution code remains lazy. Custom implementations
must explicitly declare their own semantics instead of inheriting a built-in ID.
"""

from dataclasses import replace

from .normalization import NormalizationSpec
from .parameterizations import (
    AmpWidthSpec,
    BandwidthBounds,
    DirectAmpWidthSpec,
    FixedWidthSpec,
    PolarAmpWidthSpec,
)
from .profiles import (
    BiweightSpec,
    GaussianSpec,
    TriangleSpec,
    TriweightSpec,
    WendlandC2Spec,
)
from .spec import KernelSpec, ProfileBinding, StatePolicySpec


def profile_declaration(profile):
    from .compact import Biweight, Triangle, Triweight, WendlandC2
    from .gaussian import Gaussian

    shapes = {
        Gaussian: GaussianSpec,
        Triweight: TriweightSpec,
        Biweight: BiweightSpec,
        Triangle: TriangleSpec,
        WendlandC2: WendlandC2Spec,
    }
    shape = shapes.get(type(profile))
    if shape is None:
        raise NotImplementedError("custom profiles must implement declaration()")
    normalization = NormalizationSpec()
    if profile.normalize_columns:
        normalization = NormalizationSpec(
            kind="discrete_l2",
            domain="chart_sites",
            floor=None if type(profile) is Gaussian else 1e-6,
        )
    return ProfileBinding(
        profile=shape(),
        parameterization=FixedWidthSpec(sigma=float(profile.sigma.detach())),
        normalization=normalization,
    )


def _bounds(kernel, side):
    return BandwidthBounds(
        minimum=float(getattr(kernel, f"sigma_min_{side}").detach()),
        birth=float(getattr(kernel, f"sigma_birth_{side}").detach()),
        maximum=float(getattr(kernel, f"sigma_max_{side}").detach()),
        upper_floor=float(getattr(kernel, f"upper_floor_{side}").detach()),
    )


def kernel_declaration(kernel, *, chart_count):
    from .amplitude import Amplitude
    from .amplitude_bandwidth import AmpWidth
    from .direct_amplitude_bandwidth import DirectAmpWidth
    from .polar_amplitude_bandwidth import PolarAmpWidth
    from .separable import Separable

    if type(chart_count) is not int or chart_count not in (1, 2):
        raise ValueError("chart_count must be 1 or 2")
    if type(kernel) not in (
        Amplitude,
        Separable,
        AmpWidth,
        DirectAmpWidth,
        PolarAmpWidth,
    ):
        raise NotImplementedError("custom kernels must implement declaration()")
    if chart_count == 1 and type(kernel) is not DirectAmpWidth:
        raise ValueError("this kernel requires an input/output chart pair")
    if type(kernel) is Amplitude:
        return KernelSpec(
            composition="amplitude",
            inner=kernel.kernel.declaration(chart_count=chart_count),
            initialization=StatePolicySpec(id="signed_amplitude"),
            update=StatePolicySpec(id="amplitude_and_inner"),
        )
    if type(kernel) is Separable:
        return KernelSpec(
            composition="separable",
            profiles=(
                kernel.input_profile.declaration(),
                kernel.output_profile.declaration(),
            ),
        )
    binding = kernel.profile.declaration()
    if chart_count == 1 and binding.normalization.kind != "none":
        raise ValueError("single-chart DirectAmpWidth requires an unnormalized profile")
    if type(kernel) is AmpWidth:
        parameterization = AmpWidthSpec(
            sigma_min=float(kernel.sigma_min.detach()),
            sigma_max=float(kernel.sigma_max.detach()),
            tau=float(kernel.tau.detach()),
            temperature=float(kernel.temperature.detach()),
            gate_eps=float(kernel.gate_eps.detach()),
            law=kernel.law,
            couple_bandwidth=kernel.couple_bandwidth,
        )
        initialization = StatePolicySpec(id="signed_amplitude_centers")
        update = StatePolicySpec(id="amplitude_and_geometry")
    else:
        cls = (
            DirectAmpWidthSpec if type(kernel) is DirectAmpWidth else PolarAmpWidthSpec
        )
        parameterization = cls(
            amplitude_max=float(kernel.amplitude_max.detach()),
            input_bounds=_bounds(kernel, "input"),
            output_bounds=_bounds(kernel, "output"),
            w_c=float(kernel.w_c.detach()),
            kappa=float(kernel.kappa.detach()),
            lower_kappa=float(kernel.lower_kappa.detach()),
            upper_decay_power=float(kernel.upper_decay_power.detach()),
        )
        initialization = StatePolicySpec(
            id="direct_centers" if type(kernel) is DirectAmpWidth else "polar_centers",
            settings=(("alpha_init", float(kernel.alpha_init.detach())),),
        )
        settings = [
            ("radial_regularization", float(kernel.radial_regularization.detach()))
        ]
        if type(kernel) is PolarAmpWidth:
            settings += [
                ("activity_gain", float(kernel.activity_gain.detach())),
                ("activity_mode", kernel.activity_mode),
                (
                    "dormant_expansion_rate",
                    float(kernel.dormant_expansion_rate.detach()),
                ),
            ]
        update = StatePolicySpec(id=parameterization.id, settings=tuple(settings))
    return KernelSpec(
        composition="radial" if chart_count == 1 else "separable",
        profiles=(replace(binding, parameterization=None),) * chart_count,
        parameterization=parameterization,
        initialization=initialization,
        update=update,
    )
