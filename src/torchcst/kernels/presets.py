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


def cell_mass(
    *,
    input_min_tile_sites,
    output_min_tile_sites,
    profile=None,
    sigma_init=(0.2, 40.0),
    amplitude_std=1.0,
    sigma_floor=1e-3,
):
    """Cell-integrated periodic factors with live physical sigma.

    Use CSTOptimizer to project proposed sigma into [sigma_floor, min(Q)/3].
    Plain torch optimizers may be used when the caller maintains that domain.
    No clipping or coordinate conversion occurs in the forward function.
    """
    import math

    from .parameterizations import PhysicalAmpWidthSpec
    from .profiles.fourier import ChebyshevSquareSpec
    from .spec import CellIntegratedProfileBinding

    if (
        type(sigma_init) is not tuple
        or len(sigma_init) != 2
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in sigma_init)
        or not 0 < sigma_floor <= sigma_init[0] <= sigma_init[1]
    ):
        raise ValueError(
            "sigma_init must contain ordered positive endpoints above sigma_floor"
        )
    if (
        type(amplitude_std) not in (int, float)
        or not math.isfinite(amplitude_std)
        or amplitude_std < 0
        or type(sigma_floor) not in (int, float)
        or not math.isfinite(sigma_floor)
    ):
        raise ValueError("amplitude_std and sigma_floor must be finite scalars")
    profile = (
        ChebyshevSquareSpec(modes=8, tail_bound=1e-3) if profile is None else profile
    )
    return KernelSpec(
        composition="separable",
        profiles=tuple(
            CellIntegratedProfileBinding(profile=profile, min_tile_sites=n)
            for n in (input_min_tile_sites, output_min_tile_sites)
        ),
        parameterization=PhysicalAmpWidthSpec(),
        initialization=StatePolicySpec(
            id="cell_mass_centers",
            settings=(
                ("sigma_init_min", sigma_init[0]),
                ("sigma_init_max", sigma_init[1]),
                ("amplitude_std", amplitude_std),
            ),
        ),
        update=StatePolicySpec(
            id="cell_mass_physical", settings=(("sigma_floor", sigma_floor),)
        ),
    )


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


def polar_torus_profile_product(*, profiles, **kwargs):
    """Circle/Sphere centre-fibre chord profiles on one Torus Chart.

    Profile order is circle then section, independently of logical input/output
    order. Revision 2 distinguishes this metric from revision 1's Euclidean
    coordinate product. Shared Polar width and one operator L2 floor remain.
    """
    return replace(polar_profile_product(profiles=profiles, **kwargs), revision=2)


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


def polar_profile_product(
    *,
    profiles,
    amplitude_max,
    bounds: BandwidthBounds,
    w_c,
    normalization: NormalizationSpec | None = None,
    kappa=3.0,
    lower_kappa=None,
    upper_decay_power=1.0,
    alpha_init=0.0,
    radial_regularization=0.1,
    activity_gain=1.0,
    activity_mode="finite_chord",
    dormant_expansion_rate=0.0,
):
    """Axis profiles on one Chart, with shared Polar bandwidth and update law.

    Profiles follow Euclidean coordinate order (all output coordinates first,
    then all input coordinates). L2 normalization and its floor apply once to
    the complete product over the full operator, never to individual axes.
    """
    if (
        not isinstance(profiles, tuple)
        or not profiles
        or not all(isinstance(p, ProfileSpec) for p in profiles)
    ):
        raise TypeError("profiles must be a nonempty tuple of ProfileSpec declarations")
    base = polar_activity(
        amplitude_max=amplitude_max,
        input_bounds=bounds,
        output_bounds=bounds,
        w_c=w_c,
        profile=profile(profiles[0], normalize=False),
        kappa=kappa,
        lower_kappa=lower_kappa,
        upper_decay_power=upper_decay_power,
        alpha_init=alpha_init,
        radial_regularization=radial_regularization,
        activity_gain=activity_gain,
        activity_mode=activity_mode,
        dormant_expansion_rate=dormant_expansion_rate,
    )
    return replace(
        base,
        composition="profile_product",
        profiles=tuple(profile(p, normalize=False) for p in profiles),
        normalization=normalization
        if normalization is not None
        else NormalizationSpec(kind="discrete_l2", domain="operator_sites", floor=1e-6),
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


def polar_periodic_profile_product(*, profiles, **kwargs):
    """Coordinate-wise profiles on a flat periodic grid (revision 3).

    Shared live activity width, task width stop-gradient, and one full-domain
    discrete-L2 normalization floor match the Polar product contract.
    """
    from dataclasses import replace

    return replace(polar_profile_product(profiles=profiles, **kwargs), revision=3)


def polar_fourier_profile_product(*, profiles, tile_widths, **kwargs):
    """Native Fourier profiles on three owner-relative tiles (revision 4)."""
    from .profiles.fourier import PeriodicFourierSpec
    from .spec import FourierProductKernelSpec
    from .support import FourierTileSupportSpec

    if (
        not isinstance(profiles, tuple)
        or not profiles
        or not all(isinstance(p, PeriodicFourierSpec) for p in profiles)
    ):
        raise TypeError("profiles must be a tuple of periodic Fourier declarations")
    base = polar_profile_product(profiles=(TriweightSpec(),) * len(profiles), **kwargs)
    support = FourierTileSupportSpec(tile_widths=tile_widths)
    if base.parameterization.input_bounds.maximum > min(support.tile_widths):
        raise ValueError("maximum width must not exceed any tile width")
    return FourierProductKernelSpec(
        **{
            **vars(base),
            "profiles": tuple(profile(p, normalize=False) for p in profiles),
            "revision": 4,
            "tile_support": support,
        }
    )


def polar_matern_mixture_product(*, profiles, **kwargs):
    """Full-period image-sum profiles with adjacent fixed-width mixing (rev6)."""
    from .profiles.matern import MaternMixtureSpec

    if (
        not isinstance(profiles, tuple)
        or not profiles
        or not all(type(p) is MaternMixtureSpec for p in profiles)
    ):
        raise TypeError("requires a nonempty tuple of MaternMixtureSpec")
    base = polar_profile_product(profiles=(TriweightSpec(),) * len(profiles), **kwargs)
    return replace(
        base, profiles=tuple(profile(p, normalize=False) for p in profiles), revision=6
    )
