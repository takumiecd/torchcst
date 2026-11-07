"""Kernel declarations in three parts, composed into pure ``KernelSpec`` values.

1. A *profile shape* is a ``ProfileSpec`` such as ``GaussianSpec()`` or
   ``ChebyshevSpec(modes=8)``.
2. A *factor* says how a shape is evaluated and normalized on one chart:
   ``point`` (sampled values), ``l2`` (sampled values with discrete L2
   normalization) or ``cell_mass`` (whole-cell integrals with unit total mass).
3. A *coordinate rule* says how atom parameters become amplitude, radius and
   centers and how they are updated: ``polar``, ``physical``, ``direct`` or a
   plain fixed width (``sigma=`` on a factor).

``product`` multiplies one factor per chart, in ``charts=(output, input)``
order. ``radial`` applies a single factor to the combined distance.
"""

import math
from dataclasses import replace

from .normalization import NormalizationSpec
from .parameterizations import (
    AmpWidthSpec,
    BandwidthBounds,
    DirectAmpWidthSpec,
    FixedWidthSpec,
    LogWidthSpec,
    PhysicalAmpRadiusSpec,
    PolarAmpRadiusSpec,
    PolarAmpWidthSpec,
    RadiusBounds,
)
from .profiles import (
    GaussianSpec,
    ProfileSpec,
    TriweightSpec,
)
from .profiles.fourier import ChebyshevSpec
from .spec import (
    CoordinateSpec,
    KernelSpec,
    ProfileBinding,
    RadiusIntegratedProfileBinding,
    StatePolicySpec,
)

_DEFAULT_FLOOR = object()


def _sampled(
    shape: ProfileSpec, sigma, *, normalize: bool, name: str, floor=_DEFAULT_FLOOR
) -> ProfileBinding:
    if not isinstance(shape, ProfileSpec):
        raise TypeError(f"{name} needs a ProfileSpec shape")
    if floor is _DEFAULT_FLOOR:
        floor = None if isinstance(shape, GaussianSpec) else 1e-6
    if normalize and isinstance(shape, GaussianSpec) and floor is not None:
        raise ValueError("Gaussian L2 uses stable log normalization without a floor")
    if normalize and not isinstance(shape, GaussianSpec) and floor is None:
        raise ValueError("compact L2 requires a positive norm floor")
    return ProfileBinding(
        profile=shape,
        parameterization=None if sigma is None else FixedWidthSpec(sigma=sigma),
        normalization=NormalizationSpec(
            kind="discrete_l2",
            domain="chart_sites",
            floor=floor,
        )
        if normalize
        else NormalizationSpec(),
    )


def point(shape: ProfileSpec, sigma=None) -> ProfileBinding:
    """Sampled profile values without normalization.

    ``sigma`` fixes the width; leave it out when a coordinate rule owns the width.
    """
    return _sampled(shape, sigma, normalize=False, name="point")


def l2(shape: ProfileSpec, sigma=None, *, floor=_DEFAULT_FLOOR) -> ProfileBinding:
    """Sampled profile values, L2-normalized over the sites of each chart."""
    return _sampled(shape, sigma, normalize=True, name="l2", floor=floor)


def cell_mass(
    shape: ProfileSpec | None = None, *, base_tile_sites=1, merge_factor=2
) -> RadiusIntegratedProfileBinding:
    """Whole-cell integrals of a periodic density with unit total mass.

    The radius is physical. ``base_tile_sites`` and ``merge_factor`` define the
    hierarchy of the approximation and are part of the declaration. It needs a
    one-dimensional flat-torus chart and a ``polar`` or ``physical`` rule.
    """
    return RadiusIntegratedProfileBinding(
        profile=ChebyshevSpec() if shape is None else shape,
        base_tile_sites=base_tile_sites,
        merge_factor=merge_factor,
    )


def _compose(composition, profiles, coordinates):
    if coordinates is None:
        return KernelSpec(composition=composition, profiles=profiles)
    if not isinstance(coordinates, CoordinateSpec):
        raise TypeError("coordinates must be a CoordinateSpec")
    radius = type(coordinates.parameterization) in (
        PolarAmpRadiusSpec,
        PhysicalAmpRadiusSpec,
    )
    return KernelSpec(
        composition=composition,
        profiles=profiles,
        parameterization=coordinates.parameterization,
        initialization=coordinates.initialization,
        update=coordinates.update,
        revision=7 if radius else 1,
    )


def product(output_factor, input_factor=None, *, coordinates=None) -> KernelSpec:
    """Multiply one factor per Chart in (output, input) order.

    A multidimensional Chart produces one factor over all its sites, using its
    full distance. Cell masses require physical/Polar radius coordinates and
    one-dimensional FlatTorus charts.
    """
    return _compose(
        "separable",
        (output_factor, output_factor if input_factor is None else input_factor),
        coordinates,
    )


def radial(factor, *, coordinates=None) -> KernelSpec:
    """Evaluate one profile on the sum of squared scaled Chart distances.

    L2 normalization covers the complete operator, rather than each Chart.
    Cell masses have no radial definition and are rejected.
    """
    if isinstance(factor, RadiusIntegratedProfileBinding):
        raise ValueError("radius cell masses require product composition")  # noqa: TRY004
    if (
        isinstance(factor, ProfileBinding)
        and factor.normalization.kind == "discrete_l2"
    ):
        factor = replace(
            factor, normalization=replace(factor.normalization, domain="operator_sites")
        )
    return _compose("radial", (factor,), coordinates)


def amplitude(kernel) -> KernelSpec:
    """Add one signed amplitude coordinate to any kernel."""
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
    factor=None,
    law="interpolating",
    couple_bandwidth=True,
    tau=5e-3,
    temperature=0.25,
    gate_eps=1e-12,
) -> KernelSpec:
    """Product kernel whose width follows the signed amplitude."""
    binding = factor or l2(GaussianSpec())
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


def physical(
    *,
    radius_init=(0.5, 1.0),
    amplitude_std=1.0,
    radius_floor=1e-6,
) -> CoordinateSpec:
    """Cell-mass coordinates with a directly declared physical support radius.

    Combine with ``product(cell_mass(...), coordinates=physical(...))``.
    """
    if (
        type(radius_floor) not in (int, float)
        or not math.isfinite(radius_floor)
        or radius_floor <= 0
        or type(radius_init) is not tuple
        or len(radius_init) != 2
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in radius_init)
        or not radius_floor <= radius_init[0] <= radius_init[1]
    ):
        raise ValueError(
            "radius_init must contain ordered positive endpoints above radius_floor"
        )
    if (
        type(amplitude_std) not in (int, float)
        or not math.isfinite(amplitude_std)
        or amplitude_std < 0
    ):
        raise ValueError("amplitude_std must be a finite nonnegative scalar")
    return CoordinateSpec(
        parameterization=PhysicalAmpRadiusSpec(),
        initialization=StatePolicySpec(
            id="radius_cell_mass_centers",
            settings=(
                ("radius_init_min", radius_init[0]),
                ("radius_init_max", radius_init[1]),
                ("amplitude_std", amplitude_std),
            ),
        ),
        update=StatePolicySpec(
            id="radius_cell_mass_physical", settings=(("radius_floor", radius_floor),)
        ),
    )


def _activity(
    *,
    coordinate,
    amplitude_max,
    input_bounds,
    output_bounds,
    w_c,
    kappa,
    lower_kappa,
    upper_decay_power,
    alpha_init,
    radial_regularization,
    activity_gain=1.0,
    activity_mode="finite_chord",
    dormant_expansion_rate=0.0,
    radius_envelope="rational",
    closed_lower_power=4.0,
    closed_upper_power=1.0,
):
    if type(alpha_init) not in (float, int) or not 0 <= alpha_init <= 1:
        raise ValueError("alpha_init must be in [0, 1]")
    if type(radial_regularization) not in (float, int) or radial_regularization < 0:
        raise ValueError("radial_regularization must be nonnegative")
    if kappa <= 1 or upper_decay_power > 1:
        raise ValueError("kappa must exceed 1 and upper_decay_power must not exceed 1")
    is_polar = coordinate in ("polar", "polar_radius")
    cls = (
        DirectAmpWidthSpec
        if coordinate == "direct"
        else PolarAmpRadiusSpec
        if coordinate == "polar_radius"
        else PolarAmpWidthSpec
    )
    parameterization = cls(
        amplitude_max=amplitude_max,
        input_bounds=input_bounds,
        output_bounds=output_bounds,
        w_c=w_c,
        kappa=kappa,
        lower_kappa=kappa if lower_kappa is None else lower_kappa,
        upper_decay_power=upper_decay_power,
        **(
            dict(
                radius_envelope=radius_envelope,
                closed_lower_power=closed_lower_power,
                closed_upper_power=closed_upper_power,
            )
            if coordinate == "polar_radius"
            else {}
        ),
    )
    updates = [("radial_regularization", radial_regularization)]
    if is_polar:
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
    return CoordinateSpec(
        parameterization=parameterization,
        initialization=StatePolicySpec(
            id=("polar" if is_polar else coordinate) + "_centers",
            settings=(("alpha_init", alpha_init),),
        ),
        update=StatePolicySpec(id=parameterization.id, settings=tuple(updates)),
    )


def _bounds(bounds, input_bounds, output_bounds):
    if bounds is not None:
        if input_bounds is not None or output_bounds is not None:
            raise ValueError("use shared bounds or explicit input/output bounds")
        input_bounds = output_bounds = bounds
    if input_bounds is None:
        raise TypeError("coordinates require bounds or input_bounds")
    return input_bounds, input_bounds if output_bounds is None else output_bounds


def direct(
    *,
    amplitude_max,
    w_c,
    bounds: BandwidthBounds | None = None,
    input_bounds: BandwidthBounds | None = None,
    output_bounds: BandwidthBounds | None = None,
    kappa=3.0,
    lower_kappa=None,
    upper_decay_power=1.0,
    alpha_init=0.0,
    radial_regularization=0.1,
) -> CoordinateSpec:
    """Direct amplitude and activity-dependent sigma; pair with point/l2."""
    input_bounds, output_bounds = _bounds(bounds, input_bounds, output_bounds)
    return _activity(
        coordinate="direct",
        amplitude_max=amplitude_max,
        input_bounds=input_bounds,
        output_bounds=output_bounds,
        w_c=w_c,
        kappa=kappa,
        lower_kappa=lower_kappa,
        upper_decay_power=upper_decay_power,
        alpha_init=alpha_init,
        radial_regularization=radial_regularization,
    )


def polar(
    *,
    amplitude_max,
    w_c,
    bounds: BandwidthBounds | RadiusBounds | None = None,
    input_bounds: BandwidthBounds | RadiusBounds | None = None,
    output_bounds: BandwidthBounds | RadiusBounds | None = None,
    kappa=3.0,
    lower_kappa=None,
    upper_decay_power=1.0,
    alpha_init=0.0,
    radial_regularization=0.1,
    activity_gain=1.0,
    activity_mode="finite_chord",
    dormant_expansion_rate=0.0,
    radius_envelope="rational",
    closed_lower_power=4.0,
    closed_upper_power=1.0,
) -> CoordinateSpec:
    """Angular task motion with radial activity that sets the width.

    ``BandwidthBounds`` describes sigma for point/l2; ``RadiusBounds`` describes
    physical radius for cell_mass. Widths stay detached from task gradients.
    """
    input_bounds, output_bounds = _bounds(bounds, input_bounds, output_bounds)
    if not isinstance(input_bounds, RadiusBounds) and (
        radius_envelope != "rational"
        or closed_lower_power != 4
        or closed_upper_power != 1
    ):
        raise ValueError("radius envelopes require RadiusBounds")
    return _activity(
        coordinate="polar_radius"
        if isinstance(input_bounds, RadiusBounds)
        else "polar",
        amplitude_max=amplitude_max,
        input_bounds=input_bounds,
        output_bounds=output_bounds,
        w_c=w_c,
        kappa=kappa,
        lower_kappa=lower_kappa,
        upper_decay_power=upper_decay_power,
        alpha_init=alpha_init,
        radial_regularization=radial_regularization,
        activity_gain=activity_gain,
        activity_mode=activity_mode,
        dormant_expansion_rate=dormant_expansion_rate,
        radius_envelope=radius_envelope,
        closed_lower_power=closed_lower_power,
        closed_upper_power=closed_upper_power,
    )


NORMALIZED_RADIAL_TRIWEIGHT = KernelSpec(
    composition="radial",
    profiles=(
        replace(
            l2(TriweightSpec()),
            normalization=NormalizationSpec(
                kind="discrete_l2", domain="operator_sites", floor=1e-6
            ),
        ),
    ),
    parameterization=LogWidthSpec(sigma_min=0.03, sigma_max=3.25),
    initialization=StatePolicySpec(id="provided_atoms"),
    update=StatePolicySpec(id="euclidean"),
)
