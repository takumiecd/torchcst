"""Execute compositional KernelSpec contracts using common fixed tensor state."""

from importlib import import_module

from torchcst._backends.torch.charts import execution as _charts
from torchcst.charts import StripChartState
from torchcst.geometry.spec import EuclideanGeometrySpec
from torchcst.kernels.options import KernelOptions


def family(state):
    from torchcst.kernels.parameterizations import (
        AmpWidthSpec,
        DirectAmpWidthSpec,
        LogWidthSpec,
        PhysicalAmpRadiusSpec,
        PhysicalAmpWidthSpec,
        PolarAmpRadiusSpec,
        PolarAmpWidthSpec,
    )

    spec = state.spec
    radius_contract = spec.revision == 7 and type(spec.parameterization) in (
        PhysicalAmpRadiusSpec,
        PolarAmpRadiusSpec,
    )
    if (
        spec.revision != 1
        and not radius_contract
        and not (
            spec.revision in (2, 3, 4, 6) and spec.composition == "profile_product"
        )
    ):
        raise ValueError("unsupported kernel revision")
    if spec.composition == "amplitude":
        name, policies = (
            "amplitude",
            ("signed_amplitude", "amplitude_and_inner", (), ()),
        )
    elif spec.parameterization is None:
        name = "separable" if spec.composition == "separable" else "radial"
        policies = ("profile_centers", "geometry", (), ())
    else:
        contracts = {
            AmpWidthSpec: (
                "amp_width",
                ("signed_amplitude_centers", "amplitude_and_geometry", (), ()),
            ),
            DirectAmpWidthSpec: (
                "direct_amp_width",
                (
                    "direct_centers",
                    "direct_activity_width",
                    ("alpha_init",),
                    ("radial_regularization",),
                ),
            ),
            PolarAmpWidthSpec: (
                "polar_amp_width",
                (
                    "polar_centers",
                    "polar_activity_width",
                    ("alpha_init",),
                    (
                        "radial_regularization",
                        "activity_gain",
                        "activity_mode",
                        "dormant_expansion_rate",
                    ),
                ),
            ),
            LogWidthSpec: ("log_width", ("provided_atoms", "euclidean", (), ())),
            PhysicalAmpWidthSpec: (
                "hierarchical_cell_mass",
                (
                    "cell_mass_centers",
                    "cell_mass_physical",
                    ("sigma_init_min", "sigma_init_max", "amplitude_std"),
                    ("sigma_floor",),
                ),
            ),
            PhysicalAmpRadiusSpec: (
                "radius_cell_mass",
                (
                    "radius_cell_mass_centers",
                    "radius_cell_mass_physical",
                    ("radius_init_min", "radius_init_max", "amplitude_std"),
                    ("radius_floor",),
                ),
            ),
            PolarAmpRadiusSpec: (
                "polar_radius_cell_mass",
                (
                    "polar_centers",
                    "polar_activity_radius",
                    ("alpha_init",),
                    (
                        "radial_regularization",
                        "activity_gain",
                        "activity_mode",
                        "dormant_expansion_rate",
                    ),
                ),
            ),
        }
        try:
            name, policies = contracts[type(spec.parameterization)]
        except KeyError:
            raise ValueError("unsupported kernel parameterization") from None
        if name == "polar_amp_width" and spec.composition == "profile_product":
            name = "profile_product"
        elif name == "polar_amp_width":
            from torchcst.kernels.spec import CellIntegratedProfileBinding

            if any(type(p) is CellIntegratedProfileBinding for p in spec.profiles):
                name = "polar_cell_mass"
        if name in ("amp_width", "polar_amp_width") and spec.composition != "separable":
            raise ValueError("parameterization requires separable composition")
        if name == "log_width" and spec.composition != "radial":
            raise ValueError("log width requires radial composition")
    initialization, update, init_keys, update_keys = policies
    if (
        spec.initialization.id != initialization
        or spec.update.id != update
        or set(dict(spec.initialization.settings)) != set(init_keys)
        or set(dict(spec.update.settings)) != set(update_keys)
    ):
        raise ValueError("unsupported kernel initialization or update policy")
    settings = dict((*spec.initialization.settings, *spec.update.settings))
    if "alpha_init" in settings and (
        type(settings["alpha_init"]) not in (int, float)
        or not 0 <= settings["alpha_init"] <= 1
    ):
        raise ValueError("alpha_init must be in [0, 1]")
    for key in ("radial_regularization", "dormant_expansion_rate"):
        if key in settings and (
            type(settings[key]) not in (int, float) or settings[key] < 0
        ):
            raise ValueError(f"{key} must be nonnegative")
    if name in (
        "polar_amp_width",
        "profile_product",
        "polar_cell_mass",
        "polar_radius_cell_mass",
    ) and (
        type(settings["activity_gain"]) not in (int, float)
        or settings["activity_gain"] <= 0
        or settings["activity_mode"] not in ("finite_chord", "time_energy")
    ):
        raise ValueError("invalid polar activity policy")
    return name


def _operation(state, operation, area="kernels"):
    name = family(state)
    polar_radius = name == "polar_radius_cell_mass"
    if name in ("radius_cell_mass", "polar_radius_cell_mass"):
        name = "radius_cell_mass"
    if name == "polar_cell_mass" and area == "parameterizations":
        name = "polar_amp_width"
    path = (
        "algorithms.polar_update.executor"
        if area == "updates"
        and (name in ("polar_amp_width", "polar_cell_mass") or polar_radius)
        else area + "." + name
    )
    module = import_module("torchcst._backends.torch." + path)
    return getattr(module, operation)


def validate_layout(state, charts):
    expected = 1 if state.spec.composition in ("radial", "profile_product") else 2
    if state.spec.composition == "amplitude":
        return validate_layout(state.inner, charts)
    if len(charts) != expected:
        raise ValueError("kernel composition differs from chart layout")
    name = family(state)
    if name in ("radius_cell_mass", "polar_radius_cell_mass"):
        import torch

        from torchcst.kernels.radius_cell_mass import radius_cell_mass_axes
        from torchcst.operators.metadata import fixed_state_signature

        # Coordinate projection/transport is shared by the explicit CUDA Polar
        # update. CPU-only factor evaluators and update Algorithms retain their
        # own device guards; validating immutable layout is device-independent.
        signature = fixed_state_signature((*charts, state))
        certified = state.__dict__.get("_radius_layout_signature")
        if signature is not None and (certified is signature or certified == signature):
            return
        if (
            any(t.is_cuda for t in state.buffers())
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "radius layout must be validated outside CUDA Graph capture"
            )
        if any(t.dtype not in (torch.float32, torch.float64) for t in state.buffers()):
            raise ValueError("radius cell-mass reference requires float32 or float64")
        radius_cell_mass_axes(
            tuple(c.declaration() for c in charts), state.declaration()
        )
        if signature is not None:
            state.__dict__["_radius_layout_signature"] = signature
        return
    if name in ("hierarchical_cell_mass", "polar_cell_mass"):
        import torch

        from torchcst.kernels.cell_mass_layout import cell_mass_axes
        from torchcst.operators.metadata import fixed_state_signature

        # _split asks for parameter_dim repeatedly during one Polar update.
        # Validate the live mathematical configuration once, then check only
        # identities/versions. Atom updates do not change this configuration.
        signature = fixed_state_signature((*charts, state))
        certified = state.__dict__.get("_cell_mass_layout_signature")
        if signature is not None and (certified is signature or certified == signature):
            return
        if (
            any(t.is_cuda for t in state.buffers())
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "cell-mass configuration must be validated outside CUDA graph capture"
            )
        cell_mass_axes(tuple(c.declaration() for c in charts), state.declaration())
        # Never certify a failed validation, or reuse unversioned inference data.
        if signature is not None:
            state.__dict__["_cell_mass_layout_signature"] = signature
        return
    if state.spec.composition == "profile_product":
        from torchcst.operators.spec import validate_profile_product_layout

        validate_profile_product_layout(charts[0].spec, state.spec)
    if (
        name == "log_width"
        and type(charts[0].geometry.spec) is not EuclideanGeometrySpec
    ):
        raise ValueError("log-width Euclidean updates require Euclidean geometry")
    if state.spec.composition == "radial" and name == "direct_amp_width":
        check_radial_activity(state, charts[0])


def check_radial_activity(state, chart):
    if not getattr(chart, "shape", ()):
        raise TypeError("radial activity kernel requires a shaped Chart")
    if state.profiles[0].binding.normalization.kind != "none":
        raise ValueError("radial activity kernel requires an unnormalized profile")
    parameterization = state.spec.parameterization
    if parameterization.input_bounds != parameterization.output_bounds:
        raise ValueError("radial kernel requires one set of bandwidth bounds")
    if isinstance(chart, StripChartState):
        _charts.validate_support(chart, float(state.scalar("sigma_max_input")))


def parameter_dim(state, *charts):
    validate_layout(state, charts)
    if state.spec.composition == "amplitude":
        return 1 + parameter_dim(state.inner, *charts)
    prefix = {
        "amp_width": 1,
        "direct_amp_width": 2,
        "polar_amp_width": 2,
        "profile_product": 2,
        "log_width": 2,
        "hierarchical_cell_mass": 2,
        "polar_cell_mass": 2,
        "radius_cell_mass": 2,
        "polar_radius_cell_mass": 2,
    }.get(family(state), 0)
    return prefix + sum(c.center_parameter_dim for c in charts)


def parameter_dof(state, *charts):
    validate_layout(state, charts)
    if state.spec.composition == "amplitude":
        return 1 + parameter_dof(state.inner, *charts)
    prefix = {
        "amp_width": 1,
        "direct_amp_width": 2,
        "polar_amp_width": 2,
        "profile_product": 2,
        "log_width": 2,
        "hierarchical_cell_mass": 2,
        "polar_cell_mass": 2,
        "radius_cell_mass": 2,
        "polar_radius_cell_mass": 2,
    }.get(family(state), 0)
    return prefix + sum(c.intrinsic_dim for c in charts)


def supports_factorization(state):
    if state.spec.composition == "amplitude":
        return supports_factorization(state.inner)
    return state.spec.composition in ("separable", "profile_product")


def initialize(state, *charts_and_atoms, mode="balanced"):
    charts = charts_and_atoms[:-1]
    validate_layout(state, charts)
    return _operation(state, "initialize")(state, *charts_and_atoms, mode=mode)


def materialize_atoms(state, *charts_and_p):
    return _operation(state, "materialize_atoms")(state, *charts_and_p)


def factors(state, *charts_and_p):
    if not supports_factorization(state):
        raise ValueError("kernel has no input/output factors")
    return _operation(state, "factors")(state, *charts_and_p)


def weight(state, *charts_and_p):
    return _operation(state, "weight")(state, *charts_and_p)


def weight_tile(state, chart, p, rows, columns):
    return _operation(state, "weight_tile")(state, chart, p, rows, columns)


def packed_weight(state, chart, p):
    return _operation(state, "packed_weight")(state, chart, p)


def project_parameter_gradient(state, *charts_p_gradient):
    return _operation(state, "project_parameter_gradient", "updates")(
        state, *charts_p_gradient
    )


def apply_parameter_update(state, *charts_p_displacement, step_size):
    return _operation(state, "apply_parameter_update", "updates")(
        state, *charts_p_displacement, step_size=step_size
    )


def transport_parameter_state(state, *charts_old_new_state):
    return _operation(state, "transport_parameter_state", "updates")(
        state, *charts_old_new_state
    )


def tangent_backend(state, input_chart, output_chart):
    if family(state) not in ("amplitude", "separable", "amp_width"):
        return None
    return _operation(state, "tangent_backend")(state, input_chart, output_chart)


def options(state):
    return state.options or KernelOptions()


def coordinate(state, operation, *args, **kwargs):
    """Inspect the declared coordinate law through its backend implementation."""
    area = (
        "updates"
        if operation == "_project_polar"
        else "kernels"
        if operation == "lower_half_amplitude"
        else "parameterizations"
    )
    return (
        _operation(state, operation, area)(state, *args, **kwargs)
        if operation != "_project_polar"
        else _operation(state, operation, area)(*args, **kwargs)
    )
