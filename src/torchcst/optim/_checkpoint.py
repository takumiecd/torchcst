"""Stable metadata for existing optimizer checkpoints and opt-in extensions."""

from dataclasses import asdict

from torchcst._builtin_callbacks import seal_functions as _seal_functions


def kernel_manifest(declaration):
    result = asdict(declaration)
    inner = result
    while isinstance(inner, dict):
        for profile in inner.get("profiles", ()):
            if profile.get("connection") is None:
                profile.pop("connection", None)
        inner = inner.get("inner")
    parameterization = result.get("parameterization")
    if (
        isinstance(parameterization, dict)
        and parameterization.get("radius_envelope") != "rational_zero_opening"
    ):
        parameterization.pop("zero_opening_scale", None)
    if (
        isinstance(parameterization, dict)
        and parameterization.get("id") == "polar_activity_radius"
        and parameterization.get("radius_envelope") == "rational"
    ):
        # These fields do not change the pre-existing rational law. Preserve
        # its exact saved contract, while keeping all closed-law fields.
        for name in ("radius_envelope", "closed_lower_power", "closed_upper_power"):
            parameterization.pop(name)
    return result


_seal_functions(globals())
