"""Common owners of fixed kernel tensors; computations belong to backends."""

from dataclasses import asdict, fields, replace

import torch
from torch import nn

from .options import KernelOptions
from .parameterizations import BandwidthBounds, FixedWidthSpec, LogWidthSpec
from .parameterizations.activity_width import ActivityWidthSpec
from .spec import KernelSpec, ProfileBinding


class ProfileState(nn.Module):
    """A profile binding and its optional fixed-width tensor."""

    def __init__(self, binding):
        super().__init__()
        if not isinstance(binding, ProfileBinding):
            raise TypeError("profile must be a ProfileBinding")
        self.binding = binding
        if binding.parameterization is not None:
            self.register_buffer("sigma", torch.tensor(binding.parameterization.sigma))

    def declaration(self):
        if self.binding.parameterization is None:
            return self.binding
        return replace(
            self.binding, parameterization=FixedWidthSpec(sigma=float(self.sigma))
        )

    def get_extra_state(self):
        return {
            "version": 1,
            "profile": asdict(self.binding.profile),
            "normalization": asdict(self.binding.normalization),
            "fixed_width": self.binding.parameterization is not None,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("profile checkpoint contract differs")


class KernelState(nn.Module):
    """Tensor state compiled from a single immutable KernelSpec.

    Atom parameters are owned by Atoms. No evaluation, derivative or update
    methods live here. Scalar tensors follow Module.to(); declaration snapshots
    are explicit configuration operations and never part of an execution step.
    """

    def __init__(self, spec: KernelSpec, *, options=None):
        super().__init__()
        if not isinstance(spec, KernelSpec):
            raise TypeError("kernel must be a KernelSpec")
        self.spec = spec
        if options is not None and not isinstance(options, KernelOptions):
            raise TypeError("options must be KernelOptions")
        self.options = options
        self.profiles = nn.ModuleList(ProfileState(p) for p in spec.profiles)
        self.inner = KernelState(spec.inner, options=options) if spec.inner else None
        parameterization = spec.parameterization
        scalars = {}
        if isinstance(parameterization, ActivityWidthSpec):
            for name in (
                "amplitude_max",
                "w_c",
                "kappa",
                "lower_kappa",
                "upper_decay_power",
            ):
                scalars[name] = getattr(parameterization, name)
            for side in ("input", "output"):
                bounds = getattr(parameterization, side + "_bounds")
                for name, field in [
                    ("sigma_min", "minimum"),
                    ("sigma_birth", "birth"),
                    ("sigma_max", "maximum"),
                    ("upper_floor", "upper_floor"),
                ]:
                    scalars[name + "_" + side] = getattr(bounds, field)
        elif parameterization is not None:
            scalars.update(
                {
                    f.name: getattr(parameterization, f.name)
                    for f in fields(parameterization)
                    if type(getattr(parameterization, f.name)) in (float, int)
                }
            )
        scalars.update(
            {
                k: v
                for k, v in (*spec.initialization.settings, *spec.update.settings)
                if type(v) in (float, int)
            }
        )
        for name, value in scalars.items():
            self.register_buffer(
                name,
                torch.tensor(
                    float(value),
                    dtype=(
                        torch.float64
                        if type(parameterization) is LogWidthSpec
                        else None
                    ),
                ),
            )

    def _apply(self, fn, recurse=True):
        # Log-width endpoints must survive float()/double() without quantizing
        # the declared bounds; evaluation casts them to the parameter dtype.
        preserved = (
            {name: value.detach().clone() for name, value in self._buffers.items()}
            if type(self.spec.parameterization) is LogWidthSpec
            else {}
        )
        result = super()._apply(fn, recurse=recurse)
        for name, value in preserved.items():
            self._buffers[name] = value.to(device=self._buffers[name].device)
        return result

    def setting(self, name):
        if self.spec.parameterization is not None and hasattr(
            self.spec.parameterization, name
        ):
            return getattr(self.spec.parameterization, name)
        return dict((*self.spec.initialization.settings, *self.spec.update.settings))[
            name
        ]

    def scalar(self, name):
        """Read a fixed Tensor without a host synchronization."""
        value = self._buffers.get(name)
        if value is None:
            raise ValueError(f"kernel has no scalar setting {name!r}")
        return value

    def declaration(self):
        spec = self.spec
        parameterization = spec.parameterization
        if isinstance(parameterization, ActivityWidthSpec):
            replacements = {
                name: float(self.scalar(name))
                for name in (
                    "amplitude_max",
                    "w_c",
                    "kappa",
                    "lower_kappa",
                    "upper_decay_power",
                )
            }
            for side in ("input", "output"):
                replacements[side + "_bounds"] = BandwidthBounds(
                    **{
                        field: float(self.scalar(name + "_" + side))
                        for name, field in [
                            ("sigma_min", "minimum"),
                            ("sigma_birth", "birth"),
                            ("sigma_max", "maximum"),
                            ("upper_floor", "upper_floor"),
                        ]
                    }
                )
            parameterization = replace(parameterization, **replacements)
        elif parameterization is not None:
            parameterization = replace(
                parameterization,
                **{
                    f.name: float(self.scalar(f.name))
                    for f in fields(parameterization)
                    if f.name in self._buffers
                },
            )

        def policy_snapshot(policy):
            return replace(
                policy,
                settings=tuple(
                    (k, float(self.scalar(k)) if k in self._buffers else v)
                    for k, v in policy.settings
                ),
            )

        return replace(
            spec,
            profiles=tuple(p.declaration() for p in self.profiles),
            parameterization=parameterization,
            inner=self.inner.declaration() if self.inner else None,
            initialization=policy_snapshot(spec.initialization),
            update=policy_snapshot(spec.update),
        )

    def get_extra_state(self):
        # Tensor values are stored in buffers. The structural contract must match.
        parameterization = self.spec.parameterization
        result = {
            "version": 1,
            "composition": self.spec.composition,
            "revision": self.spec.revision,
            "parameterization_type": type(parameterization).__name__
            if parameterization
            else None,
            "parameterization_flags": tuple(
                (f.name, getattr(parameterization, f.name))
                for f in fields(parameterization)
                if type(getattr(parameterization, f.name)) in (str, bool)
            )
            if parameterization
            else (),
            "initialization": (
                self.spec.initialization.id,
                tuple(k for k, v in self.spec.initialization.settings),
            ),
            "update": (
                self.spec.update.id,
                tuple(
                    (k, v if type(v) in (str, bool) else None)
                    for k, v in self.spec.update.settings
                ),
            ),
        }
        if self.spec.composition == "profile_product":
            result["normalization"] = asdict(self.spec.normalization)
        return result

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("kernel checkpoint contract differs")
