"""Amplitude-dependent shared-width operator kernels."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst.geometry import Chart

from .base import AtomInit, Kernel, Profile
from .gaussian import Gaussian

_LAWS = ("interpolating", "inverse")


class AmpWidth(Kernel):
    r"""A signed rank-one atom whose shared width narrows with ``|w|``.

    ``law="interpolating"`` uses one even amplitude gate to interpolate the
    shared input and output precision between finite ``sigma_max`` and
    ``sigma_min``. ``law="inverse"`` uses the smooth map ``σ ≈ τ / |w|``,
    softly clamped to the same bounds in log-precision. The atom row is
    ``(w, input_center, output_center)``. The default profile is ``Gaussian``;
    compact profiles such as ``WendlandC2`` and ``Triweight`` use the same
    precision convention, with ``sigma`` as the support radius.
    """

    def __init__(
        self,
        *,
        sigma_min: float,
        sigma_max: float,
        tau: float = 5e-3,
        temperature: float = 0.25,
        gate_eps: float = 1e-12,
        law: str = "interpolating",
        profile: Profile | None = None,
        couple_bandwidth: bool = True,
    ) -> None:
        super().__init__()
        if law not in _LAWS:
            raise ValueError("law must be 'interpolating' or 'inverse'")
        self.law = law
        if profile is None:
            profile = Gaussian(sigma_min)
        elif not isinstance(profile, Profile):
            raise TypeError("profile must implement the Profile contract")
        if not hasattr(profile, "evaluate_with_precision") or not hasattr(
            profile, "tangent_with_precision"
        ):
            raise TypeError(
                "bandwidth profile must implement evaluate_with_precision "
                "and tangent_with_precision"
            )
        if not hasattr(profile, "sigma"):
            raise TypeError("bandwidth profile must expose a sigma buffer")
        minimum = self._positive_scalar(sigma_min, name="sigma_min")
        if not torch.allclose(
            profile.sigma.detach().to(dtype=minimum.dtype).reshape(()),
            minimum,
        ):
            raise ValueError("profile.sigma must match sigma_min")
        self.profile = profile
        maximum = self._positive_scalar(sigma_max, name="sigma_max")
        if maximum < self.profile.sigma:
            raise ValueError("sigma_max must not be narrower than sigma_min")
        self.register_buffer("sigma_max", maximum)
        self.register_buffer("tau", self._positive_scalar(tau, name="tau"))
        self.register_buffer(
            "temperature",
            self._positive_scalar(temperature, name="temperature"),
        )
        self.register_buffer(
            "gate_eps",
            self._positive_scalar(gate_eps, name="gate_eps"),
        )
        self.couple_bandwidth = bool(couple_bandwidth)

    def get_extra_state(self) -> dict[str, object]:
        """Record non-buffer settings that select the amplitude-width law."""

        return {
            "format_version": 1,
            "kernel_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "law": self.law,
            "couple_bandwidth": self.couple_bandwidth,
        }

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("AmpWidth checkpoint contract differs from this kernel")

    @property
    def sigma_min(self) -> Tensor:
        return self.profile.sigma

    def parameter_dim(self, input_chart: Chart, output_chart: Chart) -> int:
        return (
            1
            + self.profile.parameter_dim(input_chart)
            + self.profile.parameter_dim(output_chart)
        )

    def parameter_dof(self, input_chart: Chart, output_chart: Chart) -> int:
        return (
            1
            + self.profile.parameter_dof(input_chart)
            + self.profile.parameter_dof(output_chart)
        )

    def initialize(
        self,
        input_chart: Chart,
        output_chart: Chart,
        atoms: int,
        *,
        mode: AtomInit,
    ) -> Tensor:
        from torchcst._backends.torch.kernels.amp_width import initialize

        return initialize(self, input_chart, output_chart, atoms, mode=mode)

    def materialize_atoms(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.kernels.amp_width import materialize_atoms

        return materialize_atoms(self, input_chart, output_chart, p)

    @property
    def supports_factorization(self) -> bool:
        return True

    def factors(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.kernels.amp_width import factors

        return factors(self, input_chart, output_chart, p)

    def amplitude_gate(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        """Return the interpolating commitment gate for diagnostic use."""
        from torchcst._backends.torch.parameterizations.amp_width import amplitude_gate

        return amplitude_gate(self, input_chart, output_chart, p)

    def bandwidth_precision(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        """Return the shared input/output precision for each atom."""
        from torchcst._backends.torch.parameterizations.amp_width import (
            bandwidth_precision,
        )

        return bandwidth_precision(self, input_chart, output_chart, p)

    def bandwidth_sigma(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        """Return the shared effective input/output sigma for diagnostics."""
        from torchcst._backends.torch.parameterizations.amp_width import bandwidth_sigma

        return bandwidth_sigma(self, input_chart, output_chart, p)

    def _split(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        from torchcst._backends.torch.parameterizations.amp_width import _split

        return _split(self, input_chart, output_chart, p)

    def _precision_and_jacobian(self, amplitude: Tensor) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.parameterizations.amp_width import (
            _precision_and_jacobian,
        )

        return _precision_and_jacobian(self, amplitude)

    def _magnitude_square(self, amplitude: Tensor) -> Tensor:
        from torchcst._backends.torch.parameterizations.amp_width import (
            _magnitude_square,
        )

        return _magnitude_square(self, amplitude)

    def _interpolating_gate(self, amplitude: Tensor) -> Tensor:
        from torchcst._backends.torch.parameterizations.amp_width import (
            _interpolating_gate,
        )

        return _interpolating_gate(self, amplitude)

    def _interpolating_precision_and_jacobian(
        self,
        amplitude: Tensor,
    ) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.parameterizations.amp_width import (
            _interpolating_precision_and_jacobian,
        )

        return _interpolating_precision_and_jacobian(self, amplitude)

    def _inverse_precision_and_jacobian(
        self,
        amplitude: Tensor,
    ) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.parameterizations.amp_width import (
            _inverse_precision_and_jacobian,
        )

        return _inverse_precision_and_jacobian(self, amplitude)

    @staticmethod
    def _positive_scalar(value: float, *, name: str) -> Tensor:
        result = torch.as_tensor(value, dtype=torch.get_default_dtype())
        if result.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        result = result.detach().clone().reshape(())
        if not torch.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    def tangent_backend(self, input_chart: Chart, output_chart: Chart):
        from torchcst._backends.torch.kernels.amp_width import tangent_backend

        return tangent_backend(self, input_chart, output_chart)

    def project_parameter_gradient(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.updates.amp_width import (
            project_parameter_gradient,
        )

        return project_parameter_gradient(self, input_chart, output_chart, p, gradient)

    def apply_parameter_update(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        displacement: Tensor,
        *,
        step_size: float,
    ) -> Tensor:
        from torchcst._backends.torch.updates.amp_width import apply_parameter_update

        return apply_parameter_update(
            self, input_chart, output_chart, p, displacement, step_size=step_size
        )

    def transport_parameter_state(
        self,
        input_chart: Chart,
        output_chart: Chart,
        old: Tensor,
        new: Tensor,
        state: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.updates.amp_width import transport_parameter_state

        return transport_parameter_state(
            self, input_chart, output_chart, old, new, state
        )

    def extra_repr(self) -> str:
        return (
            f"law={self.law}, "
            f"tau={self.tau.item():g}, temperature={self.temperature.item():g}, "
            f"profile={type(self.profile).__name__}, "
            f"sigma_min={self.sigma_min.item():g}, "
            f"sigma_max={self.sigma_max.item():g}, "
            f"couple_bandwidth={self.couple_bandwidth}, "
            f"supports_factorization={self.supports_factorization}"
        )


AmplitudeBandwidthSeparable = AmpWidth
