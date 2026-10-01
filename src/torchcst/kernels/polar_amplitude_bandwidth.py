"""Polar amplitude/width coordinates with radial-angular decoupling."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from torchcst.geometry import Chart

from .base import AtomInit, Kernel, Profile
from .gaussian import Gaussian


class PolarAmpWidth(Kernel):
    r"""A rank-one atom with angular amplitude and radial width control.

    The first two coordinates of each atom are ``(s, t)``.  They encode

    .. math::

        w = W s / \sqrt{s^2+t^2}, \qquad
        \alpha = \operatorname{clamp}((s^2+t^2-1)/3, 0, 1).

    Consequently ``w`` is invariant to radial rescaling while ``alpha`` is
    invariant to angular motion.  The shared input/output bandwidth is
    the geometric interpolation ``L(w) ** (1-alpha) * U(w) ** alpha``.
    At zero amplitude, ``L`` starts at ``sigma_birth`` while ``U`` starts at
    ``sigma_max``.  Both approach ``sigma_min`` as the amplitude scale grows.
    Input and output charts may use independent bandwidth limits while sharing
    the same amplitude and exploration clock.

    The hard clamp keeps every forward bandwidth inside its configured
    bounds under arbitrary optimizers.  Coordinates are initialized on the
    unit circle, so ``alpha == 0`` initially.
    """

    def __init__(
        self,
        *,
        amplitude_max: float,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        sigma_birth: float | None = None,
        input_sigma_min: float | None = None,
        input_sigma_birth: float | None = None,
        input_sigma_max: float | None = None,
        output_sigma_min: float | None = None,
        output_sigma_birth: float | None = None,
        output_sigma_max: float | None = None,
        w_c: float,
        kappa: float = 3.0,
        lower_kappa: float | None = None,
        lower_half_amplitude: float | None = None,
        upper_decay_power: float = 1.0,
        upper_floor: float | None = None,
        alpha_init: float = 0.0,
        activity_gain: float = 1.0,
        activity_mode: Literal["finite_chord", "time_energy"] = "finite_chord",
        dormant_expansion_rate: float = 0.0,
        radial_regularization: float = 0.1,
        profile: Profile | None = None,
    ) -> None:
        super().__init__()
        maximum_amplitude = self._positive_scalar(amplitude_max, name="amplitude_max")
        minimum_input, minimum_output = self._bandwidth_pair(
            sigma_min,
            input_sigma_min,
            output_sigma_min,
            name="sigma_min",
        )
        maximum_input, maximum_output = self._bandwidth_pair(
            sigma_max,
            input_sigma_max,
            output_sigma_max,
            name="sigma_max",
        )
        if (
            sigma_birth is None
            and input_sigma_birth is None
            and output_sigma_birth is None
        ):
            birth_input = maximum_input.detach().clone()
            birth_output = maximum_output.detach().clone()
        else:
            birth_input, birth_output = self._bandwidth_pair(
                sigma_birth,
                input_sigma_birth,
                output_sigma_birth,
                name="sigma_birth",
            )
        crossover = self._positive_scalar(w_c, name="w_c")
        separation = self._positive_scalar(kappa, name="kappa")
        if lower_kappa is not None and lower_half_amplitude is not None:
            raise ValueError(
                "lower_kappa and lower_half_amplitude are mutually exclusive"
            )
        if lower_half_amplitude is not None:
            lower_half = self._positive_scalar(
                lower_half_amplitude,
                name="lower_half_amplitude",
            )
            lower_separation = (crossover / lower_half).square()
        else:
            lower_separation = (
                separation.detach().clone()
                if lower_kappa is None
                else self._positive_scalar(lower_kappa, name="lower_kappa")
            )
        upper_power = self._positive_scalar(upper_decay_power, name="upper_decay_power")
        exploration_floor_input = (
            minimum_input.detach().clone()
            if upper_floor is None
            else self._positive_scalar(upper_floor, name="upper_floor")
        )
        exploration_floor_output = (
            minimum_output.detach().clone()
            if upper_floor is None
            else exploration_floor_input.detach().clone()
        )
        initial_alpha = self._unit_interval_scalar(alpha_init, name="alpha_init")
        gain = self._positive_scalar(activity_gain, name="activity_gain")
        if activity_mode not in ("finite_chord", "time_energy"):
            raise ValueError("activity_mode must be 'finite_chord' or 'time_energy'")
        dormant_rate = self._nonnegative_scalar(
            dormant_expansion_rate, name="dormant_expansion_rate"
        )
        regularization = self._nonnegative_scalar(
            radial_regularization, name="radial_regularization"
        )
        for side, minimum, birth, maximum, floor in (
            (
                "input",
                minimum_input,
                birth_input,
                maximum_input,
                exploration_floor_input,
            ),
            (
                "output",
                minimum_output,
                birth_output,
                maximum_output,
                exploration_floor_output,
            ),
        ):
            if maximum < minimum:
                raise ValueError(f"{side} sigma_max must not be smaller than sigma_min")
            if birth < minimum or birth > maximum:
                raise ValueError(
                    f"{side} sigma_birth must be in [sigma_min, sigma_max]"
                )
            if floor < minimum or floor > maximum:
                raise ValueError(
                    f"{side} upper_floor must be in [sigma_min, sigma_max]"
                )
        if separation <= 1:
            raise ValueError("kappa must be greater than 1")
        if upper_power > 1:
            raise ValueError("upper_decay_power must not be greater than 1")

        if profile is None:
            profile = Gaussian(minimum_input)
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
        if not torch.allclose(
            profile.sigma.detach().to(dtype=minimum_input.dtype).reshape(()),
            minimum_input,
        ):
            raise ValueError("profile.sigma must match input sigma_min")

        self.profile = profile
        self.register_buffer("amplitude_max", maximum_amplitude)
        self.register_buffer("sigma_min_input", minimum_input)
        self.register_buffer("sigma_birth_input", birth_input)
        self.register_buffer("sigma_max_input", maximum_input)
        self.register_buffer("sigma_min_output", minimum_output)
        self.register_buffer("sigma_birth_output", birth_output)
        self.register_buffer("sigma_max_output", maximum_output)
        self.register_buffer("w_c", crossover)
        self.register_buffer("kappa", separation)
        self.register_buffer("lower_kappa", lower_separation)
        self.register_buffer("upper_decay_power", upper_power)
        self.register_buffer("upper_floor_input", exploration_floor_input)
        self.register_buffer("upper_floor_output", exploration_floor_output)
        self.register_buffer("alpha_init", initial_alpha)
        self.register_buffer("activity_gain", gain)
        self.activity_mode = activity_mode
        self.register_buffer("dormant_expansion_rate", dormant_rate)
        self.register_buffer("radial_regularization", regularization)

    def get_extra_state(self) -> dict[str, object]:
        """Record the meaning of the first two atom coordinates in checkpoints."""

        return {
            "format_version": 1,
            "coordinate_system": "polar",
            "profile": f"{type(self.profile).__module__}.{type(self.profile).__qualname__}",
            "normalize_columns": getattr(self.profile, "normalize_columns", None),
            "activity_mode": self.activity_mode,
        }

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise RuntimeError(
                f"{type(self).__name__} checkpoint contract differs from this kernel"
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        marker = prefix + "_extra_state"
        legacy_maximum = prefix + "sigma_max"
        if marker not in state_dict:
            if (
                legacy_maximum not in state_dict
                or prefix + "sigma_max_input" in state_dict
                or prefix + "profile.sigma" not in state_dict
                or prefix + "profile._extra_state" not in state_dict
                or getattr(self.profile, "normalize_columns", True) is not True
            ):
                error_msgs.append(
                    f"{prefix[:-1]}: untagged amplitude-bandwidth checkpoint "
                    "has unverifiable coordinates or profile settings; "
                    "identify and migrate its format explicitly"
                )
                return
            # Before split bandwidths, Polar stored one sigma_max and used
            # profile.sigma as the shared sigma_min. All added controls had
            # values equivalent to these legacy bounds by default.
            minimum = state_dict[prefix + "profile.sigma"]
            maximum = state_dict.pop(legacy_maximum)
            defaults = {
                "sigma_min_input": minimum,
                "sigma_min_output": minimum,
                "sigma_birth_input": maximum,
                "sigma_birth_output": maximum,
                "sigma_max_input": maximum,
                "sigma_max_output": maximum,
                "lower_kappa": state_dict[prefix + "kappa"],
                "upper_decay_power": minimum.new_tensor(1.0),
                "upper_floor_input": minimum,
                "upper_floor_output": minimum,
                "alpha_init": minimum.new_tensor(0.0),
                "dormant_expansion_rate": minimum.new_tensor(0.0),
            }
            for name, value in defaults.items():
                state_dict[prefix + name] = value.detach().clone()
            state_dict[marker] = self.get_extra_state()
        elif state_dict[marker] != self.get_extra_state():
            error_msgs.append(
                f"{prefix[:-1]}: checkpoint contract "
                f"{state_dict[marker]!r} does not match {self.get_extra_state()!r}"
            )
            return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def sigma_min(self) -> Tensor:
        """Input-side minimum; use ``sigma_min_output`` for split limits."""

        return self.sigma_min_input

    @property
    def sigma_birth(self) -> Tensor:
        """Input-side birth width; use ``sigma_birth_output`` for output."""

        return self.sigma_birth_input

    @property
    def sigma_max(self) -> Tensor:
        """Input-side maximum; use ``sigma_max_output`` for split limits."""

        return self.sigma_max_input

    @property
    def upper_floor(self) -> Tensor:
        """Input-side upper floor retained as a compatibility alias."""

        return self.upper_floor_input

    @property
    def lower_half_amplitude(self) -> Tensor:
        """Absolute amplitude where ``L(w)`` is halfway between its limits."""
        from torchcst._backends.torch.kernels.polar_amp_width import (
            lower_half_amplitude,
        )

        return lower_half_amplitude(self)

    def parameter_dim(self, input_chart: Chart, output_chart: Chart) -> int:
        return (
            2
            + self.profile.parameter_dim(input_chart)
            + self.profile.parameter_dim(output_chart)
        )

    def parameter_dof(self, input_chart: Chart, output_chart: Chart) -> int:
        return (
            2
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
        from torchcst._backends.torch.kernels.polar_amp_width import initialize

        return initialize(self, input_chart, output_chart, atoms, mode=mode)

    def materialize_atoms(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.kernels.polar_amp_width import materialize_atoms

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
        from torchcst._backends.torch.kernels.polar_amp_width import factors

        return factors(self, input_chart, output_chart, p)

    def amplitude(self, input_chart: Chart, output_chart: Chart, p: Tensor) -> Tensor:
        """Return the bounded signed amplitude represented by each atom."""
        from torchcst._backends.torch.parameterizations.polar_amp_width import amplitude

        return amplitude(self, input_chart, output_chart, p)

    def bandwidth_alpha(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> Tensor:
        """Return the radial interpolation coordinate in ``[0, 1]``."""
        from torchcst._backends.torch.parameterizations.polar_amp_width import (
            bandwidth_alpha,
        )

        return bandwidth_alpha(self, input_chart, output_chart, p)

    def bandwidth_bounds(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``(L(w), U(w))`` for diagnostic use."""
        from torchcst._backends.torch.parameterizations.polar_amp_width import (
            bandwidth_bounds,
        )

        return bandwidth_bounds(self, input_chart, output_chart, p)

    def bandwidth_bounds_by_side(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Return ``((L_in, U_in), (L_out, U_out))``."""
        from torchcst._backends.torch.parameterizations.polar_amp_width import (
            bandwidth_bounds_by_side,
        )

        return bandwidth_bounds_by_side(self, input_chart, output_chart, p)

    def bandwidth_sigma(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> Tensor:
        """Return the shared effective input/output sigma for each atom."""
        from torchcst._backends.torch.parameterizations.polar_amp_width import (
            bandwidth_sigma,
        )

        return bandwidth_sigma(self, input_chart, output_chart, p)

    def bandwidth_sigmas(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return effective ``(sigma_input, sigma_output)`` for each atom."""
        from torchcst._backends.torch.parameterizations.polar_amp_width import (
            bandwidth_sigmas,
        )

        return bandwidth_sigmas(self, input_chart, output_chart, p)

    def bandwidth_precision(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> Tensor:
        """Return the shared input/output precision for each atom."""
        from torchcst._backends.torch.parameterizations.polar_amp_width import (
            bandwidth_precision,
        )

        return bandwidth_precision(self, input_chart, output_chart, p)

    def bandwidth_precisions(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return input and output precisions for split bandwidths."""
        from torchcst._backends.torch.parameterizations.polar_amp_width import (
            bandwidth_precisions,
        )

        return bandwidth_precisions(self, input_chart, output_chart, p)

    def apply_parameter_update(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        displacement: Tensor,
        *,
        step_size: float,
    ) -> Tensor:
        """Project task motion tangentially, decay radius, then enforce 1<=q<=4."""
        from torchcst._backends.torch.updates.polar_amp_width import (
            apply_parameter_update,
        )

        return apply_parameter_update(
            self, input_chart, output_chart, p, displacement, step_size=step_size
        )

    def project_parameter_gradient(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.updates.polar_amp_width import (
            project_parameter_gradient,
        )

        return project_parameter_gradient(self, input_chart, output_chart, p, gradient)

    def transport_parameter_state(
        self,
        input_chart: Chart,
        output_chart: Chart,
        old: Tensor,
        new: Tensor,
        state: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.updates.polar_amp_width import (
            transport_parameter_state,
        )

        return transport_parameter_state(
            self, input_chart, output_chart, old, new, state
        )

    def _split(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        from torchcst._backends.torch.parameterizations.polar_amp_width import _split

        return _split(self, input_chart, output_chart, p)

    def _amplitude_and_alpha(self, polar: Tensor) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.parameterizations.polar_amp_width import (
            _amplitude_and_alpha,
        )

        return _amplitude_and_alpha(self, polar)

    @staticmethod
    def _project_polar(polar: Tensor) -> Tensor:
        from torchcst._backends.torch.updates.polar_amp_width import _project_polar

        return _project_polar(polar)

    def _bandwidth_sigmas(
        self, amplitude: Tensor, alpha: Tensor
    ) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.parameterizations.polar_amp_width import (
            _bandwidth_sigmas,
        )

        return _bandwidth_sigmas(self, amplitude, alpha)

    def _sigma_bounds(
        self,
        amplitude: Tensor,
        alpha: Tensor | None = None,
        *,
        side: Literal["input", "output"] = "input",
    ) -> tuple[Tensor, Tensor, Tensor]:
        from torchcst._backends.torch.parameterizations.polar_amp_width import (
            _sigma_bounds,
        )

        return _sigma_bounds(self, amplitude, alpha, side=side)

    def _require_shared_bandwidths(self) -> None:
        from torchcst._backends.torch.parameterizations.polar_amp_width import (
            _require_shared_bandwidths,
        )

        return _require_shared_bandwidths(self)

    @classmethod
    def _bandwidth_pair(
        cls,
        shared: float | None,
        input_value: float | None,
        output_value: float | None,
        *,
        name: str,
    ) -> tuple[Tensor, Tensor]:
        if shared is not None:
            if input_value is not None or output_value is not None:
                raise ValueError(
                    f"{name} and side-specific {name} values are mutually exclusive"
                )
            value = cls._positive_scalar(shared, name=name)
            return value, value.detach().clone()
        if input_value is None or output_value is None:
            raise ValueError(f"provide {name} or both input_{name} and output_{name}")
        return (
            cls._positive_scalar(input_value, name=f"input_{name}"),
            cls._positive_scalar(output_value, name=f"output_{name}"),
        )

    @staticmethod
    def _positive_scalar(value: float, *, name: str) -> Tensor:
        result = torch.as_tensor(value, dtype=torch.get_default_dtype())
        if result.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        result = result.detach().clone().reshape(())
        if not torch.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @staticmethod
    def _nonnegative_scalar(value: float, *, name: str) -> Tensor:
        result = torch.as_tensor(value, dtype=torch.get_default_dtype())
        if result.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        result = result.detach().clone().reshape(())
        if not torch.isfinite(result) or result < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return result

    @staticmethod
    def _unit_interval_scalar(value: float, *, name: str) -> Tensor:
        result = torch.as_tensor(value, dtype=torch.get_default_dtype())
        if result.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        result = result.detach().clone().reshape(())
        if not torch.isfinite(result) or result < 0 or result > 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return result

    def extra_repr(self) -> str:
        return (
            f"amplitude_max={self.amplitude_max.item():g}, "
            f"sigma_min=({self.sigma_min_input.item():g}, "
            f"{self.sigma_min_output.item():g}), "
            f"sigma_birth=({self.sigma_birth_input.item():g}, "
            f"{self.sigma_birth_output.item():g}), "
            f"sigma_max=({self.sigma_max_input.item():g}, "
            f"{self.sigma_max_output.item():g}), "
            f"w_c={self.w_c.item():g}, kappa={self.kappa.item():g}, "
            f"lower_kappa={self.lower_kappa.item():g}, "
            f"lower_half_amplitude={self.lower_half_amplitude.item():g}, "
            f"upper_decay_power={self.upper_decay_power.item():g}, "
            f"upper_floor=({self.upper_floor_input.item():g}, "
            f"{self.upper_floor_output.item():g}), "
            f"alpha_init={self.alpha_init.item():g}, "
            f"activity_gain={self.activity_gain.item():g}, "
            f"activity_mode={self.activity_mode!r}, "
            f"dormant_expansion_rate={self.dormant_expansion_rate.item():g}, "
            f"radial_regularization={self.radial_regularization.item():g}, "
            f"profile={type(self.profile).__name__}, "
            f"supports_factorization={self.supports_factorization}"
        )
