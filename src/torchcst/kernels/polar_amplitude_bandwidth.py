"""Polar amplitude/width coordinates with radial-angular decoupling."""

from __future__ import annotations

import math
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
                or getattr(self.profile, "normalize_columns", True) is not True
            ):
                error_msgs.append(
                    f"{prefix[:-1]}: untagged amplitude-bandwidth checkpoint "
                    "has ambiguous atom coordinates; identify and migrate its "
                    "Polar or Direct format explicitly"
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

        return self.w_c / self.lower_kappa.sqrt()

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
        if mode not in ("balanced", "uniform"):
            raise ValueError("mode must be 'balanced' or 'uniform'")
        input_p = self.profile.initialize(input_chart, atoms, mode="uniform")
        output_p = self.profile.initialize(output_chart, atoms, mode=mode)

        amplitude = input_p.new_empty(atoms)
        amplitude.normal_(mean=0.0, std=0.1 / math.sqrt(atoms))
        maximum = self.amplitude_max.to(amplitude)
        # Keep initialization away from the angular critical points w = +/- W.
        ratio = (amplitude / maximum).clamp(-1 + 1e-6, 1 - 1e-6)
        polar = torch.stack((ratio, (1 - ratio.square()).sqrt()), dim=-1)
        # Preserve the initialized angular amplitude while giving the
        # bandwidth clock an optional, regularized exploration reserve.
        initial_radius = (1.0 + 3.0 * self.alpha_init.to(polar)).sqrt()
        polar = polar * initial_radius
        return torch.cat((polar, input_p, output_p), dim=-1)

    def materialize_atoms(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        phi_input, phi_output = self.factors(input_chart, output_chart, p)
        return torch.einsum("oa,ia->aoi", phi_output, phi_input)

    @property
    def supports_factorization(self) -> bool:
        return True

    def factors(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor]:
        polar, input_p, output_p = self._split(input_chart, output_chart, p)
        amplitude, alpha = self._amplitude_and_alpha(polar)
        sigma_input, sigma_output = self._bandwidth_sigmas(amplitude, alpha)
        # Width is a state derived from update history, not a task-loss degree
        # of freedom. Only the explicit radial regularizer may decrease alpha.
        precision_input = sigma_input.reciprocal().square().detach()
        precision_output = sigma_output.reciprocal().square().detach()
        phi_input = self.profile.evaluate_with_precision(
            input_chart, input_p, precision_input
        )
        phi_output = self.profile.evaluate_with_precision(
            output_chart, output_p, precision_output
        )
        return phi_input, phi_output * amplitude.unsqueeze(0)

    def amplitude(self, input_chart: Chart, output_chart: Chart, p: Tensor) -> Tensor:
        """Return the bounded signed amplitude represented by each atom."""

        polar, _, _ = self._split(input_chart, output_chart, p)
        amplitude, _ = self._amplitude_and_alpha(polar)
        return amplitude

    def bandwidth_alpha(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> Tensor:
        """Return the radial interpolation coordinate in ``[0, 1]``."""

        polar, _, _ = self._split(input_chart, output_chart, p)
        _, alpha = self._amplitude_and_alpha(polar)
        return alpha

    def bandwidth_bounds(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``(L(w), U(w))`` for diagnostic use."""

        (input_bounds, output_bounds) = self.bandwidth_bounds_by_side(
            input_chart, output_chart, p
        )
        self._require_shared_bandwidths()
        del output_bounds
        return input_bounds

    def bandwidth_bounds_by_side(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Return ``((L_in, U_in), (L_out, U_out))``."""

        polar, _, _ = self._split(input_chart, output_chart, p)
        amplitude, _ = self._amplitude_and_alpha(polar)
        _, lower_input, upper_input = self._sigma_bounds(amplitude, side="input")
        _, lower_output, upper_output = self._sigma_bounds(amplitude, side="output")
        return (lower_input, upper_input), (lower_output, upper_output)

    def bandwidth_sigma(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> Tensor:
        """Return the shared effective input/output sigma for each atom."""

        sigma_input, _ = self.bandwidth_sigmas(input_chart, output_chart, p)
        self._require_shared_bandwidths()
        return sigma_input

    def bandwidth_sigmas(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return effective ``(sigma_input, sigma_output)`` for each atom."""

        polar, _, _ = self._split(input_chart, output_chart, p)
        amplitude, alpha = self._amplitude_and_alpha(polar)
        return self._bandwidth_sigmas(amplitude, alpha)

    def bandwidth_precision(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> Tensor:
        """Return the shared input/output precision for each atom."""

        sigma = self.bandwidth_sigma(input_chart, output_chart, p)
        return sigma.reciprocal().square()

    def bandwidth_precisions(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return input and output precisions for split bandwidths."""

        sigma_input, sigma_output = self.bandwidth_sigmas(input_chart, output_chart, p)
        return sigma_input.reciprocal().square(), sigma_output.reciprocal().square()

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

        self._split(input_chart, output_chart, p)
        if displacement.shape != p.shape:
            raise ValueError("displacement must match the atom parameter shape")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be finite and positive")

        polar = self._project_polar(p[:, :2])
        raw = displacement[:, :2]
        radius_square = polar.square().sum(dim=-1, keepdim=True)
        radial_coefficient = (raw * polar).sum(dim=-1, keepdim=True) / radius_square
        tangent = raw - radial_coefficient * polar

        # Preserve the optimizer's angular proposal while allowing the radial
        # history clock to be calibrated independently. ``finite_chord`` is
        # the original q' = q + gamma ||tangent||^2 rule. ``time_energy``
        # interprets gamma as an activity rate and divides the squared motion
        # by the optimizer's outer time step.
        chord = polar + tangent
        chord_q = chord.square().sum(dim=-1, keepdim=True)
        direction = chord / chord_q.sqrt()
        energy = tangent.square().sum(dim=-1, keepdim=True)
        if self.activity_mode == "time_energy":
            energy = energy / step_size
        proposed_amplitude, _ = self._amplitude_and_alpha(direction)
        dormant_weight = 1.0 / (1.0 + (proposed_amplitude / self.w_c.to(p)).square())
        dormant_q = (
            3.0
            * self.dormant_expansion_rate.to(p)
            * p.new_tensor(step_size)
            * dormant_weight.unsqueeze(-1)
        )
        task_q = (radius_square + self.activity_gain.to(p) * energy + dormant_q).clamp(
            1.0, 4.0
        )
        task_polar = direction * task_q.sqrt()

        # Exact gradient flow for R(q)=lambda/2*(q-1)^2 over time step_size:
        # y=(q-1)/q decays as exp(-4*lambda*t), keeping q in [1, 4].
        decay = torch.exp(
            -4.0 * self.radial_regularization.to(p) * p.new_tensor(step_size)
        )
        activity = (task_q - 1.0) / task_q
        regularized_q = 1.0 / (1.0 - activity * decay)
        regularized_polar = task_polar * (regularized_q / task_q).sqrt()

        _, input_p, output_p = self._split(input_chart, output_chart, p)
        _, input_d, output_d = self._split(
            input_chart,
            output_chart,
            displacement,
        )
        updated_input = self.profile.apply_parameter_update(
            input_chart,
            input_p,
            input_d,
        )
        updated_output = self.profile.apply_parameter_update(
            output_chart,
            output_p,
            output_d,
        )
        return torch.cat((regularized_polar, updated_input, updated_output), dim=-1)

    def project_parameter_gradient(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        _, input_p, output_p = self._split(input_chart, output_chart, p)
        polar_g, input_g, output_g = self._split(
            input_chart,
            output_chart,
            gradient,
        )
        return torch.cat(
            (
                polar_g,
                self.profile.project_gradient(input_chart, input_p, input_g),
                self.profile.project_gradient(output_chart, output_p, output_g),
            ),
            dim=-1,
        )

    def transport_parameter_state(
        self,
        input_chart: Chart,
        output_chart: Chart,
        old: Tensor,
        new: Tensor,
        state: Tensor,
    ) -> Tensor:
        _, old_i, old_o = self._split(input_chart, output_chart, old)
        _, new_i, new_o = self._split(input_chart, output_chart, new)
        polar_s, state_i, state_o = self._split(input_chart, output_chart, state)
        return torch.cat(
            (
                polar_s,
                self.profile.transport_state(input_chart, old_i, new_i, state_i),
                self.profile.transport_state(output_chart, old_o, new_o, state_o),
            ),
            dim=-1,
        )

    def _split(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        expected_dim = self.parameter_dim(input_chart, output_chart)
        if p.ndim != 2 or p.shape[1] != expected_dim:
            raise ValueError(f"p must have shape [atoms, {expected_dim}]")
        input_dim = self.profile.parameter_dim(input_chart)
        input_end = 2 + input_dim
        return p[:, :2], p[:, 2:input_end], p[:, input_end:]

    def _amplitude_and_alpha(self, polar: Tensor) -> tuple[Tensor, Tensor]:
        radius_square = polar.square().sum(dim=-1)
        # The exact polar map is used everywhere except the singular origin.
        safe_square = radius_square.clamp_min(torch.finfo(polar.dtype).tiny)
        amplitude = self.amplitude_max.to(polar) * polar[:, 0] / safe_square.sqrt()
        alpha = ((radius_square - 1.0) / 3.0).clamp(0.0, 1.0)
        return amplitude, alpha

    @staticmethod
    def _project_polar(polar: Tensor) -> Tensor:
        radius_square = polar.square().sum(dim=-1, keepdim=True)
        tiny = torch.finfo(polar.dtype).tiny
        safe_radius = radius_square.clamp_min(tiny).sqrt()
        fallback = torch.zeros_like(polar)
        fallback[:, 1] = 1.0
        unit = torch.where(radius_square > tiny, polar / safe_radius, fallback)
        radius = safe_radius.clamp(1.0, 2.0)
        return unit * radius

    def _bandwidth_sigmas(
        self, amplitude: Tensor, alpha: Tensor
    ) -> tuple[Tensor, Tensor]:
        sigma_input, _, _ = self._sigma_bounds(amplitude, alpha, side="input")
        sigma_output, _, _ = self._sigma_bounds(amplitude, alpha, side="output")
        return sigma_input, sigma_output

    def _sigma_bounds(
        self,
        amplitude: Tensor,
        alpha: Tensor | None = None,
        *,
        side: Literal["input", "output"] = "input",
    ) -> tuple[Tensor, Tensor, Tensor]:
        if side == "input":
            minimum = self.sigma_min_input
            birth = self.sigma_birth_input
            maximum = self.sigma_max_input
            floor = self.upper_floor_input
        elif side == "output":
            minimum = self.sigma_min_output
            birth = self.sigma_birth_output
            maximum = self.sigma_max_output
            floor = self.upper_floor_output
        else:
            raise ValueError("side must be 'input' or 'output'")
        x = (amplitude / self.w_c.to(amplitude)).square()
        lower_delta = birth.to(amplitude) - minimum.to(amplitude)
        upper_delta = maximum.to(amplitude) - minimum.to(amplitude)
        kappa = self.kappa.to(amplitude)
        upper_x = x.pow(self.upper_decay_power.to(amplitude))
        upper = minimum.to(amplitude) + upper_delta * kappa / (kappa + upper_x)
        # Keep radial activity meaningful for high-amplitude atoms.  Without
        # this floor U(w) converges to sigma_min, so alpha loses all authority
        # precisely when a strong atom becomes trapped on a single site.
        upper = torch.maximum(upper, floor.to(amplitude))
        lower = minimum.to(amplitude) + lower_delta / (
            1.0 + self.lower_kappa.to(amplitude) * x
        )
        if alpha is None:
            sigma = lower
        else:
            # Width is a scale, so alpha advances a constant fraction of the
            # multiplicative range rather than a constant absolute distance.
            sigma = torch.exp((1.0 - alpha) * lower.log() + alpha * upper.log())
        return sigma, lower, upper

    def _require_shared_bandwidths(self) -> None:
        pairs = (
            (self.sigma_min_input, self.sigma_min_output),
            (self.sigma_birth_input, self.sigma_birth_output),
            (self.sigma_max_input, self.sigma_max_output),
            (self.upper_floor_input, self.upper_floor_output),
        )
        if not all(bool(torch.equal(left, right)) for left, right in pairs):
            raise ValueError("input and output bandwidths differ; use the by-side API")

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
