"""Direct amplitude coordinates with an explicit bandwidth activity state."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from torchcst.geometry import Chart
from torchcst.geometry.lazy_chart import StripChart
from torchcst.profiling import cst_span

from .base import AtomInit, Kernel, Profile
from .gaussian import Gaussian


class DirectAmpWidth(Kernel):
    r"""A rank-one atom with direct amplitude and activity coordinates.

    The first two coordinates of each atom are ``(w, q)``.  They encode

    .. math::

        w \in [-W, W], \qquad
        \alpha = \operatorname{clamp}((q-1)/3, 0, 1).

    ``w`` is the task-loss degree of freedom and therefore receives Adam
    moments directly. ``q`` is a persistent activity state rather than a
    task-loss degree of freedom. An accepted physical amplitude displacement
    grows the state by

    .. math::

        \delta^2 = q(\Delta w/W)^2, \qquad q_{\rm geom}=q+\delta^2.

    The kernel then takes an ordinary gradient step on
    :math:`R(q)=\lambda(q-1)^2/2`. Polar activity gain, time-energy scaling,
    and dormant expansion are not part of this kernel.
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
        radial_regularization: float = 0.1,
        profile: Profile | None = None,
        site_chunk: int = 2048,
        atom_chunk: int = 64,
        checkpoint_blocks: bool = True,
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
        self.register_buffer("radial_regularization", regularization)
        if (
            type(site_chunk) is not int
            or site_chunk < 1
            or type(atom_chunk) is not int
            or atom_chunk < 1
        ):
            raise ValueError("chunk sizes must be positive integers")
        self.site_chunk = site_chunk
        self.atom_chunk = atom_chunk
        self.checkpoint_blocks = checkpoint_blocks

    def get_extra_state(self) -> dict[str, object]:
        """Record the direct atom layout and profile settings in checkpoints."""

        return {
            "format_version": 2,
            "coordinate_system": "direct",
            "profile": f"{type(self.profile).__module__}.{type(self.profile).__qualname__}",
            "normalize_columns": getattr(self.profile, "normalize_columns", None),
        }

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("DirectAmpWidth checkpoint contract differs from this kernel")

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
        if marker not in state_dict:
            error_msgs.append(
                f"{prefix[:-1]}: untagged amplitude-bandwidth checkpoint "
                "has ambiguous atom coordinates; identify and migrate its "
                "Polar or Direct format explicitly"
            )
            return
        expected = self.get_extra_state()
        previous = {**expected, "format_version": 1, "activity_mode": None}
        if state_dict[marker] not in (expected, previous):
            error_msgs.append(
                f"{prefix[:-1]}: checkpoint contract "
                f"{state_dict[marker]!r} does not match {expected!r}"
            )
            return
        if state_dict[marker] == previous:
            # The previous Direct implementation inherited these unused Polar
            # buffers. Their values never influenced a Direct update.
            state_dict.pop(prefix + "activity_gain", None)
            state_dict.pop(prefix + "dormant_expansion_rate", None)
            state_dict[marker] = expected
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

    def parameter_dim(
        self, input_chart: Chart, output_chart: Chart | None = None
    ) -> int:
        if output_chart is None:
            self._check_single_chart(input_chart)
            return 2 + self.profile.parameter_dim(input_chart)
        return (
            2
            + self.profile.parameter_dim(input_chart)
            + self.profile.parameter_dim(output_chart)
        )

    def parameter_dof(
        self, input_chart: Chart, output_chart: Chart | None = None
    ) -> int:
        if output_chart is None:
            self._check_single_chart(input_chart)
            return 2 + self.profile.parameter_dof(input_chart)
        return (
            2
            + self.profile.parameter_dof(input_chart)
            + self.profile.parameter_dof(output_chart)
        )

    def initialize(
        self,
        input_chart: Chart,
        output_chart: Chart | int,
        atoms: int | None = None,
        *,
        mode: AtomInit,
    ) -> Tensor:
        if mode not in ("balanced", "uniform"):
            raise ValueError("mode must be 'balanced' or 'uniform'")
        if atoms is None:
            if type(output_chart) is not int:
                raise TypeError("single-chart initialization requires an atom count")
            self._check_single_chart(input_chart)
            atoms = output_chart
            center = self.profile.initialize(input_chart, atoms, mode=mode)
            amplitude = center.new_empty(atoms).normal_(
                mean=0.0, std=0.1 / math.sqrt(atoms)
            )
            maximum = self.amplitude_max.to(amplitude)
            amplitude = amplitude.clamp(-maximum, maximum)
            q = torch.ones_like(amplitude) + 3.0 * self.alpha_init.to(amplitude)
            return torch.cat((torch.stack((amplitude, q), dim=-1), center), dim=-1)
        if not isinstance(output_chart, Chart):
            raise TypeError("output_chart must be a Chart")
        input_p = self.profile.initialize(input_chart, atoms, mode="uniform")
        output_p = self.profile.initialize(output_chart, atoms, mode=mode)

        amplitude = input_p.new_empty(atoms)
        amplitude.normal_(mean=0.0, std=0.1 / math.sqrt(atoms))
        maximum = self.amplitude_max.to(amplitude)
        amplitude = amplitude.clamp(-maximum, maximum)
        q = torch.ones_like(amplitude) + 3.0 * self.alpha_init.to(amplitude)
        direct = torch.stack((amplitude, q), dim=-1)
        return torch.cat((direct, input_p, output_p), dim=-1)

    def materialize_atoms(
        self,
        input_chart: Chart,
        output_chart: Chart | Tensor,
        p: Tensor | None = None,
    ) -> Tensor:
        if p is None:
            if not isinstance(output_chart, Tensor):
                raise TypeError("single-chart materialization requires atom parameters")
            return self._single_materialize_atoms(input_chart, output_chart)
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
        with cst_span("cst.kernel.bandwidth"):
            direct, input_p, output_p = self._split(input_chart, output_chart, p)
            amplitude, alpha = self._amplitude_and_alpha(direct)
            sigma_input, sigma_output = self._bandwidth_sigmas(amplitude, alpha)
            # Width is derived from update history, not task-loss gradients.
            precision_input = sigma_input.reciprocal().square().detach()
            precision_output = sigma_output.reciprocal().square().detach()
        with cst_span("cst.kernel.input_profile"):
            phi_input = self.profile.evaluate_with_precision(
                input_chart, input_p, precision_input
            )
        with cst_span("cst.kernel.output_profile"):
            phi_output = self.profile.evaluate_with_precision(
                output_chart, output_p, precision_output
            )
        with cst_span("cst.kernel.amplitude_scale"):
            return phi_input, phi_output * amplitude.unsqueeze(0)

    def amplitude(
        self,
        input_chart: Chart,
        output_chart: Chart | Tensor,
        p: Tensor | None = None,
    ) -> Tensor:
        """Return the bounded signed amplitude represented by each atom."""

        if p is None:
            if not isinstance(output_chart, Tensor):
                raise TypeError("single-chart amplitude requires atom parameters")
            direct, _ = self._single_split(input_chart, output_chart)
        else:
            direct, _, _ = self._split(input_chart, output_chart, p)
        amplitude, _ = self._amplitude_and_alpha(direct)
        return amplitude

    def bandwidth_alpha(
        self,
        input_chart: Chart,
        output_chart: Chart | Tensor,
        p: Tensor | None = None,
    ) -> Tensor:
        """Return the activity interpolation coordinate in ``[0, 1]``."""

        if p is None:
            if not isinstance(output_chart, Tensor):
                raise TypeError("single-chart bandwidth requires atom parameters")
            direct, _ = self._single_split(input_chart, output_chart)
        else:
            direct, _, _ = self._split(input_chart, output_chart, p)
        _, alpha = self._amplitude_and_alpha(direct)
        return alpha

    def bandwidth_bounds(
        self,
        input_chart: Chart,
        output_chart: Chart | Tensor,
        p: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(L(w), U(w))`` for diagnostic use."""

        if p is None:
            if not isinstance(output_chart, Tensor):
                raise TypeError("single-chart bandwidth requires atom parameters")
            direct, _ = self._single_split(input_chart, output_chart)
            amplitude, _ = self._amplitude_and_alpha(direct)
            _, lower, upper = self._sigma_bounds(amplitude)
            return lower, upper
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

        direct, _, _ = self._split(input_chart, output_chart, p)
        amplitude, _ = self._amplitude_and_alpha(direct)
        _, lower_input, upper_input = self._sigma_bounds(amplitude, side="input")
        _, lower_output, upper_output = self._sigma_bounds(amplitude, side="output")
        return (lower_input, upper_input), (lower_output, upper_output)

    def bandwidth_sigma(
        self,
        input_chart: Chart,
        output_chart: Chart | Tensor,
        p: Tensor | None = None,
    ) -> Tensor:
        """Return the shared effective input/output sigma for each atom."""

        if p is None:
            if not isinstance(output_chart, Tensor):
                raise TypeError("single-chart bandwidth requires atom parameters")
            direct, _ = self._single_split(input_chart, output_chart)
            amplitude, alpha = self._amplitude_and_alpha(direct)
            sigma, _, _ = self._sigma_bounds(amplitude, alpha)
            return sigma
        sigma_input, _ = self.bandwidth_sigmas(input_chart, output_chart, p)
        self._require_shared_bandwidths()
        return sigma_input

    def bandwidth_sigmas(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return effective ``(sigma_input, sigma_output)`` for each atom."""

        direct, _, _ = self._split(input_chart, output_chart, p)
        amplitude, alpha = self._amplitude_and_alpha(direct)
        return self._bandwidth_sigmas(amplitude, alpha)

    def bandwidth_precision(
        self,
        input_chart: Chart,
        output_chart: Chart | Tensor,
        p: Tensor | None = None,
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
        output_chart: Chart | Tensor,
        p: Tensor,
        displacement: Tensor | None = None,
        *,
        step_size: float,
    ) -> Tensor:
        """Apply a direct-amplitude proposal and advance its activity state."""

        if displacement is None:
            if not isinstance(output_chart, Tensor):
                raise TypeError("single-chart update requires atom parameters")
            return self._single_apply_update(input_chart, output_chart, p, step_size)
        direct, input_p, output_p = self._split(input_chart, output_chart, p)
        if displacement.shape != p.shape:
            raise ValueError("displacement must match the atom parameter shape")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be finite and positive")

        maximum = self.amplitude_max.to(p)
        old_w = direct[:, 0].clamp(-maximum, maximum)
        old_q = direct[:, 1].clamp(1.0, 4.0)
        accepted_w = (old_w + displacement[:, 0]).clamp(-maximum, maximum)

        accepted_delta = (accepted_w - old_w) / maximum
        delta_square = old_q * accepted_delta.square()
        geometric_q = (old_q + delta_square).clamp(1.0, 4.0)

        # One ordinary gradient step for R(q)=lambda/2*(q-1)^2. The clamp
        # retains the state contract even for an unusually large step size.
        regularized_q = (
            geometric_q
            - self.radial_regularization.to(p)
            * p.new_tensor(step_size)
            * (geometric_q - 1.0)
        ).clamp(1.0, 4.0)

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
        updated_direct = torch.stack((accepted_w, regularized_q), dim=-1)
        return torch.cat((updated_direct, updated_input, updated_output), dim=-1)

    def project_parameter_gradient(
        self,
        input_chart: Chart,
        output_chart: Chart | Tensor,
        p: Tensor,
        gradient: Tensor | None = None,
    ) -> Tensor:
        if gradient is None:
            if not isinstance(output_chart, Tensor):
                raise TypeError("single-chart projection requires atom parameters")
            direct, center = self._single_split(input_chart, output_chart)
            direct_g, center_g = self._single_split(input_chart, p)
            return torch.cat(
                (
                    torch.stack(
                        (direct_g[:, 0], torch.zeros_like(direct_g[:, 1])), dim=-1
                    ),
                    self.profile.project_gradient(input_chart, center, center_g),
                ),
                dim=-1,
            )
        direct, input_p, output_p = self._split(input_chart, output_chart, p)
        del direct
        direct_g, input_g, output_g = self._split(
            input_chart,
            output_chart,
            gradient,
        )
        projected_direct = torch.stack(
            (direct_g[:, 0], torch.zeros_like(direct_g[:, 1])),
            dim=-1,
        )
        return torch.cat(
            (
                projected_direct,
                self.profile.project_gradient(input_chart, input_p, input_g),
                self.profile.project_gradient(output_chart, output_p, output_g),
            ),
            dim=-1,
        )

    def transport_parameter_state(
        self,
        input_chart: Chart,
        output_chart: Chart | Tensor,
        old: Tensor,
        new: Tensor,
        state: Tensor | None = None,
    ) -> Tensor:
        if state is None:
            if not isinstance(output_chart, Tensor):
                raise TypeError("single-chart transport requires atom parameters")
            _, old_center = self._single_split(input_chart, output_chart)
            _, new_center = self._single_split(input_chart, old)
            direct_state, center_state = self._single_split(input_chart, new)
            return torch.cat(
                (
                    direct_state,
                    self.profile.transport_state(
                        input_chart, old_center, new_center, center_state
                    ),
                ),
                dim=-1,
            )
        _, old_i, old_o = self._split(input_chart, output_chart, old)
        _, new_i, new_o = self._split(input_chart, output_chart, new)
        direct_s, state_i, state_o = self._split(input_chart, output_chart, state)
        return torch.cat(
            (
                direct_s,
                self.profile.transport_state(input_chart, old_i, new_i, state_i),
                self.profile.transport_state(output_chart, old_o, new_o, state_o),
            ),
            dim=-1,
        )

    def _check_single_chart(self, chart: Chart) -> None:
        if not isinstance(chart, Chart) or len(chart.shape) != 2:
            raise TypeError("single-chart DirectAmpWidth requires a two-dimensional Chart")
        if getattr(self.profile, "normalize_columns", True):
            raise ValueError(
                "single-chart DirectAmpWidth requires an unnormalized Profile"
            )
        if (
            type(self.profile).evaluate_with_precision_slice
            is Profile.evaluate_with_precision_slice
        ):
            raise TypeError("single-chart Profile must support sliced evaluation")
        self._require_shared_bandwidths()
        if isinstance(chart, StripChart):
            chart.validate_support(float(self.sigma_max_input))

    def _single_split(self, chart: Chart, p: Tensor) -> tuple[Tensor, Tensor]:
        self._check_single_chart(chart)
        expected_dim = 2 + self.profile.parameter_dim(chart)
        if p.ndim != 2 or p.shape[1] != expected_dim:
            raise ValueError(f"p must have shape [atoms, {expected_dim}]")
        return p[:, :2], p[:, 2:]

    def _single_values(
        self, chart: Chart, p: Tensor, selection: slice | Tensor
    ) -> Tensor:
        direct, center = self._single_split(chart, p)
        amplitude, alpha = self._amplitude_and_alpha(direct)
        sigma, _, _ = self._sigma_bounds(amplitude, alpha)
        precision = sigma.reciprocal().square().detach()
        values = self.profile.evaluate_with_precision_slice(
            chart, center, precision, selection
        )
        return values * amplitude.unsqueeze(0)

    def _single_block(
        self, chart: Chart, p: Tensor, selection: slice | Tensor
    ) -> Tensor:
        parts = []
        for start in range(0, p.shape[0], self.atom_chunk):
            selected = p[start : start + self.atom_chunk]
            if (
                self.checkpoint_blocks
                and torch.is_grad_enabled()
                and selected.requires_grad
            ):
                values = checkpoint(
                    lambda x: self._single_values(chart, x, selection),
                    selected,
                    use_reentrant=False,
                )
            else:
                values = self._single_values(chart, selected, selection)
            parts.append(values.sum(dim=-1))
        return torch.stack(parts).sum(dim=0)

    def weight(self, chart: Chart, p: Tensor) -> Tensor:
        """Sum single-chart atoms with bounded site and atom temporaries."""

        self._single_split(chart, p)
        blocks = [
            self._single_block(
                chart, p, slice(start, min(start + self.site_chunk, chart.features))
            )
            for start in range(0, chart.features, self.site_chunk)
        ]
        return torch.cat(blocks).reshape(chart.shape)

    def _single_materialize_atoms(self, chart: Chart, p: Tensor) -> Tensor:
        self._single_split(chart, p)
        blocks = [
            self._single_values(
                chart, p, slice(start, min(start + self.site_chunk, chart.features))
            )
            for start in range(0, chart.features, self.site_chunk)
        ]
        return torch.cat(blocks, dim=0).transpose(0, 1).reshape(p.shape[0], *chart.shape)

    def packed_weight(self, chart: StripChart, p: Tensor) -> Tensor:
        """Return physically ordered, contiguous tile-major weight storage."""

        if not isinstance(chart, StripChart):
            raise TypeError("packed_weight requires a StripChart")
        self._single_split(chart, p)
        tile_size = math.prod(chart.tile_shape)
        tiles = []
        for station in range(chart.tile_count):
            logical, local = chart.tile_indices(station)
            values = self._single_block(chart, p, logical)
            tile = values.new_zeros(tile_size).index_copy(0, local, values)
            tiles.append(tile.reshape(chart.tile_shape))
        return torch.stack(tiles)

    def _single_apply_update(
        self, chart: Chart, p: Tensor, displacement: Tensor, step_size: float
    ) -> Tensor:
        direct, center = self._single_split(chart, p)
        if displacement.shape != p.shape:
            raise ValueError("displacement must match the atom parameter shape")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be finite and positive")
        maximum = self.amplitude_max.to(p)
        old_w = direct[:, 0].clamp(-maximum, maximum)
        old_q = direct[:, 1].clamp(1.0, 4.0)
        accepted_w = (old_w + displacement[:, 0]).clamp(-maximum, maximum)
        delta_square = old_q * ((accepted_w - old_w) / maximum).square()
        geometric_q = (old_q + delta_square).clamp(1.0, 4.0)
        regularized_q = (
            geometric_q
            - self.radial_regularization.to(p)
            * p.new_tensor(step_size)
            * (geometric_q - 1.0)
        ).clamp(1.0, 4.0)
        updated_center = self.profile.apply_parameter_update(
            chart, center, displacement[:, 2:]
        )
        return torch.cat(
            (torch.stack((accepted_w, regularized_q), dim=-1), updated_center),
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

    def _amplitude_and_alpha(self, direct: Tensor) -> tuple[Tensor, Tensor]:
        maximum = self.amplitude_max.to(direct)
        amplitude = direct[:, 0].clamp(-maximum, maximum)
        q = direct[:, 1].clamp(1.0, 4.0)
        alpha = (q - 1.0) / 3.0
        return amplitude, alpha

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
        # Independent lower decay and upper decay can otherwise cross. The
        # effective upper envelope must never narrow below the lower curve.
        upper = torch.maximum(upper, lower)
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
            f"radial_regularization={self.radial_regularization.item():g}, "
            f"profile={type(self.profile).__name__}, "
            f"supports_factorization={self.supports_factorization}"
        )
