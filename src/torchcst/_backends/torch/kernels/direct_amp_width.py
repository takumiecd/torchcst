"""PyTorch execution for legacy direct_amplitude_bandwidth contracts."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from torchcst.geometry import Chart
from torchcst.geometry.lazy_chart import StripChart
from torchcst.kernels.base import AtomInit
from torchcst.profiling import cst_span


def lower_half_amplitude(self) -> Tensor:
    """Absolute amplitude where ``L(w)`` is halfway between its limits."""

    return self.w_c / self.lower_kappa.sqrt()


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


def _single_values(
    self,
    chart: Chart,
    center: Tensor,
    amplitude: Tensor,
    precision: Tensor,
    selection: slice | Tensor,
) -> Tensor:
    values = self.profile.evaluate_with_precision_slice(
        chart, center, precision, selection
    )
    return values * amplitude.unsqueeze(0)


def _single_block(
    self,
    chart: Chart,
    center: Tensor,
    amplitude: Tensor,
    precision: Tensor,
    selection: slice | Tensor,
) -> Tensor:
    parts = []
    for start in range(0, center.shape[0], self.atom_chunk):
        stop = start + self.atom_chunk
        selected_center = center[start:stop]
        selected_amplitude = amplitude[start:stop]
        selected_precision = precision[start:stop]
        if (
            self.checkpoint_blocks
            and torch.is_grad_enabled()
            and (selected_center.requires_grad or selected_amplitude.requires_grad)
        ):
            values = checkpoint(
                lambda c, a, prec=selected_precision: self._single_values(
                    chart, c, a, prec, selection
                ),
                selected_center,
                selected_amplitude,
                use_reentrant=False,
            )
        else:
            values = self._single_values(
                chart,
                selected_center,
                selected_amplitude,
                selected_precision,
                selection,
            )
        parts.append(values.sum(dim=-1))
    return torch.stack(parts).sum(dim=0)


def weight(self, chart: Chart, p: Tensor) -> Tensor:
    """Sum single-chart atoms with bounded site and atom temporaries."""

    center, amplitude, precision = self.tile_parameters(chart, p)
    blocks = [
        self._single_block(
            chart,
            center,
            amplitude,
            precision,
            slice(start, min(start + self.site_chunk, chart.features)),
        )
        for start in range(0, chart.features, self.site_chunk)
    ]
    return torch.cat(blocks).reshape(chart.shape)


def weight_tile(self, chart: Chart, p: Tensor, rows: Tensor, columns: Tensor) -> Tensor:
    """Sum atom contributions on one [output rows, input columns] tile.

    The tile is a temporary PyTorch tensor. A native tiled GEMM may
    evaluate the same entries directly in registers or shared memory.
    """

    if not hasattr(chart, "shape") or len(chart.shape) != 2:
        raise ValueError("weight_tile requires a two-dimensional operator chart")
    if (
        rows.ndim != 1
        or columns.ndim != 1
        or rows.dtype != torch.long
        or columns.dtype != torch.long
        or rows.device != p.device
        or columns.device != p.device
    ):
        raise ValueError("rows and columns must be long vectors on the atom device")
    if bool(((rows < 0) | (rows >= chart.shape[0])).any()) or bool(
        ((columns < 0) | (columns >= chart.shape[1])).any()
    ):
        raise IndexError("weight tile index out of bounds")
    if p.shape[0] == 0 or rows.numel() == 0 or columns.numel() == 0:
        return p.new_zeros((rows.numel(), columns.numel()))
    center, amplitude, precision = self.tile_parameters(chart, p)
    sites = (rows[:, None] * chart.shape[1] + columns[None, :]).reshape(-1)
    pieces = [
        self._single_block(
            chart, center, amplitude, precision, sites[start : start + self.site_chunk]
        )
        for start in range(0, sites.numel(), self.site_chunk)
    ]
    return torch.cat(pieces).reshape(rows.numel(), columns.numel())


def _single_materialize_atoms(self, chart: Chart, p: Tensor) -> Tensor:
    center, amplitude, precision = self.tile_parameters(chart, p)
    blocks = [
        self._single_values(
            chart,
            center,
            amplitude,
            precision,
            slice(start, min(start + self.site_chunk, chart.features)),
        )
        for start in range(0, chart.features, self.site_chunk)
    ]
    return torch.cat(blocks, dim=0).transpose(0, 1).reshape(p.shape[0], *chart.shape)


def packed_weight(self, chart: StripChart, p: Tensor) -> Tensor:
    """Return physically ordered, contiguous tile-major weight storage."""

    if not isinstance(chart, StripChart):
        raise TypeError("packed_weight requires a StripChart")
    center, amplitude, precision = self.tile_parameters(chart, p)
    tile_size = math.prod(chart.tile_shape)
    tiles = []
    for station in range(chart.tile_count):
        logical, local = chart.tile_indices(station)
        values = self._single_block(chart, center, amplitude, precision, logical)
        tile = values.new_zeros(tile_size).index_copy(0, local, values)
        tiles.append(tile.reshape(chart.tile_shape))
    return torch.stack(tiles)
