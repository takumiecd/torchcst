"""Fresh atom preparation for Strip + Torus CUDA execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from torchcst._backends.torch.operators.strip_torus.layout import _station_layout
from torchcst._backends.torch.operators.strip_torus.preparation import execution_plan
from torchcst.kernels import DirectAmpWidth
from torchcst.profiling import cst_span

if TYPE_CHECKING:
    from torchcst.nn.linear import CSTLinear


def prepare(
    site: CSTLinear, p: Tensor, *, use_triton: bool = True, support_layout: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    plan = execution_plan(site)
    if p.ndim != 2 or p.shape[1] != plan.parameter_dim:
        raise ValueError(f"p must have shape [atoms, {plan.parameter_dim}]")
    fused = (
        use_triton and p.is_cuda and p.dtype == torch.float32 and not torch.version.hip
    )
    with cst_span("cst.linear.prepare_atoms"):
        # Preserve overrides on custom kernels; the built-in evaluator can
        # reuse the plan's configuration validation without GPU/CPU sync.
        if fused and type(site.kernel) is DirectAmpWidth:
            from torchcst._backends.cuda.algorithms.strip_torus.fused.preparation import (
                tile_parameters,
            )

            center, amplitude, precision = tile_parameters(site.kernel, p)
        elif type(site.kernel).tile_parameters is DirectAmpWidth.tile_parameters:
            center, amplitude, precision = site.kernel._tile_parameters(p)
        else:
            center, amplitude, precision = site.kernel.tile_parameters(site.chart, p)
        decoded = site.chart.geometry.decode_centers(center)
    if (
        fused
        and decoded.is_cuda
        and decoded.dtype == torch.float32
        and amplitude.dtype == torch.float32
        and precision.dtype == torch.float32
    ):
        from torchcst._backends.cuda.algorithms.strip_torus.fused.preparation import (
            Pack,
            route_and_layout,
        )

        with cst_span("cst.linear.route_atoms"):
            _, order, offsets = route_and_layout(
                plan.routing,
                decoded,
                support=(plan.circle, plan.section, precision, site.chart.tile_shape[0])
                if support_layout
                else None,
                retain_owners=False,
            )
        with cst_span("cst.linear.pack_atoms"):
            packed = Pack.apply(amplitude, precision, decoded, order)
        return packed, plan.circle, plan.section, offsets
    with cst_span("cst.linear.route_atoms"):
        if support_layout:
            from torchcst._backends.torch.operators.strip_torus.support import (
                support_layout as classify,
            )

            layout = classify(plan, decoded, precision, site.chart.tile_shape[0])
        else:
            owners = plan.routing.owners(decoded)
            layout = _station_layout(owners, site.chart.tile_count)
    with cst_span("cst.linear.pack_atoms"):
        packed = layout.pack(
            torch.cat((amplitude[:, None], precision[:, None], decoded), dim=-1)
        )
    return packed, plan.circle, plan.section, layout.offsets
