"""Diagnostic C/site timing for the unchanged N128 radius shared route.

This is not a registered execution Plan. The original fused transform remains
in use for complete-step measurements. Split probes export exactly the C slots
owned by each original CTA, preserving repeated coarse-node work and atom order.
"""

from dataclasses import dataclass

import torch


@dataclass
class ExpansionBuffers:
    coefficients: torch.Tensor
    masks: torch.Tensor
    output: torch.Tensor
    timestamps: torch.Tensor

    @property
    def payload_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (self.coefficients, self.masks, self.output, self.timestamps)
        )


def allocate_buffers(atoms, response, axis, start, tile_sites):
    """Keep all allocations outside CUDA-event intervals."""
    blocks = (axis.site_count + tile_sites - 1) // tile_sites
    elements = axis.active_nodes(start) * axis.basis_count
    batch = response.shape[0]
    return ExpansionBuffers(
        torch.empty(
            (batch, blocks, elements), dtype=torch.float64, device=atoms.device
        ),
        torch.empty((batch, blocks), dtype=torch.int32, device=atoms.device),
        torch.empty((batch, axis.site_count), dtype=atoms.dtype, device=atoms.device),
        torch.empty((batch, blocks, 4), dtype=torch.int64, device=atoms.device),
    )


def run_probe(
    native,
    atoms,
    rows,
    offsets,
    response,
    axis,
    start,
    tile_sites,
    output_side,
    buffers,
    mode,
    repeats=1,
):
    native.run_expansion_probe(
        atoms,
        rows,
        offsets,
        response,
        buffers.coefficients,
        buffers.masks,
        buffers.output,
        buffers.timestamps,
        axis.native_config(),
        start,
        tile_sites,
        output_side,
        mode,
        repeats,
    )
    return buffers.output


def timestamp_summary(timestamps):
    """CTA latency is not additive: report the last-finished CTA separately.

    The tool-only PTX globaltimer register reports nanoseconds on the measured
    L4 target. The site interval includes an added final barrier. Never sum CTA
    latencies to estimate kernel wall time or subtract these from a full step.
    """
    t = timestamps.detach().cpu().reshape(-1, 4)
    if not t.numel():
        return {"ctas": 0}
    if not bool((t[:, 1:] >= t[:, :-1]).all()):
        raise ValueError("nonmonotonic diagnostic GPU timestamps")
    durations = (t[:, 1:] - t[:, :-1]).double() / 1000
    last = int(t[:, 3].argmax())
    return {
        "ctas": len(t),
        "cta_median_us": dict(
            zip(
                ("setup", "coefficients", "site_with_barrier"),
                durations.median(dim=0).values.tolist(),
            )
        ),
        "last_finished_cta": last,
        "last_finished_cta_us": dict(
            zip(
                ("setup", "coefficients", "site_with_barrier"), durations[last].tolist()
            )
        ),
        "last_cta_start_offset_us": float((t[last, 0] - t[:, 0].min()) / 1000),
        "device_span_us": float((t[:, 3].max() - t[:, 0].min()) / 1000),
    }


class SplitExpansionNative:
    """Benchmark-only split schedule; no persistent C or duplicated response.

    All other operations, including parameter gradients, delegate unchanged.
    Each call projects once, then frees its temporary exported C after the
    consumer has been enqueued on the same stream. This is not a public Plan.
    """

    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _apply(
        self,
        signal,
        atoms,
        rows,
        offsets,
        source_config,
        dest_config,
        source_start,
        dest_start,
        merge,
        tile_sites,
        source_output_side,
    ):
        from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
            AxisPlan,
        )

        axis = AxisPlan(
            site_count=int(dest_config[0]),
            spacing=dest_config[1],
            base_tile_sites=int(dest_config[2]),
            origin=dest_config[3],
            modes=int(dest_config[4]),
            merge_factor=int(dest_config[5]),
            polar=tuple(dest_config[6:]) or None,
        )
        moments, response = self.native.shared_project_parallel(
            signal, atoms, source_config, source_output_side, source_start, merge
        )
        buffers = allocate_buffers(atoms, response, axis, dest_start, tile_sites)
        run_probe(
            self.native,
            atoms,
            rows,
            offsets,
            response,
            axis,
            dest_start,
            tile_sites,
            not source_output_side,
            buffers,
            4,
        )
        return buffers.output, moments, response

    def shared_forward_parallel(
        self,
        x,
        atoms,
        rows,
        offsets,
        input_config,
        output_config,
        input_start,
        output_start,
        merge,
        tile_sites,
    ):
        return self._apply(
            x,
            atoms,
            rows,
            offsets,
            input_config,
            output_config,
            input_start,
            output_start,
            merge,
            tile_sites,
            False,
        )

    def shared_backward_parallel(
        self,
        dy,
        atoms,
        rows,
        offsets,
        input_config,
        output_config,
        input_start,
        output_start,
        merge,
        tile_sites,
    ):
        return self._apply(
            dy,
            atoms,
            rows,
            offsets,
            output_config,
            input_config,
            output_start,
            input_start,
            merge,
            tile_sites,
            True,
        )
