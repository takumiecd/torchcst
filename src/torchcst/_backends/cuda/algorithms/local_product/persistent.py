"""Model-owned topology; normalized values are refreshed independently of slots."""

import torch
from torch import nn

from .layout import LayoutSnapshot


class PersistentLayout(nn.Module):
    """Fixed-capacity bucket segments with incremental atom-ID migration.

    This owner retains topology across steps, not numerical coefficients/H.
    Calls must be serialized on the model's CUDA stream. No autograd invocation
    saves the mutable topology: refresh returns independent LayoutSnapshots.
    Initialization may synchronize; refresh/rebuild never reads device data on host.
    """

    def __init__(self, parameters, value, domain, recipe):
        super().__init__()
        import triton as tr

        from .executor import prepare_metadata, tile_layout
        from .preparation import polar_scalars

        self.atoms, self.domain, self.recipe = len(parameters), domain, recipe
        self.stride = (
            max(tr.cdiv(domain.input_count, 16), tr.cdiv(domain.output_count, 16)) + 5
        )
        self.slots = tr.cdiv(self.atoms + 31 * (self.stride - 1), 16) * 16
        packed = prepare_metadata(
            parameters,
            domain,
            sparse=True,
            scalars=polar_scalars(value),
            support_bounded=recipe.support_prepare,
        )
        compact = tile_layout(packed, domain, recipe).starts.cpu()
        starts = torch.zeros((2, self.stride), dtype=torch.int32)
        for direction, size in enumerate((domain.output_count, domain.input_count)):
            buckets = tr.cdiv(size, 16) + 4
            counts = compact[direction, 1 : buckets + 1] - compact[direction, :buckets]
            capacities = ((counts + 15) // 16) * 16 + 16
            starts[direction, 1 : buckets + 1] = capacities.cumsum(0)

        def buffer(name, shape, fill, dtype=torch.int32):
            self.register_buffer(
                name,
                torch.full(shape, fill, device=parameters.device, dtype=dtype),
                persistent=False,
            )

        buffer("ids", (2, self.slots), -1)
        buffer("reverse", (2, self.atoms), -1)
        buffer("keys", (2, self.atoms), -1)
        self.register_buffer("starts", starts.to(parameters.device), persistent=False)
        buffer("ends", (2, self.stride), 0)
        buffer("free", (2, self.slots), 0)
        buffer("stats", (2, 4), 0, torch.int64)
        self.refresh(packed)
        self.stats.zero_()  # Diagnostics distinguish configuration from timed work.

    def refresh(self, packed):
        import triton as tr

        from . import layout_kernels
        from .executor import _report

        if packed.shape != (13, self.atoms) or packed.device != self.ids.device:
            raise ValueError("persistent layout atom shape/device changed")
        snapshot = LayoutSnapshot.allocate(
            packed, slots=self.slots, stride=self.stride, persistent=True
        )
        compiled = layout_kernels.refresh[(2,)](
            packed,
            self.ids,
            self.reverse,
            self.keys,
            self.starts,
            self.ends,
            self.free,
            self.stats,
            snapshot.views,
            snapshot.orders,
            snapshot.starts,
            snapshot.ends,
            self.atoms,
            self.slots,
            self.domain.input_count,
            self.domain.output_count,
            self.domain.input_start,
            self.domain.output_start,
            self.domain.spacing,
            self.recipe.rho_upper[1],
            max(1, tr.next_power_of_2(self.atoms)),
            tr.next_power_of_2(self.slots),
            tr.next_power_of_2(self.stride),
            self.stride,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report("persistent_layout", compiled)
        return snapshot

    def report(self):
        """Host diagnostics outside the measured step."""
        counters = self.stats.cpu().tolist()
        return {
            "slot_capacity_per_view": self.slots,
            "canonical_atoms": self.atoms,
            "counter_scope": "since initialization; includes warmup/capture/replay; each direction separately",
            "counter_names": [
                "refreshes",
                "moved_atoms",
                "incremental_repairs",
                "overflow_rebuilds",
            ],
            "forward": counters[0],
            "dx": counters[1],
        }
