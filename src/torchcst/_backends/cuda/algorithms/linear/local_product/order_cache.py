"""Model-owned exact permutation reuse; values/support remain per-call snapshots."""

import torch
from torch import nn


def cache_buffer_dtypes(atoms, domain, recipe):
    """Bound every logical key and canonical ID, including the signed limits."""
    groups = (max(domain.input_count, domain.output_count) + 15) // 16
    logical_limit = (groups + 5) * (max(domain.input_count, domain.output_count) + 1)
    if recipe.compact_cached_order:
        key_dtype = (
            torch.int16
            if logical_limit < 32768
            else torch.int32
            if logical_limit < 2147483647
            else torch.int64
        )
        order_dtype = torch.int16 if atoms <= 32768 else torch.int32
    else:
        key_dtype = (
            torch.int32
            if recipe.compact_order_key and logical_limit * atoms < 2147483647
            else torch.int64
        )
        order_dtype = torch.int32
    return key_dtype, order_dtype


class OrderKeyCache(nn.Module):
    """Serialize refreshes on the model stream; never save mutable cache in autograd.

    A changed canonical composite key rebuilds that direction's entire order.
    Unchanged keys reuse IDs/offsets only. Payloads, normalizers and exact support
    envelopes are still refreshed, including changes to support ends alone.
    """

    def __init__(self, parameters, value, domain, recipe):
        super().__init__()
        import triton as tr

        from .executor import prepare_metadata
        from .preparation import polar_scalars

        if not recipe.cached_order or not recipe.parallel_owner_ranges:
            raise ValueError("key cache requires fresh parallel owner ranges")
        self.atoms, self.domain, self.recipe = len(parameters), domain, recipe
        self.stride = (
            max(tr.cdiv(domain.input_count, 16), tr.cdiv(domain.output_count, 16)) + 5
        )
        key_dtype, order_dtype = cache_buffer_dtypes(self.atoms, domain, recipe)
        self.register_buffer(
            "keys",
            torch.full((2, self.atoms), -1, device=parameters.device, dtype=key_dtype),
            persistent=False,
        )
        self.register_buffer(
            "order",
            torch.empty((2, self.atoms), device=parameters.device, dtype=order_dtype),
            persistent=False,
        )
        self.register_buffer(
            "offsets",
            torch.empty((2, self.stride), device=parameters.device, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "stats",
            torch.zeros((2, 3), device=parameters.device, dtype=torch.int64),
            persistent=False,
        )
        packed = prepare_metadata(
            parameters,
            domain,
            sparse=True,
            scalars=polar_scalars(value),
            support_bounded=True,
        )
        self.refresh(packed)
        self.stats.zero_()

    def refresh(self, packed):
        from .executor import ordered_layout

        if packed.shape != (13, self.atoms) or packed.device != self.keys.device:
            raise ValueError("cached order atom shape/device changed")
        return ordered_layout(packed, self.domain, self.recipe, cache=self)

    def report(self):
        counters = self.stats.cpu().tolist()
        return {
            "kind": "exact-order-key-cache",
            "canonical_atoms": self.atoms,
            "slot_capacity_per_view": self.atoms,
            "key_bits": self.keys.element_size() * 8,
            "key_encoding": "bucket-position"
            if self.recipe.compact_cached_order
            else "canonical-composite",
            "order_bits": self.order.element_size() * 8,
            "counter_scope": "since initialization; includes warmup/capture/replay/diagnostics",
            "counter_names": ["refreshes", "rebuilds", "reuses"],
            "forward": counters[0],
            "dx": counters[1],
        }
