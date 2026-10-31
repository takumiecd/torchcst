#pragma once
#include "common.cuh"

namespace radius_cell_mass {
// Small by-value launch metadata, never a device allocation/table. Each bank
// keeps exactly its meaningful K parts and direct-sites or Fourier features.
struct OwnerPartialLayout {
    int offsets[32] = {};
    int fields[32] = {};
    int total_fields = 0;
};
inline OwnerPartialLayout compact_owner_partial_layout(const Axis &axis, int start) {
    TORCH_CHECK(start >= 0 && start <= axis.bank_count / 2, "invalid compact owner cutoff");
    OwnerPartialLayout layout;
    int64_t total = 0;
    for (int bi = 0; bi < axis.bank_count; ++bi) {
        const Bank &bank = axis.banks[bi];
        layout.offsets[bi] = int(total);
        layout.fields[bi] = bank.level < start ? bank.tile_sites : axis.basis_count();
        total += int64_t(bank.tile_count) * bank.count * layout.fields[bi];
        TORCH_CHECK(total <= INT_MAX, "compact owner partial index limit");
    }
    layout.total_fields = int(total);
    return layout;
}
__device__ inline int64_t compact_owner_partial_index(int b, int owner, int bi,
    int part, int field, const Axis &axis, const OwnerPartialLayout &layout) {
    const Bank &bank = axis.banks[bi];
    return int64_t(b) * layout.total_fields + layout.offsets[bi] +
        ((owner - bank.node_offset) * bank.count + part) * layout.fields[bi] + field;
}
} // namespace radius_cell_mass
