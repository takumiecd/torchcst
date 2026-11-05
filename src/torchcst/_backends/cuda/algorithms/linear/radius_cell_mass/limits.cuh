#pragma once
#include <cstdint>
#include <limits>

namespace radius_cell_mass {
constexpr int MAX_SITES = 16384;
constexpr int MAX_ATOMS = 1 << 24;
constexpr int MAX_OWNER_NODES = 4 * MAX_SITES - 2;
constexpr int SHARED_MAX_SITES = 128;
constexpr int SHARED_MAX_ATOMS = 4096;
constexpr int BLOCK_LAYOUT_LIMIT = 1024;
constexpr unsigned OWNER_ROW_BITS = 12;
constexpr unsigned OWNER_ROW_MASK = (1u << OWNER_ROW_BITS) - 1;

// The owner-window research path retains its M8/base1/merge2/B4 contract.
// Wider grids are explicit cohorts, never permission for arbitrary capture.
constexpr bool owner_window_sites(int sites) {
    return sites >= 1024 && sites <= MAX_SITES && (sites & (sites - 1)) == 0;
}
constexpr int64_t owner_window_cohort_atoms(int sites) {
    return int64_t(sites) * sites / 20;
}

// Separate finite HIGGS density study. Existing MNIST/private 5% entrances
// retain owner_window_cohort_atoms; public execution still rejects capture.
constexpr bool higgs_owner_window_cohort(int sites, int64_t atoms) {
    if (!owner_window_sites(sites) || sites > 8192) return false;
    const int64_t area = int64_t(sites) * sites;
    return atoms == area / 200 || atoms == area / 100 || atoms == area / 50 ||
        atoms == area / 20 || atoms == area / 10;
}

// Low-density confirmation has its own finite permission; it does not widen
// either the old HIGGS exploratory path or the MNIST five-percent entrance.
constexpr bool higgs_confirmation_owner_window_cohort(int sites, int64_t atoms) {
    if (sites != 1024 && sites != 2048) return false;
    const int64_t area = int64_t(sites) * sites;
    return atoms == area / 1000 || atoms == area / 500 ||
        atoms == area / 200 || atoms == area / 100;
}

// The final owner boundary is key_count << OWNER_ROW_BITS. It must also fit:
// UINT_MAX is reserved for padding, never a wrapped valid owner/row key.
constexpr bool can_use_block_layout(int64_t atoms, int input_nodes, int output_nodes) {
    return atoms >= 0 && atoms <= BLOCK_LAYOUT_LIMIT && input_nodes > 0 && output_nodes > 0 &&
        uint64_t(input_nodes) * uint64_t(output_nodes) <=
            (std::numeric_limits<uint32_t>::max() >> OWNER_ROW_BITS);
}
} // namespace radius_cell_mass
