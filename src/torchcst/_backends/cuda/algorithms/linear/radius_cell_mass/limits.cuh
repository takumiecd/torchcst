#pragma once
#include <cstdint>
#include <limits>

namespace radius_cell_mass {
constexpr int MAX_SITES = 1024;
constexpr int MAX_ATOMS = 65536;
constexpr int MAX_OWNER_NODES = 4 * MAX_SITES - 2;
constexpr int SHARED_MAX_SITES = 128;
constexpr int SHARED_MAX_ATOMS = 4096;
constexpr int BLOCK_LAYOUT_LIMIT = 1024;
constexpr unsigned OWNER_ROW_BITS = 12;
constexpr unsigned OWNER_ROW_MASK = (1u << OWNER_ROW_BITS) - 1;

// The final owner boundary is key_count << OWNER_ROW_BITS. It must also fit:
// UINT_MAX is reserved for padding, never a wrapped valid owner/row key.
constexpr bool can_use_block_layout(int64_t atoms, int input_nodes, int output_nodes) {
    return atoms >= 0 && atoms <= BLOCK_LAYOUT_LIMIT && input_nodes > 0 && output_nodes > 0 &&
        uint64_t(input_nodes) * uint64_t(output_nodes) <=
            (std::numeric_limits<uint32_t>::max() >> OWNER_ROW_BITS);
}
} // namespace radius_cell_mass
