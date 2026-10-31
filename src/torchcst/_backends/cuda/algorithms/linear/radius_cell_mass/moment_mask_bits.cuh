#pragma once
#include <cstdint>

namespace radius_cell_mass {
// Two adjacent banks at each level have K=2 and K=3. Moving down by two
// preserves K, exactly matching compact_moment_element's child dependency.
// Invalid metadata returns no bits instead of permitting an undefined shift.
#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr uint32_t moment_bank_dependencies(int selected_bank, int start, bool merge) {
    if (selected_bank < 0 || selected_bank >= 32 || start < 0 || start > 16) return 0;
    uint32_t bits = 0;
    for (int bank = selected_bank; bank >= 2 * start; bank -= 2) {
        bits |= uint32_t(1) << bank;
        if (!merge) break;
    }
    return bits;
}
} // namespace radius_cell_mass
