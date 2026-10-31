#pragma once
#include "moment_mask_bits.cuh"

namespace radius_cell_mass {
// This is integer execution metadata only. Original Parameter rows stay put;
// no response/gradient reduction or atom-value table is introduced.
template<class T>
__global__ void active_moment_banks_kernel(const T *p, uint32_t *required,
    int atoms, Axis axis, int start, bool merge) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bits = 0;
    if (atom < atoms) {
        bool valid = true;
        for (int column = 0; column < 4; ++column)
            valid &= isfinite(double(p[4 * atom + column]));
        double R = radius(p, atom, axis);
        valid &= isfinite(R) && R > 0 && R <= axis.period / 2 &&
            (2 * R / (axis.base_tile_sites * axis.spacing)) > 0;
        CUDA_KERNEL_ASSERT_MSG(valid, "invalid radius atom: finite coordinates and representable supported radii required");
        if (valid) {
            // Only the radius selects a bank. This neutral center's tile is
            // unused; classification and its radius/period convention agree
            // with the unchanged caller's validated current-P layout.
            int bank = classify(R, 0., axis).bank;
            bits = moment_bank_dependencies(bank, start, merge);
        }
    }
    // Tail lanes carry zero but still participate: every CTA has full warps.
    // One global OR per warp replaces a contended atomic for every atom.
    for (int distance = 16; distance; distance /= 2)
        bits |= __shfl_down_sync(0xffffffffu, bits, distance);
    if ((threadIdx.x & 31) == 0 && bits) atomicOr(required, bits);
}

inline Tensor active_moment_banks(Tensor p, Config config, bool merge, int start) {
    check_atoms(p);
    Axis axis = parse_axis(config);
    checked_start(axis, start);
    c10::cuda::CUDAGuard guard(p.device());
    auto required = torch::zeros({1}, p.options().dtype(torch::kUInt32));
    if (!p.numel() || start == axis.bank_count / 2) return required;
    static_assert(THREADS % 32 == 0, "moment-mask reduction requires full warps");
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_active_moment_banks", [&] {
        active_moment_banks_kernel<scalar_t><<<(p.size(0) + THREADS - 1) / THREADS,
            THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            p.data_ptr<scalar_t>(), required.data_ptr<uint32_t>(), p.size(0), axis, start, merge);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return required;
}

inline Tensor build_compact_moments_active(Tensor signal, Tensor p, Config config,
    bool merge, int start) {
    check_atoms(p);
    Axis axis = parse_axis(config);
    check_hybrid_signal(signal, p, axis);
    int nodes = axis.node_count - checked_start(axis, start);
    TORCH_CHECK(!nodes || signal.size(0) <= (INT_MAX - THREADS) / (nodes * axis.basis_count()),
        "moment work exceeds native index limit");
    auto required = active_moment_banks(p, config, merge, start);
    // Producer, level launches and consumers share one stream. Releasing this
    // four-byte Tensor after enqueue is safe under the stream-ordered allocator.
    // The global Z shape stays unchanged; inactive bank values are never read.
    return build_compact_moments_impl(signal, config, merge, start,
        required.data_ptr<uint32_t>());
}
} // namespace radius_cell_mass
