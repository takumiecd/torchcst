#pragma once
#include "owner_response_coefficients.cuh"

namespace radius_cell_mass {
// Forward classification already validated all four P fields and both decoded
// radii. A key is output_node*input_nodes+input_node, never an atom-value copy.
// Backward can reuse it only after the ordinary saved-P/version/metadata checks.
__device__ inline uint32_t key_moment_dependencies(int64_t key, const Axis &input,
    const Axis &output, bool output_side, int start, bool merge, bool &valid) {
    valid = key >= 0 && key < int64_t(input.node_count) * output.node_count;
    if (!valid) return 0;
    const Axis &axis = output_side ? output : input;
    static_assert(int64_t(MAX_OWNER_NODES) * MAX_OWNER_NODES <= INT_MAX,
        "validated owner keys fit32-bit division");
    int encoded = int(key);
    int node = output_side ? encoded / input.node_count : encoded % input.node_count;
    return moment_bank_dependencies(bank_for_node(node, axis), start, merge);
}

__global__ void key_moment_banks_kernel(const int64_t *keys, uint32_t *required,
    int atoms, Axis input, Axis output, bool output_side, int start, bool merge) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bits = 0;
    if (atom < atoms) {
        bool valid;
        bits = key_moment_dependencies(keys[atom], input, output, output_side, start, merge, valid);
        CUDA_KERNEL_ASSERT_MSG(valid, "invalid radius owner key range");
    }
    // Inactive tail lanes participate with zero. Only integer execution
    // metadata uses atomics; no response, parameter or gradient atomic exists.
    for (int distance = 16; distance; distance /= 2)
        bits |= __shfl_down_sync(0xffffffffu, bits, distance);
    if ((threadIdx.x & 31) == 0 && bits) atomicOr(required, bits);
}

inline Tensor active_moment_banks_from_keys(Tensor keys, Config input_config,
    Config output_config, bool output_side, bool merge, int start) {
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    check_layout_keys(keys, input.node_count, output.node_count);
    checked_start(output_side ? output : input, start);
    c10::cuda::CUDAGuard guard(keys.device());
    auto required = torch::zeros({1}, keys.options().dtype(torch::kUInt32));
    if (!keys.numel()) return required;
    static_assert(THREADS % 32 == 0, "key-mask reduction requires full warps");
    key_moment_banks_kernel<<<(keys.numel() + THREADS - 1) / THREADS,
        THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(keys.data_ptr<int64_t>(),
        required.data_ptr<uint32_t>(), keys.numel(), input, output, output_side, start, merge);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return required;
}

inline Tensor build_compact_moments_from_keys(Tensor signal, Tensor p, Tensor keys,
    Config input_config, Config output_config, bool output_side, bool merge, int start) {
    check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    const Axis &axis = output_side ? output : input;
    check_hybrid_signal(signal, p, axis);
    check_layout_keys(keys, input.node_count, output.node_count);
    TORCH_CHECK(keys.device() == p.device() && keys.numel() == p.size(0), "owner keys differ from atoms");
    int nodes = axis.node_count - checked_start(axis, start);
    TORCH_CHECK(!nodes || signal.size(0) <= (INT_MAX - THREADS) / (nodes * axis.basis_count()),
        "moment work exceeds native index limit");
    auto required = active_moment_banks_from_keys(keys, input_config, output_config, output_side, merge, start);
    // The same compact kernel/level order reads this four-byte stream-local
    // word. Z's shape/allocation and every retained bank's arithmetic are old.
    return build_compact_moments_impl(signal, output_side ? output_config : input_config,
        merge, start, required.data_ptr<uint32_t>());
}

// Mode8 reuses the qualified mode6 owner response/C and site kernels literally.
// Only the producer of the active-bank word changes; no P/radius revalidation
// or reclassification is necessary after this forward's validated keys exist.
std::vector<Tensor> autograd_owner_key_transform(Tensor x, Tensor p, Tensor keys,
    Tensor rows, Tensor offsets, Config input_config, Config output_config, bool reverse) {
    check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    const Axis &source = reverse ? output : input, &destination = reverse ? input : output;
    check_hybrid_signal(x, p, source); check_ranges(rows, offsets, p, destination);
    check_owner_axes(input, output, x.size(0), 2, 2, true, 32);
    c10::cuda::CUDAGuard guard(p.device());
    auto z = build_compact_moments_from_keys(x, p, keys, input_config, output_config, reverse, true, 2);
    auto response = torch::empty({x.size(0), p.size(0)}, x.options().dtype(torch::kDouble));
    auto partials = torch::empty({x.size(0), destination.node_count, OWNER_PARTS, OWNER_FEATURES},
        x.options().dtype(torch::kDouble));
    auto y = torch::empty({x.size(0), destination.sites}, x.options());
    if (x.size(0)) {
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_owner_key_masks", [&] {
            owner_response_coefficients_kernel<scalar_t><<<destination.node_count, THREADS,
                OWNER_SHARED_BYTES, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(), z.data_ptr<double>(), response.data_ptr<double>(),
                partials.data_ptr<double>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                p.size(0), x.size(0), source, destination, 2, reverse);
            owner_expand_kernel<scalar_t><<<(y.numel() + THREADS - 1) / THREADS, THREADS, 0,
                at::cuda::getCurrentCUDAStream()>>>(partials.data_ptr<double>(), offsets.data_ptr<int64_t>(),
                y.data_ptr<scalar_t>(), x.size(0), destination, 2);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {y, z, response};
}

std::vector<Tensor> autograd_owner_key_project(Tensor dy, Tensor p, Tensor keys,
    Config input_config, Config output_config) {
    // Atom-only backward retains keys too, but needs no owner sorting/C/dX.
    auto z = build_compact_moments_from_keys(dy, p, keys, input_config, output_config, true, true, 2);
    auto response = torch::empty({dy.size(0), p.size(0)}, dy.options().dtype(torch::kDouble));
    parallel_response_into(dy, p, z, response, output_config, true, 2);
    return {z, response};
}
} // namespace radius_cell_mass
