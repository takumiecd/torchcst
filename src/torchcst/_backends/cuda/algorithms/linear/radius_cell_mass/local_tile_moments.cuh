#pragma once
#include "local_tile_geometry.cuh"

namespace radius_cell_mass {
template<class T, int Stage>
__global__ void local_tile_owner_kernel(const T *p, const T *signal,
    const double *retained, double *response, const double *h, const double *g,
    T *gradient, const int64_t *rows, const int64_t *offsets,
    int atoms, int batch, Axis axis, bool output_side) {
    int node = blockIdx.x;
    int begin = int(offsets[node]), end = int(offsets[node + 1]);
    if (begin == end) return; // uniform; no participant skips a later barrier
    int bi = bank_for_node(node, axis);
    Bank bank = axis.banks[bi];
    int first_tile = node - bank.node_offset;
    extern __shared__ double scratch[];
    if (bank.level >= LOCAL_TILE_DIRECT && bank.level < LOCAL_TILE_RETAIN) {
        int elements = batch * bank.count * LOCAL_TILE_FEATURES;
        double *phi = scratch + elements;
        double *filters = phi + bank.tile_sites * LOCAL_TILE_FEATURES;
        for (int f = threadIdx.x; f < LOCAL_TILE_FEATURES; f += blockDim.x)
            local_tile_basis_column(phi, filters, f, axis, bank);
        __syncthreads();
        for (int i = threadIdx.x; i < elements; i += blockDim.x) {
            int f = i % LOCAL_TILE_FEATURES;
            int part = (i / LOCAL_TILE_FEATURES) % bank.count;
            int b = i / (LOCAL_TILE_FEATURES * bank.count);
            scratch[i] = local_tile_dot(signal, b, first_tile, part, f, axis, bank, phi, filters);
        }
        __syncthreads();
    }
    // Original Parameter row IDs are read directly; no atom table is packed.
    // Each atom belongs to exactly one owner, and each column has one writer.
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x)
        local_tile_atom<T, Stage>(p, signal, retained, scratch, response, h, g,
            gradient, int(rows[i]), atoms, batch, axis, output_side);
}

inline void check_local_tile_axes(const Axis &input, const Axis &output, int batch,
    int input_start, int output_start, int tile_sites) {
    TORCH_CHECK(batch <= LOCAL_TILE_BATCH && input_start == LOCAL_TILE_DIRECT &&
        output_start == LOCAL_TILE_DIRECT && tile_sites == 32,
        "local tile moments require B<=32, direct levels2 and output tiles32");
    for (const Axis *axis : {&input, &output})
        TORCH_CHECK(axis->sites == 128 && axis->polar && axis->modes == 8 &&
            axis->base_tile_sites == 1 && axis->merge_factor == 2,
            "local tile moments require N128 Polar M8 base1 merge2");
    TORCH_CHECK(input.amplitude_max == output.amplitude_max && input.w_c == output.w_c &&
        input.kappa == output.kappa && input.lower_kappa == output.lower_kappa &&
        input.upper_decay_power == output.upper_decay_power,
        "local tile moments require matching Polar global parameters");
    TORCH_CHECK(local_tile_shared_bytes(batch) <= 48 * 1024,
        "local tile moments exceed shared memory budget");
}

inline Tensor local_tile_retained(Tensor signal, Tensor p, Axis axis, bool output_side) {
    int elements = (axis.node_count - prefix_nodes(axis, LOCAL_TILE_RETAIN)) * axis.basis_count();
    auto z = torch::empty({signal.size(0), elements / axis.basis_count(), axis.basis_count()},
        signal.options().dtype(torch::kDouble));
    if (!signal.size(0)) return z;
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_local_tile_retained", [&] {
        // This kernel initializes shared scratch and writes ALL retained slots;
        // inactive banks are defined zeros. It does not perform old H work.
        shared_projection_kernel<scalar_t, false><<<signal.size(0), THREADS,
            elements * sizeof(double), at::cuda::getCurrentCUDAStream()>>>(
            signal.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(), z.data_ptr<double>(),
            nullptr, p.size(0), axis, LOCAL_TILE_RETAIN, true, output_side);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return z;
}

template<int Stage>
inline void local_tile_owners(Tensor signal, Tensor p, Tensor z, Tensor response,
    Tensor h, Tensor g, Tensor gradient, Tensor rows, Tensor offsets, Axis axis, bool output_side) {
    if (!signal.size(0) || !p.size(0)) return;
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_local_tile_owner", [&] {
        local_tile_owner_kernel<scalar_t, Stage><<<axis.node_count, LOCAL_TILE_THREADS,
            local_tile_shared_bytes(signal.size(0)), at::cuda::getCurrentCUDAStream()>>>(
            p.data_ptr<scalar_t>(), signal.data_ptr<scalar_t>(), z.data_ptr<double>(),
            response.defined() ? response.data_ptr<double>() : nullptr,
            h.defined() ? h.data_ptr<double>() : nullptr, g.defined() ? g.data_ptr<double>() : nullptr,
            gradient.defined() ? gradient.data_ptr<scalar_t>() : nullptr,
            rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), p.size(0), signal.size(0), axis, output_side);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<Tensor> local_tile_forward(Tensor x, Tensor p, Tensor keys, Tensor rows,
    Tensor offsets, Config input_config, Config output_config) {
    c10::cuda::CUDAGuard guard(p.device());
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    auto input_ranges = layout_ranges(keys, input.node_count, output.node_count, true);
    auto z = local_tile_retained(x, p, input, false);
    auto h = torch::empty({x.size(0), p.size(0)}, x.options().dtype(torch::kDouble));
    local_tile_owners<0>(x, p, z, h, Tensor(), Tensor(), Tensor(),
        input_ranges[0], input_ranges[1], input, false);
    auto y = batched_cached_expansion(p, rows, offsets, h, output_config, LOCAL_TILE_DIRECT, 32, true);
    return {y, z, h};
}

std::vector<Tensor> local_tile_backward(Tensor x, Tensor p, Tensor zx, Tensor h,
    Tensor keys, Tensor dy, Config input_config, Config output_config, bool needs_x, bool needs_p) {
    // Unpack hooks may return different Tensor metadata. Validate every saved
    // buffer before any allocation, sort or CUDA launch, as the old wrappers do.
    check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    check_hybrid_signal(x, p, input); check_hybrid_signal(dy, p, output);
    check_local_tile_axes(input, output, x.size(0), LOCAL_TILE_DIRECT, LOCAL_TILE_DIRECT, 32);
    check_compact(zx, p, input, LOCAL_TILE_RETAIN);
    TORCH_CHECK(x.size(0) == dy.size(0) && zx.size(0) == x.size(0), "local tile saved batches differ");
    check_response(h, p, x.size(0));
    check_layout_keys(keys, input.node_count, output.node_count);
    TORCH_CHECK(keys.device() == p.device() && keys.numel() == p.size(0), "local tile saved keys differ");
    c10::cuda::CUDAGuard guard(p.device());
    auto output_ranges = layout_ranges(keys, input.node_count, output.node_count, false);
    auto zy = local_tile_retained(dy, p, output, true);
    auto g = torch::empty({dy.size(0), p.size(0)}, dy.options().dtype(torch::kDouble));
    Tensor gradient, dx;
    if (needs_p) gradient = torch::empty_like(p);
    if (needs_p) local_tile_owners<1>(dy, p, zy, g, h, Tensor(), gradient,
        output_ranges[0], output_ranges[1], output, true);
    else local_tile_owners<0>(dy, p, zy, g, Tensor(), Tensor(), Tensor(),
        output_ranges[0], output_ranges[1], output, true);
    auto input_ranges = layout_ranges(keys, input.node_count, output.node_count, true);
    if (needs_x) dx = batched_cached_expansion(p, input_ranges[0], input_ranges[1],
        g, input_config, LOCAL_TILE_DIRECT, 32, false);
    if (needs_p) local_tile_owners<2>(x, p, zx, Tensor(), h, g, gradient,
        input_ranges[0], input_ranges[1], input, false);
    // Empty batches still have defined zero task gradients, like the baseline.
    if (needs_p && !dy.size(0)) gradient.zero_();
    return {dx, gradient};
}
} // namespace radius_cell_mass
