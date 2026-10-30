#pragma once
#include "block_local.cuh"
#include "parallel_projection.cuh"
#include "cached_coefficients.cuh"

namespace radius_cell_mass {

// One CTA owns a batch row's complete projection. Output tiles NEVER rebuild
// these moments or responses. The next launch is the cross-block dependency
// boundary; there is no inter-block spin wait or output atomic accumulation.
template<class T, bool ProduceResponse = true>
__global__ void shared_projection_kernel(
    const T *signal, const T *p, double *moments, double *response,
    int atoms, Axis axis, int start, bool merge, bool output_side) {
    extern __shared__ double scratch[];
    __shared__ unsigned required_banks;
    int b = blockIdx.x;
    const T *row = signal + int64_t(b) * axis.sites;
    int prefix = prefix_nodes(axis, start);
    int elements = (axis.node_count - prefix) * axis.basis_count();
    if (threadIdx.x == 0) required_banks = 0;
    for (int i = threadIdx.x; i < elements; i += blockDim.x) scratch[i] = 0;
    __syncthreads();

    // Membership depends on the current atom, not a previous-step cache. Only
    // retained banks need moments; lower-level atoms project directly from X.
    unsigned thread_banks = 0;
    for (int atom = threadIdx.x; atom < atoms; atom += blockDim.x) {
        Membership member = classify(radius(p, atom, axis),
            center(p, atom, axis, output_side), axis);
        if (axis.banks[member.bank].level >= start)
            thread_banks |= 1u << member.bank;
    }
    // Many atoms often occupy the SAME bank. Publish one mask per warp,
    // instead of serializing every atom on one shared-memory atomic address.
    for (int distance = 16; distance; distance /= 2)
        thread_banks |= __shfl_down_sync(0xffffffff, thread_banks, distance);
    if (threadIdx.x % 32 == 0) atomicOr(&required_banks, thread_banks);
    __syncthreads();
    if (threadIdx.x == 0 && merge) {
        // Even/divisible harmonics inherit from the same K in the child level.
        // Keep those producers even when no atom directly consumes that bank.
        for (int bi = axis.bank_count - 1; bi >= 2 * start; --bi)
            if ((required_banks & (1u << bi)) && bi >= 2 * (start + 1))
                required_banks |= 1u << (bi - 2);
    }
    __syncthreads();

    int levels = axis.bank_count / 2;
    for (int level = start; level < (merge ? levels : start + 1); ++level) {
        // The dependency barrier is between LEVELS, not elements. Traverse
        // only selected banks in this level; zeroed inactive slots stay zero.
        // K=2/K=3 banks at the same level are independent of each other.
        int first_bank = 2 * level;
        int last_bank = merge ? first_bank + 2 : axis.bank_count;
        for (int bi = first_bank; bi < last_bank; ++bi) {
            if (!(required_banks & (1u << bi))) continue;
            Bank bank = axis.banks[bi];
            int begin = (bank.node_offset - prefix) * axis.basis_count();
            int end = begin + bank.tile_count * axis.basis_count();
            for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
                int node = prefix + i / axis.basis_count();
                scratch[i] = compact_moment_element(row, scratch, 0, node,
                    i % axis.basis_count(), axis, start, merge);
            }
        }
        __syncthreads();
    }

    // Inactive slots are defined zeros. Required child-bank slots are retained
    // for exact merge dependencies and backward's selected-bank contractions.
    for (int i = threadIdx.x; i < elements; i += blockDim.x)
        moments[int64_t(b) * elements + i] = scratch[i];
    if constexpr (ProduceResponse) {
        for (int atom = threadIdx.x; atom < atoms; atom += blockDim.x) {
            double R = radius(p, atom, axis);
            double original = center(p, atom, axis, output_side);
            Membership member = classify(R, original, axis);
            Bank bank = axis.banks[member.bank];
            double c[17], dr[17], dR, dmu;
            profile_coefficients(R, bank.period, axis.modes, c, dr);
            double mu = wrapped_center(original, axis);
            double value = bank.level < start
                ? direct_project(row, 0, axis, member, mu, c, dr, dR, dmu)
                : compact_project(scratch, 0, axis, member, mu, c, dr, start, dR, dmu);
            response[int64_t(b) * atoms + atom] = value;
        }
    }
}

// Coefficients live only in shared memory. All destination CTAs read the same
// completed global response row from the projection launch above.
template<class T>
__global__ void shared_expand_kernel(
    const T *p, const int64_t *rows, const int64_t *offsets,
    const double *responses, T *y, int atoms, Axis axis,
    int start, int tile_sites, int coefficient_warps, bool output_side) {
    extern __shared__ double scratch[];
    __shared__ unsigned active_banks;
    int b = blockIdx.y;
    int prefix = prefix_nodes(axis, start);
    int features = axis.basis_count();
    int elements = (axis.node_count - prefix) * features;
    int first_site = blockIdx.x * tile_sites;
    int end_site = min(first_site + tile_sites, axis.sites);
    const double *response = responses + int64_t(b) * atoms;
    if (threadIdx.x == 0) active_banks = 0;
    __syncthreads();
    if (threadIdx.x < axis.bank_count) {
        int bi = threadIdx.x;
        Bank bank = axis.banks[bi];
        if (offsets[bank.node_offset + bank.tile_count] != offsets[bank.node_offset])
            atomicOr(&active_banks, 1u << bi);
    }
    __syncthreads();

    if (coefficient_warps) {
        int warp = threadIdx.x / BLOCK_LOCAL_C_CHUNK;
        if (warp < coefficient_warps) {
            double *columns = scratch + elements + warp * features * BLOCK_LOCAL_C_STRIDE;
            int job = 0;
            for (int bi = 2 * start; bi < axis.bank_count; ++bi) {
                if (!(active_banks & (1u << bi))) continue;
                Bank bank = axis.banks[bi];
                int first_tile = first_site / bank.tile_sites;
                int count = (end_site - 1) / bank.tile_sites - first_tile + 1;
                int first_job = warp - job % coefficient_warps;
                if (first_job < 0) first_job += coefficient_warps;
                for (int local = first_job; local < count; local += coefficient_warps) {
                    int node = bank.node_offset + first_tile + local;
                    block_local_coefficient_tile(p, rows, offsets, response,
                        scratch + int64_t(node - prefix) * features, columns,
                        node, axis, output_side);
                }
                job += count;
            }
        }
    } else {
        for (int node = prefix + threadIdx.x; node < axis.node_count; node += blockDim.x) {
            int bi = bank_for_node(node, axis);
            if (!(active_banks & (1u << bi))) continue;
            Bank bank = axis.banks[bi];
            int first = (node - bank.node_offset) * bank.tile_sites;
            if (first >= end_site || first + bank.tile_sites <= first_site) continue;
            compact_coefficient_tile(p, rows, offsets, response,
                scratch + int64_t(node - prefix) * features,
                atoms, 0, node, axis, output_side);
        }
    }
    __syncthreads();
    for (int site = first_site + threadIdx.x; site < end_site; site += blockDim.x)
        y[int64_t(b) * axis.sites + site] = T(hybrid_site(
            p, rows, offsets, response, scratch, atoms, 0, site,
            axis, output_side, start, active_banks));
}

inline std::vector<Tensor> shared_apply(
    Tensor x, Tensor p, Tensor rows, Tensor offsets,
    Config input_config, Config output_config, int input_start, int output_start,
    bool merge, int output_tile_sites, bool projection_output_side, bool emit_output,
    bool parallel_projection = false, bool cache_coefficients = false) {
    check_matrix(x); check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(x.device() == p.device() && x.scalar_type() == p.scalar_type() &&
        x.size(1) == input.sites, "signal and atoms must match radius projection axis/device/dtype");
    checked_start(input, input_start); checked_start(output, output_start);
    TORCH_CHECK(output_tile_sites >= 1 && output_tile_sites <= 128,
        "output_tile_sites must lie in [1,128]");
    TORCH_CHECK(x.size(0) <= 65535, "shared projection batch exceeds grid.y limit");
    if (emit_output) check_ranges(rows, offsets, p, output);
    int input_elements = (input.node_count - prefix_nodes(input, input_start)) * input.basis_count();
    int output_elements = emit_output
        ? (output.node_count - prefix_nodes(output, output_start)) * output.basis_count() : 0;
    TORCH_CHECK(input_elements == 0 || x.size(0) <= (INT_MAX - THREADS) / input_elements,
        "shared projection moments exceed native index limit");
    TORCH_CHECK(p.size(0) == 0 || x.size(0) <= (INT_MAX - THREADS) / p.size(0),
        "shared projection response exceeds native index limit");
    size_t projection_bytes = size_t(input_elements) * sizeof(double);
    size_t expansion_bytes = size_t(output_elements) * sizeof(double);
    TORCH_CHECK(projection_bytes <= 48 * 1024 && expansion_bytes <= 48 * 1024,
        "shared radius route exceeds the 48-KiB shared-memory budget");
    int coefficient_warps = 0;
    if (emit_output && output_elements) {
        size_t warp_bytes = size_t(output.basis_count()) * BLOCK_LOCAL_C_STRIDE * sizeof(double);
        coefficient_warps = int(std::min(size_t(THREADS / BLOCK_LOCAL_C_CHUNK),
            (48 * 1024 - expansion_bytes) / warp_bytes));
        expansion_bytes += coefficient_warps * warp_bytes;
    }
    CoefficientCacheLayout cache_layout{};
    if (emit_output && cache_coefficients) {
        cache_layout = coefficient_cache_layout(output, output_start);
        expansion_bytes = cache_layout.bytes;
    }
    c10::cuda::CUDAGuard guard(x.device());
    auto z = torch::empty({x.size(0), input_elements / input.basis_count(), input.basis_count()},
        x.options().dtype(torch::kDouble));
    auto h = torch::empty({x.size(0), p.size(0)}, x.options().dtype(torch::kDouble));
    Tensor y;
    if (emit_output) y = torch::empty({x.size(0), output.sites}, x.options());
    if (!x.size(0)) return emit_output ? std::vector<Tensor>{y, z, h} : std::vector<Tensor>{z, h};
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "radius_shared_apply", [&] {
        if (parallel_projection) {
            if (input_elements) {
                shared_projection_kernel<scalar_t, false><<<x.size(0), THREADS, projection_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(), z.data_ptr<double>(),
                    h.data_ptr<double>(), p.size(0), input, input_start, merge, projection_output_side);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            if (p.size(0)) {
                dim3 response_grid((p.size(0) + PARALLEL_RESPONSE_THREADS - 1) / PARALLEL_RESPONSE_THREADS,
                    (x.size(0) + PARALLEL_RESPONSE_BATCH - 1) / PARALLEL_RESPONSE_BATCH);
                parallel_response_kernel<<<response_grid, PARALLEL_RESPONSE_THREADS, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(), z.data_ptr<double>(),
                    h.data_ptr<double>(), p.size(0), x.size(0), input, input_start, projection_output_side);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        } else {
            shared_projection_kernel<<<x.size(0), THREADS, projection_bytes, at::cuda::getCurrentCUDAStream()>>>(
                x.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(), z.data_ptr<double>(), h.data_ptr<double>(),
                p.size(0), input, input_start, merge, projection_output_side);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if (emit_output) {
            dim3 grid((output.sites + output_tile_sites - 1) / output_tile_sites, x.size(0));
            if (cache_coefficients) {
                cached_expand_kernel<scalar_t><<<grid, THREADS, expansion_bytes, at::cuda::getCurrentCUDAStream()>>>(
                    p.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                    h.data_ptr<double>(), y.data_ptr<scalar_t>(), p.size(0), output, output_start,
                    output_tile_sites, cache_layout.capacity, cache_layout.warps, !projection_output_side);
            } else shared_expand_kernel<<<grid, THREADS, expansion_bytes, at::cuda::getCurrentCUDAStream()>>>(
                p.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                h.data_ptr<double>(), y.data_ptr<scalar_t>(), p.size(0), output, output_start,
                output_tile_sites, coefficient_warps, !projection_output_side);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });
    return emit_output ? std::vector<Tensor>{y, z, h} : std::vector<Tensor>{z, h};
}

inline std::vector<Tensor> shared_forward(Tensor x, Tensor p, Tensor rows, Tensor offsets,
    Config input, Config output, int input_start, int output_start, bool merge, int tile_sites) {
    return shared_apply(x, p, rows, offsets, input, output, input_start, output_start,
        merge, tile_sites, false, true);
}
inline std::vector<Tensor> shared_backward(Tensor dy, Tensor p, Tensor rows, Tensor offsets,
    Config input, Config output, int input_start, int output_start, bool merge, int tile_sites) {
    return shared_apply(dy, p, rows, offsets, output, input, output_start, input_start,
        merge, tile_sites, true, true);
}
inline std::vector<Tensor> shared_project(Tensor signal, Tensor p, Config axis,
    bool output_side, int start, bool merge) {
    return shared_apply(signal, p, Tensor(), Tensor(), axis, axis,
        start, start, merge, 1, output_side, false);
}

// Explicit opt-in schedule. The caches and their lifetime match shared_apply;
// only the response ownership and the additional launch boundary differ.
inline std::vector<Tensor> shared_forward_parallel(Tensor x, Tensor p, Tensor rows, Tensor offsets,
    Config input, Config output, int input_start, int output_start, bool merge, int tile_sites) {
    return shared_apply(x, p, rows, offsets, input, output, input_start, output_start,
        merge, tile_sites, false, true, true);
}
inline std::vector<Tensor> shared_backward_parallel(Tensor dy, Tensor p, Tensor rows, Tensor offsets,
    Config input, Config output, int input_start, int output_start, bool merge, int tile_sites) {
    return shared_apply(dy, p, rows, offsets, output, input, output_start, input_start,
        merge, tile_sites, true, true, true);
}
inline std::vector<Tensor> shared_project_parallel(Tensor signal, Tensor p, Config axis,
    bool output_side, int start, bool merge) {
    return shared_apply(signal, p, Tensor(), Tensor(), axis, axis,
        start, start, merge, 1, output_side, false, true);
}

inline std::vector<Tensor> shared_forward_cached_parallel(Tensor x, Tensor p, Tensor rows, Tensor offsets,
    Config input, Config output, int input_start, int output_start, bool merge, int tile_sites) {
    return shared_apply(x, p, rows, offsets, input, output, input_start, output_start,
        merge, tile_sites, false, true, true, true);
}
inline std::vector<Tensor> shared_backward_cached_parallel(Tensor dy, Tensor p, Tensor rows, Tensor offsets,
    Config input, Config output, int input_start, int output_start, bool merge, int tile_sites) {
    return shared_apply(dy, p, rows, offsets, output, input, output_start, input_start,
        merge, tile_sites, true, true, true, true);
}

// Tests only: a separate specialization counts producers without adding a
// counter allocation, atomic, or instrumentation to normal training.
inline std::vector<Tensor> counted_cached_expansion(Tensor p, Tensor rows, Tensor offsets,
    Tensor response, Config config, int start, int tile_sites, bool output_side) {
    check_atoms(p);
    Axis axis = parse_axis(config);
    checked_start(axis, start); check_ranges(rows, offsets, p, axis);
    TORCH_CHECK(tile_sites >= 1 && tile_sites <= 128, "invalid cached tile size");
    TORCH_CHECK(response.device() == p.device() && response.scalar_type() == torch::kDouble &&
        response.dim() == 2 && response.is_contiguous() && response.size(1) == p.size(0) &&
        response.size(0) <= 65535, "invalid counted cache response");
    c10::cuda::CUDAGuard guard(p.device());
    auto y = torch::empty({response.size(0), axis.sites}, p.options());
    int blocks = (axis.sites + tile_sites - 1) / tile_sites;
    auto counts = torch::zeros({response.size(0), blocks, p.size(0)}, p.options().dtype(torch::kInt));
    auto layout = coefficient_cache_layout(axis, start);
    if (response.size(0)) {
        dim3 grid(blocks, response.size(0));
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "counted_cached_expansion", [&] {
            cached_expand_kernel<scalar_t, true><<<grid, THREADS, layout.bytes, at::cuda::getCurrentCUDAStream()>>>(
                p.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                response.data_ptr<double>(), y.data_ptr<scalar_t>(), p.size(0), axis, start,
                tile_sites, layout.capacity, layout.warps, output_side, counts.data_ptr<int>());
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {y, counts};
}
} // namespace radius_cell_mass
