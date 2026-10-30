#pragma once
#include "shared_projection.cuh"

// Measurement-only entrypoints. Normal shared_forward/backward never call this
// file and allocate neither exported C nor timestamp buffers. Each probe keeps
// the original (destination interval, batch row) CTA and its duplicate nodes.
namespace radius_cell_mass {

__device__ inline unsigned long long expansion_probe_time() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
    return value;
}

__device__ inline bool expansion_probe_owns(
    int node, unsigned active_banks, Axis axis, int first_site, int end_site) {
    int bi = bank_for_node(node, axis);
    Bank bank = axis.banks[bi];
    int first = (node - bank.node_offset) * bank.tile_sites;
    return (active_banks & (1u << bi)) &&
        first < end_site && first + bank.tile_sites > first_site;
}

template<class T>
__device__ inline void expansion_probe_coefficients(
    const T *p, const int64_t *rows, const int64_t *offsets,
    const double *response, double *scratch, int atoms, Axis axis,
    int start, int first_site, int end_site, int coefficient_warps,
    unsigned active_banks, bool output_side) {
    int prefix = prefix_nodes(axis, start);
    int features = axis.basis_count();
    int elements = (axis.node_count - prefix) * features;
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
}

// Mode 0: instrument the shared-resident fused chain.
// Mode 1: compute the SAME per-CTA C and export its valid slots to global.
// Mode 2: reload those slots into shared, then run the SAME site/direct body.
// Split timings necessarily include this export/reload traffic. They are not
// pure arithmetic timings of the fused kernel; report both experiments.
template<int Mode>
__global__ void expansion_probe_kernel(
    const float *p, const int64_t *rows, const int64_t *offsets,
    const double *responses, double *exported, int *masks, float *y,
    int64_t *times, int atoms, Axis axis, int start, int tile_sites,
    int coefficient_warps, bool output_side) {
    extern __shared__ double scratch[];
    __shared__ unsigned active_banks;
    unsigned long long t0 = 0, t1 = 0, t2 = 0;
    if constexpr (Mode == 0) if (threadIdx.x == 0) t0 = expansion_probe_time();
    int b = blockIdx.y;
    int prefix = prefix_nodes(axis, start);
    int features = axis.basis_count();
    int elements = (axis.node_count - prefix) * features;
    int first_site = blockIdx.x * tile_sites;
    int end_site = min(first_site + tile_sites, axis.sites);
    int cta = b * gridDim.x + blockIdx.x;
    const double *response = responses + int64_t(b) * atoms;

    if constexpr (Mode == 2) {
        if (threadIdx.x == 0) active_banks = unsigned(masks[cta]);
        __syncthreads();
        for (int i = threadIdx.x; i < elements; i += blockDim.x) {
            int node = prefix + i / features;
            if (expansion_probe_owns(node, active_banks, axis, first_site, end_site))
                scratch[i] = exported[int64_t(cta) * elements + i];
        }
        __syncthreads();
    } else {
        if (threadIdx.x == 0) active_banks = 0;
        __syncthreads();
        if (threadIdx.x < axis.bank_count) {
            int bi = threadIdx.x;
            Bank bank = axis.banks[bi];
            if (offsets[bank.node_offset + bank.tile_count] != offsets[bank.node_offset])
                atomicOr(&active_banks, 1u << bi);
        }
        // On the scoped 128-site axes all bank-mask producers are in warp 0.
        // Timestamp before the existing barrier: no coefficient warp can start
        // before t1, and no new start barrier is added.
        if constexpr (Mode == 0) if (threadIdx.x == 0) t1 = expansion_probe_time();
        __syncthreads();
        expansion_probe_coefficients(p, rows, offsets, response, scratch, atoms,
            axis, start, first_site, end_site, coefficient_warps, active_banks, output_side);
        __syncthreads();
        if constexpr (Mode == 0) if (threadIdx.x == 0) t2 = expansion_probe_time();
        if constexpr (Mode == 1) {
            for (int i = threadIdx.x; i < elements; i += blockDim.x) {
                int node = prefix + i / features;
                if (expansion_probe_owns(node, active_banks, axis, first_site, end_site))
                    exported[int64_t(cta) * elements + i] = scratch[i];
            }
            if (threadIdx.x == 0) masks[cta] = int(active_banks);
        }
    }
    if constexpr (Mode != 1) {
        for (int site = first_site + threadIdx.x; site < end_site; site += blockDim.x)
            y[int64_t(b) * axis.sites + site] = float(hybrid_site(
                p, rows, offsets, response, scratch, atoms, 0, site,
                axis, output_side, start, active_banks));
    }
    if constexpr (Mode == 0) {
        // The scoped 32-site interval is written by warp 0 after its t2 read.
        // Unlike the original kernel, the probe adds a final barrier so the
        // timestamp covers every site's store. This overhead is disclosed.
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long t3 = expansion_probe_time();
            times[int64_t(cta) * 4] = int64_t(t0);
            times[int64_t(cta) * 4 + 1] = int64_t(t1);
            times[int64_t(cta) * 4 + 2] = int64_t(t2);
            times[int64_t(cta) * 4 + 3] = int64_t(t3);
        }
    }
}

// mode: 0=original fused kernel, 1=fused timestamp probe,
//       2=C plus export, 3=reload C plus site/direct expansion, 4=split pair.
// Buffers are caller-owned to keep allocation outside device timing. Repeats
// amortize host enqueue/setup gaps; per-kernel profiler times are also saved.
void run_expansion_probe(Tensor p, Tensor rows, Tensor offsets, Tensor responses,
    Tensor exported, Tensor masks, Tensor y, Tensor times, Config config,
    int start, int tile_sites, bool output_side, int mode, int repeats) {
    check_atoms(p);
    TORCH_CHECK(p.scalar_type() == torch::kFloat, "expansion probe requires FP32 atoms");
    Axis axis = parse_axis(config);
    checked_start(axis, start); check_ranges(rows, offsets, p, axis);
    TORCH_CHECK(axis.sites == 128, "expansion probe is scoped to 128 sites");
    TORCH_CHECK(tile_sites == 32, "expansion probe requires the existing 32-site CTA");
    TORCH_CHECK(mode >= 0 && mode <= 4 && repeats >= 1 && repeats <= 256,
        "invalid probe mode or repeat count");
    TORCH_CHECK(responses.device() == p.device() && responses.scalar_type() == torch::kDouble &&
        responses.is_contiguous() && responses.dim() == 2 && responses.size(1) == p.size(0) &&
        responses.size(0) <= 65535, "invalid probe responses");
    int batch = responses.size(0);
    int blocks = (axis.sites + tile_sites - 1) / tile_sites;
    int elements = (axis.node_count - prefix_nodes(axis, start)) * axis.basis_count();
    TORCH_CHECK(exported.device() == p.device() && exported.scalar_type() == torch::kDouble &&
        exported.is_contiguous() && exported.dim() == 3 && exported.size(0) == batch &&
        exported.size(1) == blocks && exported.size(2) == elements, "invalid exported C workspace");
    TORCH_CHECK(exported.numel() * int64_t(sizeof(double)) <= 8 * 1024 * 1024,
        "diagnostic C workspace exceeds 8 MiB");
    TORCH_CHECK(masks.device() == p.device() && masks.scalar_type() == torch::kInt &&
        masks.is_contiguous() && masks.dim() == 2 && masks.size(0) == batch &&
        masks.size(1) == blocks, "invalid probe mask workspace");
    TORCH_CHECK(y.device() == p.device() && y.scalar_type() == p.scalar_type() &&
        y.is_contiguous() && y.dim() == 2 && y.size(0) == batch &&
        y.size(1) == axis.sites, "invalid probe output");
    TORCH_CHECK(times.device() == p.device() && times.scalar_type() == torch::kLong &&
        times.is_contiguous() && times.dim() == 3 && times.size(0) == batch &&
        times.size(1) == blocks && times.size(2) == 4, "invalid probe timestamps");
    size_t bytes = size_t(elements) * sizeof(double);
    TORCH_CHECK(bytes <= 48 * 1024, "probe exceeds shared-memory budget");
    int warps = 0;
    if (elements) {
        size_t warp_bytes = size_t(axis.basis_count()) * BLOCK_LOCAL_C_STRIDE * sizeof(double);
        warps = int(std::min(size_t(THREADS / BLOCK_LOCAL_C_CHUNK),
            (48 * 1024 - bytes) / warp_bytes));
        bytes += warps * warp_bytes;
    }
    if (!batch) return;
    c10::cuda::CUDAGuard guard(p.device());
    dim3 grid(blocks, batch);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto coefficients = [&] {
        expansion_probe_kernel<1><<<grid, THREADS, bytes, stream>>>(
            p.data_ptr<float>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
            responses.data_ptr<double>(), exported.data_ptr<double>(), masks.data_ptr<int>(),
            y.data_ptr<float>(), times.data_ptr<int64_t>(), p.size(0), axis,
            start, tile_sites, warps, output_side);
    };
    auto expand = [&] {
        expansion_probe_kernel<2><<<grid, THREADS, bytes, stream>>>(
            p.data_ptr<float>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
            responses.data_ptr<double>(), exported.data_ptr<double>(), masks.data_ptr<int>(),
            y.data_ptr<float>(), times.data_ptr<int64_t>(), p.size(0), axis,
            start, tile_sites, warps, output_side);
    };
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if (mode == 0)
            shared_expand_kernel<<<grid, THREADS, bytes, stream>>>(
                p.data_ptr<float>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                responses.data_ptr<double>(), y.data_ptr<float>(), p.size(0), axis,
                start, tile_sites, warps, output_side);
        else if (mode == 1)
            expansion_probe_kernel<0><<<grid, THREADS, bytes, stream>>>(
                p.data_ptr<float>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                responses.data_ptr<double>(), exported.data_ptr<double>(), masks.data_ptr<int>(),
                y.data_ptr<float>(), times.data_ptr<int64_t>(), p.size(0), axis,
                start, tile_sites, warps, output_side);
        else if (mode == 2) coefficients();
        else if (mode == 3) expand();
        else { coefficients(); expand(); }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

std::vector<std::vector<int64_t>> expansion_probe_resources(Tensor p) {
    check_atoms(p);
    c10::cuda::CUDAGuard guard(p.device());
    std::vector<std::vector<int64_t>> result;
    for (int mode = 0; mode < 4; ++mode) {
        cudaFuncAttributes attr;
        cudaError_t status;
        if (mode == 0) status = cudaFuncGetAttributes(&attr, shared_expand_kernel<float>);
        else if (mode == 1) status = cudaFuncGetAttributes(&attr, expansion_probe_kernel<0>);
        else if (mode == 2) status = cudaFuncGetAttributes(&attr, expansion_probe_kernel<1>);
        else status = cudaFuncGetAttributes(&attr, expansion_probe_kernel<2>);
        C10_CUDA_CHECK(status);
        result.push_back({attr.numRegs, int64_t(attr.localSizeBytes),
            int64_t(attr.sharedSizeBytes), attr.maxThreadsPerBlock, attr.binaryVersion});
    }
    return result;
}
} // namespace radius_cell_mass
