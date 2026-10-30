#pragma once
#include "cached_coefficients.cuh"

namespace radius_cell_mass {
constexpr int EXPANSION_BATCH = 4;

struct BatchedCoefficientLayout {
    int capacity, warps, destinations;
    size_t bytes;
};

// C holds only the current bank's destination nodes, not the whole hierarchy.
// Both the cache and C are reused after every thread has consumed this bank.
inline BatchedCoefficientLayout batched_coefficient_layout(Axis axis, int start, int tile_sites) {
    int destinations = 0;
    for (int bi = 0; bi < axis.bank_count; ++bi) {
        Bank bank = axis.banks[bi];
        if (bank.level < start) continue;
        for (int first = 0; first < axis.sites; first += tile_sites) {
            int end = std::min(first + tile_sites, axis.sites);
            destinations = std::max(destinations,
                (end - 1) / bank.tile_sites - first / bank.tile_sites + 1);
        }
    }
    size_t coefficients = size_t(destinations) * EXPANSION_BATCH * axis.basis_count() * sizeof(double);
    size_t warp_bytes = size_t(axis.basis_count()) * BLOCK_LOCAL_C_STRIDE * sizeof(double);
    for (int capacity = THREADS; capacity >= 32; capacity /= 2) {
        size_t base = coefficients + size_t(capacity) *
            ((axis.modes + 2 + EXPANSION_BATCH) * sizeof(double) + sizeof(int));
        if (base > 48 * 1024) continue;
        int warps = destinations ? int(std::min(size_t(THREADS / 32),
            (48 * 1024 - base) / warp_bytes)) : 0;
        if (destinations && !warps) continue;
        return {capacity, warps, destinations, base + warps * warp_bytes};
    }
    TORCH_CHECK(false, "batched coefficient cache exceeds the 48-KiB shared-memory budget");
    return {0, 0, 0, 0};
}

// One warp owns one destination node. Generate its unscaled Fourier columns
// once, then each feature lane contracts the same columns with four H rows.
__device__ inline void batched_coefficient_tile(
    const double *cache, const int *first_tiles, const int64_t *offsets,
    double *out, double *columns, int64_t chunk, int retained, int node,
    Axis axis, Bank bank, int stride) {
    int lane = threadIdx.x % 32, features = axis.basis_count();
    int tile = node - bank.node_offset;
    double sum[EXPANSION_BATCH] = {}, last[EXPANSION_BATCH] = {};
    for (int part = 0; part < bank.count; ++part) {
        int owner = bank.node_offset + positive_mod(tile - part, bank.tile_count);
        int64_t begin = offsets[owner] > chunk ? offsets[owner] : chunk;
        int64_t end = offsets[owner + 1] < chunk + retained ? offsets[owner + 1] : chunk + retained;
        for (int64_t group = begin; group < end; group += 32) {
            int count = int(end - group < 32 ? end - group : 32);
            if (lane < count) {
                int slot = int(group + lane - chunk);
                const double *c = cache + slot * stride;
                int unwrapped = first_tiles[slot] + part;
                double displacement = c[axis.modes + 1] - unwrapped * bank.tile_sites * axis.spacing;
                columns[lane] = c[0];
                HarmonicPhase phase(2 * PI * displacement / bank.period);
                for (int k = 1; k <= axis.modes; ++k) {
                    double cp, sp; phase.next(cp, sp);
                    columns[(2 * k - 1) * BLOCK_LOCAL_C_STRIDE + lane] = 2 * c[k] * cp;
                    columns[2 * k * BLOCK_LOCAL_C_STRIDE + lane] = 2 * c[k] * sp;
                }
            }
            __syncwarp();
            if (lane < features) {
                for (int atom = 0; atom < count; ++atom) {
                    const double *c = cache + int(group + atom - chunk) * stride;
                    double coefficient = columns[lane * BLOCK_LOCAL_C_STRIDE + atom];
                    #pragma unroll
                    for (int r = 0; r < EXPANSION_BATCH; ++r)
                        sum[r] += c[axis.modes + 2 + r] * coefficient;
                }
            }
            if (lane == 0 && features > 32) {
                for (int atom = 0; atom < count; ++atom) {
                    const double *c = cache + int(group + atom - chunk) * stride;
                    double coefficient = columns[32 * BLOCK_LOCAL_C_STRIDE + atom];
                    #pragma unroll
                    for (int r = 0; r < EXPANSION_BATCH; ++r)
                        last[r] += c[axis.modes + 2 + r] * coefficient;
                }
            }
            __syncwarp();
        }
    }
    #pragma unroll
    for (int r = 0; r < EXPANSION_BATCH; ++r) {
        if (lane < features) out[r * features + lane] += sum[r];
        if (lane == 0 && features > 32) out[r * features + 32] += last[r];
    }
}

// Same cell integral and repeated-image sum as cached_direct_site. The mass
// does not depend on batch; multiply the four completed responses afterwards.
__device__ inline void batched_direct_site(
    const double *cache, const int *first_tiles, const int64_t *offsets,
    int64_t chunk, int retained, int site, Axis axis, Bank bank, int stride, double *result) {
    int tile = site / bank.tile_sites, local = site % bank.tile_sites;
    for (int part = 0; part < bank.count; ++part) {
        int owner = bank.node_offset + positive_mod(tile - part, bank.tile_count);
        int64_t begin = offsets[owner] > chunk ? offsets[owner] : chunk;
        int64_t end = offsets[owner + 1] < chunk + retained ? offsets[owner + 1] : chunk + retained;
        for (int64_t row = begin; row < end; ++row) {
            int slot = int(row - chunk);
            const double *c = cache + slot * stride;
            int unwrapped_site = (first_tiles[slot] + part) * bank.tile_sites + local;
            double mass = axis.spacing * c[0];
            HarmonicPhase phase(2 * PI * (c[axis.modes + 1] - unwrapped_site * axis.spacing) / bank.period);
            double half_cell = PI * axis.spacing / bank.period;
            HarmonicPhase cell_phase(half_cell);
            for (int k = 1; k <= axis.modes; ++k) {
                double cp, sp, cell_cosine, cell_sine;
                phase.next(cp, sp); cell_phase.next(cell_cosine, cell_sine);
                double argument = PI * k * axis.spacing / bank.period;
                mass += 2 * c[k] * axis.spacing * cell_sine / argument * cp;
            }
            #pragma unroll
            for (int r = 0; r < EXPANSION_BATCH; ++r)
                result[r] += c[axis.modes + 2 + r] * mass;
        }
    }
}

template<class T, bool Count = false>
__global__ void batched_cached_expand_kernel(
    const T *p, const int64_t *rows, const int64_t *offsets,
    const double *responses, T *y, int atoms, int batch, Axis axis, int start,
    int tile_sites, int capacity, int warps, int max_destinations, bool output_side,
    int *counts = nullptr) {
    extern __shared__ double scratch[];
    int b = blockIdx.y * EXPANSION_BATCH;
    int features = axis.basis_count(), stride = axis.modes + 2 + EXPANSION_BATCH;
    double *columns = scratch + max_destinations * EXPANSION_BATCH * features;
    double *cache = columns + warps * features * BLOCK_LOCAL_C_STRIDE;
    int *first_tiles = reinterpret_cast<int *>(cache + capacity * stride);
    int first_site = blockIdx.x * tile_sites;
    int end_site = min(first_site + tile_sites, axis.sites);
    int site = first_site + threadIdx.x;
    double value[EXPANSION_BATCH] = {};
    for (int bi = 0; bi < axis.bank_count; ++bi) {
        Bank bank = axis.banks[bi];
        // This branch and all subsequent CTA barriers are batch/site invariant.
        if (offsets[bank.node_offset] == offsets[bank.node_offset + bank.tile_count]) continue;
        int first_tile = first_site / bank.tile_sites;
        int destinations = (end_site - 1) / bank.tile_sites - first_tile + 1;
        if (bank.level >= start)
            for (int i = threadIdx.x; i < destinations * EXPANSION_BATCH * features; i += blockDim.x)
                scratch[i] = 0;
        __syncthreads();
        // Owner segments are disjoint even when K exceeds the chart tile count.
        // The repeated images are summed by the consumer, never duplicated here.
        int owner_count = min(bank.tile_count, destinations + bank.count - 1);
        int first_owner = owner_count == bank.tile_count ? 0 :
            positive_mod(first_tile - (bank.count - 1), bank.tile_count);
        int first_length = min(owner_count, bank.tile_count - first_owner);
        for (int segment = 0; segment < (first_length < owner_count ? 2 : 1); ++segment) {
            int owner = segment ? 0 : first_owner;
            int length = segment ? owner_count - first_length : first_length;
            int64_t begin = offsets[bank.node_offset + owner];
            int64_t end = offsets[bank.node_offset + owner + length];
            for (int64_t chunk = begin; chunk < end; chunk += capacity) {
                int retained = int(end - chunk < capacity ? end - chunk : capacity);
                if (threadIdx.x < retained) {
                    int atom = rows[chunk + threadIdx.x];
                    double R = radius(p, atom, axis), original = center(p, atom, axis, output_side);
                    Membership member = classify(R, original, axis);
                    double c[17], dr[17];
                    profile_coefficients(R, bank.period, axis.modes, c, dr);
                    double *saved = cache + threadIdx.x * stride;
                    for (int k = 0; k <= axis.modes; ++k) saved[k] = c[k];
                    saved[axis.modes + 1] = wrapped_center(original, axis);
                    double amp = amplitude(p, atom, axis);
                    #pragma unroll
                    for (int r = 0; r < EXPANSION_BATCH; ++r)
                        saved[axis.modes + 2 + r] = b + r < batch
                            ? amp * responses[int64_t(b + r) * atoms + atom] : 0;
                    first_tiles[threadIdx.x] = member.first_tile;
                    if constexpr (Count) counts[(int64_t(blockIdx.y) * gridDim.x + blockIdx.x) * atoms + atom] += 1;
                }
                __syncthreads();
                if (bank.level >= start) {
                    int warp = threadIdx.x / 32;
                    if (warp < warps)
                        for (int local = warp; local < destinations; local += warps)
                            batched_coefficient_tile(cache, first_tiles, offsets,
                                scratch + local * EXPANSION_BATCH * features,
                                columns + warp * features * BLOCK_LOCAL_C_STRIDE,
                                chunk, retained, bank.node_offset + first_tile + local, axis, bank, stride);
                } else if (site < end_site) {
                    batched_direct_site(cache, first_tiles, offsets, chunk, retained,
                        site, axis, bank, stride, value);
                }
                __syncthreads(); // All cache/column reads finish before the next chunk.
            }
        }
        if (bank.level >= start && site < end_site) {
            int local = site % bank.tile_sites;
            const double *node = scratch + (site / bank.tile_sites - first_tile) * EXPANSION_BATCH * features;
            #pragma unroll
            for (int r = 0; r < EXPANSION_BATCH; ++r) value[r] += axis.spacing * node[r * features];
            double half_cell = PI * axis.spacing / bank.period;
            HarmonicPhase phase(2 * half_cell * local), cell_phase(half_cell);
            for (int k = 1; k <= axis.modes; ++k) {
                double argument = PI * k * axis.spacing / bank.period;
                double cp, sp, cell_cosine, cell_sine;
                phase.next(cp, sp); cell_phase.next(cell_cosine, cell_sine);
                #pragma unroll
                for (int r = 0; r < EXPANSION_BATCH; ++r)
                    value[r] += axis.spacing * cell_sine / argument *
                        (node[r * features + 2 * k - 1] * cp + node[r * features + 2 * k] * sp);
            }
        }
        __syncthreads(); // Expansion is complete before the next bank overwrites C.
    }
    if (site < end_site) {
        #pragma unroll
        for (int r = 0; r < EXPANSION_BATCH; ++r)
            if (b + r < batch) y[int64_t(b + r) * axis.sites + site] = T(value[r]);
    }
}
// Independent consumer of a completed global response. This host helper has
// no projection or all-atom shared-storage dependency; larger projection routes
// can reuse it once their separate axis/atom validation has been qualified.
inline Tensor batched_cached_expansion(Tensor p, Tensor rows, Tensor offsets,
    Tensor response, Config config, int start, int tile_sites, bool output_side) {
    check_atoms(p);
    Axis axis = parse_axis(config);
    checked_start(axis, start); check_ranges(rows, offsets, p, axis);
    TORCH_CHECK(tile_sites >= 1 && tile_sites <= 128, "invalid batched cached tile size");
    TORCH_CHECK(response.device() == p.device() && response.scalar_type() == torch::kDouble &&
        response.dim() == 2 && response.is_contiguous() && response.size(1) == p.size(0) &&
        response.size(0) <= 65535, "invalid batched cache response");
    c10::cuda::CUDAGuard guard(p.device());
    auto y = torch::empty({response.size(0), axis.sites}, p.options());
    auto layout = batched_coefficient_layout(axis, start, tile_sites);
    if (response.size(0)) {
        dim3 grid((axis.sites + tile_sites - 1) / tile_sites,
            (response.size(0) + EXPANSION_BATCH - 1) / EXPANSION_BATCH);
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "batched_cached_expansion", [&] {
            batched_cached_expand_kernel<scalar_t><<<grid, THREADS, layout.bytes, at::cuda::getCurrentCUDAStream()>>>(
                p.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                response.data_ptr<double>(), y.data_ptr<scalar_t>(), p.size(0), response.size(0), axis, start,
                tile_sites, layout.capacity, layout.warps, layout.destinations, output_side);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return y;
}
} // namespace radius_cell_mass
