#pragma once
#include "cached_coefficients.cuh"

namespace radius_cell_mass {
#ifndef RADIUS_ROW_BATCH
#define RADIUS_ROW_BATCH 4
#endif
static_assert(RADIUS_ROW_BATCH == 4 || RADIUS_ROW_BATCH == 8,
    "only explicit row4 and row8 builds are qualified candidates");
constexpr int EXPANSION_BATCH = RADIUS_ROW_BATCH;

struct BatchedCoefficientLayout {
    int capacity, warps, destinations;
    size_t bytes;
};

// C holds only the current bank's destination nodes, not the whole hierarchy.
// Both the cache and C are reused after every thread has consumed this bank.
template<class Storage = double>
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
    size_t coefficients = size_t(destinations) * EXPANSION_BATCH * axis.basis_count() * sizeof(Storage);
    size_t warp_bytes = size_t(axis.basis_count()) * BLOCK_LOCAL_C_STRIDE * sizeof(Storage);
    for (int capacity = THREADS; capacity >= 32; capacity /= 2) {
        size_t base = coefficients + size_t(capacity) *
            ((axis.modes + 2 + EXPANSION_BATCH) * sizeof(Storage) + sizeof(int));
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
// once, then each feature lane contracts the same columns with a group of H rows.
template<class Storage = double>
__device__ inline void batched_coefficient_tile(
    const Storage *cache, const int *first_tiles, const int64_t *offsets,
    Storage *out, Storage *columns, int64_t chunk, int retained, int node,
    Axis axis, Bank bank, int stride) {
    int lane = threadIdx.x % 32, features = axis.basis_count();
    int tile = node - bank.node_offset;
    Storage sum[EXPANSION_BATCH] = {}, last[EXPANSION_BATCH] = {};
    for (int part = 0; part < bank.count; ++part) {
        int owner = bank.node_offset + positive_mod(tile - part, bank.tile_count);
        int64_t begin = offsets[owner] > chunk ? offsets[owner] : chunk;
        int64_t end = offsets[owner + 1] < chunk + retained ? offsets[owner + 1] : chunk + retained;
        for (int64_t group = begin; group < end; group += 32) {
            int count = int(end - group < 32 ? end - group : 32);
            if (lane < count) {
                int slot = int(group + lane - chunk);
                const Storage *c = cache + slot * stride;
                int unwrapped = first_tiles[slot] + part;
                Storage displacement = c[axis.modes + 1] - unwrapped * bank.tile_sites * Storage(axis.spacing);
                columns[lane] = c[0];
                SignalPhase<Storage> phase(2 * Storage(PI) * displacement / Storage(bank.period));
                for (int k = 1; k <= axis.modes; ++k) {
                    Storage cp, sp; phase.next(cp, sp);
                    columns[(2 * k - 1) * BLOCK_LOCAL_C_STRIDE + lane] = 2 * c[k] * cp;
                    columns[2 * k * BLOCK_LOCAL_C_STRIDE + lane] = 2 * c[k] * sp;
                }
            }
            __syncwarp();
            if (lane < features) {
                for (int atom = 0; atom < count; ++atom) {
                    const Storage *c = cache + int(group + atom - chunk) * stride;
                    Storage coefficient = columns[lane * BLOCK_LOCAL_C_STRIDE + atom];
                    #pragma unroll
                    for (int r = 0; r < EXPANSION_BATCH; ++r)
                        sum[r] += c[axis.modes + 2 + r] * coefficient;
                }
            }
            if (lane == 0 && features > 32) {
                for (int atom = 0; atom < count; ++atom) {
                    const Storage *c = cache + int(group + atom - chunk) * stride;
                    Storage coefficient = columns[32 * BLOCK_LOCAL_C_STRIDE + atom];
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
template<class Storage = double>
__device__ inline void batched_direct_site(
    const Storage *cache, const int *first_tiles, const int64_t *offsets,
    int64_t chunk, int retained, int site, Axis axis, Bank bank, int stride, Storage *result) {
    int tile = site / bank.tile_sites, local = site % bank.tile_sites;
    for (int part = 0; part < bank.count; ++part) {
        int owner = bank.node_offset + positive_mod(tile - part, bank.tile_count);
        int64_t begin = offsets[owner] > chunk ? offsets[owner] : chunk;
        int64_t end = offsets[owner + 1] < chunk + retained ? offsets[owner + 1] : chunk + retained;
        for (int64_t row = begin; row < end; ++row) {
            int slot = int(row - chunk);
            const Storage *c = cache + slot * stride;
            int unwrapped_site = (first_tiles[slot] + part) * bank.tile_sites + local;
            Storage mass = Storage(axis.spacing) * c[0];
            SignalPhase<Storage> phase(2 * Storage(PI) * (c[axis.modes + 1] - unwrapped_site * Storage(axis.spacing)) / Storage(bank.period));
            Storage half_cell = Storage(PI) * Storage(axis.spacing) / Storage(bank.period);
            SignalPhase<Storage> cell_phase(half_cell);
            for (int k = 1; k <= axis.modes; ++k) {
                Storage cp, sp, cell_cosine, cell_sine;
                phase.next(cp, sp); cell_phase.next(cell_cosine, cell_sine);
                Storage argument = Storage(PI) * k * Storage(axis.spacing) / Storage(bank.period);
                mass += 2 * c[k] * Storage(axis.spacing) * cell_sine / argument * cp;
            }
            #pragma unroll
            for (int r = 0; r < EXPANSION_BATCH; ++r)
                result[r] += c[axis.modes + 2 + r] * mass;
        }
    }
}

template<class T, class Storage, bool CachedGeometry>
__device__ inline void batched_load_atom(
    const T *p, const int64_t *rows, const Storage *responses, const Storage *geometry,
    Storage *cache, int *first_tiles, int64_t row, int slot, int stride,
    int b, int batch, int atoms, Axis axis, Bank bank, bool output_side) {
    int atom = rows[row];
    Membership member;
    Storage c[17], mu, amp;
    if constexpr (CachedGeometry) {
        const Storage *v = geometry + int64_t(atom) * 25;
        int offset = output_side ? 9 : 0;
        member = Membership{int(v[output_side ? 22 : 20]), int(v[output_side ? 23 : 21])};
        for (int k = 0; k <= axis.modes; ++k) c[k] = v[offset + k];
        mu = v[output_side ? 19 : 18]; amp = v[24];
    } else {
        Storage R = signal_radius<Storage>(p, atom, axis), original = center(p, atom, axis, output_side);
        member = signal_classify<Storage>(R, original, axis);
        profile_values(R, Storage(bank.period), axis.modes, c);
        mu = signal_wrapped<Storage>(original, axis);
        amp = signal_amplitude<Storage>(p, atom, axis);
    }
    Storage *saved = cache + slot * stride;
    for (int k = 0; k <= axis.modes; ++k) saved[k] = c[k];
    saved[axis.modes + 1] = mu;
    #pragma unroll
    for (int r = 0; r < EXPANSION_BATCH; ++r)
        saved[axis.modes + 2 + r] = b + r < batch
            ? amp * responses[int64_t(b + r) * atoms + atom] : 0;
    first_tiles[slot] = member.first_tile;
}

template<class Storage>
__device__ inline void batched_expand_coefficients(
    const Storage *node, Storage *value, Axis axis, Bank bank, int local, int features) {
    #pragma unroll
    for (int r = 0; r < EXPANSION_BATCH; ++r) value[r] += Storage(axis.spacing) * node[r * features];
    Storage half_cell = Storage(PI) * Storage(axis.spacing) / Storage(bank.period);
    SignalPhase<Storage> phase(2 * half_cell * local), cell_phase(half_cell);
    for (int k = 1; k <= axis.modes; ++k) {
        Storage argument = Storage(PI) * k * Storage(axis.spacing) / Storage(bank.period);
        Storage cp, sp, cell_cosine, cell_sine;
        phase.next(cp, sp); cell_phase.next(cell_cosine, cell_sine);
        #pragma unroll
        for (int r = 0; r < EXPANSION_BATCH; ++r)
            value[r] += Storage(axis.spacing) * cell_sine / argument *
                (node[r * features + 2 * k - 1] * cp + node[r * features + 2 * k] * sp);
    }
}

struct BatchedOwnerSpan {
    int first, first_length, count;
};

__device__ inline BatchedOwnerSpan batched_owner_span(Bank bank, int tile, int destinations) {
    int count = min(bank.tile_count, destinations + bank.count - 1);
    int first = count == bank.tile_count ? 0 :
        positive_mod(tile - (bank.count - 1), bank.tile_count);
    return {first, min(count, bank.tile_count - first), count};
}

constexpr int SHARED_WIDE_LEVEL = 6; // 64 sites on the admitted unit-base dyadic grids.

#ifdef RADIUS_LM_WIDTH
__global__ void batched_wide_tile_coefficients(
    const float *p, const int64_t *rows, const int64_t *offsets,
    const float *responses, const float *geometry, float *wide, int atoms,
    int batch, Axis axis, int prefix, int capacity, bool output_side) {
    int slot = blockIdx.x;
    int node = prefix + slot;
    Bank bank = axis.banks[bank_for_node(node, axis)];
    if (offsets[bank.node_offset] == offsets[bank.node_offset + bank.tile_count]) return;

    extern __shared__ double scratch[];
    float *work = reinterpret_cast<float *>(scratch);
    int features = axis.basis_count(), stride = axis.modes + 2 + EXPANSION_BATCH;
    float *columns = work + EXPANSION_BATCH * features;
    float *cache = columns + features * BLOCK_LOCAL_C_STRIDE;
    int *first_tiles = reinterpret_cast<int *>(cache + capacity * stride);
    int b = blockIdx.y * EXPANSION_BATCH;
    for (int i = threadIdx.x; i < EXPANSION_BATCH * features; i += blockDim.x) work[i] = 0;
    __syncthreads();

    // Keep the original owner order and chunk boundaries for this destination.
    auto owners = batched_owner_span(bank, node - bank.node_offset, 1);
    for (int segment = 0; segment < (owners.first_length < owners.count ? 2 : 1); ++segment) {
        int owner = segment ? 0 : owners.first;
        int length = segment ? owners.count - owners.first_length : owners.first_length;
        int64_t begin = offsets[bank.node_offset + owner];
        int64_t end = offsets[bank.node_offset + owner + length];
        for (int64_t chunk = begin; chunk < end; chunk += capacity) {
            int retained = int(end - chunk < capacity ? end - chunk : capacity);
            if (threadIdx.x < retained)
                batched_load_atom<float, float, true>(
                    p, rows, responses, geometry, cache, first_tiles, chunk + threadIdx.x,
                    threadIdx.x, stride, b, batch, atoms, axis, bank, output_side);
            __syncthreads();
            if (threadIdx.x < 32)
                batched_coefficient_tile(cache, first_tiles, offsets, work, columns,
                    chunk, retained, node, axis, bank, stride);
            __syncthreads();
        }
    }
    int64_t output = (int64_t(blockIdx.y) * gridDim.x + slot) * EXPANSION_BATCH * features;
    for (int i = threadIdx.x; i < EXPANSION_BATCH * features; i += blockDim.x) wide[output + i] = work[i];
}
#endif

template<class T, bool Count = false, class Storage = double,
    bool CachedGeometry = false, bool SharedWide = false>
__global__ void batched_cached_expand_kernel(
    const T *p, const int64_t *rows, const int64_t *offsets,
    const Storage *responses, T *y, int atoms, int batch, Axis axis, int start,
    int tile_sites, int capacity, int warps, int max_destinations, bool output_side,
    int *counts = nullptr, const Storage *geometry = nullptr, const Storage *wide = nullptr) {
    extern __shared__ double scratch[];
    Storage *work = reinterpret_cast<Storage *>(scratch);
    int b = blockIdx.y * EXPANSION_BATCH;
    int features = axis.basis_count(), stride = axis.modes + 2 + EXPANSION_BATCH;
    Storage *columns = work + max_destinations * EXPANSION_BATCH * features;
    Storage *cache = columns + warps * features * BLOCK_LOCAL_C_STRIDE;
    int *first_tiles = reinterpret_cast<int *>(cache + capacity * stride);
    int first_site = blockIdx.x * tile_sites;
    int end_site = min(first_site + tile_sites, axis.sites);
    int site = first_site + threadIdx.x;
    Storage value[EXPANSION_BATCH] = {};
    for (int bi = 0; bi < axis.bank_count; ++bi) {
        Bank bank = axis.banks[bi];
        // This branch and all subsequent CTA barriers are batch/site invariant.
        if (offsets[bank.node_offset] == offsets[bank.node_offset + bank.tile_count]) continue;
        int first_tile = first_site / bank.tile_sites;
        if constexpr (SharedWide) {
            if (bank.level >= SHARED_WIDE_LEVEL) {
                int prefix = prefix_nodes(axis, SHARED_WIDE_LEVEL);
                int nodes = axis.node_count - prefix;
                int slot = bank.node_offset + first_tile - prefix;
                const Storage *node = wide + (int64_t(blockIdx.y) * nodes + slot) * EXPANSION_BATCH * features;
                if (site < end_site)
                    batched_expand_coefficients(node, value, axis, bank, site % bank.tile_sites, features);
                continue;
            }
        }
        int destinations = (end_site - 1) / bank.tile_sites - first_tile + 1;
        if (bank.level >= start)
            for (int i = threadIdx.x; i < destinations * EXPANSION_BATCH * features; i += blockDim.x)
                work[i] = 0;
        __syncthreads();
        // Owner segments are disjoint even when K exceeds the chart tile count.
        // The repeated images are summed by the consumer, never duplicated here.
        auto owners = batched_owner_span(bank, first_tile, destinations);
        for (int segment = 0; segment < (owners.first_length < owners.count ? 2 : 1); ++segment) {
            int owner = segment ? 0 : owners.first;
            int length = segment ? owners.count - owners.first_length : owners.first_length;
            int64_t begin = offsets[bank.node_offset + owner];
            int64_t end = offsets[bank.node_offset + owner + length];
            for (int64_t chunk = begin; chunk < end; chunk += capacity) {
                int retained = int(end - chunk < capacity ? end - chunk : capacity);
                if (threadIdx.x < retained) {
                    batched_load_atom<T, Storage, CachedGeometry>(
                        p, rows, responses, geometry, cache, first_tiles, chunk + threadIdx.x,
                        threadIdx.x, stride, b, batch, atoms, axis, bank, output_side);
                    if constexpr (Count)
                        counts[(int64_t(blockIdx.y) * gridDim.x + blockIdx.x) * atoms + rows[chunk + threadIdx.x]] += 1;
                }
                __syncthreads();
                if (bank.level >= start) {
                    int warp = threadIdx.x / 32;
                    if (warp < warps)
                        for (int local = warp; local < destinations; local += warps)
                            batched_coefficient_tile(cache, first_tiles, offsets,
                                work + local * EXPANSION_BATCH * features,
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
            const Storage *node = work + (site / bank.tile_sites - first_tile) * EXPANSION_BATCH * features;
            batched_expand_coefficients(node, value, axis, bank, local, features);
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
