#pragma once
#include "block_local.cuh"

namespace radius_cell_mass {

// Shared storage is bounded independently of atom count. Each selected atom is
// loaded once per destination CTA, then all of its local tile images consume
// this cache before the next chunk overwrites it. No global profile table.
struct CoefficientCacheLayout {
    int capacity, warps;
    size_t bytes;
};
inline CoefficientCacheLayout coefficient_cache_layout(Axis axis, int start) {
    size_t coefficients = size_t(axis.node_count - prefix_nodes(axis, start)) *
        axis.basis_count() * sizeof(double);
    size_t warp_bytes = size_t(axis.basis_count()) * BLOCK_LOCAL_C_STRIDE * sizeof(double);
    for (int capacity = THREADS; capacity >= 32; capacity /= 2) {
        size_t cache_bytes = size_t(capacity) *
            ((axis.modes + 3) * sizeof(double) + sizeof(int));
        size_t base = coefficients + cache_bytes;
        if (base > 48 * 1024) continue;
        int warps = coefficients ? int(std::min(size_t(THREADS / 32),
            (48 * 1024 - base) / warp_bytes)) : 0;
        if (coefficients && !warps) continue;
        return {capacity, warps, base + warps * warp_bytes};
    }
    TORCH_CHECK(false, "coefficient cache exceeds the 48-KiB shared-memory budget");
    return {0, 0, 0};
}

// One warp accumulates one node's modes. Cached atoms retain their sorted row
// order within each range. Across cache chunks the order of different first-
// tile ranges may differ from the uncached kernel; accumulation stays FP64.
__device__ inline void cached_coefficient_tile(
    const double *cache, const int *first_tiles, const int64_t *offsets,
    double *out, double *columns, int64_t chunk, int retained, int node,
    Axis axis, Bank bank, int stride) {
    int lane = threadIdx.x % 32;
    int tile = node - bank.node_offset;
    int features = axis.basis_count();
    double sum = 0, last = 0;
    for (int part = 0; part < bank.count; ++part) {
        int owner = bank.node_offset + positive_mod(tile - part, bank.tile_count);
        int64_t begin = offsets[owner] > chunk ? offsets[owner] : chunk;
        int64_t end = offsets[owner + 1] < chunk + retained
            ? offsets[owner + 1] : chunk + retained;
        for (int64_t group = begin; group < end; group += 32) {
            int count = int(end - group < 32 ? end - group : 32);
            if (lane < count) {
                int slot = int(group + lane - chunk);
                const double *c = cache + slot * stride;
                double mu = c[axis.modes + 1], value = c[axis.modes + 2];
                int unwrapped = first_tiles[slot] + part;
                double displacement = mu - unwrapped * bank.tile_sites * axis.spacing;
                columns[lane] = value * c[0];
                HarmonicPhase phase(2 * PI * displacement / bank.period);
                for (int k = 1; k <= axis.modes; ++k) {
                    double cp, sp; phase.next(cp, sp);
                    columns[(2 * k - 1) * BLOCK_LOCAL_C_STRIDE + lane] = value * 2 * c[k] * cp;
                    columns[2 * k * BLOCK_LOCAL_C_STRIDE + lane] = value * 2 * c[k] * sp;
                }
            }
            __syncwarp();
            if (lane < features)
                for (int atom = 0; atom < count; ++atom)
                    sum += columns[lane * BLOCK_LOCAL_C_STRIDE + atom];
            if (lane == 0 && features > 32)
                for (int atom = 0; atom < count; ++atom)
                    last += columns[32 * BLOCK_LOCAL_C_STRIDE + atom];
            __syncwarp();
        }
    }
    if (lane < features) out[lane] += sum;
    if (lane == 0 && features > 32) out[32] += last;
}

// The direct banks consume the SAME atom cache. Only site integration remains
// site-dependent; radius decoding and profile generation never repeat here.
__device__ inline double cached_direct_site(
    const double *cache, const int *first_tiles, const int64_t *offsets,
    int64_t chunk, int retained, int site, Axis axis, Bank bank, int stride) {
    int tile = site / bank.tile_sites, local = site % bank.tile_sites;
    double result = 0;
    for (int part = 0; part < bank.count; ++part) {
        int owner = bank.node_offset + positive_mod(tile - part, bank.tile_count);
        int64_t begin = offsets[owner] > chunk ? offsets[owner] : chunk;
        int64_t end = offsets[owner + 1] < chunk + retained
            ? offsets[owner + 1] : chunk + retained;
        for (int64_t row = begin; row < end; ++row) {
            int slot = int(row - chunk);
            const double *c = cache + slot * stride;
            double mu = c[axis.modes + 1], value = c[axis.modes + 2];
            int unwrapped_site = (first_tiles[slot] + part) * bank.tile_sites + local;
            double mass = axis.spacing * c[0];
            HarmonicPhase phase(2 * PI * (mu - unwrapped_site * axis.spacing) / bank.period);
            double half_cell = PI * axis.spacing / bank.period;
            HarmonicPhase cell_phase(half_cell);
            for (int k = 1; k <= axis.modes; ++k) {
                double cp, sp, cell_cosine, cell_sine;
                phase.next(cp, sp); cell_phase.next(cell_cosine, cell_sine);
                double argument = PI * k * axis.spacing / bank.period;
                mass += 2 * c[k] * axis.spacing * cell_sine / argument * cp;
            }
            result += value * mass;
        }
    }
    return result;
}

template<class T, bool Count = false>
__global__ void cached_expand_kernel(
    const T *p, const int64_t *rows, const int64_t *offsets,
    const double *responses, T *y, int atoms, Axis axis, int start,
    int tile_sites, int capacity, int warps, bool output_side, int *counts = nullptr) {
    extern __shared__ double scratch[];
    int b = blockIdx.y;
    int features = axis.basis_count(), stride = axis.modes + 3;
    int prefix = prefix_nodes(axis, start);
    int elements = (axis.node_count - prefix) * features;
    double *columns = scratch + elements;
    double *cache = columns + warps * features * BLOCK_LOCAL_C_STRIDE;
    int *first_tiles = reinterpret_cast<int *>(cache + capacity * stride);
    int first_site = blockIdx.x * tile_sites;
    int end_site = min(first_site + tile_sites, axis.sites);
    int site = first_site + threadIdx.x;
    const double *response = responses + int64_t(b) * atoms;
    double direct = 0;
    unsigned retained_banks = 0;
    for (int i = threadIdx.x; i < elements; i += blockDim.x) scratch[i] = 0;
    __syncthreads();

    // A bank's radius/period is shared by its tile ranges. The padded K-tile
    // window means only this interval of first-tile owners can reach the CTA.
    // Modulo wrapping is split into disjoint ranges; repeated chart images do
    // NOT load the same atom twice, even when K exceeds the chart tile count.
    for (int bi = 0; bi < axis.bank_count; ++bi) {
        Bank bank = axis.banks[bi];
        if (offsets[bank.node_offset] == offsets[bank.node_offset + bank.tile_count]) continue;
        if (bank.level >= start) retained_banks |= 1u << bi;
        int first_tile = first_site / bank.tile_sites;
        int destinations = (end_site - 1) / bank.tile_sites - first_tile + 1;
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
                    double R = radius(p, atom, axis);
                    double original = center(p, atom, axis, output_side);
                    Membership member = classify(R, original, axis);
                    double c[17];
                    profile_values(R, bank.period, axis.modes, c);
                    double *saved = cache + threadIdx.x * stride;
                    for (int k = 0; k <= axis.modes; ++k) saved[k] = c[k];
                    saved[axis.modes + 1] = wrapped_center(original, axis);
                    saved[axis.modes + 2] = amplitude(p, atom, axis) * response[atom];
                    first_tiles[threadIdx.x] = member.first_tile;
                    if constexpr (Count) counts[(int64_t(b) * gridDim.x + blockIdx.x) * atoms + atom] += 1;
                }
                __syncthreads(); // Publish this chunk to every consumer warp.
                if (bank.level >= start) {
                    int warp = threadIdx.x / 32;
                    if (warp < warps)
                        for (int local = warp; local < destinations; local += warps) {
                            int node = bank.node_offset + first_tile + local;
                            cached_coefficient_tile(cache, first_tiles, offsets,
                                scratch + int64_t(node - prefix) * features,
                                columns + warp * features * BLOCK_LOCAL_C_STRIDE,
                                chunk, retained, node, axis, bank, stride);
                        }
                } else if (site < end_site) {
                    direct += cached_direct_site(cache, first_tiles, offsets,
                        chunk, retained, site, axis, bank, stride);
                }
                __syncthreads(); // Finish all reads before reusing the cache.
            }
        }
    }
    if (site < end_site)
        y[int64_t(b) * axis.sites + site] = T(direct + hybrid_site(
            p, rows, offsets, response, scratch, atoms, 0, site,
            axis, output_side, start, retained_banks));
}
} // namespace radius_cell_mass
