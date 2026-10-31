#pragma once
#include "owner_response_derivatives.cuh"
#include "owner_partial_layout.cuh"

namespace radius_cell_mass {
// Explicit mode9: literal mode7 floating expressions/order; only temporary
// destination partial addresses differ. Every response remains complete FP64.
template<class T>
__global__ void owner_compact_partials_kernel(const T *p, const T *signal,
    const double *z, double *response, double *derivative, double *partials, const int64_t *rows,
    const int64_t *offsets, int atoms, int batch, Axis source, Axis destination,
    int start, bool source_output_side, bool need_derivative, OwnerPartialLayout layout) {
    int owner = blockIdx.x;
    int begin = int(offsets[owner]), end = int(offsets[owner + 1]);
    if (begin == end) return; // uniform, before every CTA barrier
    int bank_index = bank_for_node(owner, destination);
    Bank bank = destination.banks[bank_index];
    extern __shared__ double owner_scratch[];
    double *coefficients = owner_scratch;
    double *columns = coefficients + OWNER_BATCH * OWNER_PARTS * OWNER_FEATURES;
    double *cache = columns + OWNER_WARPS * OWNER_FEATURES * BLOCK_LOCAL_C_STRIDE;
    int *first_tiles = reinterpret_cast<int *>(cache + OWNER_CAPACITY * OWNER_STRIDE);
    for (int i = threadIdx.x; i < OWNER_BATCH * OWNER_PARTS * OWNER_FEATURES; i += blockDim.x)
        coefficients[i] = 0;
    __syncthreads();
    for (int chunk = begin; chunk < end; chunk += OWNER_CAPACITY) {
        int retained = min(OWNER_CAPACITY, end - chunk);
        if (threadIdx.x < retained) {
            int atom = int(rows[chunk + threadIdx.x]);
            double R = radius(p, atom, source), original = center(p, atom, source, source_output_side);
            Membership member = classify(R, original, source);
            double c[17], values[OWNER_BATCH] = {}, derivatives[OWNER_BATCH] = {};
            profile_values(R, source.banks[member.bank].period, source.modes, c);
            owner_complete_response_derivatives(signal, z, values, derivatives, batch, source, member,
                wrapped_center(original, source), c, start, need_derivative);
            double *saved = cache + threadIdx.x * OWNER_STRIDE;
            for (int k = 0; k <= destination.modes; ++k) saved[k] = c[k];
            double dest_original = center(p, atom, destination, !source_output_side);
            Membership dest_member = classify(R, dest_original, destination);
            saved[destination.modes + 1] = wrapped_center(dest_original, destination);
            first_tiles[threadIdx.x] = dest_member.first_tile;
            double amp = amplitude(p, atom, source);
            #pragma unroll
            for (int b = 0; b < OWNER_BATCH; ++b) {
                if (b < batch) {
                    response[int64_t(b) * atoms + atom] = values[b];
                    if (need_derivative) derivative[int64_t(b) * atoms + atom] = derivatives[b];
                }
                saved[destination.modes + 2 + b] = b < batch ? amp * values[b] : 0;
            }
        }
        __syncthreads();
        int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        if (bank.level >= start) {
            if (warp < bank.count) {
                double sum[OWNER_BATCH] = {};
                double *warp_columns = columns + warp * OWNER_FEATURES * BLOCK_LOCAL_C_STRIDE;
                for (int group = 0; group < retained; group += 32) {
                    int count = min(32, retained - group);
                    if (lane < count) owner_columns(cache + (group + lane) * OWNER_STRIDE,
                        first_tiles[group + lane], warp, destination, bank,
                        warp_columns, BLOCK_LOCAL_C_STRIDE, lane);
                    __syncwarp();
                    if (lane < OWNER_FEATURES) {
                        for (int atom = 0; atom < count; ++atom) {
                            const double *saved = cache + (group + atom) * OWNER_STRIDE;
                            double coefficient = warp_columns[lane * BLOCK_LOCAL_C_STRIDE + atom];
                            #pragma unroll
                            for (int b = 0; b < OWNER_BATCH; ++b)
                                sum[b] += saved[destination.modes + 2 + b] * coefficient;
                        }
                    }
                    __syncwarp();
                }
                if (lane < OWNER_FEATURES) {
                    #pragma unroll
                    for (int b = 0; b < OWNER_BATCH; ++b)
                        coefficients[(b * OWNER_PARTS + warp) * OWNER_FEATURES + lane] += sum[b];
                }
            }
        } else if (threadIdx.x < bank.count * bank.tile_sites) {
            int part = threadIdx.x / bank.tile_sites, local = threadIdx.x % bank.tile_sites;
            double sum[OWNER_BATCH] = {};
            for (int atom = 0; atom < retained; ++atom) {
                const double *saved = cache + atom * OWNER_STRIDE;
                double mass = owner_direct_mass(saved, first_tiles[atom], part, local, destination, bank);
                #pragma unroll
                for (int b = 0; b < OWNER_BATCH; ++b)
                    sum[b] += saved[destination.modes + 2 + b] * mass;
            }
            #pragma unroll
            for (int b = 0; b < OWNER_BATCH; ++b)
                coefficients[(b * OWNER_PARTS + part) * OWNER_FEATURES + local] += sum[b];
        }
        __syncthreads(); // all cache/column readers finish before the next chunk
    }
    int fields = bank.level < start ? bank.tile_sites : OWNER_FEATURES;
    for (int i = threadIdx.x; i < batch * bank.count * fields; i += blockDim.x) {
        int f = i % fields, part = (i / fields) % bank.count, b = i / (fields * bank.count);
        partials[compact_owner_partial_index(b, owner, bank_index, part, f, destination, layout)] =
            coefficients[(b * OWNER_PARTS + part) * OWNER_FEATURES + f];
    }
}

// Sum the K coefficient contributions before integrating the bank at a site.
__device__ inline double compact_owner_partial_site(const double *partials, const int64_t *offsets,
    int b, int site, Axis axis, int start, const OwnerPartialLayout &layout) {
    double value = 0;
    for (int bi = 0; bi < axis.bank_count; ++bi) {
        Bank bank = axis.banks[bi];
        if (offsets[bank.node_offset] == offsets[bank.node_offset + bank.tile_count]) continue;
        int tile = site / bank.tile_sites, local = site % bank.tile_sites;
        int owners[OWNER_PARTS], active = 0;
        double dc = 0;
        for (int part = 0; part < bank.count; ++part) {
            int owner = bank.node_offset + positive_mod(tile - part, bank.tile_count);
            owners[part] = owner;
            if (offsets[owner] == offsets[owner + 1]) continue;
            active |= 1 << part;
            double entry = partials[compact_owner_partial_index(b, owner, bi, part,
                bank.level < start ? local : 0, axis, layout)];
            if (bank.level < start) value += entry;
            else dc += entry;
        }
        if (bank.level < start) continue;
        value += axis.spacing * dc;
        double half_cell = PI * axis.spacing / bank.period;
        HarmonicPhase phase(2 * half_cell * local), cell_phase(half_cell);
        for (int k = 1; k <= axis.modes; ++k) {
            double c = 0, s = 0;
            for (int part = 0; part < bank.count; ++part) {
                if (!(active & (1 << part))) continue;
                c += partials[compact_owner_partial_index(b, owners[part], bi, part, 2 * k - 1, axis, layout)];
                s += partials[compact_owner_partial_index(b, owners[part], bi, part, 2 * k, axis, layout)];
            }
            double cp, sp, cc, ss; phase.next(cp, sp); cell_phase.next(cc, ss);
            double argument = PI * k * axis.spacing / bank.period;
            value += axis.spacing * ss / argument * (c * cp + s * sp);
        }
    }
    return value;
}

template<class T>
__global__ void owner_compact_expand_kernel(const double *partials, const int64_t *offsets,
    T *y, int batch, Axis axis, int start, OwnerPartialLayout layout) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch * axis.sites)
        y[index] = T(compact_owner_partial_site(partials, offsets, index / axis.sites,
            index % axis.sites, axis, start, layout));
}

std::vector<Tensor> owner_compact_partials_transform(Tensor x, Tensor p, Tensor rows,
    Tensor offsets, Config source_config, Config destination_config,
    bool reverse, bool need_derivative) {
    check_atoms(p);
    Axis source = parse_axis(source_config), destination = parse_axis(destination_config);
    check_hybrid_signal(x, p, source); check_ranges(rows, offsets, p, destination);
    check_owner_axes(source, destination, x.size(0), 2, 2, true, 32);
    c10::cuda::CUDAGuard guard(p.device());
    auto z = build_compact_moments_active(x, p, source_config, true, 2);
    auto response = torch::empty({x.size(0), p.size(0)}, x.options().dtype(torch::kDouble));
    auto derivative = need_derivative ? torch::empty_like(response) : Tensor();
    auto layout = compact_owner_partial_layout(destination, 2);
    TORCH_CHECK(x.size(0) <= (INT_MAX - THREADS) / layout.total_fields, "compact owner work limit");
    auto partials = torch::empty({x.size(0), layout.total_fields},
        x.options().dtype(torch::kDouble));
    auto y = torch::empty({x.size(0), destination.sites}, x.options());
    if (x.size(0)) {
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_owner_response_derivatives", [&] {
            owner_compact_partials_kernel<scalar_t><<<destination.node_count, THREADS,
                OWNER_SHARED_BYTES, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(), z.data_ptr<double>(), response.data_ptr<double>(),
                need_derivative ? derivative.data_ptr<double>() : nullptr,
                partials.data_ptr<double>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                p.size(0), x.size(0), source, destination, 2, reverse, need_derivative, layout);
            owner_compact_expand_kernel<scalar_t><<<(y.numel() + THREADS - 1) / THREADS, THREADS, 0,
                at::cuda::getCurrentCUDAStream()>>>(partials.data_ptr<double>(), offsets.data_ptr<int64_t>(),
                y.data_ptr<scalar_t>(), x.size(0), destination, 2, layout);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {y, response, derivative};
}

} // namespace radius_cell_mass
