#pragma once
// Destination-owner CTAs finish each source response once, then reuse that
// atom's profile for destination coefficients. No atom values are reordered.
namespace radius_cell_mass {
constexpr int OWNER_BATCH = 4, OWNER_FEATURES = 17, OWNER_PARTS = 3;
constexpr int OWNER_CAPACITY = 128, OWNER_STRIDE = 14, OWNER_WARPS = 4;
static_assert(THREADS == OWNER_CAPACITY, "one thread must publish every cached atom");
constexpr size_t OWNER_SHARED_BYTES = sizeof(double) *
    (OWNER_BATCH * OWNER_PARTS * OWNER_FEATURES +
     OWNER_WARPS * OWNER_FEATURES * BLOCK_LOCAL_C_STRIDE + OWNER_CAPACITY * OWNER_STRIDE) +
    sizeof(int) * OWNER_CAPACITY;

// Same part/harmonic/batch order and expressions as parallel_response_kernel.
// c is generated once by the owning atom thread and then reused for C below.
template<class T>
__device__ inline void owner_complete_response(const T *signal, const double *z,
    double *values, int batch, Axis axis, Membership member, double mu,
    const double *c, int start) {
    Bank bank = axis.banks[member.bank];
    int features = axis.basis_count();
    int elements = (axis.node_count - prefix_nodes(axis, start)) * features;
    for (int part = 0; part < bank.count; ++part) {
        if (bank.level < start) {
            for (int local = 0; local < bank.tile_sites; ++local) {
                int site = (member.first_tile + part) * bank.tile_sites + local;
                double mass = axis.spacing * c[0];
                HarmonicPhase phase(2 * PI / bank.period * (mu - site * axis.spacing));
                double half_cell = PI * axis.spacing / bank.period;
                HarmonicPhase cell_phase(half_cell);
                for (int k = 1; k <= axis.modes; ++k) {
                    double argument = PI * k * axis.spacing / bank.period;
                    double cp, sp, cc, ss; phase.next(cp, sp); cell_phase.next(cc, ss);
                    double filter = axis.spacing * ss / argument;
                    mass += 2 * c[k] * filter * cp;
                }
                #pragma unroll
                for (int b = 0; b < OWNER_BATCH; ++b)
                    if (b < batch) values[b] += double(signal[int64_t(b) * axis.sites +
                        positive_mod(site, axis.sites)]) * mass;
            }
        } else {
            int unwrapped_tile = member.first_tile + part;
            int node = bank.node_offset + positive_mod(unwrapped_tile, bank.tile_count);
            int relative = (node - prefix_nodes(axis, start)) * features;
            double displacement = mu - unwrapped_tile * bank.tile_sites * axis.spacing;
            #pragma unroll
            for (int b = 0; b < OWNER_BATCH; ++b)
                if (b < batch) values[b] += c[0] * z[int64_t(b) * elements + relative];
            HarmonicPhase phase(2 * PI / bank.period * displacement);
            for (int k = 1; k <= axis.modes; ++k) {
                double cp, sp; phase.next(cp, sp);
                #pragma unroll
                for (int b = 0; b < OWNER_BATCH; ++b) {
                    if (b < batch) {
                        const double *tile = z + int64_t(b) * elements + relative;
                        double moment = cp * tile[2 * k - 1] + sp * tile[2 * k];
                        values[b] += 2 * c[k] * moment;
                    }
                }
            }
        }
    }
}

__device__ inline void owner_columns(const double *c, int first_tile, int part,
    Axis axis, Bank bank, double *columns, int stride, int lane) {
    int unwrapped = first_tile + part;
    double displacement = c[axis.modes + 1] - unwrapped * bank.tile_sites * axis.spacing;
    columns[lane] = c[0];
    HarmonicPhase phase(2 * PI * displacement / bank.period);
    for (int k = 1; k <= axis.modes; ++k) {
        double cp, sp; phase.next(cp, sp);
        columns[(2 * k - 1) * stride + lane] = 2 * c[k] * cp;
        columns[2 * k * stride + lane] = 2 * c[k] * sp;
    }
}

__device__ inline double owner_direct_mass(const double *c, int first_tile,
    int part, int local, Axis axis, Bank bank) {
    int site = (first_tile + part) * bank.tile_sites + local;
    double mass = axis.spacing * c[0];
    HarmonicPhase phase(2 * PI * (c[axis.modes + 1] - site * axis.spacing) / bank.period);
    double half_cell = PI * axis.spacing / bank.period;
    HarmonicPhase cell_phase(half_cell);
    for (int k = 1; k <= axis.modes; ++k) {
        double cp, sp, cc, ss; phase.next(cp, sp); cell_phase.next(cc, ss);
        double argument = PI * k * axis.spacing / bank.period;
        mass += 2 * c[k] * axis.spacing * ss / argument * cp;
    }
    return mass;
}

__device__ inline int64_t owner_partial_index(int b, int owner, int part, int f, Axis axis) {
    return ((int64_t(b) * axis.node_count + owner) * OWNER_PARTS + part) * OWNER_FEATURES + f;
}

// Each atom thread publishes COMPLETE H/G, not an owner-local response. Each
// warp then reduces one distinct part, including repeated chart images.
template<class T>
__global__ void owner_response_coefficients_kernel(const T *p, const T *signal,
    const double *z, double *response, double *partials, const int64_t *rows,
    const int64_t *offsets, int atoms, int batch, Axis source, Axis destination,
    int start, bool source_output_side) {
    int owner = blockIdx.x;
    int begin = int(offsets[owner]), end = int(offsets[owner + 1]);
    if (begin == end) return; // uniform, before every CTA barrier
    Bank bank = destination.banks[bank_for_node(owner, destination)];
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
            double c[17], values[OWNER_BATCH] = {};
            profile_values(R, source.banks[member.bank].period, source.modes, c);
            owner_complete_response(signal, z, values, batch, source, member,
                wrapped_center(original, source), c, start);
            double *saved = cache + threadIdx.x * OWNER_STRIDE;
            for (int k = 0; k <= destination.modes; ++k) saved[k] = c[k];
            double dest_original = center(p, atom, destination, !source_output_side);
            Membership dest_member = classify(R, dest_original, destination);
            saved[destination.modes + 1] = wrapped_center(dest_original, destination);
            first_tiles[threadIdx.x] = dest_member.first_tile;
            double amp = amplitude(p, atom, source);
            #pragma unroll
            for (int b = 0; b < OWNER_BATCH; ++b) {
                if (b < batch) response[int64_t(b) * atoms + atom] = values[b];
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
        partials[owner_partial_index(b, owner, part, f, destination)] =
            coefficients[(b * OWNER_PARTS + part) * OWNER_FEATURES + f];
    }
}

// Unwritten empty-owner and direct-tier padding slots must never be read.
// Sum the K coefficient contributions before integrating the bank at a site.
__device__ inline double owner_partial_site(const double *partials, const int64_t *offsets,
    int b, int site, Axis axis, int start) {
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
            double entry = partials[owner_partial_index(b, owner, part,
                bank.level < start ? local : 0, axis)];
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
                c += partials[owner_partial_index(b, owners[part], part, 2 * k - 1, axis)];
                s += partials[owner_partial_index(b, owners[part], part, 2 * k, axis)];
            }
            double cp, sp, cc, ss; phase.next(cp, sp); cell_phase.next(cc, ss);
            double argument = PI * k * axis.spacing / bank.period;
            value += axis.spacing * ss / argument * (c * cp + s * sp);
        }
    }
    return value;
}

template<class T>
__global__ void owner_expand_kernel(const double *partials, const int64_t *offsets,
    T *y, int batch, Axis axis, int start) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch * axis.sites)
        y[index] = T(owner_partial_site(partials, offsets, index / axis.sites,
            index % axis.sites, axis, start));
}

// Host orchestration: this temporary is released after Y/dX is enqueued and
// is never saved in the autograd context. Everything uses the current stream.
inline void check_owner_axes(const Axis &input, const Axis &output, int batch,
    int input_start, int output_start, bool merge, int tile_sites) {
    TORCH_CHECK(batch <= OWNER_BATCH && input_start == 2 && output_start == 2 &&
        merge && tile_sites == 32, "owner response coefficients require B<=4, merge, cutoffs2 and tile32");
    for (const Axis *a : {&input, &output})
        TORCH_CHECK(a->sites == 1024 && a->polar && a->modes == 8 &&
            a->base_tile_sites == 1 && a->merge_factor == 2,
            "owner response coefficients require N1024 Polar M8 base1 merge2");
    TORCH_CHECK(input.spacing == output.spacing && input.origin == output.origin &&
        input.amplitude_max == output.amplitude_max && input.w_c == output.w_c &&
        input.kappa == output.kappa && input.lower_kappa == output.lower_kappa &&
        input.upper_decay_power == output.upper_decay_power && input.radius_min == output.radius_min &&
        input.radius_birth == output.radius_birth && input.radius_max == output.radius_max &&
        input.upper_floor == output.upper_floor, "owner response coefficients require identical live axes");
    static_assert(OWNER_SHARED_BYTES <= 48 * 1024);
}

inline void check_owner_saved(Tensor x, Tensor p, Tensor zx, Tensor h, Tensor keys,
    Tensor dy, Config input_config, Config output_config, bool needs_x) {
    // Saved-tensor hooks may replace metadata. Check ALL retained operands
    // before sorting or launching the reverse owner stage, including Zx/H
    // that will only be consumed by the later dP launch.
    check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    check_hybrid_signal(x, p, input); check_hybrid_signal(dy, p, output);
    check_owner_axes(input, output, x.size(0), 2, 2, true, 32);
    check_compact(zx, p, input, 2); check_response(h, p, x.size(0));
    TORCH_CHECK(x.size(0) == dy.size(0) && zx.size(0) == x.size(0), "owner saved batches differ");
    if (needs_x) {
        check_layout_keys(keys, input.node_count, output.node_count);
        TORCH_CHECK(keys.device() == p.device() && keys.numel() == p.size(0), "owner saved keys differ");
    }
}

inline std::vector<Tensor> owner_transform(Tensor x, Tensor p, Tensor rows, Tensor offsets,
    Config source_config, Config destination_config, int source_start, int destination_start,
    bool merge, int tile_sites, bool reverse) {
    check_atoms(p);
    Axis source = parse_axis(source_config), destination = parse_axis(destination_config);
    check_hybrid_signal(x, p, source); check_ranges(rows, offsets, p, destination);
    check_owner_axes(source, destination, x.size(0), source_start, destination_start, merge, tile_sites);
    c10::cuda::CUDAGuard guard(p.device());
    auto z = build_compact_moments_active(x, p, source_config, merge, source_start);
    auto response = torch::empty({x.size(0), p.size(0)}, x.options().dtype(torch::kDouble));
    auto partials = torch::empty({x.size(0), destination.node_count, OWNER_PARTS, OWNER_FEATURES},
        x.options().dtype(torch::kDouble));
    auto y = torch::empty({x.size(0), destination.sites}, x.options());
    if (x.size(0)) {
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_owner_response_coefficients", [&] {
            owner_response_coefficients_kernel<scalar_t><<<destination.node_count, THREADS,
                OWNER_SHARED_BYTES, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(), z.data_ptr<double>(), response.data_ptr<double>(),
                partials.data_ptr<double>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                p.size(0), x.size(0), source, destination, source_start, reverse);
            owner_expand_kernel<scalar_t><<<(y.numel() + THREADS - 1) / THREADS, THREADS, 0,
                at::cuda::getCurrentCUDAStream()>>>(partials.data_ptr<double>(), offsets.data_ptr<int64_t>(),
                y.data_ptr<scalar_t>(), x.size(0), destination, destination_start);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {y, z, response};
}
} // namespace radius_cell_mass
