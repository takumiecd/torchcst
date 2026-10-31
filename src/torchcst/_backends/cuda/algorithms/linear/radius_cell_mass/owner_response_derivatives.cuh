#pragma once
// Opt-in derivative responses. The original owner kernels stay literal.
// H/G remain complete. D_H/D_G use the same moments and phase, preserving each
// center derivative's part/k/row order from compact_derivatives_batch4.
namespace radius_cell_mass {
template<class T>
__device__ inline void owner_complete_response_derivatives(const T *signal, const double *z,
    double *values, double *derivatives, int batch, Axis axis, Membership member, double mu,
    const double *c, int start, bool need_derivative) {
    Bank bank = axis.banks[member.bank];
    int features = axis.basis_count();
    int elements = (axis.node_count - prefix_nodes(axis, start)) * features;
    for (int part = 0; part < bank.count; ++part) {
        if (bank.level < start) {
            for (int local = 0; local < bank.tile_sites; ++local) {
                int site = (member.first_tile + part) * bank.tile_sites + local;
                double mass = axis.spacing * c[0], slope = 0;
                HarmonicPhase phase(2 * PI / bank.period * (mu - site * axis.spacing));
                double half_cell = PI * axis.spacing / bank.period;
                HarmonicPhase cell_phase(half_cell);
                for (int k = 1; k <= axis.modes; ++k) {
                    double argument = PI * k * axis.spacing / bank.period;
                    double cp, sp, cc, ss; phase.next(cp, sp); cell_phase.next(cc, ss);
                    double filter = axis.spacing * ss / argument;
                    mass += 2 * c[k] * filter * cp;
                    if (need_derivative) slope -= 2 * c[k] * filter *
                        (2 * PI * k / bank.period) * sp;
                }
                #pragma unroll
                for (int b = 0; b < OWNER_BATCH; ++b)
                    if (b < batch) {
                        double value = double(signal[int64_t(b) * axis.sites +
                            positive_mod(site, axis.sites)]);
                        values[b] += value * mass;
                        if (need_derivative) derivatives[b] += value * slope;
                    }
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
                        if (need_derivative) derivatives[b] += 2 * c[k] *
                            (2 * PI * k / bank.period) *
                            (-sp * tile[2 * k - 1] + cp * tile[2 * k]);
                    }
                }
            }
        }
    }
}

template<class T>
__global__ void owner_response_derivatives_kernel(const T *p, const T *signal,
    const double *z, double *response, double *derivative, double *partials, const int64_t *rows,
    const int64_t *offsets, int atoms, int batch, Axis source, Axis destination,
    int start, bool source_output_side, bool need_derivative) {
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
        partials[owner_partial_index(b, owner, part, f, destination)] =
            coefficients[(b * OWNER_PARTS + part) * OWNER_FEATURES + f];
    }
}

// Atom-only backward completes G/D_G without destination C, sorting or dX.
template<class T>
__global__ void owner_derivative_project_kernel(const T *p, const T *signal,
    const double *z, double *response, double *derivative, int atoms, int batch,
    Axis axis, bool output_side) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    double R = radius(p, atom, axis), original = center(p, atom, axis, output_side);
    Membership member = classify(R, original, axis);
    double c[17], values[OWNER_BATCH] = {}, slopes[OWNER_BATCH] = {};
    profile_values(R, axis.banks[member.bank].period, axis.modes, c);
    owner_complete_response_derivatives(signal, z, values, slopes, batch, axis,
        member, wrapped_center(original, axis), c, 2, true);
    for (int b = 0; b < batch; ++b) {
        response[int64_t(b) * atoms + atom] = values[b];
        derivative[int64_t(b) * atoms + atom] = slopes[b];
    }
}

// Geometry was consumed while making H/D_H and G/D_G. Only the amplitude
// coordinate chain rule and the original batch-ordered contractions remain.
template<class T>
__global__ void owner_derivative_parameter_kernel(const T *p, const double *h,
    const double *g, const double *dh, const double *dg, T *gradient,
    int atoms, int batch, Axis input) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    double da = 0, dmi = 0, dmo = 0, amp = amplitude(p, atom, input);
    for (int b = 0; b < batch; ++b) {
        int64_t index = int64_t(b) * atoms + atom;
        da += h[index] * g[index];
        dmi += amp * g[index] * dh[index];
        dmo += amp * h[index] * dg[index];
    }
    // Polar radius is detached from task gradients; no dR or decay is added.
    store_gradient(p, gradient, atom, input, da, 0, dmi, dmo);
}

// Host orchestration: unlike mode6, the returned tensors never contain Z.
// The local active hierarchy is released after all of its readers are enqueued
// on the current stream. PyTorch's allocator preserves that stream dependency.
// Saved D_H replaces saved Zx; no atom or profile table is retained.
std::vector<Tensor> owner_derivative_transform(Tensor x, Tensor p, Tensor rows,
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
    auto partials = torch::empty({x.size(0), destination.node_count, OWNER_PARTS, OWNER_FEATURES},
        x.options().dtype(torch::kDouble));
    auto y = torch::empty({x.size(0), destination.sites}, x.options());
    if (x.size(0)) {
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_owner_response_derivatives", [&] {
            owner_response_derivatives_kernel<scalar_t><<<destination.node_count, THREADS,
                OWNER_SHARED_BYTES, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(), z.data_ptr<double>(), response.data_ptr<double>(),
                need_derivative ? derivative.data_ptr<double>() : nullptr,
                partials.data_ptr<double>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                p.size(0), x.size(0), source, destination, 2, reverse, need_derivative);
            owner_expand_kernel<scalar_t><<<(y.numel() + THREADS - 1) / THREADS, THREADS, 0,
                at::cuda::getCurrentCUDAStream()>>>(partials.data_ptr<double>(), offsets.data_ptr<int64_t>(),
                y.data_ptr<scalar_t>(), x.size(0), destination, 2);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {y, response, derivative};
}

std::vector<Tensor> owner_derivative_project(Tensor x, Tensor p, Config config) {
    check_atoms(p);
    Axis axis = parse_axis(config);
    check_hybrid_signal(x, p, axis);
    check_owner_axes(axis, axis, x.size(0), 2, 2, true, 32);
    c10::cuda::CUDAGuard guard(p.device());
    auto z = build_compact_moments_active(x, p, config, true, 2);
    auto response = torch::empty({x.size(0), p.size(0)}, x.options().dtype(torch::kDouble));
    auto derivative = torch::empty_like(response);
    if (p.size(0) && x.size(0)) {
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_owner_derivative_project", [&] {
            owner_derivative_project_kernel<scalar_t><<<(p.size(0) + THREADS - 1) / THREADS,
                THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(), z.data_ptr<double>(), response.data_ptr<double>(),
                derivative.data_ptr<double>(), p.size(0), x.size(0), axis, true);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {response, derivative};
}

void check_owner_derivative_saved(Tensor x, Tensor p, Tensor h, Tensor dh,
    Tensor keys, Tensor dy, Config input_config, Config output_config,
    bool needs_x, bool needs_p) {
    check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    check_hybrid_signal(x, p, input); check_hybrid_signal(dy, p, output);
    check_owner_axes(input, output, x.size(0), 2, 2, true, 32);
    TORCH_CHECK(x.size(0) == dy.size(0), "owner derivative saved batches differ");
    check_response(h, p, x.size(0));
    if (needs_p) check_response(dh, p, x.size(0));
    if (needs_x) {
        check_layout_keys(keys, input.node_count, output.node_count);
        TORCH_CHECK(keys.device() == p.device() && keys.numel() == p.size(0),
            "owner derivative saved keys differ");
    }
}

Tensor owner_derivative_parameter_grad(Tensor p, Tensor h, Tensor g, Tensor dh,
    Tensor dg, Config input_config, Config output_config) {
    check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(h.dim() == 2, "expected complete response matrix");
    check_owner_axes(input, output, h.size(0), 2, 2, true, 32);
    for (const Tensor &value : {h, g, dh, dg}) check_response(value, p, h.size(0));
    c10::cuda::CUDAGuard guard(p.device());
    auto gradient = torch::empty_like(p);
    if (p.size(0)) {
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_owner_derivative_parameter", [&] {
            owner_derivative_parameter_kernel<scalar_t><<<(p.size(0) + THREADS - 1) / THREADS,
                THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<scalar_t>(),
                h.data_ptr<double>(), g.data_ptr<double>(), dh.data_ptr<double>(),
                dg.data_ptr<double>(), gradient.data_ptr<scalar_t>(), p.size(0), h.size(0), input);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return gradient;
}
} // namespace radius_cell_mass

