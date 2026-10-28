#pragma once
// Low hierarchy levels use direct integrated cells; retained levels have compact
// FP64 storage. H is always complete before its output contribution is consumed.
namespace radius_cell_mass {
__host__ __device__ inline int prefix_nodes(const Axis &a, int start) {
    return start == a.bank_count / 2 ? a.node_count : a.banks[2 * start].node_offset;
}
inline int checked_start(const Axis &a, int start) {
    TORCH_CHECK(start >= 0 && start <= a.bank_count / 2, "invalid compact start level");
    return prefix_nodes(a, start);
}
inline void check_compact(const Tensor &z, const Tensor &p, const Axis &a, int start) {
    int prefix = checked_start(a, start);
    TORCH_CHECK(z.device() == p.device() && z.scalar_type() == torch::kDouble &&
        z.is_contiguous() && z.dim() == 3 && z.size(1) == a.node_count - prefix &&
        z.size(2) == a.basis_count(), "invalid compact moments/coefficients");
    TORCH_CHECK(z.numel() <= INT_MAX - THREADS, "compact work exceeds native index limit");
}
inline void check_response(const Tensor &h, const Tensor &p, int64_t batch) {
    TORCH_CHECK(h.device() == p.device() && h.scalar_type() == torch::kDouble &&
        h.is_contiguous() && h.dim() == 2 && h.size(0) == batch &&
        h.size(1) == p.size(0) && h.numel() <= INT_MAX - THREADS, "invalid response");
}
inline void check_ranges(const Tensor &rows, const Tensor &offsets, const Tensor &p, const Axis &a) {
    TORCH_CHECK(rows.device() == p.device() && offsets.device() == p.device() &&
        rows.scalar_type() == torch::kLong && offsets.scalar_type() == torch::kLong &&
        rows.is_contiguous() && offsets.is_contiguous() && rows.dim() == 1 && offsets.dim() == 1 &&
        rows.numel() == p.size(0) && offsets.numel() == a.node_count + 1, "invalid sorted ranges");
}
template<class T>
__device__ inline double compact_moment_element(const T *signal, const double *z,
    int b, int node, int basis, Axis axis, int start, bool merge) {
    int features = axis.basis_count(), bank_index = bank_for_node(node, axis);
    Bank bank = axis.banks[bank_index];
    int tile = node - bank.node_offset;
    int harmonic = (basis + 1) / 2;
    bool sine = basis != 0 && basis % 2 == 0;
    double sum = 0;
    if (merge && bank.level > start && harmonic % axis.merge_factor == 0) {
        Bank child = axis.banks[bank_index - 2];
        int child_harmonic = harmonic / axis.merge_factor;
        for (int part = 0; part < axis.merge_factor; ++part) {
            int child_node = child.node_offset + tile * axis.merge_factor + part;
            int64_t offset = (int64_t(b) * (axis.node_count - prefix_nodes(axis, start)) + child_node - prefix_nodes(axis, start)) * features;
            if (basis == 0) { sum += z[offset]; continue; }
            double phase = 2 * PI * harmonic * part * child.tile_sites * axis.spacing / bank.period;
            double c = z[offset + 2 * child_harmonic - 1];
            double s = z[offset + 2 * child_harmonic];
            sum += sine ? sin(phase) * c + cos(phase) * s : cos(phase) * c - sin(phase) * s;
        }
    } else {
        double filter = axis.spacing;
        if (harmonic) {
            double argument = PI * harmonic * axis.spacing / bank.period;
            filter *= sin(argument) / argument;
        }
        for (int local_site = 0; local_site < bank.tile_sites; ++local_site) {
            int site = tile * bank.tile_sites + local_site;
            double phase = 2 * PI * harmonic * local_site * axis.spacing / bank.period;
            double phi = basis == 0 ? 1 : (sine ? sin(phase) : cos(phase));
            sum += double(signal[b * axis.sites + site]) * filter * phi;
        }
    }
    return sum;
}
template<class T>
__global__ void compact_moments_kernel(const T *signal, double *z, int batch,
    Axis axis, int start, int level, bool merge) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nodes = axis.node_count - prefix_nodes(axis, start), F = axis.basis_count();
    if (index >= batch * nodes * F) return;
    int node = prefix_nodes(axis, start) + (index / F) % nodes;
    if (level >= 0 && axis.banks[bank_for_node(node, axis)].level != level) return;
    z[index] = compact_moment_element(signal, z, index / (nodes * F), node, index % F, axis, start, merge);
}
__device__ inline double compact_project(const double *z, int b, const Axis &axis,
    Membership member, double mu, const double *c, const double *dr,
    int start, double &radius_gradient, double &center_gradient) {
    Bank bank = axis.banks[member.bank];
    double value = 0;
    radius_gradient = center_gradient = 0;
    for (int part = 0; part < bank.count; ++part) {
        int unwrapped_tile = member.first_tile + part;
        int node = bank.node_offset + positive_mod(unwrapped_tile, bank.tile_count);
        const double *tile = z + (int64_t(b) * (axis.node_count - prefix_nodes(axis, start)) + node - prefix_nodes(axis, start)) * axis.basis_count();
        double displacement = mu - unwrapped_tile * bank.tile_sites * axis.spacing;
        value += c[0] * tile[0];
        for (int k = 1; k <= axis.modes; ++k) {
            double omega = 2 * PI * k / bank.period;
            double cp = cos(omega * displacement), sp = sin(omega * displacement);
            double moment = cp * tile[2 * k - 1] + sp * tile[2 * k];
            value += 2 * c[k] * moment;
            radius_gradient += 2 * dr[k] * moment;
            center_gradient += 2 * c[k] * omega * (-sp * tile[2 * k - 1] + cp * tile[2 * k]);
        }
    }
    return value;
}
// Wrap the signal index only: unwrapped phases retain repeated chart images
// whenever the padded local Fourier period exceeds the chart period.
template<class T>
__device__ inline double direct_project(const T *signal, int b, Axis axis,
    Membership member, double mu, const double *c, const double *dr, double &dR, double &dmu) {
    Bank bank = axis.banks[member.bank];
    double value = 0; dR = dmu = 0;
    for (int part = 0; part < bank.count; ++part) {
        for (int local = 0; local < bank.tile_sites; ++local) {
            int site = (member.first_tile + part) * bank.tile_sites + local;
            double mass = axis.spacing * c[0], mr = 0, mm = 0;
            for (int k = 1; k <= axis.modes; ++k) {
                double omega = 2 * PI * k / bank.period;
                double argument = PI * k * axis.spacing / bank.period;
                double filter = axis.spacing * sin(argument) / argument;
                double phase = omega * (mu - site * axis.spacing);
                mass += 2 * c[k] * filter * cos(phase);
                mr += 2 * dr[k] * filter * cos(phase);
                mm -= 2 * c[k] * filter * omega * sin(phase);
            }
            double x = double(signal[int64_t(b) * axis.sites + positive_mod(site, axis.sites)]);
            value += x * mass; dR += x * mr; dmu += x * mm;
        }
    }
    return value;
}
template<class T, bool Direct>
__global__ void hybrid_response_kernel(const T *p, const T *signal, const double *z,
    double *h, int atoms, int batch, Axis axis, bool output_side, int start) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= atoms * batch) return;
    int atom = index % atoms, b = index / atoms;
    double R = radius(p, atom, axis), original = center(p, atom, axis, output_side);
    Membership member = classify(R, original, axis);
    if ((axis.banks[member.bank].level < start) != Direct) return;
    double c[17], dr[17], dR, dmu;
    profile_coefficients(R, axis.banks[member.bank].period, axis.modes, c, dr);
    double mu = wrapped_center(original, axis);
    h[index] = Direct ? direct_project(signal, b, axis, member, mu, c, dr, dR, dmu)
                     : compact_project(z, b, axis, member, mu, c, dr, start, dR, dmu);
}
template<class T>
__device__ inline void compact_coefficient_tile(const T *p, const int64_t *rows,
    const int64_t *offsets, const double *response, double *out, int atoms,
    int b, int node, Axis axis, bool output_side) {
    Bank bank = axis.banks[bank_for_node(node, axis)];
    int tile = node - bank.node_offset;
    double accumulator[33] = {};
    for (int part = 0; part < bank.count; ++part) {
        int first = positive_mod(tile - part, bank.tile_count);
        int first_node = bank.node_offset + first;
        for (int64_t slot = offsets[first_node]; slot < offsets[first_node + 1]; ++slot) {
            int atom = rows[slot];
            double R = radius(p, atom, axis);
            double original_center = center(p, atom, axis, output_side);
            Membership member = classify(R, original_center, axis);
            int unwrapped_tile = member.first_tile + part;
            double displacement = wrapped_center(original_center, axis) - unwrapped_tile * bank.tile_sites * axis.spacing;
            double c[17], dr[17];
            profile_coefficients(R, bank.period, axis.modes, c, dr);
            double value = amplitude(p, atom, axis) * response[b * atoms + atom];
            accumulator[0] += value * c[0];
            for (int k = 1; k <= axis.modes; ++k) {
                double phase = 2 * PI * k * displacement / bank.period;
                accumulator[2 * k - 1] += value * 2 * c[k] * cos(phase);
                accumulator[2 * k] += value * 2 * c[k] * sin(phase);
            }
        }
    }
    for (int f = 0; f < axis.basis_count(); ++f) out[f] = accumulator[f];
}
template<class T>
__global__ void compact_coefficients_kernel(const T *p, const int64_t *rows,
    const int64_t *offsets, const double *h, double *c, int atoms, int batch,
    Axis axis, bool output_side, int start) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nodes = axis.node_count - prefix_nodes(axis, start);
    if (index >= batch * nodes) return;
    compact_coefficient_tile(p, rows, offsets, h, c + int64_t(index) * axis.basis_count(),
        atoms, index / nodes, prefix_nodes(axis, start) + index % nodes, axis, output_side);
}
template<class T>
__device__ inline double hybrid_site(const T *p, const int64_t *rows,
    const int64_t *offsets, const double *h, const double *coefficients, int atoms,
    int b, int site, Axis axis, bool output_side, int start) {
    double value = 0;
    int prefix = prefix_nodes(axis, start), nodes = axis.node_count - prefix;
    for (int bi = 0; bi < axis.bank_count; ++bi) {
        Bank bank = axis.banks[bi];
        int tile = site / bank.tile_sites, local = site % bank.tile_sites;
        if (bank.level >= start) {
            int node = bank.node_offset + tile;
            const double *c = coefficients + (int64_t(b) * nodes + node - prefix) * axis.basis_count();
            value += axis.spacing * c[0];
            for (int k = 1; k <= axis.modes; ++k) {
                double argument = PI * k * axis.spacing / bank.period;
                double phase = 2 * argument * local;
                value += axis.spacing * sin(argument) / argument *
                    (c[2 * k - 1] * cos(phase) + c[2 * k] * sin(phase));
            }
            continue;
        }
        // Repeated wrapped first tiles represent separate unwrapped cells.
        for (int part = 0; part < bank.count; ++part) {
            int first_node = bank.node_offset + positive_mod(tile - part, bank.tile_count);
            for (int64_t slot = offsets[first_node]; slot < offsets[first_node + 1]; ++slot) {
                int atom = rows[slot];
                double R = radius(p, atom, axis), original = center(p, atom, axis, output_side);
                Membership member = classify(R, original, axis);
                int unwrapped_site = (member.first_tile + part) * bank.tile_sites + local;
                double c[17], dr[17];
                profile_coefficients(R, bank.period, axis.modes, c, dr);
                double mass = axis.spacing * c[0];
                for (int k = 1; k <= axis.modes; ++k) {
                    double argument = PI * k * axis.spacing / bank.period;
                    double phase = 2 * PI * k * (wrapped_center(original, axis) - unwrapped_site * axis.spacing) / bank.period;
                    mass += 2 * c[k] * axis.spacing * sin(argument) / argument * cos(phase);
                }
                value += amplitude(p, atom, axis) * h[int64_t(b) * atoms + atom] * mass;
            }
        }
    }
    return value;
}
template<class T>
__global__ void hybrid_expand_kernel(const T *p, const int64_t *rows, const int64_t *offsets,
    const double *h, const double *c, T *y, int atoms, int batch, Axis axis, bool output_side, int start) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch * axis.sites)
        y[index] = T(hybrid_site(p, rows, offsets, h, c, atoms, index / axis.sites,
                               index % axis.sites, axis, output_side, start));
}
template<class T>
__global__ void hybrid_parameter_kernel(const T *p, const T *x, const T *dy, const double *zx, const double *zy,
    const double *h, const double *g, T *gradient, int atoms, int batch, Axis input, Axis output, int input_start, int output_start) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    double ri = radius(p, atom, input), ro = radius(p, atom, output);
    double mui = wrapped_center(center(p, atom, input, false), input);
    double muo = wrapped_center(center(p, atom, output, true), output);
    Membership mi = classify(ri, center(p, atom, input, false), input);
    Membership mo = classify(ro, center(p, atom, output, true), output);
    double ci[17], di[17], co[17], dout[17];
    profile_coefficients(ri, input.banks[mi.bank].period, input.modes, ci, di);
    profile_coefficients(ro, output.banks[mo.bank].period, output.modes, co, dout);
    double da = 0, dR = 0, dmi = 0, dmo = 0, amp = amplitude(p, atom, input);
    for (int b = 0; b < batch; ++b) {
        double dri, dci, dro, dco;
        if (input.banks[mi.bank].level < input_start)
            direct_project(x, b, input, mi, mui, ci, di, dri, dci);
        else compact_project(zx, b, input, mi, mui, ci, di, input_start, dri, dci);
        if (output.banks[mo.bank].level < output_start)
            direct_project(dy, b, output, mo, muo, co, dout, dro, dco);
        else compact_project(zy, b, output, mo, muo, co, dout, output_start, dro, dco);
        double H = h[b * atoms + atom], G = g[b * atoms + atom];
        da += H * G;
        dR += amp * (G * dri + H * dro);
        dmi += amp * G * dci;
        dmo += amp * H * dco;
    }
    store_gradient(p, gradient, atom, input, da, dR, dmi, dmo);
}

Tensor build_compact_moments(Tensor signal, Config config, bool merge, int start) {
    check_matrix(signal);
    Axis axis = parse_axis(config);
    int nodes = axis.node_count - checked_start(axis, start), F = axis.basis_count();
    TORCH_CHECK(signal.size(1) == axis.sites, "signal must match axis site count");
    TORCH_CHECK(signal.size(0) <= (INT_MAX - THREADS) / axis.sites &&
        (!nodes || signal.size(0) <= (INT_MAX - THREADS) / (nodes * F)),
        "moment work exceeds native index limit");
    c10::cuda::CUDAGuard guard(signal.device());
    auto z = torch::empty({signal.size(0), nodes, F}, signal.options().dtype(torch::kDouble));
    if (!z.numel()) return z;
    int blocks = (z.numel() + THREADS - 1) / THREADS;
    AT_DISPATCH_FLOATING_TYPES(signal.scalar_type(), "radius_compact_moments", [&] {
        if (merge) {
            for (int level = start; level < axis.bank_count / 2; ++level)
                compact_moments_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                    signal.data_ptr<scalar_t>(), z.data_ptr<double>(), signal.size(0), axis, start, level, true);
        } else {
            compact_moments_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                signal.data_ptr<scalar_t>(), z.data_ptr<double>(), signal.size(0), axis, start, -1, false);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return z;
}

inline void check_hybrid_signal(const Tensor &signal, const Tensor &p, const Axis &axis) {
    check_matrix(signal);
    TORCH_CHECK(signal.device() == p.device() && signal.scalar_type() == p.scalar_type() &&
        signal.size(1) == axis.sites && signal.size(0) <= (INT_MAX - THREADS) / axis.sites,
        "invalid hybrid signal");
    TORCH_CHECK(p.size(0) == 0 || signal.size(0) <= (INT_MAX - THREADS) / p.size(0),
        "response work exceeds native index limit");
}
void direct_response_into(Tensor signal, Tensor p, Tensor h, Config config, bool output_side, int start) {
    check_atoms(p);
    Axis axis = parse_axis(config);
    checked_start(axis, start);
    check_hybrid_signal(signal, p, axis);
    check_response(h, p, signal.size(0));
    if (!h.numel() || start == 0) return;
    c10::cuda::CUDAGuard guard(p.device());
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_direct_response", [&] {
        hybrid_response_kernel<scalar_t, true><<<(h.numel() + THREADS - 1) / THREADS, THREADS, 0,
            at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<scalar_t>(), signal.data_ptr<scalar_t>(),
            nullptr, h.data_ptr<double>(), p.size(0), signal.size(0), axis, output_side, start);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
Tensor direct_response(Tensor signal, Tensor p, Config config, bool output_side, int start) {
    check_atoms(p);
    Axis axis = parse_axis(config);
    checked_start(axis, start);
    check_hybrid_signal(signal, p, axis);
    c10::cuda::CUDAGuard guard(p.device());
    auto h = torch::empty({signal.size(0), p.size(0)}, p.options().dtype(torch::kDouble));
    direct_response_into(signal, p, h, config, output_side, start);
    return h;
}
void complete_response(Tensor p, Tensor z, Tensor h, Config config, bool output_side, int start) {
    check_atoms(p);
    Axis axis = parse_axis(config);
    check_compact(z, p, axis, start);
    check_response(h, p, z.size(0));
    if (!h.numel() || start == axis.bank_count / 2) return;
    c10::cuda::CUDAGuard guard(p.device());
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_complete_response", [&] {
        hybrid_response_kernel<scalar_t, false><<<(h.numel() + THREADS - 1) / THREADS, THREADS, 0,
            at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<scalar_t>(), nullptr, z.data_ptr<double>(),
            h.data_ptr<double>(), p.size(0), z.size(0), axis, output_side, start);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
Tensor compact_coefficients(Tensor p, Tensor rows, Tensor offsets, Tensor h,
    Config config, bool output_side, int start) {
    check_atoms(p);
    Axis axis = parse_axis(config);
    int nodes = axis.node_count - checked_start(axis, start), F = axis.basis_count();
    check_ranges(rows, offsets, p, axis);
    TORCH_CHECK(h.dim() == 2, "invalid response rank");
    check_response(h, p, h.size(0));
    TORCH_CHECK(!nodes || h.size(0) <= (INT_MAX - THREADS) / (nodes * F),
        "coefficient work exceeds native index limit");
    c10::cuda::CUDAGuard guard(p.device());
    auto c = torch::empty({h.size(0), nodes, F}, h.options());
    if (!c.numel()) return c;
    int work = h.size(0) * nodes;
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_compact_coefficients", [&] {
        compact_coefficients_kernel<<<(work + THREADS - 1) / THREADS, THREADS, 0,
            at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(),
            offsets.data_ptr<int64_t>(), h.data_ptr<double>(), c.data_ptr<double>(), p.size(0),
            h.size(0), axis, output_side, start);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
Tensor expand_hybrid(Tensor p, Tensor rows, Tensor offsets, Tensor h, Tensor c,
    Tensor like, Config config, bool output_side, int start) {
    check_atoms(p); check_matrix(like);
    Axis axis = parse_axis(config);
    check_compact(c, p, axis, start);
    check_ranges(rows, offsets, p, axis);
    check_response(h, p, c.size(0));
    TORCH_CHECK(like.device() == p.device() && like.scalar_type() == p.scalar_type() &&
        c.size(0) <= (INT_MAX - THREADS) / axis.sites, "invalid expansion shape or dtype");
    c10::cuda::CUDAGuard guard(p.device());
    auto y = torch::empty({c.size(0), axis.sites}, like.options());
    if (!y.numel()) return y;
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_hybrid_expand", [&] {
        hybrid_expand_kernel<<<(y.numel() + THREADS - 1) / THREADS, THREADS, 0,
            at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(),
            offsets.data_ptr<int64_t>(), h.data_ptr<double>(), c.data_ptr<double>(),
            y.data_ptr<scalar_t>(), p.size(0), c.size(0), axis, output_side, start);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
Tensor hybrid_parameter_grad(Tensor p, Tensor x, Tensor dy, Tensor zx, Tensor zy,
    Tensor h, Tensor g, Config input_config, Config output_config, int input_start, int output_start) {
    check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    check_compact(zx, p, input, input_start); check_compact(zy, p, output, output_start);
    check_hybrid_signal(x, p, input); check_hybrid_signal(dy, p, output);
    TORCH_CHECK(x.size(0) == dy.size(0) && zx.size(0) == x.size(0) && zy.size(0) == x.size(0),
        "hybrid gradient batches differ");
    check_response(h, p, x.size(0)); check_response(g, p, x.size(0));
    c10::cuda::CUDAGuard guard(p.device());
    auto grad = torch::empty_like(p);
    if (!p.numel()) return grad;
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_hybrid_parameter_grad", [&] {
        hybrid_parameter_kernel<<<(p.size(0) + THREADS - 1) / THREADS, THREADS, 0,
            at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
            dy.data_ptr<scalar_t>(), zx.data_ptr<double>(), zy.data_ptr<double>(),
            h.data_ptr<double>(), g.data_ptr<double>(), grad.data_ptr<scalar_t>(), p.size(0),
            x.size(0), input, output, input_start, output_start);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}
} // namespace radius_cell_mass
