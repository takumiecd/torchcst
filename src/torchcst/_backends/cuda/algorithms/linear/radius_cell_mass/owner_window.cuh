#pragma once
#include "owner_compact_fused.cuh"

namespace radius_cell_mass {
struct OwnerWindowLayout {
    int offsets[32] = {}, ranks[32] = {}, total_fields = 0;
};
inline OwnerWindowLayout owner_window_layout(const Axis &axis, bool fold) {
    OwnerWindowLayout layout;
    for (int bi = 0; bi < axis.bank_count; ++bi) {
        const Bank &bank = axis.banks[bi];
        layout.offsets[bi] = layout.total_fields;
        layout.ranks[bi] =
            fold ? std::min(bank.count * bank.tile_sites, axis.basis_count()) : axis.basis_count();
        layout.total_fields += bank.tile_count * layout.ranks[bi];
    }
    return layout;
}
__device__ inline int64_t owner_window_index(int b, int bi, int owner, int f, const Axis &axis,
                                             const OwnerWindowLayout &layout) {
    return int64_t(b) * layout.total_fields + layout.offsets[bi] +
           (owner - axis.banks[bi].node_offset) * layout.ranks[bi] + f;
}
// Integration belongs to atom coefficients exactly once. Window moments here
// are raw sampled Fourier moments. Fold derivatives at the ORIGINAL frequency.
__device__ inline void owner_window_coefficients(const double *c, double mu, int first_tile,
                                                 Axis axis, Bank bank, bool fold, double *q,
                                                 double *dq) {
    int L = bank.count * bank.tile_sites;
    int rank = fold ? min(L, axis.basis_count()) : axis.basis_count();
    for (int f = 0; f < rank; ++f)
        q[f] = dq[f] = 0;
    q[0] = axis.spacing * c[0];
    HarmonicPhase phase(2 * PI / bank.period * (mu - first_tile * bank.tile_sites * axis.spacing));
    HarmonicPhase cell_phase(PI * axis.spacing / bank.period);
    for (int k = 1; k <= axis.modes; ++k) {
        double cp, sp, cc, ss;
        phase.next(cp, sp);
        cell_phase.next(cc, ss);
        double w = 2 * c[k] * axis.spacing * ss / (PI * k * axis.spacing / bank.period);
        double omega = 2 * PI * k / bank.period;
        int alias = fold ? k % L : k;
        double sign = 1;
        if (fold && alias > L / 2) {
            alias = L - alias;
            sign = -1;
        }
        int cosine = alias == 0 ? 0 : 2 * alias - 1;
        q[cosine] += w * cp;
        dq[cosine] -= w * omega * sp;
        if (alias != 0 && (!fold || 2 * alias != L)) {
            q[2 * alias] += sign * w * sp;
            dq[2 * alias] += sign * w * omega * cp;
        }
    }
}

template <class T>
__global__ void owner_window_moments_kernel(const T *signal, const double *z,
                                            const uint32_t *required, double *windows, int batch,
                                            Axis axis, OwnerWindowLayout layout) {
    int owner = blockIdx.x, bi = bank_for_node(owner, axis);
    if (!(*required & (uint32_t(1) << bi)))
        return;
    Bank bank = axis.banks[bi];
    int rank = layout.ranks[bi], L = bank.count * bank.tile_sites;
    int tile = owner - bank.node_offset;
    int features = axis.basis_count(),
        elements = (axis.node_count - prefix_nodes(axis, 2)) * features;
    for (int index = threadIdx.x; index < batch * rank; index += blockDim.x) {
        int b = index / rank, f = index % rank;
        int k = (f + 1) / 2;
        bool sine = f && !(f & 1);
        double value = 0;
        if (L <= 16) {
            for (int j = 0; j < L; ++j) {
                double cp, sp;
                sincos(2 * PI * k * j / L, &sp, &cp);
                value += double(signal[int64_t(b) * axis.sites +
                                       positive_mod(tile * bank.tile_sites + j, axis.sites)]) *
                         (sine ? sp : cp);
            }
        } else {
            double argument = PI * k / L;
            double filter = k ? axis.spacing * sin(argument) / argument : axis.spacing;
            for (int part = 0; part < bank.count; ++part) {
                int node = bank.node_offset + positive_mod(tile + part, bank.tile_count);
                const double *moments =
                    z + int64_t(b) * elements + (node - prefix_nodes(axis, 2)) * features;
                if (!k)
                    value += moments[0] / filter;
                else {
                    double cp, sp;
                    sincos(2 * PI * k * part / bank.count, &sp, &cp);
                    value += (sine ? sp * moments[2 * k - 1] + cp * moments[2 * k]
                                   : cp * moments[2 * k - 1] - sp * moments[2 * k]) /
                             filter;
                }
            }
        }
        windows[owner_window_index(b, bi, owner, f, axis, layout)] = value;
    }
}
inline Tensor build_owner_windows(Tensor signal, Tensor p, Tensor keys, Config input_config,
                                  Config output_config, bool reverse, bool fold) {
    Axis axis = parse_axis(reverse ? output_config : input_config);
    auto layout = owner_window_layout(axis, fold);
    auto required =
        active_moment_banks_from_keys(keys, input_config, output_config, reverse, false, 0);
    auto z = build_compact_moments_from_keys(signal, p, keys, input_config, output_config, reverse,
                                             true, 2);
    auto windows =
        torch::empty({signal.size(0), layout.total_fields}, signal.options().dtype(torch::kDouble));
    if (signal.size(0)) {
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_window_moments", [&] {
            owner_window_moments_kernel<scalar_t>
                <<<axis.node_count, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                    signal.data_ptr<scalar_t>(), z.data_ptr<double>(),
                    required.data_ptr<uint32_t>(), windows.data_ptr<double>(), signal.size(0), axis,
                    layout);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return windows;
}

constexpr int WINDOW_CACHE_STRIDE = OWNER_FEATURES + OWNER_BATCH;
constexpr int WINDOW_SHARED_BYTES =
    sizeof(double) * (OWNER_CAPACITY * WINDOW_CACHE_STRIDE + OWNER_BATCH * OWNER_FEATURES);
// One CTA owns a destination window; atoms remain in original parameter rows.
// No per-atom Fourier table, H/G atomics, or dense effective W is allocated.
template <class T, bool Reverse, bool Recompute, bool FlatC = false>
__global__ void owner_window_contract_kernel(
    const T *p, const double *windows, const double *input_windows, const double *h,
    const double *dh, double *response, double *derivative, T *gradient, double *partials,
    const int64_t *rows, const int64_t *offsets, int atoms, int batch, Axis source,
    Axis destination, OwnerWindowLayout layout, bool fold, bool need_derivative, bool emit_x) {
    int owner = blockIdx.x, begin = int(offsets[owner]), end = int(offsets[owner + 1]);
    if (begin == end)
        return;
    int bi = bank_for_node(owner, destination), rank = layout.ranks[bi];
    extern __shared__ double window_cache[];
    double *dest_windows = window_cache + OWNER_CAPACITY * WINDOW_CACHE_STRIDE;
    if constexpr (Reverse && Recompute) {
        if (need_derivative) {
            for (int i = threadIdx.x; i < batch * rank; i += blockDim.x)
                dest_windows[i] = input_windows[owner_window_index(i / rank, bi, owner, i % rank,
                                                                   destination, layout)];
        }
        __syncthreads();
    }
    double sums[FlatC ? 1 : OWNER_BATCH] = {};
    for (int chunk = begin; chunk < end; chunk += OWNER_CAPACITY) {
        int retained = min(OWNER_CAPACITY, end - chunk);
        if (threadIdx.x < retained) {
            int atom = int(rows[chunk + threadIdx.x]);
            double R = radius(p, atom, source), c[17], q[17], dq[17];
            double mu = center(p, atom, source, Reverse);
            Membership sm = classify(R, mu, source);
            Bank sb = source.banks[sm.bank];
            profile_values(R, sb.period, source.modes, c);
            owner_window_coefficients(c, wrapped_center(mu, source), sm.first_tile, source, sb,
                                      fold, q, dq);
            int source_owner = sb.node_offset + positive_mod(sm.first_tile, sb.tile_count);
            double values[OWNER_BATCH] = {}, slopes[OWNER_BATCH] = {};
            for (int f = 0; f < layout.ranks[sm.bank]; ++f) {
                for (int b = 0; b < batch; ++b) {
                    double z =
                        windows[owner_window_index(b, sm.bank, source_owner, f, source, layout)];
                    values[b] += z * q[f];
                    if (need_derivative && (Reverse || !Recompute))
                        slopes[b] += z * dq[f];
                }
            }
            double dest_mu = center(p, atom, destination, !Reverse);
            Membership dm = classify(R, dest_mu, destination);
            owner_window_coefficients(c, wrapped_center(dest_mu, destination), dm.first_tile,
                                      destination, destination.banks[dm.bank], fold, q, dq);
            double amp = amplitude(p, atom, source), da = 0, dmi = 0, dmo = 0;
            for (int b = 0; b < batch; ++b) {
                int64_t index = int64_t(b) * atoms + atom;
                if constexpr (Reverse) {
                    if (need_derivative) {
                        double H = 0, DH = 0;
                        if constexpr (Recompute) {
                            for (int f = 0; f < rank; ++f) {
                                double z = dest_windows[b * rank + f];
                                H += z * q[f];
                                DH += z * dq[f];
                            }
                        } else {
                            H = h[index];
                            DH = dh[index];
                        }
                        da += H * values[b];
                        dmi += amp * values[b] * DH;
                        dmo += amp * H * slopes[b];
                    }
                } else {
                    if constexpr (!Recompute) {
                        response[index] = values[b];
                        if (need_derivative)
                            derivative[index] = slopes[b];
                    }
                }
                window_cache[threadIdx.x * WINDOW_CACHE_STRIDE + OWNER_FEATURES + b] =
                    amp * values[b];
            }
            if constexpr (Reverse)
                if (need_derivative)
                    store_gradient(p, gradient, atom, destination, da, 0, dmi, dmo);
            for (int f = 0; f < rank; ++f)
                window_cache[threadIdx.x * WINDOW_CACHE_STRIDE + f] = q[f];
        }
        __syncthreads();
        if constexpr (FlatC) {
            // One thread per (batch, feature), preserving the atom/chunk order
            // of every individual output. At B4/r12 this uses 48 lanes.
            if (emit_x && threadIdx.x < batch * rank) {
                int b = threadIdx.x / rank, f = threadIdx.x % rank;
                for (int a = 0; a < retained; ++a) {
                    const double *v = window_cache + a * WINDOW_CACHE_STRIDE;
                    sums[0] += v[f] * v[OWNER_FEATURES + b];
                }
            }
        } else {
            if (emit_x && threadIdx.x < rank) {
                for (int a = 0; a < retained; ++a) {
                    const double *v = window_cache + a * WINDOW_CACHE_STRIDE;
                    for (int b = 0; b < batch; ++b)
                        sums[b] += v[threadIdx.x] * v[OWNER_FEATURES + b];
                }
            }
        }
        __syncthreads();
    }
    if constexpr (FlatC) {
        if (emit_x && threadIdx.x < batch * rank)
            partials[owner_window_index(threadIdx.x / rank, bi, owner, threadIdx.x % rank,
                                        destination, layout)] = sums[0];
    } else {
        if (emit_x && threadIdx.x < rank)
            for (int b = 0; b < batch; ++b)
                partials[owner_window_index(b, bi, owner, threadIdx.x, destination, layout)] =
                    sums[b];
    }
}

template <class T>
__global__ void owner_window_expand_kernel(const double *partials, const int64_t *offsets, T *y,
                                           int batch, Axis axis, OwnerWindowLayout layout) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * axis.sites)
        return;
    int b = index / axis.sites, site = index % axis.sites;
    double value = 0;
    for (int bi = 0; bi < axis.bank_count; ++bi) {
        Bank bank = axis.banks[bi];
        if (offsets[bank.node_offset] == offsets[bank.node_offset + bank.tile_count])
            continue;
        int tile = site / bank.tile_sites, local = site % bank.tile_sites;
        for (int part = 0; part < bank.count; ++part) {
            int owner = bank.node_offset + positive_mod(tile - part, bank.tile_count);
            if (offsets[owner] == offsets[owner + 1])
                continue;
            const double *v = partials + owner_window_index(b, bi, owner, 0, axis, layout);
            double entry = v[0];
            HarmonicPhase phase(2 * PI * (part * bank.tile_sites + local) /
                                (bank.count * bank.tile_sites));
            for (int k = 1; 2 * k - 1 < layout.ranks[bi]; ++k) {
                double cp, sp;
                phase.next(cp, sp);
                entry += v[2 * k - 1] * cp;
                if (2 * k < layout.ranks[bi])
                    entry += v[2 * k] * sp;
            }
            value += entry;
        }
    }
    y[index] = T(value);
}
std::vector<Tensor> owner_window_transform(Tensor signal, Tensor p, Tensor keys, Tensor rows,
                                           Tensor offsets, Config input_config,
                                           Config output_config, bool reverse, bool need_derivative,
                                           bool fold, bool recompute, bool emit_x, Tensor h,
                                           Tensor dh, Tensor saved_windows, bool flat_c) {
    Axis source = parse_axis(reverse ? output_config : input_config);
    Axis destination = parse_axis(reverse ? input_config : output_config);
    check_atoms(p);
    check_hybrid_signal(signal, p, source);
    check_ranges(rows, offsets, p, destination);
    check_owner_axes(source, destination, signal.size(0), 2, 2, true, 32);
    c10::cuda::CUDAGuard guard(p.device());
    TORCH_CHECK(!flat_c || (fold && recompute), "batch-feature C requires folded recomputation");
    auto layout = owner_window_layout(destination, fold);
    auto windows = build_owner_windows(signal, p, keys, input_config, output_config, reverse, fold);
    auto response = !reverse && !recompute ? torch::empty({signal.size(0), p.size(0)},
                                                          signal.options().dtype(torch::kDouble))
                                           : Tensor();
    auto derivative =
        response.defined() && need_derivative ? torch::empty_like(response) : Tensor();
    auto gradient = reverse && need_derivative
                        ? (signal.size(0) ? torch::empty_like(p) : torch::zeros_like(p))
                        : Tensor();
    auto partials = emit_x ? torch::empty({signal.size(0), layout.total_fields},
                                          signal.options().dtype(torch::kDouble))
                           : Tensor();
    auto y =
        emit_x ? torch::empty({signal.size(0), destination.sites}, signal.options()) : Tensor();
    if (signal.size(0)) {
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_window_contract", [&] {
            auto launch = [&](auto rev, auto recomp, auto flat) {
                owner_window_contract_kernel<scalar_t, decltype(rev)::value,
                                             decltype(recomp)::value, decltype(flat)::value>
                    <<<destination.node_count, THREADS, WINDOW_SHARED_BYTES,
                       at::cuda::getCurrentCUDAStream()>>>(
                        p.data_ptr<scalar_t>(), windows.data_ptr<double>(),
                        saved_windows.defined() ? saved_windows.data_ptr<double>() : nullptr,
                        h.defined() ? h.data_ptr<double>() : nullptr,
                        dh.defined() ? dh.data_ptr<double>() : nullptr,
                        response.defined() ? response.data_ptr<double>() : nullptr,
                        derivative.defined() ? derivative.data_ptr<double>() : nullptr,
                        gradient.defined() ? gradient.data_ptr<scalar_t>() : nullptr,
                        partials.defined() ? partials.data_ptr<double>() : nullptr,
                        rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), p.size(0),
                        signal.size(0), source, destination, layout, fold, need_derivative, emit_x);
            };
            if (flat_c) {
                if (reverse)
                    launch(std::true_type{}, std::true_type{}, std::true_type{});
                else
                    launch(std::false_type{}, std::true_type{}, std::true_type{});
            } else if (reverse) {
                if (recompute)
                    launch(std::true_type{}, std::true_type{}, std::false_type{});
                else
                    launch(std::true_type{}, std::false_type{}, std::false_type{});
            } else {
                if (recompute)
                    launch(std::false_type{}, std::true_type{}, std::false_type{});
                else
                    launch(std::false_type{}, std::false_type{}, std::false_type{});
            }
            if (emit_x)
                owner_window_expand_kernel<scalar_t><<<(y.numel() + THREADS - 1) / THREADS, THREADS,
                                                       0, at::cuda::getCurrentCUDAStream()>>>(
                    partials.data_ptr<double>(), offsets.data_ptr<int64_t>(),
                    y.data_ptr<scalar_t>(), signal.size(0), destination, layout);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {y, response, derivative, gradient, windows};
}
void check_owner_window_saved(Tensor x, Tensor atoms, Tensor window, Tensor keys, Tensor dy,
                              Config input, Config output) {
    check_atoms(atoms);
    Axis ai = parse_axis(input), ao = parse_axis(output);
    check_hybrid_signal(x, atoms, ai);
    check_hybrid_signal(dy, atoms, ao);
    check_owner_axes(ai, ao, x.size(0), 2, 2, true, 32);
    TORCH_CHECK(x.size(0) == dy.size(0), "owner window saved batches differ");
    auto w = window;
    // Match the actual fixed folded layout before sorting or launching.
    TORCH_CHECK(w.defined() && w.is_cuda() && w.device() == atoms.device() &&
                    w.scalar_type() == torch::kDouble && w.is_contiguous() && w.dim() == 2 &&
                    w.size(0) == x.size(0) &&
                    w.size(1) == owner_window_layout(ai, true).total_fields,
                "owner window saved payload differs");
    check_layout_keys(keys, ai.node_count, ao.node_count);
    TORCH_CHECK(keys.device() == atoms.device() && keys.numel() == atoms.size(0),
                "owner window saved keys differ");
}
} // namespace radius_cell_mass
