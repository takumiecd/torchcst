#pragma once
#include "owner_window.cuh"

namespace radius_cell_mass {
namespace fp32_body {
// Geometry decode/classification and store_gradient remain the original FP64.
// Continuous arithmetic/storage below is FP32, with unchanged iteration order.
constexpr float PI32 = 3.14159265358979323846f;
__device__ __constant__ float profile_m8_terms32[8][8] = {
    {8, 252, 2520, 11550, 27720, 36036, 24024, 6435},
    {9, 270, 2475, 9900, 19305, 18018, 6435},
    {10, 275, 2200, 7150, 10010, 5005},
    {11, 264, 1716, 4004, 3003},
    {12, 234, 1092, 1365},
    {13, 182, 455},
    {14, 105},
    {15},
};

struct HarmonicPhase {
    float step_sine, step_cosine, sine = 0, cosine = 1;
    __device__ explicit HarmonicPhase(float phase) { sincosf(phase, &step_sine, &step_cosine); }
    __device__ inline void next(float &c, float &s) {
        float next_cosine = cosine * step_cosine - sine * step_sine;
        sine = sine * step_cosine + cosine * step_sine;
        cosine = next_cosine;
        c = cosine;
        s = sine;
    }
};
__device__ __noinline__ void profile_values_m8(float R, float P, float *c) {
    float angle = PI32 * (R / P), cb = cosf(angle);
    float z = cb * cb, u = 1 - z;
    float zp[9], up[9];
    zp[0] = up[0] = 1;
    for (int j = 1; j <= 8; ++j) {
        zp[j] = zp[j - 1] * z;
        up[j] = up[j - 1] * u;
    }
    // Homogenization cancels the common z^-8 before it can overflow.
    float zero = 0;
    for (int k = 0; k < 8; ++k) {
        int degree = 8 - k;
        float raw = k == 0 ? 0.5f * zp[8] : 0;
        float value = 0;
        for (int j = 1; j <= degree; ++j) {
            int power = degree - j;
            float coefficient = profile_m8_terms32[k][j - 1];
            value += coefficient * zp[power] * up[j];
        }
        float factor = 4.0f / (8 + k);
        raw += factor * value;
        if (k == 0) {
            zero = raw;
            c[0] = 1 / P;
        } else {
            c[k] = (raw / zero) / P;
        }
    }
    c[8] = (0.25f / zero) / P;
}

__device__ inline void owner_window_coefficients(const float *c, double mu, int first_tile,
                                                 Axis axis, Bank bank, bool fold, float *q,
                                                 float *dq) {
    int L = bank.count * bank.tile_sites;
    int rank = fold ? min(L, axis.basis_count()) : axis.basis_count();
    for (int f = 0; f < rank; ++f)
        q[f] = dq[f] = 0;
    q[0] = float(axis.spacing) * c[0];
    HarmonicPhase phase(2 * PI32 / float(bank.period) *
                        float(mu - first_tile * bank.tile_sites * axis.spacing));
    HarmonicPhase cell_phase(PI32 * float(axis.spacing) / float(bank.period));
    for (int k = 1; k <= axis.modes; ++k) {
        float cp, sp, cc, ss;
        phase.next(cp, sp);
        cell_phase.next(cc, ss);
        float w = 2 * c[k] * float(axis.spacing) * ss /
                  (PI32 * k * float(axis.spacing) / float(bank.period));
        float omega = 2 * PI32 * k / float(bank.period);
        int alias = fold ? k % L : k;
        float sign = 1;
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
__device__ inline float compact_moment_element(const T *signal, const float *z, int b, int node,
                                               int basis, Axis axis, int start, bool merge) {
    int features = axis.basis_count(), bank_index = bank_for_node(node, axis);
    Bank bank = axis.banks[bank_index];
    int tile = node - bank.node_offset;
    int harmonic = (basis + 1) / 2;
    bool sine = basis != 0 && basis % 2 == 0;
    float sum = 0;
    if (merge && bank.level > start && harmonic % axis.merge_factor == 0) {
        Bank child = axis.banks[bank_index - 2];
        int child_harmonic = harmonic / axis.merge_factor;
        for (int part = 0; part < axis.merge_factor; ++part) {
            int child_node = child.node_offset + tile * axis.merge_factor + part;
            int64_t offset = (int64_t(b) * (axis.node_count - prefix_nodes(axis, start)) +
                              child_node - prefix_nodes(axis, start)) *
                             features;
            if (basis == 0) {
                sum += z[offset];
                continue;
            }
            float phase = 2 * PI32 * harmonic * part * child.tile_sites * float(axis.spacing) /
                          float(bank.period);
            float c = z[offset + 2 * child_harmonic - 1];
            float s = z[offset + 2 * child_harmonic];
            sum += sine ? sinf(phase) * c + cosf(phase) * s : cosf(phase) * c - sinf(phase) * s;
        }
    } else {
        float filter = float(axis.spacing);
        if (harmonic) {
            float argument = PI32 * harmonic * float(axis.spacing) / float(bank.period);
            filter *= sinf(argument) / argument;
        }
        int first_site = tile * bank.tile_sites;
        if (basis == 0) {
            for (int local_site = 0; local_site < bank.tile_sites; ++local_site)
                sum += float(signal[b * axis.sites + first_site + local_site]) * filter;
        } else if (bank.tile_sites == 1) {
            // Tile-relative phase is exactly zero at the only site.
            sum += float(signal[b * axis.sites + first_site]) * filter * (sine ? 0 : 1);
        } else {
            HarmonicPhase phase(2 * PI32 * harmonic * float(axis.spacing) / float(bank.period));
            for (int local_site = 0; local_site < bank.tile_sites; ++local_site) {
                float phi = sine ? phase.sine : phase.cosine;
                sum += float(signal[b * axis.sites + first_site + local_site]) * filter * phi;
                float unused_cosine, unused_sine;
                phase.next(unused_cosine, unused_sine);
            }
        }
    }
    return sum;
}
template <class T>
__global__ void compact_moments_kernel(const T *signal, float *z, int batch, Axis axis, int start,
                                       int level, bool merge,
                                       const uint32_t *required_banks = nullptr) {
    // The immutable mask includes all required lower child banks before launch.
    // Empty levels return uniformly; no barrier or global-layout change occurs.
    if (required_banks && level >= 0 && !(*required_banks & (3u << (2 * level))))
        return;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nodes = axis.node_count - prefix_nodes(axis, start), F = axis.basis_count();
    if (index >= batch * nodes * F)
        return;
    int node = prefix_nodes(axis, start) + (index / F) % nodes;
    int bank = bank_for_node(node, axis);
    if (level >= 0 && axis.banks[bank].level != level)
        return;
    if (required_banks && !(*required_banks & (uint32_t(1) << bank)))
        return;
    z[index] =
        compact_moment_element(signal, z, index / (nodes * F), node, index % F, axis, start, merge);
}

// END_FP32_SCALARS

inline Tensor build_raw(Tensor signal, Tensor keys, Config input, Config output, bool reverse) {
    Axis axis = parse_axis(reverse ? output : input);
    auto required = active_moment_banks_from_keys(keys, input, output, reverse, true, 2);
    int nodes = axis.node_count - checked_start(axis, 2), features = axis.basis_count();
    auto z = torch::empty({signal.size(0), nodes, features}, signal.options());
    if (z.numel()) {
        int blocks = (z.numel() + THREADS - 1) / THREADS;
        for (int level = 2; level < axis.bank_count / 2; ++level)
            compact_moments_kernel<float><<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                signal.data_ptr<float>(), z.data_ptr<float>(), signal.size(0), axis, 2, level, true,
                required.data_ptr<uint32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return z;
}
template <class T>
__global__ void owner_window_moments_kernel(const T *signal, const float *z,
                                            const uint32_t *required, float *windows, int batch,
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
        float value = 0;
        if (L <= 16) {
            for (int j = 0; j < L; ++j) {
                float cp, sp;
                sincosf(2 * PI32 * k * j / L, &sp, &cp);
                value += float(signal[int64_t(b) * axis.sites +
                                      positive_mod(tile * bank.tile_sites + j, axis.sites)]) *
                         (sine ? sp : cp);
            }
        } else {
            float argument = PI32 * k / L;
            float filter =
                k ? float(axis.spacing) * sinf(argument) / argument : float(axis.spacing);
            for (int part = 0; part < bank.count; ++part) {
                int node = bank.node_offset + positive_mod(tile + part, bank.tile_count);
                const float *moments =
                    z + int64_t(b) * elements + (node - prefix_nodes(axis, 2)) * features;
                if (!k)
                    value += moments[0] / filter;
                else {
                    float cp, sp;
                    sincosf(2 * PI32 * k * part / bank.count, &sp, &cp);
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
    auto z = build_raw(signal, keys, input_config, output_config, reverse);
    auto windows =
        torch::empty({signal.size(0), layout.total_fields}, signal.options().dtype(torch::kFloat));
    if (signal.size(0)) {
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_window_moments", [&] {
            owner_window_moments_kernel<scalar_t>
                <<<axis.node_count, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                    signal.data_ptr<scalar_t>(), z.data_ptr<float>(), required.data_ptr<uint32_t>(),
                    windows.data_ptr<float>(), signal.size(0), axis, layout);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return windows;
}

constexpr int WINDOW_CACHE_STRIDE = OWNER_FEATURES + OWNER_BATCH;
constexpr int WINDOW_SHARED_BYTES =
    sizeof(float) * (OWNER_CAPACITY * WINDOW_CACHE_STRIDE + OWNER_BATCH * OWNER_FEATURES);
// One CTA owns a destination window; atoms remain in original parameter rows.
// No per-atom Fourier table, H/G atomics, or dense effective W is allocated.
template <class T, bool Reverse, bool Recompute, bool GeometryShared, bool FlatC = false>
__global__ void owner_window_contract_kernel(
    const T *p, const float *windows, const float *input_windows, const float *h, const float *dh,
    float *response, float *derivative, T *gradient, float *partials, const int64_t *rows,
    const int64_t *offsets, int atoms, int batch, Axis source, Axis destination,
    OwnerWindowLayout layout, bool fold, bool need_derivative, bool emit_x,
    const double *geometry) {
    int owner = blockIdx.x, begin = int(offsets[owner]), end = int(offsets[owner + 1]);
    if (begin == end)
        return;
    int bi = bank_for_node(owner, destination), rank = layout.ranks[bi];
    extern __shared__ float window_cache[];
    float *dest_windows = window_cache + OWNER_CAPACITY * WINDOW_CACHE_STRIDE;
    if constexpr (Reverse && Recompute) {
        if (need_derivative) {
            for (int i = threadIdx.x; i < batch * rank; i += blockDim.x)
                dest_windows[i] = input_windows[owner_window_index(i / rank, bi, owner, i % rank,
                                                                   destination, layout)];
        }
        __syncthreads();
    }
    float sums[FlatC ? 1 : OWNER_BATCH] = {};
    const int c_feature = FlatC ? int(threadIdx.x) % rank : int(threadIdx.x);
    const int c_batch = FlatC ? int(threadIdx.x) / rank : 0;
    for (int chunk = begin; chunk < end; chunk += OWNER_CAPACITY) {
        int retained = min(OWNER_CAPACITY, end - chunk);
        if (threadIdx.x < retained) {
            int atom = int(rows[chunk + threadIdx.x]);
            double R;
            if constexpr (GeometryShared) R = geometry[atom];
            else R = radius(p, atom, source);
            float c[17], q[17], dq[17];
            double mu = center(p, atom, source, Reverse);
            Membership sm = classify(R, mu, source);
            Bank sb = source.banks[sm.bank];
            profile_values_m8(float(R), float(sb.period), c);
            owner_window_coefficients(c, wrapped_center(mu, source), sm.first_tile, source, sb,
                                      fold, q, dq);
            int source_owner = sb.node_offset + positive_mod(sm.first_tile, sb.tile_count);
            float values[OWNER_BATCH] = {}, slopes[OWNER_BATCH] = {};
            for (int f = 0; f < layout.ranks[sm.bank]; ++f) {
                for (int b = 0; b < batch; ++b) {
                    float z =
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
            float amp;
            if constexpr (GeometryShared) amp = float(geometry[atoms + atom]);
            else amp = float(amplitude(p, atom, source));
            float da = 0, dmi = 0, dmo = 0;
            for (int b = 0; b < batch; ++b) {
                int64_t index = int64_t(b) * atoms + atom;
                if constexpr (Reverse) {
                    if (need_derivative) {
                        float H = 0, DH = 0;
                        if constexpr (Recompute) {
                            for (int f = 0; f < rank; ++f) {
                                float z = dest_windows[b * rank + f];
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
            if (emit_x && threadIdx.x < batch * rank) {
                for (int a = 0; a < retained; ++a) {
                    const float *v = window_cache + a * WINDOW_CACHE_STRIDE;
                    sums[0] += v[c_feature] * v[OWNER_FEATURES + c_batch];
                }
            }
        } else {
            if (emit_x && threadIdx.x < rank) {
                for (int a = 0; a < retained; ++a) {
                    const float *v = window_cache + a * WINDOW_CACHE_STRIDE;
                    for (int b = 0; b < batch; ++b)
                        sums[b] += v[threadIdx.x] * v[OWNER_FEATURES + b];
                }
            }
        }
        __syncthreads();
    }
    if constexpr (FlatC) {
        if (emit_x && threadIdx.x < batch * rank)
            partials[owner_window_index(c_batch, bi, owner, c_feature, destination, layout)] = sums[0];
    } else {
        if (emit_x && threadIdx.x < rank)
            for (int b = 0; b < batch; ++b)
                partials[owner_window_index(b, bi, owner, threadIdx.x, destination, layout)] = sums[b];
    }
}

template <class T>
__global__ void owner_window_expand_kernel(const float *partials, const int64_t *offsets, T *y,
                                           int batch, Axis axis, OwnerWindowLayout layout) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * axis.sites)
        return;
    int b = index / axis.sites, site = index % axis.sites;
    float value = 0;
    for (int bi = 0; bi < axis.bank_count; ++bi) {
        Bank bank = axis.banks[bi];
        if (offsets[bank.node_offset] == offsets[bank.node_offset + bank.tile_count])
            continue;
        int tile = site / bank.tile_sites, local = site % bank.tile_sites;
        for (int part = 0; part < bank.count; ++part) {
            int owner = bank.node_offset + positive_mod(tile - part, bank.tile_count);
            if (offsets[owner] == offsets[owner + 1])
                continue;
            const float *v = partials + owner_window_index(b, bi, owner, 0, axis, layout);
            float entry = v[0];
            HarmonicPhase phase(2 * PI32 * (part * bank.tile_sites + local) /
                                (bank.count * bank.tile_sites));
            for (int k = 1; 2 * k - 1 < layout.ranks[bi]; ++k) {
                float cp, sp;
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
template <bool FlatC>
std::vector<Tensor> owner_window_transform_impl(Tensor signal, Tensor p, Tensor keys, Tensor rows,
                                           Tensor offsets, Config input_config,
                                           Config output_config, bool reverse, bool need_derivative,
                                           bool fold, bool recompute, bool emit_x, Tensor h,
                                           Tensor dh, Tensor saved_windows, Tensor geometry) {
    Axis source = parse_axis(reverse ? output_config : input_config);
    Axis destination = parse_axis(reverse ? input_config : output_config);
    check_atoms(p);
    check_hybrid_signal(signal, p, source);
    check_ranges(rows, offsets, p, destination);
    check_owner_axes(source, destination, signal.size(0), 2, 2, true, 32);
    c10::cuda::CUDAGuard guard(p.device());
    auto layout = owner_window_layout(destination, fold);
    auto windows =
        fp32_body::build_owner_windows(signal, p, keys, input_config, output_config, reverse, fold);
    auto response = !reverse && !recompute ? torch::empty({signal.size(0), p.size(0)},
                                                          signal.options().dtype(torch::kFloat))
                                           : Tensor();
    auto derivative =
        response.defined() && need_derivative ? torch::empty_like(response) : Tensor();
    auto gradient = reverse && need_derivative
                        ? (signal.size(0) ? torch::empty_like(p) : torch::zeros_like(p))
                        : Tensor();
    auto partials = emit_x ? torch::empty({signal.size(0), layout.total_fields},
                                          signal.options().dtype(torch::kFloat))
                           : Tensor();
    auto y =
        emit_x ? torch::empty({signal.size(0), destination.sites}, signal.options()) : Tensor();
    if (signal.size(0)) {
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_window_contract", [&] {
            auto launch = [&](auto rev, auto recomp, auto geom) {
                owner_window_contract_kernel<scalar_t, decltype(rev)::value,
                                             decltype(recomp)::value, decltype(geom)::value, FlatC>
                    <<<destination.node_count, THREADS, WINDOW_SHARED_BYTES,
                       at::cuda::getCurrentCUDAStream()>>>(
                        p.data_ptr<scalar_t>(), windows.data_ptr<float>(),
                        saved_windows.defined() ? saved_windows.data_ptr<float>() : nullptr,
                        h.defined() ? h.data_ptr<float>() : nullptr,
                        dh.defined() ? dh.data_ptr<float>() : nullptr,
                        response.defined() ? response.data_ptr<float>() : nullptr,
                        derivative.defined() ? derivative.data_ptr<float>() : nullptr,
                        gradient.defined() ? gradient.data_ptr<scalar_t>() : nullptr,
                        partials.defined() ? partials.data_ptr<float>() : nullptr,
                        rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), p.size(0),
                        signal.size(0), source, destination, layout, fold, need_derivative, emit_x,
                        geometry.defined() ? geometry.data_ptr<double>() : nullptr);
            };
            auto dispatch = [&](auto geom) {
            if (reverse) {
                if (recompute)
                    launch(std::true_type{}, std::true_type{}, geom);
                else
                    launch(std::true_type{}, std::false_type{}, geom);
            } else {
                if (recompute)
                    launch(std::false_type{}, std::true_type{}, geom);
                else
                    launch(std::false_type{}, std::false_type{}, geom);
            }
            };
            if constexpr (FlatC) {
                TORCH_CHECK(geometry.defined() && recompute, "flat C requires shared geometry and recomputation");
                if (reverse) launch(std::true_type{}, std::true_type{}, std::true_type{});
                else launch(std::false_type{}, std::true_type{}, std::true_type{});
            } else {
                if (geometry.defined()) dispatch(std::true_type{});
                else dispatch(std::false_type{});
            }
            if (emit_x)
                owner_window_expand_kernel<scalar_t><<<(y.numel() + THREADS - 1) / THREADS, THREADS,
                                                       0, at::cuda::getCurrentCUDAStream()>>>(
                    partials.data_ptr<float>(), offsets.data_ptr<int64_t>(), y.data_ptr<scalar_t>(),
                    signal.size(0), destination, layout);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {y, response, derivative, gradient, windows};
}
std::vector<Tensor> owner_window_transform(Tensor signal, Tensor p, Tensor keys, Tensor rows,
    Tensor offsets, Config input, Config output, bool reverse, bool need_derivative,
    bool fold, bool recompute, bool emit_x, Tensor h, Tensor dh, Tensor saved_windows,
    Tensor geometry) {
    return owner_window_transform_impl<false>(signal, p, keys, rows, offsets, input, output,
        reverse, need_derivative, fold, recompute, emit_x, h, dh, saved_windows, geometry);
}

std::vector<Tensor> owner_window_transform_flat_c(Tensor signal, Tensor p, Tensor keys, Tensor rows,
    Tensor offsets, Config input, Config output, bool reverse, bool need_derivative,
    bool fold, bool recompute, bool emit_x, Tensor h, Tensor dh, Tensor saved_windows,
    Tensor geometry) {
    return owner_window_transform_impl<true>(signal, p, keys, rows, offsets, input, output,
        reverse, need_derivative, fold, recompute, emit_x, h, dh, saved_windows, geometry);
}

void check_geometry_saved(Tensor atoms, Tensor geometry) {
    TORCH_CHECK(geometry.defined() && geometry.is_cuda() &&
        geometry.device() == atoms.device() && geometry.scalar_type() == torch::kDouble &&
        geometry.dim() == 2 && geometry.size(0) == 2 && geometry.size(1) == atoms.size(0) &&
        geometry.is_contiguous(), "owner geometry saved payload differs");
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
                    w.scalar_type() == torch::kFloat && w.is_contiguous() && w.dim() == 2 &&
                    w.size(0) == x.size(0) &&
                    w.size(1) == owner_window_layout(ai, true).total_fields,
                "owner window saved payload differs");
    check_layout_keys(keys, ai.node_count, ao.node_count);
    TORCH_CHECK(keys.device() == atoms.device() && keys.numel() == atoms.size(0),
                "owner window saved keys differ");
}

// Diagnostic-only allocations are never part of a timed implementation.
template <bool Single, class W>
__global__ void diagnostic_kernel(const float *p, const W *wx, const W *wy, double *table,
                                  int64_t *meta, int atoms, int batch, Axis ai, Axis ao,
                                  OwnerWindowLayout li, OwnerWindowLayout lo) {
    using S = std::conditional_t<Single, float, double>;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= atoms)
        return;
    double R = radius(p, a, ai);
    double mi = center(p, a, ai, false), mo = center(p, a, ao, true);
    auto im = classify(R, mi, ai), om = classify(R, mo, ao);
    auto ib = ai.banks[im.bank], ob = ao.banks[om.bank];
    int io = ib.node_offset + positive_mod(im.first_tile, ib.tile_count);
    int oo = ob.node_offset + positive_mod(om.first_tile, ob.tile_count);
    meta[4 * a] = im.bank;
    meta[4 * a + 1] = io;
    meta[4 * a + 2] = om.bank;
    meta[4 * a + 3] = oo;
    S c[17] = {}, qi[17] = {}, di[17] = {}, qo[17] = {}, dout[17] = {};
    if constexpr (Single) {
        fp32_body::profile_values_m8(float(R), float(ib.period), c);
        fp32_body::owner_window_coefficients(c, wrapped_center(mi, ai), im.first_tile, ai, ib, true,
                                             qi, di);
        fp32_body::owner_window_coefficients(c, wrapped_center(mo, ao), om.first_tile, ao, ob, true,
                                             qo, dout);
    } else {
        radius_cell_mass::profile_values_m8(R, ib.period, c);
        radius_cell_mass::owner_window_coefficients(c, wrapped_center(mi, ai), im.first_tile, ai,
                                                    ib, true, qi, di);
        radius_cell_mass::owner_window_coefficients(c, wrapped_center(mo, ao), om.first_tile, ao,
                                                    ob, true, qo, dout);
    }
    double *r = table + int64_t(a) * (80 + 4 * batch);
    for (int f = 0; f < 9; ++f)
        r[f] = c[f];
    for (int f = 0; f < 17; ++f) {
        r[9 + f] = qi[f];
        r[26 + f] = di[f];
        r[43 + f] = qo[f];
        r[60 + f] = dout[f];
    }
    S da = 0, dmi = 0, dmo = 0, amp = S(amplitude(p, a, ai));
    for (int b = 0; b < batch; ++b) {
        S h = 0, dh = 0, g = 0, dg = 0;
        for (int f = 0; f < li.ranks[im.bank]; ++f) {
            S z = S(wx[owner_window_index(b, im.bank, io, f, ai, li)]);
            h += z * qi[f];
            dh += z * di[f];
        }
        for (int f = 0; f < lo.ranks[om.bank]; ++f) {
            S z = S(wy[owner_window_index(b, om.bank, oo, f, ao, lo)]);
            g += z * qo[f];
            dg += z * dout[f];
        }
        r[77 + b] = h;
        r[77 + batch + b] = dh;
        r[77 + 2 * batch + b] = g;
        r[77 + 3 * batch + b] = dg;
        da += h * g;
        dmi += amp * g * dh;
        dmo += amp * h * dg;
    }
    r[77 + 4 * batch] = da;
    r[78 + 4 * batch] = dmi;
    r[79 + 4 * batch] = dmo;
}
std::vector<Tensor> diagnose(Tensor x, Tensor dy, Tensor p, Config input, Config output) {
    c10::cuda::CUDAGuard guard(p.device());
    auto keys = classify_atoms_async(p, input, output);
    auto wx = fp32_body::build_owner_windows(x, p, keys, input, output, false, true);
    auto wy = fp32_body::build_owner_windows(dy, p, keys, input, output, true, true);
    auto dx = radius_cell_mass::build_owner_windows(x, p, keys, input, output, false, true);
    auto dyw = radius_cell_mass::build_owner_windows(dy, p, keys, input, output, true, true);
    Axis ai = parse_axis(input), ao = parse_axis(output);
    auto li = owner_window_layout(ai, true), lo = owner_window_layout(ao, true);
    auto table = torch::empty({p.size(0), 80 + 4 * x.size(0)}, p.options().dtype(torch::kDouble));
    auto baseline = torch::empty_like(table), shared = torch::empty_like(table);
    auto meta = torch::empty({p.size(0), 4}, p.options().dtype(torch::kInt64));
    if (p.size(0)) {
        int blocks = (p.size(0) + THREADS - 1) / THREADS;
        auto stream = at::cuda::getCurrentCUDAStream();
        diagnostic_kernel<true><<<blocks, THREADS, 0, stream>>>(
            p.data_ptr<float>(), wx.data_ptr<float>(), wy.data_ptr<float>(),
            table.data_ptr<double>(), meta.data_ptr<int64_t>(), p.size(0), x.size(0), ai, ao, li,
            lo);
        diagnostic_kernel<false><<<blocks, THREADS, 0, stream>>>(
            p.data_ptr<float>(), dx.data_ptr<double>(), dyw.data_ptr<double>(),
            baseline.data_ptr<double>(), meta.data_ptr<int64_t>(), p.size(0), x.size(0), ai, ao, li,
            lo);
        diagnostic_kernel<true><<<blocks, THREADS, 0, stream>>>(
            p.data_ptr<float>(), dx.data_ptr<double>(), dyw.data_ptr<double>(),
            shared.data_ptr<double>(), meta.data_ptr<int64_t>(), p.size(0), x.size(0), ai, ao, li,
            lo);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {table, baseline, shared, meta, wx, wy, dx, dyw};
}

void check_scope(Tensor p, Config input, Config output) {
    TORCH_CHECK(p.scalar_type() == torch::kFloat, "FP32 body requires FP32 public tensors");
    for (const auto &config : {input, output}) {
        Axis a = parse_axis(config);
        TORCH_CHECK(owner_window_sites(a.sites) && a.modes == 8 && a.spacing == 1 && a.polar &&
                        a.radius_min >= .5 && a.radius_max <= 50 && a.amplitude_max <= 1,
                    "FP32 body research scope: N1024..16384 powers-of-two/M8/spacing1/Polar/radius[.5,50]/amplitude<=1");
    }
}
} // namespace fp32_body
} // namespace radius_cell_mass
