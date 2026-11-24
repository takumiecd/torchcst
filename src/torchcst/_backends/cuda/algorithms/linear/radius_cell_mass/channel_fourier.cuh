#pragma once

#ifdef RADIUS_RATIONAL_WIDE
namespace radius_cell_mass {

constexpr int CHANNEL_MAX_FOURIER_BANKS = 14;

struct ChannelFourierLayout {
    int offsets[CHANNEL_MAX_FOURIER_BANKS];
    int size;
};

inline bool channel_fourier_sites(int sites) {
    return sites == 16 || sites == 32 || sites == 64 || sites == 96 || sites == 128;
}

inline void check_channel_fourier_chart(int ni, int no, int radius_cap) {
    TORCH_CHECK(channel_fourier_sites(ni) && channel_fourier_sites(no),
        "expected 16, 32, 64, 96 or 128 sites per channel axis");
    TORCH_CHECK(radius_cap > 0 && (radius_cap & (radius_cap - 1)) == 0 &&
        radius_cap <= std::min(ni, no) / 2 && ni % radius_cap == 0 && no % radius_cap == 0,
        "expected a dyadic radius cap dividing both channel tori and at most their half-period");
}

__host__ __device__ inline int channel_period(Bank bank) {
    return bank.count * bank.tile_sites;
}

__host__ __device__ inline int channel_rank(Bank bank) {
    int period = channel_period(bank);
    return period < COTANGENT_FEATURES ? period : COTANGENT_FEATURES;
}

inline ChannelFourierLayout channel_fourier_layout(Axis ai, Axis ao, int taps, int mask) {
    ChannelFourierLayout layout{};
    for (int b = 0; b < ai.bank_count; ++b) {
        // On a dyadic hierarchy a selected non-base tile has R > T/2.
        if (!(mask & (1 << b)) || (b != 0 && ai.banks[b].count == 2)) {
            layout.offsets[b] = -1;
            continue;
        }
        int rank = channel_rank(ai.banks[b]);
        layout.offsets[b] = layout.size;
        layout.size += taps * ai.banks[b].tile_count * ao.banks[b].tile_count * rank * rank;
    }
    return layout;
}

__global__ void channel_active_fourier_banks(const float *geometry, int *mask, int atoms) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom < atoms) atomicOr(mask, 1 << int(geometry[int64_t(atom) * 25 + 20]));
}

__device__ inline void channel_add(float &a, float b, float sign) { a += sign * b; }

__device__ inline void channel_add(CotangentFactor &a, CotangentFactor b, float sign) {
    a.value += sign * b.value;
    a.radius += sign * b.radius;
    a.center += sign * b.center;
}

// At integer sites, k and k+P are the same frequency. The signed aliases
// combine only after integrating the original harmonics over whole cells.
template<class T>
__device__ inline void channel_alias(const T *series, T *folded, int period, int rank) {
    for (int j = 0; j < rank; ++j) folded[j] = {};
    folded[0] = series[0];
    #pragma unroll
    for (int h = 1; h <= 8; ++h) {
        int k = h % period;
        if (!k) {
            channel_add(folded[0], series[2 * h - 1], 1);
            continue;
        }
        bool conjugate = 2 * k > period;
        if (conjugate) k = period - k;
        channel_add(folded[2 * k - 1], series[2 * h - 1], 1);
        if (2 * k < rank)
            channel_add(folded[2 * k], series[2 * h], conjugate ? -1 : 1);
    }
}

__device__ inline int channel_owner_offset(const float *v, int tap, Axis ai, Axis ao,
    ChannelFourierLayout layout) {
    int bank = int(v[20]);
    Bank bi = ai.banks[bank], bo = ao.banks[bank];
    int i = positive_mod(int(v[21]), bi.tile_count);
    int o = positive_mod(int(v[23]), bo.tile_count);
    int rank = channel_rank(bi);
    return layout.offsets[bank] + ((tap * bo.tile_count + o) * bi.tile_count + i) * rank * rank;
}

__global__ void channel_fourier_keys(const float *geometry, int64_t *keys,
    int atoms, int tap_atoms, Axis ai, Axis ao, ChannelFourierLayout layout) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom < atoms)
        keys[atom] = channel_owner_offset(geometry + int64_t(atom) * 25,
            atom / tap_atoms, ai, ao, layout);
}

template<bool Sorted>
__global__ void channel_owner_coefficients(const float *geometry, const int64_t *keys,
    const int64_t *order, float *owners, int atoms, int tap_atoms, Axis ai, Axis ao,
    ChannelFourierLayout layout) {
    int position = blockIdx.x;
    if constexpr (Sorted)
        if (position && keys[position] == keys[position - 1]) return;
    int first_atom = Sorted ? int(order[position]) : position;
    const float *v = geometry + int64_t(first_atom) * 25;
    int bank = int(v[20]);
    Bank bi = ai.banks[bank], bo = ao.banks[bank];
    int period = channel_period(bi), rank = channel_rank(bi);
    int owner = Sorted ? int(keys[position])
        : channel_owner_offset(v, position / tap_atoms, ai, ao, layout);

    constexpr int load = 32, F = COTANGENT_FEATURES;
    __shared__ float si[load][F], so[load][F];
    __shared__ int end;
    int tid = threadIdx.x;
    if (tid == 0) {
        int lo = position + 1;
        if constexpr (Sorted) {
            int hi = atoms;
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if (keys[mid] == owner) lo = mid + 1;
                else hi = mid;
            }
        }
        end = lo;
    }
    __syncthreads();
    float accumulation = 0;
    int in_feature = tid / rank, out_feature = tid % rank;
    for (int j = position; j < end; j += load) {
        int count = min(load, end - j);
        if (tid < count) {
            int atom = Sorted ? int(order[j + tid]) : j + tid;
            const float *g = geometry + int64_t(atom) * 25;
            float series[F];
            dense_weight_series(g, g[18] - int(g[21]) * bi.tile_sites,
                float(period), g[24], series);
            channel_alias(series, si[tid], period, rank);
            dense_weight_series(g, g[19] - int(g[23]) * bo.tile_sites,
                float(period), 1.f, series);
            channel_alias(series, so[tid], period, rank);
        }
        __syncthreads();
        if (tid < rank * rank)
            for (int a = 0; a < count; ++a)
                accumulation += si[a][in_feature] * so[a][out_feature];
        __syncthreads();
    }
    if (tid < rank * rank) {
        if constexpr (Sorted) owners[owner + tid] = accumulation;
        else atomicAdd(owners + owner + tid, accumulation);
    }
}

__global__ void channel_fourier_basis(float *basis, int sites, int period, int rank) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= sites * rank) return;
    int site = index / rank, f = index % rank;
    float angle = (2 * float(PI) / period) * ((f + 1) / 2) * site;
    basis[index] = (f == 0 || f % 2) ? cosf(angle) : sinf(angle);
}

template<bool InputSide, bool Transpose = false>
__global__ void channel_fourier_fold(const float *source, float *dest,
    int taps, int ni, int no, int rank, int windows, int period) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_size = rank * rank;
    if (index >= taps * ni * no * pair_size) return;
    int f = index % pair_size, owner = index / pair_size;
    int i = owner % ni, o = (owner / ni) % no, tap = owner / (ni * no);
    int mode = InputSide ? f / rank : f % rank;
    int stride = InputSide ? rank : 1;
    int harmonic = (mode + 1) / 2;
    bool sine = mode && mode % 2 == 0;
    float sum = source[index];
    for (int part = 1; part < windows; ++part) {
        int delta = Transpose ? -part : part;
        int j = InputSide ? positive_mod(i + delta, ni) : i;
        int p = InputSide ? o : positive_mod(o + delta, no);
        const float *other = source + ((tap * no + p) * ni + j) * pair_size;
        int turn = (harmonic * part) % windows;
        if (!turn) {
            sum += other[f];
        } else if (windows == 2) {
            sum -= other[f];
        } else {
            // The alternating Nyquist basis has no sine companion.
            bool nyquist = 2 * harmonic == period;
            constexpr float half_sqrt3 = 0.8660254037844386f;
            float phase = turn == 1 ? half_sqrt3 : -half_sqrt3;
            if ((!sine) != Transpose) phase = -phase;
            float cross = nyquist ? 0 : phase * other[f + (sine ? -stride : stride)];
            sum += -0.5f * other[f] + cross;
        }
    }
    dest[index] = sum;
}

__global__ void channel_fourier_parameter_grad(const float *p, const float *geometry,
    const float *moments, const int64_t *order, float *grad, int atoms, int tap_atoms,
    Axis ai, Axis ao, ChannelFourierLayout layout) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= atoms) return;
    int atom = order ? int(order[row]) : row;
    const float *v = geometry + int64_t(atom) * 25;
    int bank = int(v[20]);
    Bank bi = ai.banks[bank], bo = ao.banks[bank];
    int period = channel_period(bi), rank = channel_rank(bi);
    float c[9], dr[9];
    cotangent_cell_coefficients(p[4 * atom + 1], float(period), c, dr);
    CotangentFactor series[COTANGENT_FEATURES], ci[COTANGENT_FEATURES], co[COTANGENT_FEATURES];
    cotangent_center_phase(c, dr, v[18] - int(v[21]) * bi.tile_sites, float(period), series);
    channel_alias(series, ci, period, rank);
    cotangent_center_phase(c, dr, v[19] - int(v[23]) * bo.tile_sites, float(period), series);
    channel_alias(series, co, period, rank);
    const float *tile = moments + channel_owner_offset(v, atom / tap_atoms, ai, ao, layout);
    float sum[4] = {};
    for (int i = 0; i < rank; ++i) {
        CotangentFactor accum{};
        for (int o = 0; o < rank; ++o)
            cotangent_accumulate(accum, tile[i * rank + o], co[o]);
        cotangent_combine(sum, accum, ci[i]);
    }
    cotangent_store(grad, atom, v[24], sum);
}

std::tuple<Tensor, Tensor, Tensor, int> packed_channel_fourier_forward(
    Tensor p, int input_sites, int output_sites, int radius_cap, bool sorted, bool active_only) {
    check_channel_atoms(p);
    check_channel_fourier_chart(input_sites, output_sites, radius_cap);
    c10::cuda::CUDAGuard guard(p.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    Axis ai = channel_axis(input_sites, radius_cap), ao = channel_axis(output_sites, radius_cap);
    int taps = p.size(0), tap_atoms = p.size(1), atoms = taps * tap_atoms;
    auto geometry = packed_make_geometry<true>(p.view({-1, 4}), ai, ao, radius_cap);
    int mask = (1 << ai.bank_count) - 1;
    if (active_only) {
        auto occupied = torch::zeros({1}, p.options().dtype(torch::kInt));
        channel_active_fourier_banks<<<(atoms + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
            geometry.data_ptr<float>(), occupied.data_ptr<int>(), atoms);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        mask = occupied.item<int>();
    }
    auto layout = channel_fourier_layout(ai, ao, taps, mask);
    auto owners = torch::zeros({layout.size}, p.options());
    auto order = torch::empty({0}, p.options().dtype(torch::kLong));
    if (sorted) {
        auto keys = torch::empty({atoms}, order.options());
        channel_fourier_keys<<<(atoms + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
            geometry.data_ptr<float>(), keys.data_ptr<int64_t>(), atoms, tap_atoms, ai, ao, layout);
        auto [ordered_keys, rows] = at::sort(keys, true, 0, false);
        order = rows;
        channel_owner_coefficients<true><<<atoms, DENSE_WEIGHT_THREADS, 0, stream>>>(
            geometry.data_ptr<float>(), ordered_keys.data_ptr<int64_t>(), order.data_ptr<int64_t>(),
            owners.data_ptr<float>(), atoms, tap_atoms, ai, ao, layout);
    } else {
        channel_owner_coefficients<false><<<atoms, DENSE_WEIGHT_THREADS, 0, stream>>>(
            geometry.data_ptr<float>(), nullptr, nullptr, owners.data_ptr<float>(),
            atoms, tap_atoms, ai, ao, layout);
    }
    auto weight = torch::zeros({taps, ao.sites, ai.sites}, p.options());
    for (int b = 0; b < ai.bank_count; ++b) {
        if (layout.offsets[b] < 0) continue;
        Bank bi = ai.banks[b], bo = ao.banks[b];
        int tile = bi.tile_sites, period = channel_period(bi), rank = channel_rank(bi);
        int ni = bi.tile_count, no = bo.tile_count, count = taps * ni * no * rank * rank;
        auto basis = torch::empty({tile, rank}, p.options());
        channel_fourier_basis<<<(tile * rank + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
            basis.data_ptr<float>(), tile, period, rank);
        auto coeff = owners.narrow(0, layout.offsets[b], count);
        auto shifted_input = torch::empty_like(coeff), shifted = torch::empty_like(coeff);
        int blocks = (count + THREADS - 1) / THREADS;
        channel_fourier_fold<true, true><<<blocks, THREADS, 0, stream>>>(
            coeff.data_ptr<float>(), shifted_input.data_ptr<float>(), taps, ni, no, rank, bi.count, period);
        channel_fourier_fold<false, true><<<blocks, THREADS, 0, stream>>>(
            shifted_input.data_ptr<float>(), shifted.data_ptr<float>(), taps, ni, no, rank, bi.count, period);
        auto output = at::mm(shifted.view({-1, rank}), basis.t());
        auto input = output.view({taps * no, ni * rank, tile})
            .transpose(1, 2).contiguous().view({-1, rank});
        weight.view({-1, tile}).addmm_(input, basis.t());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {weight, geometry, order, mask};
}

Tensor packed_channel_fourier_grad(Tensor p, Tensor dense, Tensor geometry, Tensor order,
    int input_sites, int output_sites, int radius_cap, int mask) {
    check_channel_atoms(p);
    check_channel_fourier_chart(input_sites, output_sites, radius_cap);
    c10::cuda::CUDAGuard guard(p.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    Axis ai = channel_axis(input_sites, radius_cap), ao = channel_axis(output_sites, radius_cap);
    int taps = p.size(0), tap_atoms = p.size(1), atoms = taps * tap_atoms;
    packed_check_tensor(dense, p, {taps, ao.sites, ai.sites});
    packed_check_tensor(geometry, p, {atoms, PACKED_GEOMETRY_FIELDS});
    TORCH_CHECK(order.device() == p.device() && order.is_contiguous() &&
        order.scalar_type() == torch::kLong && order.dim() == 1 &&
        (order.numel() == 0 || order.numel() == atoms), "invalid Fourier owner order");
    TORCH_CHECK(mask > 0 && mask < (1 << ai.bank_count), "invalid Fourier bank mask");
    auto layout = channel_fourier_layout(ai, ao, taps, mask);
    auto all_moments = torch::empty({layout.size}, p.options());
    for (int b = 0; b < ai.bank_count; ++b) {
        if (layout.offsets[b] < 0) continue;
        Bank bi = ai.banks[b], bo = ao.banks[b];
        int tile = bi.tile_sites, period = channel_period(bi), rank = channel_rank(bi);
        int ni = bi.tile_count, no = bo.tile_count, count = taps * ni * no * rank * rank;
        auto basis = torch::empty({tile, rank}, p.options());
        channel_fourier_basis<<<(tile * rank + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
            basis.data_ptr<float>(), tile, period, rank);
        auto input = at::mm(dense.view({-1, tile}), basis);
        auto output_input = input.view({taps * no, tile, ni * rank})
            .transpose(1, 2).contiguous().view({-1, tile});
        auto moments = at::mm(output_input, basis);
        auto folded_input = torch::empty_like(moments);
        int blocks = (count + THREADS - 1) / THREADS;
        channel_fourier_fold<true><<<blocks, THREADS, 0, stream>>>(
            moments.data_ptr<float>(), folded_input.data_ptr<float>(), taps, ni, no, rank, bi.count, period);
        channel_fourier_fold<false><<<blocks, THREADS, 0, stream>>>(
            folded_input.data_ptr<float>(), all_moments.data_ptr<float>() + layout.offsets[b],
            taps, ni, no, rank, bi.count, period);
    }
    auto grad = torch::empty_like(p);
    channel_fourier_parameter_grad<<<(atoms + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        p.data_ptr<float>(), geometry.data_ptr<float>(), all_moments.data_ptr<float>(),
        order.numel() ? order.data_ptr<int64_t>() : nullptr, grad.data_ptr<float>(),
        atoms, tap_atoms, ai, ao, layout);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}

} // namespace radius_cell_mass
#endif
