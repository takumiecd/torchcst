#pragma once

#ifdef RADIUS_RATIONAL_WIDE
namespace radius_cell_mass {

constexpr int COTANGENT_FEATURES = 17;
constexpr int COTANGENT_FIRST_LEVEL = 4;
constexpr int COTANGENT_LAST_LEVEL = 9;
constexpr int COTANGENT_MAX_DIRECT_SITES = 3 << (COTANGENT_FIRST_LEVEL - 1);

struct CotangentFactor {
    float value, radius, center;
};

// The frequency is fixed during differentiation within a classified bank.
__device__ inline void cotangent_series(
    float radius, float mu, float period, CotangentFactor *factor) {
    float c[9], dr[9];
    profile_coefficients_m8(radius, period, c, dr);
    float omega = 2 * float(PI) / period;
    SignalPhase<float> phase(omega * mu), cell_phase(omega / 2);
    factor[0] = {c[0], 0, 0};
    #pragma unroll
    for (int k = 1; k <= 8; ++k) {
        float cp, sp, cs, sn;
        phase.next(cp, sp);
        cell_phase.next(cs, sn);
        float frequency = omega * k, filter = 2 * sn / (frequency / 2);
        float mass = filter * c[k], radius_derivative = filter * dr[k];
        factor[2 * k - 1] = {mass * cp, radius_derivative * cp, -frequency * mass * sp};
        factor[2 * k] = {mass * sp, radius_derivative * sp, frequency * mass * cp};
    }
}

// The transpose of each local-site Fourier basis, with the cell integral
// carried by the Atom coefficients.
__global__ void cotangent_basis(float *table, int sites) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= sites * COTANGENT_FEATURES) return;
    int site = index / COTANGENT_FEATURES, basis = index % COTANGENT_FEATURES;
    if (!basis) {
        table[index] = 1;
        return;
    }
    float angle = (2 * float(PI) / (3 * sites)) * ((basis + 1) / 2) * site;
    table[index] = basis % 2 ? cosf(angle) : sinf(angle);
}

// Shift each projection from target-tile to owner-tile phase. The phase of
// harmonic k across tiles of a three-tile bank is 2*pi*k/3 per tile.
template<bool InputSide, bool Transpose = false>
__global__ void cotangent_fold_axis(const float *source, float *dest, int ni, int no) {
    constexpr int F = COTANGENT_FEATURES, pair_size = F * F;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ni * no * pair_size) return;
    int feature = index % pair_size, pair_index = index / pair_size;
    int input_tile = pair_index % ni, output_tile = pair_index / ni;
    int basis = InputSide ? feature / F : feature % F;
    int companion_offset = (basis % 2 ? 1 : -1) * (InputSide ? F : 1);
    int k = (basis + 1) / 2;
    float total = source[index];
    for (int part = 1; part < 3; ++part) {
        // The transpose sends owner coefficients to the target tile.
        int delta = Transpose ? -part : part;
        int i = InputSide ? positive_mod(input_tile + delta, ni) : input_tile;
        int o = InputSide ? output_tile : positive_mod(output_tile + delta, no);
        const float *tile = source + (o * ni + i) * pair_size;
        int turn = (k * part) % 3;
        if (!turn) {
            total += tile[feature];
        } else {
            constexpr float half_sqrt3 = 0.8660254037844386f;
            float sin_phase = (turn == 1 ? half_sqrt3 : -half_sqrt3);
            if ((basis % 2 != 0) != Transpose) sin_phase = -sin_phase;
            total += -0.5f * tile[feature] + sin_phase * tile[feature + companion_offset];
        }
    }
    dest[index] = total;
}

__device__ inline void cotangent_sites(float radius, float mu, Bank bank,
    int first_site, CotangentFactor *cells) {
    float c[9], dr[9], value[9], derivative[9], center_derivative[9];
    float period = float(bank.period), omega = 2 * float(PI) / period;
    profile_coefficients_m8(radius, period, c, dr);
    SignalPhase<float> cell_phase(omega / 2);
    #pragma unroll
    for (int k = 1; k <= 8; ++k) {
        float unused, sn;
        cell_phase.next(unused, sn);
        float frequency = omega * k, filter = 2 * sn / (frequency / 2);
        value[k] = filter * c[k];
        derivative[k] = filter * dr[k];
        center_derivative[k] = -frequency * value[k];
    }
    for (int site = 0; site < bank.count * bank.tile_sites; ++site) {
        CotangentFactor cell{c[0], 0, 0};
        SignalPhase<float> phase(omega * (mu - (first_site + site)));
        #pragma unroll
        for (int k = 1; k <= 8; ++k) {
            float cp, sp;
            phase.next(cp, sp);
            cell.value += value[k] * cp;
            cell.radius += derivative[k] * cp;
            cell.center += center_derivative[k] * sp;
        }
        cells[site] = cell;
    }
}

__device__ inline void cotangent_accumulate(CotangentFactor &sum, float entry, CotangentFactor factor) {
    sum.value += entry * factor.value;
    sum.radius += entry * factor.radius;
    sum.center += entry * factor.center;
}

__device__ inline void cotangent_combine(float *sum, CotangentFactor row, CotangentFactor column) {
    sum[0] += row.value * column.value;
    sum[1] += row.radius * column.value + row.value * column.radius;
    sum[2] += row.center * column.value;
    sum[3] += row.value * column.center;
}

__device__ inline void cotangent_store(float *gradient, int atom, float amp, const float *sum) {
    gradient[4 * atom] = sum[0];
    #pragma unroll
    for (int f = 1; f < 4; ++f) gradient[4 * atom + f] = amp * sum[f];
}

__global__ void cotangent_direct_atoms(const float *p, const float *geometry, const float *dense,
    float *gradient, int atoms, Axis input, Axis output) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    const float *v = geometry + int64_t(atom) * 25;
    int bank = int(v[20]);
    if (bank >= 2 * COTANGENT_FIRST_LEVEL) return;
    Bank bi = input.banks[bank], bo = output.banks[int(v[22])];
    int first_i = int(v[21]) * bi.tile_sites, first_o = int(v[23]) * bo.tile_sites;
    float radius = p[4 * atom + 1];
    CotangentFactor ci[COTANGENT_MAX_DIRECT_SITES], co[COTANGENT_MAX_DIRECT_SITES];
    cotangent_sites(radius, v[18], bi, first_i, ci);
    cotangent_sites(radius, v[19], bo, first_o, co);
    float sum[4] = {};
    for (int o = 0; o < bo.count * bo.tile_sites; ++o) {
        int row = positive_mod(first_o + o, output.sites);
        const float *d = dense + int64_t(row) * input.sites;
        CotangentFactor accum{};
        for (int i = 0; i < bi.count * bi.tile_sites; ++i)
            cotangent_accumulate(accum, d[positive_mod(first_i + i, input.sites)], ci[i]);
        cotangent_combine(sum, co[o], accum);
    }
    cotangent_store(gradient, atom, v[24], sum);
}

__global__ void cotangent_wide_atoms(const float *p, const float *geometry,
    const float *pair_moments, float *gradient, int atoms, int level, Axis input, Axis output) {
    constexpr int F = COTANGENT_FEATURES;
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    const float *v = geometry + int64_t(atom) * 25;
    int bank = int(v[20]);
    if (bank != 2 * level + 1) return;
    Bank bi = input.banks[bank], bo = output.banks[bank];
    int first_i = int(v[21]), first_o = int(v[23]);
    int tile_i = positive_mod(first_i, bi.tile_count), tile_o = positive_mod(first_o, bo.tile_count);
    const float *tile = pair_moments + (tile_o * bi.tile_count + tile_i) * F * F;
    float radius = p[4 * atom + 1];
    CotangentFactor ci[F], co[F];
    cotangent_series(radius, v[18] - first_i * bi.tile_sites, float(bi.period), ci);
    cotangent_series(radius, v[19] - first_o * bo.tile_sites, float(bo.period), co);
    float sum[4] = {};
    // Input basis varies more slowly in the matrix produced by the two GEMMs.
    #pragma unroll
    for (int i = 0; i < F; ++i) {
        CotangentFactor accum{};
        #pragma unroll
        for (int o = 0; o < F; ++o)
            cotangent_accumulate(accum, tile[i * F + o], co[o]);
        cotangent_combine(sum, accum, ci[i]);
    }
    cotangent_store(gradient, atom, v[24], sum);
}

Tensor packed_dense_weight_grad(Tensor p, Tensor dense, Tensor geometry,
    Config ci, Config co) {
    c10::cuda::CUDAGuard guard(p.device());
    TORCH_CHECK(!at::globalContext().allowTF32CuBLAS(),
        "pair cotangents require IEEE FP32 matmul");
    Axis input = parse_axis(ci), output = parse_axis(co);
    constexpr int F = COTANGENT_FEATURES;
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto grad = torch::empty_like(p);
    int atom_blocks = (p.size(0) + THREADS - 1) / THREADS;
    cotangent_direct_atoms<<<atom_blocks, THREADS, 0, stream>>>(p.data_ptr<float>(),
        geometry.data_ptr<float>(), dense.data_ptr<float>(), grad.data_ptr<float>(),
        p.size(0), input, output);
    for (int level = COTANGENT_FIRST_LEVEL; level <= COTANGENT_LAST_LEVEL; ++level) {
        int sites = 1 << level, ni = input.sites / sites, no = output.sites / sites;
        auto basis = torch::empty({sites, F}, p.options());
        cotangent_basis<<<(sites * F + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
            basis.data_ptr<float>(), sites);
        auto input_projection = at::mm(dense.view({-1, sites}), basis);
        auto output_input = input_projection.view({no, sites, ni * F})
            .transpose(1, 2).contiguous().view({-1, sites});
        auto moments = at::mm(output_input, basis);
        auto folded_input = torch::empty_like(moments), folded = torch::empty_like(moments);
        int blocks = (ni * no * F * F + THREADS - 1) / THREADS;
        cotangent_fold_axis<true><<<blocks, THREADS, 0, stream>>>(moments.data_ptr<float>(),
            folded_input.data_ptr<float>(), ni, no);
        cotangent_fold_axis<false><<<blocks, THREADS, 0, stream>>>(folded_input.data_ptr<float>(),
            folded.data_ptr<float>(), ni, no);
        cotangent_wide_atoms<<<atom_blocks, THREADS, 0, stream>>>(p.data_ptr<float>(),
            geometry.data_ptr<float>(), folded.data_ptr<float>(), grad.data_ptr<float>(),
            p.size(0), level, input, output);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}

Tensor packed_dense_cotangent_grad(Tensor p, Tensor x, Tensor dy, Tensor geometry,
    Config ci, Config co) {
    return packed_dense_weight_grad(p, at::mm(dy.t(), x), geometry, ci, co);
}

} // namespace radius_cell_mass
#endif
