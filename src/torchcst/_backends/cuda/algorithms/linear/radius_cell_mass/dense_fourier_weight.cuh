#pragma once

#ifdef RADIUS_LM_WIDTH
namespace radius_cell_mass {

constexpr int DENSE_WEIGHT_CHUNK = 256;
constexpr int DENSE_WEIGHT_LOADING = 64;
constexpr int DENSE_WEIGHT_THREADS = 320;
constexpr int PAIR_FEATURES = COTANGENT_FEATURES * COTANGENT_FEATURES;

__host__ __device__ inline int dense_pair_prefix(int ni, int no, int level) {
    int groups = 0;
    for (int l = COTANGENT_FIRST_LEVEL; l < level; ++l)
        groups += (ni >> l) * (no >> l);
    return groups;
}

// Group zero holds the direct atoms, then each level has a contiguous block
// per pair of periodic owner tiles.
__global__ void dense_weight_keys(const float *geometry, int64_t *keys, int atoms, Axis ai, Axis ao) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    const float *v = geometry + int64_t(atom) * 25;
    int bank = int(v[20]), level = bank / 2;
    if (level < COTANGENT_FIRST_LEVEL) {
        keys[atom] = 0;
        return;
    }
    int ni = ai.banks[bank].tile_count, no = ao.banks[bank].tile_count;
    int input_tile = positive_mod(int(v[21]), ni), output_tile = positive_mod(int(v[23]), no);
    keys[atom] = 1 + dense_pair_prefix(ai.sites, ao.sites, level) + output_tile * ni + input_tile;
}

__device__ inline void dense_weight_cell_masses(const float *c, float mu, Bank bank,
    int first_site, float *mass) {
    float omega = 2 * float(PI) / float(bank.period), scaled[9];
    SignalPhase<float> cell_phase(omega / 2);
    #pragma unroll
    for (int k = 1; k <= 8; ++k) {
        float cosine, sine;
        cell_phase.next(cosine, sine);
        scaled[k] = 2 * c[k] * sine / (omega * k / 2);
    }
    for (int site = 0; site < bank.count * bank.tile_sites; ++site) {
        float value = c[0];
        SignalPhase<float> phase(omega * (mu - (first_site + site)));
        #pragma unroll
        for (int k = 1; k <= 8; ++k) {
            float cp, sp;
            phase.next(cp, sp);
            value += scaled[k] * cp;
        }
        mass[site] = value;
    }
}

__global__ void dense_weight_direct(const float *geometry, const int64_t *order,
    const int64_t *offsets, float *weight, Axis ai, Axis ao) {
    int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= offsets[1]) return;
    int atom = order[position];
    const float *v = geometry + int64_t(atom) * 25;
    Bank bi = ai.banks[int(v[20])], bo = ao.banks[int(v[22])];
    int first_i = int(v[21]) * bi.tile_sites, first_o = int(v[23]) * bo.tile_sites;
    float ci[COTANGENT_MAX_DIRECT_SITES], co[COTANGENT_MAX_DIRECT_SITES];
    dense_weight_cell_masses(v, v[18], bi, first_i, ci);
    dense_weight_cell_masses(v + 9, v[19], bo, first_o, co);
    for (int o = 0; o < bo.count * bo.tile_sites; ++o) {
        int row = positive_mod(first_o + o, ao.sites);
        float *dest = weight + int64_t(row) * ai.sites;
        float mass = v[24] * co[o];
        for (int i = 0; i < bi.count * bi.tile_sites; ++i)
            atomicAdd(&dest[positive_mod(first_i + i, ai.sites)], mass * ci[i]);
    }
}

__device__ inline void dense_weight_series(
    const float *c, float delta, float period, float scale, float *series) {
    float omega = 2 * float(PI) / period;
    SignalPhase<float> phase(omega * delta), cell_phase(omega / 2);
    series[0] = scale * c[0];
    #pragma unroll
    for (int k = 1; k <= 8; ++k) {
        float cp, sp, cell_cosine, cell_sine;
        phase.next(cp, sp);
        cell_phase.next(cell_cosine, cell_sine);
        float factor = scale * (2 * c[k] * cell_sine / (omega * k / 2));
        series[2 * k - 1] = factor * cp;
        series[2 * k] = factor * sp;
    }
}

__device__ inline int dense_weight_task_begin(const int64_t *offsets, int group) {
    return group - 1 + int((offsets[group] - offsets[1]) / DENSE_WEIGHT_CHUNK);
}

__global__ void dense_weight_owner_partials(const float *geometry, const int64_t *order,
    const int64_t *offsets, float *partials, int groups, Axis ai, Axis ao) {
    int task = blockIdx.x, low = 1, high = groups + 1;
    if (task >= dense_weight_task_begin(offsets, high)) return;
    while (low + 1 < high) {
        int middle = (low + high) / 2;
        if (dense_weight_task_begin(offsets, middle) <= task) low = middle;
        else high = middle;
    }
    int first = offsets[low] + (task - dense_weight_task_begin(offsets, low)) * DENSE_WEIGHT_CHUNK;
    int last = min(int(offsets[low + 1]), first + DENSE_WEIGHT_CHUNK);
    if (first >= last) return;

    constexpr int F = COTANGENT_FEATURES, load = DENSE_WEIGHT_LOADING;
    __shared__ float ci[load][F], co[load][F];
    int tid = threadIdx.x, input_feature = tid / F, output_feature = tid % F;
    float sum = 0;
    for (int chunk = first; chunk < last; chunk += load) {
        int count = min(load, last - chunk);
        if (tid < count) {
            int atom = order[chunk + tid];
            const float *v = geometry + int64_t(atom) * 25;
            Bank bi = ai.banks[int(v[20])], bo = ao.banks[int(v[22])];
            dense_weight_series(v, v[18] - int(v[21]) * bi.tile_sites,
                float(bi.period), v[24], ci[tid]);
            dense_weight_series(v + 9, v[19] - int(v[23]) * bo.tile_sites,
                float(bo.period), 1.f, co[tid]);
        }
        __syncthreads();
        if (tid < PAIR_FEATURES)
            for (int a = 0; a < count; ++a)
                sum += ci[a][input_feature] * co[a][output_feature];
        __syncthreads();
    }
    if (tid < PAIR_FEATURES) partials[task * PAIR_FEATURES + tid] = sum;
}

__global__ void dense_weight_reduce(const int64_t *offsets, const float *partials,
    float *owners, int pairs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pairs * PAIR_FEATURES) return;
    int group = index / PAIR_FEATURES + 1, coefficient = index % PAIR_FEATURES;
    int task = dense_weight_task_begin(offsets, group);
    int count = int((offsets[group + 1] - offsets[group] + DENSE_WEIGHT_CHUNK - 1) / DENSE_WEIGHT_CHUNK);
    float sum = 0;
    for (int t = 0; t < count; ++t) sum += partials[(task + t) * PAIR_FEATURES + coefficient];
    owners[index] = sum;
}

Tensor packed_make_dense_weight(Tensor p, Tensor geometry, Config ci, Config co) {
    c10::cuda::CUDAGuard guard(p.device());
    TORCH_CHECK(!at::globalContext().allowTF32CuBLAS(), "dense tiles require IEEE FP32 matmul");
    Axis ai = parse_axis(ci), ao = parse_axis(co);
    const auto stream = at::cuda::getCurrentCUDAStream();
    int atoms = p.size(0), atom_blocks = (atoms + THREADS - 1) / THREADS;
    int groups = dense_pair_prefix(ai.sites, ao.sites, COTANGENT_LAST_LEVEL + 1);
    auto keys = torch::empty({atoms}, p.options().dtype(torch::kLong));
    dense_weight_keys<<<atom_blocks, THREADS, 0, stream>>>(
        geometry.data_ptr<float>(), keys.data_ptr<int64_t>(), atoms, ai, ao);
    auto [sorted, order] = at::sort(keys, true, 0, false);
    auto offsets = at::searchsorted(sorted, torch::arange(groups + 2, keys.options()));
    auto weight = torch::zeros({ao.sites, ai.sites}, p.options());
    dense_weight_direct<<<atom_blocks, THREADS, 0, stream>>>(
        geometry.data_ptr<float>(), order.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        weight.data_ptr<float>(), ai, ao);
    int max_tasks = (atoms + DENSE_WEIGHT_CHUNK - 1) / DENSE_WEIGHT_CHUNK + groups;
    auto partials = torch::empty({max_tasks, PAIR_FEATURES}, p.options());
    dense_weight_owner_partials<<<max_tasks, DENSE_WEIGHT_THREADS, 0, stream>>>(
        geometry.data_ptr<float>(), order.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        partials.data_ptr<float>(), groups, ai, ao);
    auto owners = torch::empty({groups, PAIR_FEATURES}, p.options());
    dense_weight_reduce<<<(owners.numel() + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        offsets.data_ptr<int64_t>(), partials.data_ptr<float>(), owners.data_ptr<float>(), groups);
    for (int level = COTANGENT_FIRST_LEVEL; level <= COTANGENT_LAST_LEVEL; ++level) {
        constexpr int F = COTANGENT_FEATURES;
        int sites = 1 << level, ni = ai.sites / sites, no = ao.sites / sites;
        auto basis = torch::empty({sites, F}, p.options());
        cotangent_basis<<<(sites * F + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
            basis.data_ptr<float>(), sites);
        auto coefficients = owners.narrow(0, dense_pair_prefix(ai.sites, ao.sites, level), ni * no);
        auto shifted_input = torch::empty_like(coefficients), shifted = torch::empty_like(coefficients);
        int blocks = (ni * no * F * F + THREADS - 1) / THREADS;
        cotangent_fold_axis<true, true><<<blocks, THREADS, 0, stream>>>(coefficients.data_ptr<float>(),
            shifted_input.data_ptr<float>(), ni, no);
        cotangent_fold_axis<false, true><<<blocks, THREADS, 0, stream>>>(shifted_input.data_ptr<float>(),
            shifted.data_ptr<float>(), ni, no);
        auto output_expansion = at::mm(shifted.view({-1, F}), basis.t());
        auto input_features = output_expansion.view({no, ni * F, sites})
            .transpose(1, 2).contiguous().view({-1, F});
        weight.add_(at::mm(input_features, basis.t()).view_as(weight));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return weight;
}

} // namespace radius_cell_mass
#endif
