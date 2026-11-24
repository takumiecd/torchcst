#pragma once
#include "dense_pair_cotangent.cuh"
#include "dense_fourier_weight.cuh"
#ifdef RADIUS_RATIONAL_WIDE
#include "axis_parameter_grad.cuh"
#endif
// FP32 Linear arithmetic and Z/H/G, with optional recomputation.
// Host configuration scalars retain the common metadata representation.
namespace radius_cell_mass {
__global__ void packed_classify_kernel(const float *p,int64_t *keys,int atoms,Axis ai,Axis ao,float radius_limit) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=atoms)return;
    float ri=signal_radius<float>(p,a,ai),ro=signal_radius<float>(p,a,ao);
    bool valid=isfinite(ri)&&isfinite(ro)&&ri>0&&ro>0;
    if (radius_limit > 0) valid &= ri <= radius_limit && ro <= radius_limit;
    for(int c=0;c<4;++c)valid &= isfinite(p[4*a+c]);
    valid &= isfinite(p[4*a]*p[4*a]+p[4*a+1]*p[4*a+1]);
    CUDA_KERNEL_ASSERT_MSG(valid,"invalid FP32 packed geometry");
    if(!valid){keys[a]=0;return;}
    auto mi=signal_classify<float>(ri,p[4*a+3],ai),mo=signal_classify<float>(ro,p[4*a+2],ao);
    Bank bi=ai.banks[mi.bank],bo=ao.banks[mo.bank];
    int ni=bi.node_offset+positive_mod(mi.first_tile,bi.tile_count),no=bo.node_offset+positive_mod(mo.first_tile,bo.tile_count);
    keys[a]=int64_t(no)*ai.node_count+ni;
}
Tensor packed_classify(Tensor p,Config ci,Config co,float radius_limit) {
    c10::cuda::CUDAGuard guard(p.device());
    auto keys=torch::empty({p.size(0)},p.options().dtype(torch::kLong));
    packed_classify_kernel<<<(p.size(0)+THREADS-1)/THREADS,THREADS,0,at::cuda::getCurrentCUDAStream()>>>(
        p.data_ptr<float>(),keys.data_ptr<int64_t>(),p.size(0),parse_axis(ci),parse_axis(co),radius_limit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return keys;
}
#ifdef RADIUS_RATIONAL_WIDE
bool packed_scale_cohort(double input, double output, int64_t atoms) {
    return input == output && ((input == 4096 && atoms == 4096 * 32) ||
        (input == 8192 && (atoms == 8192 * 16 || atoms == 8192 * 32)) ||
        (input == 16384 && atoms == 16384 * 32));
}
#endif

void packed_check_scope(Tensor x, Tensor p, Config ci, Config co) {
    check_atoms(p); check_matrix(x);
    Axis ai = parse_axis(ci), ao = parse_axis(co);
    TORCH_CHECK(EXPANSION_BATCH == 8 && PARAMETER_BATCH == 8,
        "packed rows require the distinct row8 build");
    bool ffn = p.size(0) == 4096 &&
        ((ai.sites == 256 && ao.sites == 1024) || (ai.sites == 1024 && ao.sites == 256));
    bool large = ai.sites == ao.sites &&
        (ai.sites == 2048 || ai.sites == 8192 || ai.sites == 16384) &&
        p.size(0) >= 4 && p.size(0) <= 8 * ai.sites && ai.origin == 0 && ao.origin == 0;
    bool physical = false;
#ifdef RADIUS_RATIONAL_WIDE
    physical = !ai.polar && !ao.polar && ai.sites == ao.sites &&
        ai.origin == 0 && ao.origin == 0 &&
        (packed_scale_cohort(ai.sites, ao.sites, p.size(0)) ||
            ((ai.sites == 2048 || ai.sites == 4096) &&
                (p.size(0) == 32 * ai.sites || p.size(0) == 48 * ai.sites)));
#endif
    TORCH_CHECK(x.device() == p.device() && x.size(1) == ai.sites &&
        x.scalar_type() == torch::kFloat && p.scalar_type() == torch::kFloat &&
        x.size(0) >= 1 && x.size(0) <= 256 && (ffn || large || physical),
        "packed rows require an admitted FP32 projection");
    for (auto a : {ai, ao}) TORCH_CHECK((a.polar || physical) && a.modes == 8 &&
        a.base_tile_sites == 1 && a.merge_factor == 2 && a.spacing == 1 &&
        (physical || (a.amplitude_max <= 1 && a.radius_min >= .5 && a.radius_max <= 50)),
        "packed rows require admitted M8/base1/merge2/spacing1 coordinates");
}

// Index only the two banks of this level; write the unchanged compact Z layout.
__global__ void packed_level_moments_kernel(const float *signal, float *z, int batch,
    Axis axis, int start, int level, const uint32_t *required_banks) {
    if (!(*required_banks & (3u << (2 * level)))) return;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int level_nodes = 2 * axis.banks[2 * level].tile_count, F = axis.basis_count();
    if (index >= batch * level_nodes * F) return;
    int local = (index / F) % level_nodes;
    int bank = 2 * level + int(local >= axis.banks[2 * level].tile_count);
    if (!(*required_banks & (uint32_t(1) << bank))) return;
    int node = axis.banks[2 * level].node_offset + local;
    int row = index / (level_nodes * F), basis = index % F;
    int64_t slot = (int64_t(row) * (axis.node_count - prefix_nodes(axis, start)) +
        node - prefix_nodes(axis, start)) * F + basis;
    z[slot] = compact_moment_element(signal, z, row, node, basis, axis, start, true);
}

Tensor packed_moments(Tensor x, Tensor p, Config config, bool level_launch) {
    Axis a = parse_axis(config); check_hybrid_signal(x, p, a);
    int nodes = a.node_count - checked_start(a, 2), f = a.basis_count();
    auto z = torch::empty({x.size(0), nodes, f}, x.options());
    auto mask = torch::zeros({1}, p.options().dtype(torch::kUInt32));
    active_moment_banks_kernel<float, float><<<(p.size(0)+THREADS-1)/THREADS,THREADS,0,
        at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),mask.data_ptr<uint32_t>(),p.size(0),a,2,true);
    for (int level = 2; level < a.bank_count / 2; ++level) {
        if (level_launch) {
            int64_t work = x.size(0) * 2 * a.banks[2 * level].tile_count * f;
            packed_level_moments_kernel<<<(work + THREADS - 1) / THREADS, THREADS, 0,
                at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), z.data_ptr<float>(),
                x.size(0), a, 2, level, mask.data_ptr<uint32_t>());
        } else {
        compact_moments_kernel<float, float><<<(z.numel() + THREADS - 1) / THREADS,
            THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),
            z.data_ptr<float>(), x.size(0), a, 2, level, true, mask.data_ptr<uint32_t>());
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return z;
}

std::vector<Tensor> packed_project(Tensor x, Tensor p, Config config, bool output_side, Tensor geometry, bool level_launch, Tensor response_order) {
    c10::cuda::CUDAGuard guard(x.device());
    auto z = packed_moments(x, p, config, level_launch);
    auto h = torch::empty({x.size(0), p.size(0)}, x.options());
    Axis a = parse_axis(config);
    dim3 grid((p.size(0) + PARALLEL_RESPONSE_THREADS - 1) / PARALLEL_RESPONSE_THREADS,
        (x.size(0) + 7) / 8);
#ifdef RADIUS_RATIONAL_WIDE
    if (response_order.defined()) {
        TORCH_CHECK(geometry.defined() && response_order.device() == p.device() &&
            response_order.scalar_type() == at::kLong && response_order.is_contiguous() &&
            response_order.dim() == 1 && response_order.size(0) == p.size(0),
            "invalid packed response order");
        parallel_response_kernel<float, float, 8, true, true><<<grid,
            PARALLEL_RESPONSE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(), p.data_ptr<float>(), z.data_ptr<float>(), h.data_ptr<float>(),
            p.size(0), x.size(0), a, 2, output_side, geometry.data_ptr<float>(),
            response_order.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {z, h};
    }
#endif
    if (geometry.defined()) {
        parallel_response_kernel<float, float, 8, true><<<grid, PARALLEL_RESPONSE_THREADS, 0,
            at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), p.data_ptr<float>(),
            z.data_ptr<float>(), h.data_ptr<float>(), p.size(0), x.size(0), a, 2, output_side,
            geometry.data_ptr<float>());
    } else {
    parallel_response_kernel<float, float, 8><<<grid, PARALLEL_RESPONSE_THREADS, 0,
        at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), p.data_ptr<float>(),
        z.data_ptr<float>(), h.data_ptr<float>(), p.size(0), x.size(0), a, 2, output_side);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {z, h};
}

std::vector<Tensor> packed_transform(Tensor x, Tensor p, Tensor rows, Tensor offsets,
    Config source, Config destination, bool output_side, Tensor geometry, bool level_launch, bool shared_wide, Tensor response_order) {
    c10::cuda::CUDAGuard guard(x.device());
    auto projection = packed_project(x, p, source, output_side, geometry, level_launch, response_order);
    Axis a = parse_axis(destination); check_ranges(rows, offsets, p, a);
    auto y = torch::empty({x.size(0), a.sites}, x.options());
    auto layout = batched_coefficient_layout<float>(a, 2, 32);
    dim3 grid((a.sites + 31) / 32, (x.size(0) + 7) / 8);
#ifdef RADIUS_RATIONAL_WIDE
    if (shared_wide) {
        int prefix = prefix_nodes(a, SHARED_WIDE_LEVEL), nodes = a.node_count - prefix;
        int features = a.basis_count();
        auto wide = torch::empty({grid.y, nodes, EXPANSION_BATCH, features}, x.options());
        size_t scratch = sizeof(float) * (EXPANSION_BATCH * features +
            features * BLOCK_LOCAL_C_STRIDE + layout.capacity * (a.modes + 2 + EXPANSION_BATCH)) +
            sizeof(int) * layout.capacity;
        auto stream = at::cuda::getCurrentCUDAStream();
        batched_wide_tile_coefficients<<<dim3(nodes, grid.y), THREADS, scratch, stream>>>(
            p.data_ptr<float>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
            projection[1].data_ptr<float>(), geometry.data_ptr<float>(), wide.data_ptr<float>(),
            p.size(0), x.size(0), a, prefix, layout.capacity, !output_side);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        batched_cached_expand_kernel<float, false, float, true, true><<<grid, THREADS, layout.bytes, stream>>>(
            p.data_ptr<float>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
            projection[1].data_ptr<float>(), y.data_ptr<float>(), p.size(0), x.size(0), a, 2, 32,
            layout.capacity, layout.warps, layout.destinations, !output_side,
            nullptr, geometry.data_ptr<float>(), wide.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {y, projection[0], projection[1]};
    }
#endif
    if (geometry.defined()) {
        batched_cached_expand_kernel<float, false, float, true><<<grid, THREADS, layout.bytes,
            at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(), rows.data_ptr<int64_t>(),
            offsets.data_ptr<int64_t>(), projection[1].data_ptr<float>(), y.data_ptr<float>(),
            p.size(0), x.size(0), a, 2, 32, layout.capacity, layout.warps,
            layout.destinations, !output_side, nullptr, geometry.data_ptr<float>());
    } else {
    batched_cached_expand_kernel<float, false, float><<<grid, THREADS, layout.bytes,
        at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(), rows.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(), projection[1].data_ptr<float>(), y.data_ptr<float>(),
        p.size(0), x.size(0), a, 2, 32, layout.capacity, layout.warps,
        layout.destinations, !output_side, nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, projection[0], projection[1]};
}

// A small transient geometry table avoids repeating profile decode in each row
// group. Linear arithmetic, scratch and retained tensors are FP32.
constexpr int PACKED_GEOMETRY_FIELDS = 25;
template<bool SharedProfile>
__global__ void packed_gradient_geometry(const float *p, float *geometry, int atoms,
    Axis ai, Axis ao, float radius_limit) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    float *v = geometry + int64_t(atom) * PACKED_GEOMETRY_FIELDS;
    float ri = signal_radius<float>(p, atom, ai), ro = signal_radius<float>(p, atom, ao);
    if (radius_limit > 0) {
        bool valid = ri > 0 && ro > 0 && ri <= radius_limit && ro <= radius_limit;
        for (int c = 0; c < 4; ++c) valid &= isfinite(p[4 * atom + c]);
        CUDA_KERNEL_ASSERT_MSG(valid, "invalid capped radius Atom");
        if (!valid) {
            for (int c = 0; c < PACKED_GEOMETRY_FIELDS; ++c) v[c] = 0;
            return;
        }
    }
    Membership mi = signal_classify<float>(ri, float(center(p, atom, ai, false)), ai);
    Membership mo = signal_classify<float>(ro, float(center(p, atom, ao, true)), ao);
    profile_values(ri, float(ai.banks[mi.bank].period), 8, v);
    if constexpr (!SharedProfile)
        profile_values(ro, float(ao.banks[mo.bank].period), 8, v + 9);
    v[18] = signal_wrapped<float>(float(center(p, atom, ai, false)), ai);
    v[19] = signal_wrapped<float>(float(center(p, atom, ao, true)), ao);
    v[20] = mi.bank; v[21] = mi.first_tile;
    v[22] = mo.bank; v[23] = mo.first_tile;
    v[24] = signal_amplitude<float>(p, atom, ai);
}

template<bool SharedProfile = false>
Tensor packed_make_geometry(Tensor p, Axis ai, Axis ao, float radius_limit = 0) {
    c10::cuda::CUDAGuard guard(p.device());
    auto geometry = torch::empty({p.size(0), PACKED_GEOMETRY_FIELDS}, p.options());
    packed_gradient_geometry<SharedProfile><<<(p.size(0) + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        p.data_ptr<float>(), geometry.data_ptr<float>(), p.size(0), ai, ao, radius_limit);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return geometry;
}

Tensor packed_make_geometry(Tensor p, Config ci, Config co) {
    return packed_make_geometry(p, parse_axis(ci), parse_axis(co));
}

__global__ void packed_gradient_groups(const float *x, const float *dy,
    const float *zx, const float *zy, const float *h, const float *g,
    const float *geometry, float *partial, int atoms, int batch, Axis ai, Axis ao) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    int first = blockIdx.y * 8;
    const float *v = geometry + int64_t(atom) * PACKED_GEOMETRY_FIELDS;
    Membership mi{int(v[20]), int(v[21])}, mo{int(v[22]), int(v[23])};
    float dci[8] = {}, dco[8] = {};
    if (ai.banks[mi.bank].level < 2)
        direct_derivatives_batch4<float, false, float>(x, first, batch, ai, mi, v[18], v,
            nullptr, nullptr, dci);
    else compact_derivatives_batch4<false, float>(zx, first, batch, ai, mi, v[18], v,
        nullptr, 2, nullptr, dci);
    if (ao.banks[mo.bank].level < 2)
        direct_derivatives_batch4<float, false, float>(dy, first, batch, ao, mo, v[19], v + 9,
            nullptr, nullptr, dco);
    else compact_derivatives_batch4<false, float>(zy, first, batch, ao, mo, v[19], v + 9,
        nullptr, 2, nullptr, dco);
    float da = 0, dmi = 0, dmo = 0;
    #pragma unroll
    for (int r = 0; r < 8; ++r) if (first + r < batch) {
        int64_t index = int64_t(first + r) * atoms + atom;
        float H = h[index], G = g[index];
        da += H * G; dmi += v[24] * G * dci[r]; dmo += v[24] * H * dco[r];
    }
    float *out = partial + (int64_t(blockIdx.y) * atoms + atom) * 3;
    out[0] = da; out[1] = dmi; out[2] = dmo;
}

__global__ void packed_gradient_reduce(const float *p, const float *partial,
    float *gradient, int atoms, int groups, Axis ai) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    float da = 0, dmi = 0, dmo = 0;
    for (int group = 0; group < groups; ++group) {
        const float *v = partial + (int64_t(group) * atoms + atom) * 3;
        da += v[0]; dmi += v[1]; dmo += v[2];
    }
    signal_store_gradient<float>(p, gradient, atom, ai, da, 0, dmi, dmo);
}

void packed_check_tensor(Tensor t, Tensor x, std::vector<int64_t> shape) {
    TORCH_CHECK(t.defined() && t.device() == x.device() && t.scalar_type() == torch::kFloat &&
        t.is_contiguous() && t.sizes().vec() == shape, "invalid packed saved tensor");
}

Tensor packed_parameter_grad(Tensor p, Tensor x, Tensor dy, Tensor zx, Tensor zy,
    Tensor h, Tensor g, Config ci, Config co, bool parallel, Tensor geometry, Tensor gradient_order) {
    c10::cuda::CUDAGuard guard(p.device());
    Axis ai = parse_axis(ci), ao = parse_axis(co);
    packed_check_tensor(zx, x, {x.size(0), ai.node_count - checked_start(ai, 2), ai.basis_count()});
    packed_check_tensor(zy, x, {x.size(0), ao.node_count - checked_start(ao, 2), ao.basis_count()});
    packed_check_tensor(h, x, {x.size(0), p.size(0)});
    packed_check_tensor(g, x, {x.size(0), p.size(0)});
    auto grad = torch::empty_like(p);
#ifdef RADIUS_RATIONAL_WIDE
    if (gradient_order.defined()) {
        TORCH_CHECK(!ai.polar && !ao.polar && gradient_order.device() == p.device() &&
            gradient_order.scalar_type() == at::kLong && gradient_order.is_contiguous() &&
            gradient_order.dim() == 1 && gradient_order.size(0) == p.size(0),
            "invalid physical gradient order");
        auto partial = torch::empty({2, p.size(0)}, p.options());
        constexpr int threads = 128;
        int blocks = (p.size(0) + threads - 1) / threads;
        auto stream = at::cuda::getCurrentCUDAStream();
        axis_parameter_gradient<true><<<blocks, threads, 0, stream>>>(
            p.data_ptr<float>(), dy.data_ptr<float>(), zy.data_ptr<float>(), h.data_ptr<float>(),
            g.data_ptr<float>(), gradient_order.data_ptr<int64_t>(), partial.data_ptr<float>(),
            nullptr, p.size(0), x.size(0), ao);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        axis_parameter_gradient<false><<<blocks, threads, 0, stream>>>(
            p.data_ptr<float>(), x.data_ptr<float>(), zx.data_ptr<float>(), h.data_ptr<float>(),
            g.data_ptr<float>(), nullptr, partial.data_ptr<float>(), grad.data_ptr<float>(),
            p.size(0), x.size(0), ai);
    } else if (!ai.polar) {
        // Reuse the complete physical-radius derivative. The packed Polar
        // reduction intentionally has no radius channel and must not be used.
        batched_parameter_kernel<float, true, float><<<(p.size(0) + 31) / 32, 32, 0,
            at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(), x.data_ptr<float>(),
            dy.data_ptr<float>(), zx.data_ptr<float>(), zy.data_ptr<float>(),
            h.data_ptr<float>(), g.data_ptr<float>(), grad.data_ptr<float>(),
            p.size(0), x.size(0), ai, ao, 2, 2);
    } else
#endif
    if (parallel) {
        if (!geometry.defined()) geometry = packed_make_geometry(p, ci, co);
        packed_check_tensor(geometry, x, {p.size(0), PACKED_GEOMETRY_FIELDS});
        int groups = (x.size(0) + 7) / 8;
        auto partial = torch::empty({groups, p.size(0), 3}, geometry.options());
        dim3 grid((p.size(0) + 31) / 32, groups);
        packed_gradient_groups<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(), dy.data_ptr<float>(), zx.data_ptr<float>(), zy.data_ptr<float>(),
            h.data_ptr<float>(), g.data_ptr<float>(), geometry.data_ptr<float>(), partial.data_ptr<float>(),
            p.size(0), x.size(0), ai, ao);
        packed_gradient_reduce<<<(p.size(0) + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            p.data_ptr<float>(), partial.data_ptr<float>(), grad.data_ptr<float>(), p.size(0), groups, ai);
    } else {
        batched_parameter_kernel<float, false, float><<<(p.size(0) + 31) / 32, 32, 0,
            at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(), x.data_ptr<float>(), dy.data_ptr<float>(),
            zx.data_ptr<float>(), zy.data_ptr<float>(), h.data_ptr<float>(), g.data_ptr<float>(),
            grad.data_ptr<float>(), p.size(0), x.size(0), ai, ao, 2, 2);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}

void packed_gradient_partial_into(Tensor p, Tensor x, Tensor dy, Tensor zx, Tensor zy,
    Tensor h, Tensor g, Tensor geometry, Tensor partial, int64_t first_group,
    Config ci, Config co) {
    c10::cuda::CUDAGuard guard(p.device());
    Axis ai = parse_axis(ci), ao = parse_axis(co);
    int groups = (x.size(0) + 7) / 8;
    packed_check_tensor(zx, x, {x.size(0), ai.node_count - checked_start(ai, 2), ai.basis_count()});
    packed_check_tensor(zy, x, {x.size(0), ao.node_count - checked_start(ao, 2), ao.basis_count()});
    packed_check_tensor(h, x, {x.size(0), p.size(0)});
    packed_check_tensor(g, x, {x.size(0), p.size(0)});
    packed_check_tensor(geometry, x, {p.size(0), PACKED_GEOMETRY_FIELDS});
    TORCH_CHECK(partial.defined() && partial.dim() == 3 && partial.size(0) >= first_group + groups && first_group >= 0,
        "invalid streamed partial range");
    packed_check_tensor(partial, x, {partial.size(0), p.size(0), 3});
    dim3 grid((p.size(0) + 31) / 32, groups);
    packed_gradient_groups<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), dy.data_ptr<float>(), zx.data_ptr<float>(), zy.data_ptr<float>(),
        h.data_ptr<float>(), g.data_ptr<float>(), geometry.data_ptr<float>(),
        partial.data_ptr<float>() + first_group * p.size(0) * 3, p.size(0), x.size(0), ai, ao);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
Tensor packed_finish_gradient(Tensor p, Tensor partial, Config ci) {
    c10::cuda::CUDAGuard guard(p.device());
    TORCH_CHECK(partial.defined() && partial.dim() == 3 && partial.size(0) >= 1 && partial.size(0) <= 32,
        "invalid streamed partial groups");
    packed_check_tensor(partial, p, {partial.size(0), p.size(0), 3});
    auto gradient = torch::empty_like(p);
    packed_gradient_reduce<<<(p.size(0) + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        p.data_ptr<float>(), partial.data_ptr<float>(), gradient.data_ptr<float>(), p.size(0), partial.size(0), parse_axis(ci));
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return gradient;
}

#ifdef RADIUS_RATIONAL_WIDE
constexpr int CHANNEL_RADIUS_TILE_LIMIT = 32;
constexpr int CHANNEL_LAST_FOURIER_LEVEL = 5;
constexpr int CHANNEL_DIRECT_LEVEL = CHANNEL_LAST_FOURIER_LEVEL + 1;

void check_channel_atoms(const Tensor &p) {
    TORCH_CHECK(p.is_cuda() && p.is_contiguous() && p.scalar_type() == torch::kFloat &&
        p.dim() == 3 && (p.size(0) == 1 || p.size(0) == 9) &&
        p.size(1) > 0 && p.size(1) <= SHARED_MAX_ATOMS && p.size(2) == 4,
        "expected contiguous CUDA FP32 [1 or 9, 1..4096, 4] channel Atoms");
    TORCH_CHECK(!at::globalContext().allowTF32CuBLAS(),
        "channel weights require IEEE FP32 matmul");
}

void check_legacy_channel_atoms(const Tensor &p, int input_sites, int first_level) {
    check_channel_atoms(p);
    TORCH_CHECK((input_sites == 96 || input_sites == 128) &&
        3 <= first_level && first_level <= CHANNEL_DIRECT_LEVEL,
        "expected 96 or 128 input sites and first Fourier level 3..6");
}

Axis channel_axis(int sites, int radius_cap = CHANNEL_RADIUS_TILE_LIMIT) {
    return parse_axis({double(sites), 1., 1., 0., 8., 2.}, radius_cap);
}

std::vector<Tensor> packed_channel_weight_forward(Tensor p, int input_sites, int first_level) {
    check_legacy_channel_atoms(p, input_sites, first_level);
    c10::cuda::CUDAGuard guard(p.device());
    Axis ai = channel_axis(input_sites), ao = channel_axis(128);
    auto geometry = packed_make_geometry(p.view({-1, 4}), ai, ao, CHANNEL_RADIUS_TILE_LIMIT);
    if (first_level == CHANNEL_DIRECT_LEVEL)
        return {packed_make_direct_channel_weights(p, geometry, ai, ao), geometry};
    std::vector<Tensor> weights;
    for (int64_t tap = 0; tap < p.size(0); ++tap) {
        auto tap_geometry = geometry.narrow(0, tap * p.size(1), p.size(1));
        weights.push_back(packed_make_dense_weight(
            p[tap], tap_geometry, ai, ao, first_level, CHANNEL_LAST_FOURIER_LEVEL));
    }
    return {torch::stack(weights), geometry};
}

Tensor packed_channel_weight_grad(Tensor p, Tensor dense, Tensor geometry,
    int input_sites, int first_level) {
    check_legacy_channel_atoms(p, input_sites, first_level);
    c10::cuda::CUDAGuard guard(p.device());
    packed_check_tensor(dense, p, {p.size(0), 128, input_sites});
    packed_check_tensor(geometry, p, {p.size(0) * p.size(1), PACKED_GEOMETRY_FIELDS});
    Axis ai = channel_axis(input_sites), ao = channel_axis(128);
    if (first_level == CHANNEL_DIRECT_LEVEL)
        return packed_direct_channel_weight_grad(p, dense, geometry, ai, ao);
    std::vector<Tensor> grads;
    for (int64_t tap = 0; tap < p.size(0); ++tap) {
        auto tap_geometry = geometry.narrow(0, tap * p.size(1), p.size(1));
        grads.push_back(packed_dense_weight_grad(
            p[tap], dense[tap], tap_geometry, ai, ao, first_level, CHANNEL_LAST_FOURIER_LEVEL));
    }
    return torch::stack(grads);
}
#endif

} // namespace radius_cell_mass

#include "channel_fourier.cuh"
