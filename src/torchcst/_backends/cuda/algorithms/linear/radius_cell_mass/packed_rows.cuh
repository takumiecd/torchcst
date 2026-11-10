#pragma once
// FP32 Linear arithmetic and Z/H/G, with optional recomputation.
// Host configuration scalars retain the common metadata representation.
namespace radius_cell_mass {
__global__ void packed_classify_kernel(const float *p,int64_t *keys,int atoms,Axis ai,Axis ao) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=atoms)return;
    float ri=signal_radius<float>(p,a,ai),ro=signal_radius<float>(p,a,ao);
    bool valid=isfinite(ri)&&isfinite(ro)&&ri>0&&ro>0;
    for(int c=0;c<4;++c)valid &= isfinite(p[4*a+c]);
    valid &= isfinite(p[4*a]*p[4*a]+p[4*a+1]*p[4*a+1]);
    CUDA_KERNEL_ASSERT_MSG(valid,"invalid FP32 packed geometry");
    if(!valid){keys[a]=0;return;}
    auto mi=signal_classify<float>(ri,p[4*a+3],ai),mo=signal_classify<float>(ro,p[4*a+2],ao);
    Bank bi=ai.banks[mi.bank],bo=ao.banks[mo.bank];
    int ni=bi.node_offset+positive_mod(mi.first_tile,bi.tile_count),no=bo.node_offset+positive_mod(mo.first_tile,bo.tile_count);
    keys[a]=int64_t(no)*ai.node_count+ni;
}
Tensor packed_classify(Tensor p,Config ci,Config co) {
    c10::cuda::CUDAGuard guard(p.device());
    auto keys=torch::empty({p.size(0)},p.options().dtype(torch::kLong));
    packed_classify_kernel<<<(p.size(0)+THREADS-1)/THREADS,THREADS,0,at::cuda::getCurrentCUDAStream()>>>(
        p.data_ptr<float>(),keys.data_ptr<int64_t>(),p.size(0),parse_axis(ci),parse_axis(co));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return keys;
}
void packed_check_scope(Tensor x, Tensor p, Config ci, Config co) {
    check_atoms(p); check_matrix(x);
    Axis ai = parse_axis(ci), ao = parse_axis(co);
    TORCH_CHECK(EXPANSION_BATCH == 8 && PARAMETER_BATCH == 8,
        "packed rows require the distinct row8 build");
    bool ffn = p.size(0) == 4096 &&
        ((ai.sites == 256 && ao.sites == 1024) || (ai.sites == 1024 && ao.sites == 256));
    bool lm = ai.origin == 0 && ao.origin == 0 && (
        (ai.sites == ao.sites && (ai.sites == 256 || ai.sites == 512) &&
         p.size(0) == 8 * ai.sites) ||
        (((ai.sites == 512 && ao.sites == 2048) ||
          (ai.sites == 2048 && ao.sites == 512)) && p.size(0) == 8192));
    bool large = ai.sites == ao.sites &&
        (ai.sites == 2048 || ai.sites == 8192 || ai.sites == 16384) &&
        p.size(0) >= 4 && p.size(0) <= 8 * ai.sites && ai.origin == 0 && ao.origin == 0;
#ifdef RADIUS_LM_C16
    // Separate physical c16 build; ordinary builds retain their old guards.
    ffn = false; large = false;
    lm = !ai.polar && !ao.polar && ai.origin == 0 && ao.origin == 0 &&
        ((ai.sites == 512 && ao.sites == 2048) ||
         (ai.sites == 2048 && ao.sites == 512)) && p.size(0) == 16384;
#endif
#ifdef RADIUS_LM_BUDGET
    // Separate M8 capacity build; the ordinary and c16 guards stay unchanged.
    ffn = false; large = false;
    lm = !ai.polar && !ao.polar && ai.origin == 0 && ao.origin == 0 &&
        ((ai.sites == 512 && ao.sites == 2048) ||
         (ai.sites == 2048 && ao.sites == 512)) &&
        (p.size(0) == 16384 || p.size(0) == 24576 || p.size(0) == 32768 ||
         p.size(0) == 49152 || p.size(0) == 65536 ||
         p.size(0) == 98304 || p.size(0) == 131072);
#endif
#ifdef RADIUS_CONNECTED_CONTROL
    // Separate research build only. Unchanged arithmetic is admitted to the
    // matched physical square controls; ordinary binaries retain old guards.
    bool physical_control = !ai.polar && !ao.polar && ai.origin == 0 && ao.origin == 0 &&
        ai.sites == ao.sites && ai.sites >= 16 && ai.sites <= 16384 &&
        p.size(0) >= 1 && p.size(0) <= 8 * ai.sites;
    large = large || physical_control;
#endif
    TORCH_CHECK(x.device() == p.device() && x.size(1) == ai.sites &&
        x.scalar_type() == torch::kFloat && p.scalar_type() == torch::kFloat &&
        x.size(0) >= 1 && x.size(0) <= 256 && (ffn || lm || large),
        "packed pilot requires admitted FP32 FFN, LM or square large-multirow cohort");
    bool physical_lm = lm && !ai.polar && !ao.polar;
#ifdef RADIUS_CONNECTED_CONTROL
    physical_lm = physical_lm || physical_control;
#endif
    for (auto a : {ai, ao}) TORCH_CHECK((a.polar || physical_lm) && a.modes == 8 &&
        a.base_tile_sites == 1 && a.merge_factor == 2 && a.spacing == 1 &&
        (physical_lm || (a.amplitude_max <= 1 && a.radius_min >= .5 && a.radius_max <= 50)),
        "packed pilot requires admitted Polar or physical LM M8 grid");
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

std::vector<Tensor> packed_project(Tensor x, Tensor p, Config config, bool output_side, Tensor geometry, bool level_launch) {
    c10::cuda::CUDAGuard guard(x.device());
    auto z = packed_moments(x, p, config, level_launch);
    auto h = torch::empty({x.size(0), p.size(0)}, x.options());
    Axis a = parse_axis(config);
    dim3 grid((p.size(0) + PARALLEL_RESPONSE_THREADS - 1) / PARALLEL_RESPONSE_THREADS,
        (x.size(0) + 7) / 8);
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
    Config source, Config destination, bool output_side, Tensor geometry, bool level_launch) {
    c10::cuda::CUDAGuard guard(x.device());
    auto projection = packed_project(x, p, source, output_side, geometry, level_launch);
    Axis a = parse_axis(destination); check_ranges(rows, offsets, p, a);
    auto y = torch::empty({x.size(0), a.sites}, x.options());
    auto layout = batched_coefficient_layout<float>(a, 2, 32);
    dim3 grid((a.sites + 31) / 32, (x.size(0) + 7) / 8);
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
__global__ void packed_gradient_geometry(const float *p, float *geometry, int atoms,
    Axis ai, Axis ao) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    float *v = geometry + int64_t(atom) * PACKED_GEOMETRY_FIELDS;
    float ri = signal_radius<float>(p, atom, ai), ro = signal_radius<float>(p, atom, ao);
    Membership mi = signal_classify<float>(ri, float(center(p, atom, ai, false)), ai);
    Membership mo = signal_classify<float>(ro, float(center(p, atom, ao, true)), ao);
    profile_values(ri, float(ai.banks[mi.bank].period), 8, v);
    profile_values(ro, float(ao.banks[mo.bank].period), 8, v + 9);
    v[18] = signal_wrapped<float>(float(center(p, atom, ai, false)), ai);
    v[19] = signal_wrapped<float>(float(center(p, atom, ao, true)), ao);
    v[20] = mi.bank; v[21] = mi.first_tile;
    v[22] = mo.bank; v[23] = mo.first_tile;
    v[24] = signal_amplitude<float>(p, atom, ai);
}

Tensor packed_make_geometry(Tensor p, Config ci, Config co) {
    c10::cuda::CUDAGuard guard(p.device());
    auto geometry = torch::empty({p.size(0), PACKED_GEOMETRY_FIELDS}, p.options());
    packed_gradient_geometry<<<(p.size(0) + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        p.data_ptr<float>(), geometry.data_ptr<float>(), p.size(0), parse_axis(ci), parse_axis(co));
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return geometry;
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
    Tensor h, Tensor g, Config ci, Config co, bool parallel, Tensor geometry) {
    c10::cuda::CUDAGuard guard(p.device());
    Axis ai = parse_axis(ci), ao = parse_axis(co);
    packed_check_tensor(zx, x, {x.size(0), ai.node_count - checked_start(ai, 2), ai.basis_count()});
    packed_check_tensor(zy, x, {x.size(0), ao.node_count - checked_start(ao, 2), ao.basis_count()});
    packed_check_tensor(h, x, {x.size(0), p.size(0)});
    packed_check_tensor(g, x, {x.size(0), p.size(0)});
    auto grad = torch::empty_like(p);
    if (!ai.polar) {
        // Reuse the complete physical-radius derivative. The packed Polar
        // reduction intentionally has no radius channel and must not be used.
        batched_parameter_kernel<float, true, float><<<(p.size(0) + 31) / 32, 32, 0,
            at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(), x.data_ptr<float>(),
            dy.data_ptr<float>(), zx.data_ptr<float>(), zy.data_ptr<float>(),
            h.data_ptr<float>(), g.data_ptr<float>(), grad.data_ptr<float>(),
            p.size(0), x.size(0), ai, ao, 2, 2);
    } else if (parallel) {
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
} // namespace radius_cell_mass
