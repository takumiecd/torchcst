// A correctness-first radius pipeline. Each launch is one dependency boundary;
// no block waits for another block and no output needs atomic accumulation.
#include "common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>

namespace radius_cell_mass {
using Tensor = torch::Tensor;
using Config = std::vector<double>;

inline void check_matrix(const Tensor &x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2,
                "expected a contiguous CUDA matrix");
    TORCH_CHECK(x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kDouble,
                "expected float32 or float64");
}
inline void check_atoms(const Tensor &p) {
    check_matrix(p);
    TORCH_CHECK(p.size(1) == 4 && p.size(0) <= 4096, "expected at most 4096 four-column atoms");
}
inline void check_moments(const Tensor &z, const Tensor &p, const Axis &axis) {
    TORCH_CHECK(z.device() == p.device() && z.scalar_type() == torch::kDouble &&
        z.is_contiguous() && z.dim() == 3 && z.size(1) == axis.node_count &&
        z.size(2) == axis.basis_count(), "invalid moments/coefficients shape or device");
    TORCH_CHECK(z.numel() <= INT_MAX - THREADS, "moment work exceeds native index limit");
}
__device__ inline int bank_for_node(int node, const Axis &axis) {
    int bank = 0;
    while (bank + 1 < axis.bank_count && node >= axis.banks[bank + 1].node_offset) ++bank;
    return bank;
}

// Temporary sortable keys refer to original Parameter rows; no packed atoms.
template<class T, bool Async = false>
__global__ void classify_kernel(const T *p, int64_t *keys, int *error, int atoms,
                               Axis input, Axis output) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    bool valid = true;
    for (int c = 0; c < 4; ++c) valid &= isfinite(double(p[4 * atom + c]));
    double ri = radius(p, atom, input), ro = radius(p, atom, output);
    valid &= isfinite(ri) && isfinite(ro) && ri > 0 && ro > 0 &&
             ri <= input.period / 2 && ro <= output.period / 2;
    // A positive mathematical ratio must have ceil >= 1. Reject scales that
    // underflow to zero, rather than indexing a nonexistent K=1 bank.
    valid &= (2 * ri / (input.base_tile_sites * input.spacing)) > 0 &&
             (2 * ro / (output.base_tile_sites * output.spacing)) > 0;
    if (!valid) {
        // Define the key before rejecting it: invalid data never reaches bank
        // indexing. The normal route reports a device-side assertion through
        // CUDA's asynchronous error handling instead of reading a flag on CPU.
        keys[atom] = 0;
        if constexpr (Async) {
            CUDA_KERNEL_ASSERT_MSG(valid, "invalid radius atom: finite coordinates and representable supported radii required");
        } else {
            atomicExch(error, 1);
        }
        return;
    }
    Membership mi = classify(ri, center(p, atom, input, false), input);
    Membership mo = classify(ro, center(p, atom, output, true), output);
    Bank bi = input.banks[mi.bank], bo = output.banks[mo.bank];
    int ni = bi.node_offset + positive_mod(mi.first_tile, bi.tile_count);
    int no = bo.node_offset + positive_mod(mo.first_tile, bo.tile_count);
    keys[atom] = int64_t(no) * input.node_count + ni;
}

// Each thread owns one (batch, tile, basis) sum. A tile's basis integrates
// whole cells: d*sinc(k*d/P)*cos/sin(2*pi*k*local_site*d/P).
// A parent can inherit DC and harmonics divisible by merge_factor exactly.
// Its other frequencies are evaluated from X: finite child modes do not
// contain the missing lower frequencies, so interpolation would change math.
template<class T>
__global__ void moments_kernel(const T *signal, double *z, int batch, Axis axis,
                               int level, bool merge) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int features = axis.basis_count();
    if (index >= batch * axis.node_count * features) return;
    int basis = index % features;
    int node = (index / features) % axis.node_count;
    int b = index / (features * axis.node_count);
    int bank_index = bank_for_node(node, axis);
    Bank bank = axis.banks[bank_index];
    if (level >= 0 && bank.level != level) return;
    int tile = node - bank.node_offset;
    int harmonic = (basis + 1) / 2;
    bool sine = basis != 0 && basis % 2 == 0;
    double sum = 0;
    if (merge && bank.level > 0 && harmonic % axis.merge_factor == 0) {
        Bank child = axis.banks[bank_index - 2];
        int child_harmonic = harmonic / axis.merge_factor;
        for (int part = 0; part < axis.merge_factor; ++part) {
            int child_node = child.node_offset + tile * axis.merge_factor + part;
            int64_t offset = (int64_t(b) * axis.node_count + child_node) * features;
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
    z[index] = sum;
}

// One thread owns a complete response for one atom and batch. It sums ALL K
// input tiles in registers before publishing H/G. dr/dmu are optional outputs
// of the same contraction used later by parameter_grad.
__device__ inline double project(const double *z, int b, const Axis &axis,
    Membership member, double mu, const double *c, const double *dr,
    double &radius_gradient, double &center_gradient) {
    Bank bank = axis.banks[member.bank];
    double value = 0;
    radius_gradient = center_gradient = 0;
    for (int part = 0; part < bank.count; ++part) {
        int unwrapped_tile = member.first_tile + part;
        int node = bank.node_offset + positive_mod(unwrapped_tile, bank.tile_count);
        const double *tile = z + (int64_t(b) * axis.node_count + node) * axis.basis_count();
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
template<class T>
__global__ void response_kernel(const T *p, const double *z, double *h,
    int atoms, int batch, Axis axis, bool output_side) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= atoms * batch) return;
    int atom = index % atoms, b = index / atoms;
    double R = radius(p, atom, axis), mu = wrapped_center(center(p, atom, axis, output_side), axis);
    Membership member = classify(R, center(p, atom, axis, output_side), axis);
    double c[17], dr[17], dR, dmu;
    profile_coefficients(R, axis.banks[member.bank].period, axis.modes, c, dr);
    h[index] = project(z, b, axis, member, mu, c, dr, dR, dmu);
}

// A thread owns one complete (batch, output tile) coefficient vector. It reads
// at most K sorted first-tile ranges. Every matching atom contributes its final
// response; C is never constructed from a partial H. No global atomics.
template<class T>
__global__ void coefficients_kernel(const T *p, const int64_t *rows,
    const int64_t *offsets, const double *response, double *coefficients,
    int atoms, int batch, Axis axis, bool output_side) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * axis.node_count) return;
    int node = index % axis.node_count, b = index / axis.node_count;
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
    for (int f = 0; f < axis.basis_count(); ++f)
        coefficients[int64_t(index) * axis.basis_count() + f] = accumulator[f];
}

// Each thread owns one final site. All levels/period families are accumulated
// in its register, then written once to Y/dX in the requested storage dtype.
template<class T>
__global__ void expand_kernel(const double *coefficients, T *result, int batch, Axis axis) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * axis.sites) return;
    int site = index % axis.sites, b = index / axis.sites;
    double value = 0;
    for (int bi = 0; bi < axis.bank_count; ++bi) {
        Bank bank = axis.banks[bi];
        int tile = site / bank.tile_sites, local_site = site % bank.tile_sites;
        int node = bank.node_offset + tile;
        const double *c = coefficients + (int64_t(b) * axis.node_count + node) * axis.basis_count();
        value += axis.spacing * c[0];
        for (int k = 1; k <= axis.modes; ++k) {
            double argument = PI * k * axis.spacing / bank.period;
            double phase = 2 * argument * local_site;
            value += axis.spacing * sin(argument) / argument *
                     (c[2 * k - 1] * cos(phase) + c[2 * k] * sin(phase));
        }
    }
    result[index] = T(value);
}

// One atom/thread accumulates the whole batch. H/G and the derivative moments
// are reused; radius differentiates both axes for physical shared-radius rows.
template<class T>
__global__ void parameter_kernel(const T *p, const double *zx, const double *zy,
    const double *h, const double *g, T *gradient, int atoms, int batch, Axis input, Axis output) {
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
        project(zx, b, input, mi, mui, ci, di, dri, dci);
        project(zy, b, output, mo, muo, co, dout, dro, dco);
        double H = h[b * atoms + atom], G = g[b * atoms + atom];
        da += H * G;
        dR += amp * (G * dri + H * dro);
        dmi += amp * G * dci;
        dmo += amp * H * dco;
    }
    store_gradient(p, gradient, atom, input, da, dR, dmi, dmo);
}

std::vector<Tensor> classify_atoms(Tensor p, Config input_config, Config output_config) {
    check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    c10::cuda::CUDAGuard guard(p.device());
    auto keys = torch::empty({p.size(0)}, p.options().dtype(torch::kLong));
    auto error = torch::zeros({}, p.options().dtype(torch::kInt));
    if (p.size(0)) AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_classify", [&] {
        classify_kernel<<<(p.size(0) + THREADS - 1) / THREADS, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            p.data_ptr<scalar_t>(), keys.data_ptr<int64_t>(), error.data_ptr<int>(), p.size(0), input, output);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {keys, error};
}
Tensor classify_atoms_async(Tensor p, Config input_config, Config output_config) {
    check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    c10::cuda::CUDAGuard guard(p.device());
    auto keys = torch::empty({p.size(0)}, p.options().dtype(torch::kLong));
    if (p.size(0)) AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_classify_async", [&] {
        classify_kernel<scalar_t, true><<<(p.size(0) + THREADS - 1) / THREADS,
            THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            p.data_ptr<scalar_t>(), keys.data_ptr<int64_t>(), nullptr,
            p.size(0), input, output);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return keys;
}
Tensor build_moments(Tensor signal, Config config, bool merge) {
    check_matrix(signal);
    Axis axis = parse_axis(config);
    TORCH_CHECK(signal.size(1) == axis.sites, "signal must match axis site count");
    c10::cuda::CUDAGuard guard(signal.device());
    TORCH_CHECK(signal.size(0) <= (INT_MAX - THREADS) / (axis.node_count * axis.basis_count()),
                "moment work exceeds native index limit");
    auto z = torch::empty({signal.size(0), axis.node_count, axis.basis_count()}, signal.options().dtype(torch::kDouble));
    if (!z.numel()) return z;
    int blocks = (z.numel() + THREADS - 1) / THREADS;
    AT_DISPATCH_FLOATING_TYPES(signal.scalar_type(), "radius_moments", [&] {
        int levels = merge ? axis.bank_count / 2 : 1;
        for (int level = 0; level < levels; ++level)
            moments_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                signal.data_ptr<scalar_t>(), z.data_ptr<double>(), signal.size(0), axis, merge ? level : -1, merge);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return z;
}
Tensor compute_response(Tensor p, Tensor z, Config config, bool output_side) {
    check_atoms(p);
    Axis axis = parse_axis(config);
    check_moments(z, p, axis);
    TORCH_CHECK(p.size(0) == 0 || z.size(0) <= (INT_MAX - THREADS) / p.size(0),
                "response work exceeds native index limit");
    c10::cuda::CUDAGuard guard(p.device());
    auto h = torch::empty({z.size(0), p.size(0)}, z.options());
    if (!h.numel()) return h;
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_response", [&] {
        response_kernel<<<(h.numel() + THREADS - 1) / THREADS, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            p.data_ptr<scalar_t>(), z.data_ptr<double>(), h.data_ptr<double>(), p.size(0), z.size(0), axis, output_side);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return h;
}
Tensor compute_coefficients(Tensor p, Tensor rows, Tensor offsets, Tensor h, Config config, bool output_side) {
    check_atoms(p);
    Axis axis = parse_axis(config);
    TORCH_CHECK(rows.device() == p.device() && offsets.device() == p.device() &&
        rows.scalar_type() == torch::kLong && offsets.scalar_type() == torch::kLong &&
        rows.is_contiguous() && offsets.is_contiguous() && rows.dim() == 1 && offsets.dim() == 1 &&
        rows.numel() == p.size(0) && offsets.numel() == axis.node_count + 1, "invalid sorted ranges");
    TORCH_CHECK(h.device() == p.device() && h.scalar_type() == torch::kDouble && h.dim() == 2 &&
        h.size(1) == p.size(0) && h.is_contiguous(), "invalid response");
    TORCH_CHECK(h.size(0) <= (INT_MAX - THREADS) / (axis.node_count * axis.basis_count()),
                "coefficient work exceeds native index limit");
    c10::cuda::CUDAGuard guard(p.device());
    auto c = torch::empty({h.size(0), axis.node_count, axis.basis_count()}, h.options());
    if (!c.numel()) return c;
    int work = h.size(0) * axis.node_count;
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_coefficients", [&] {
        coefficients_kernel<<<(work + THREADS - 1) / THREADS, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            p.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), h.data_ptr<double>(),
            c.data_ptr<double>(), p.size(0), h.size(0), axis, output_side);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
Tensor expand_coefficients(Tensor c, Tensor like, Config config) {
    check_matrix(like);
    Axis axis = parse_axis(config);
    check_moments(c, like, axis);
    c10::cuda::CUDAGuard guard(like.device());
    auto y = torch::empty({c.size(0), axis.sites}, like.options());
    if (!y.numel()) return y;
    AT_DISPATCH_FLOATING_TYPES(like.scalar_type(), "radius_expand", [&] {
        expand_kernel<<<(y.numel() + THREADS - 1) / THREADS, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            c.data_ptr<double>(), y.data_ptr<scalar_t>(), c.size(0), axis);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
Tensor parameter_grad(Tensor p, Tensor zx, Tensor zy, Tensor h, Tensor g, Config input_config, Config output_config) {
    check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    check_moments(zx, p, input); check_moments(zy, p, output);
    TORCH_CHECK(zx.size(0) == zy.size(0), "moment batches differ");
    for (const Tensor &v : {h, g}) TORCH_CHECK(v.device() == p.device() &&
        v.scalar_type() == torch::kDouble && v.is_contiguous() && v.dim() == 2 &&
        v.size(0) == zx.size(0) && v.size(1) == p.size(0), "invalid parameter-gradient response");
    c10::cuda::CUDAGuard guard(p.device());
    auto grad = torch::empty_like(p);
    if (!p.numel()) return grad;
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_parameter_grad", [&] {
        parameter_kernel<<<(p.size(0) + THREADS - 1) / THREADS, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            p.data_ptr<scalar_t>(), zx.data_ptr<double>(), zy.data_ptr<double>(), h.data_ptr<double>(),
            g.data_ptr<double>(), grad.data_ptr<scalar_t>(), p.size(0), zx.size(0), input, output);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}
} // namespace radius_cell_mass

#include "hybrid.cuh"
#include "block_local.cuh"
#include "shared_projection.cuh"
#include "layout.cuh"
#include "expansion_probe.cuh"

namespace radius_cell_mass {
void bind_host_autograd(pybind11::module_ &);

// Host autograd links to these wrappers, reusing the exact existing CUDA
// implementations instead of crossing Python/pybind for every stage.
std::vector<Tensor> autograd_ranges(Tensor keys, Config input, Config output, bool reverse) {
    return layout_ranges(keys, parse_axis(input).node_count,
        parse_axis(output).node_count, reverse);
}
std::vector<Tensor> autograd_transform(Tensor x, Tensor p, Tensor rows, Tensor offsets,
    Config input, Config output, int input_start, int output_start,
    bool merge, int tile_sites, int schedule, bool reverse) {
    return shared_apply(x, p, rows, offsets, reverse ? output : input,
        reverse ? input : output, reverse ? output_start : input_start,
        reverse ? input_start : output_start, merge, tile_sites, reverse,
        true, schedule >= 1, schedule == 2);
}
std::vector<Tensor> autograd_project(Tensor x, Tensor p, Config output,
    int start, bool merge, int schedule) {
    return shared_apply(x, p, Tensor(), Tensor(), output, output, start, start,
        merge, 1, true, false, schedule >= 1);
}
Tensor autograd_parameter_grad(Tensor p, Tensor x, Tensor dy, Tensor zx, Tensor zy,
    Tensor h, Tensor g, Config input, Config output, int input_start,
    int output_start, int schedule) {
    return hybrid_parameter_grad_impl(p, x, dy, zx, zy, h, g, input, output,
        input_start, output_start, schedule >= 1 ? 32 : THREADS);
}
} // namespace radius_cell_mass

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    radius_cell_mass::bind_host_autograd(m);
    m.def("shared_forward_cached_parallel", &radius_cell_mass::shared_forward_cached_parallel);
    m.def("shared_backward_cached_parallel", &radius_cell_mass::shared_backward_cached_parallel);
    m.def("counted_cached_expansion", &radius_cell_mass::counted_cached_expansion);
    m.def("run_expansion_probe", &radius_cell_mass::run_expansion_probe);
    m.def("expansion_probe_resources", &radius_cell_mass::expansion_probe_resources);
    m.def("layout_ranges", &radius_cell_mass::layout_ranges);
    m.def("layout_ranges_aten", &radius_cell_mass::layout_ranges_aten);
    m.def("classify_atoms", &radius_cell_mass::classify_atoms);
    m.def("classify_atoms_async", &radius_cell_mass::classify_atoms_async);
    m.def("build_moments", &radius_cell_mass::build_moments);
    m.def("compute_response", &radius_cell_mass::compute_response);
    m.def("compute_coefficients", &radius_cell_mass::compute_coefficients);
    m.def("expand_coefficients", &radius_cell_mass::expand_coefficients);
    m.def("parameter_grad", &radius_cell_mass::parameter_grad);
    m.def("build_compact_moments", &radius_cell_mass::build_compact_moments);
    m.def("direct_response", &radius_cell_mass::direct_response);
    m.def("direct_response_into", &radius_cell_mass::direct_response_into);
    m.def("complete_response", &radius_cell_mass::complete_response);
    m.def("compact_coefficients", &radius_cell_mass::compact_coefficients);
    m.def("expand_hybrid", &radius_cell_mass::expand_hybrid);
    m.def("hybrid_parameter_grad", &radius_cell_mass::hybrid_parameter_grad);
    m.def("block_local_forward", &radius_cell_mass::block_local_forward);
    m.def("block_local_backward", &radius_cell_mass::block_local_backward);
    m.def("block_local_project", &radius_cell_mass::block_local_project);
    m.def("shared_forward", &radius_cell_mass::shared_forward);
    m.def("shared_backward", &radius_cell_mass::shared_backward);
    m.def("shared_project", &radius_cell_mass::shared_project);
    m.def("shared_forward_parallel", &radius_cell_mass::shared_forward_parallel);
    m.def("shared_backward_parallel", &radius_cell_mass::shared_backward_parallel);
    m.def("shared_project_parallel", &radius_cell_mass::shared_project_parallel);
    m.def("hybrid_parameter_grad_parallel", &radius_cell_mass::hybrid_parameter_grad_parallel);
}
