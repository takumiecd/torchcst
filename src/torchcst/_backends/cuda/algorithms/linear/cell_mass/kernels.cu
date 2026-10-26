// Tile-window cell masses: Fourier tails are approximated as zero outside
// selected whole tiles. No atom-centred cell clipping or fringe integration.
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <torch/extension.h>
#include <vector>

namespace {
constexpr int THREADS = 128;
constexpr int BATCH_TILE = 4;
constexpr int OUTPUT_TILE = 32;
constexpr int ATOM_CHUNK = 32;
constexpr int MAX_MODES = 16;
constexpr int MAX_BASIS = 1 + 2 * MAX_MODES;
constexpr int LEAVES_PER_BLOCK = 4;
constexpr int COEFFICIENT_SLOTS =
    (LEAVES_PER_BLOCK * BATCH_TILE * MAX_BASIS + THREADS - 1) / THREADS;
constexpr double PI = 3.1415926535897932384626433832795;

struct Axis {
    int sites, modes, leaf, leaves, nodes, tile_sites, period_tiles;
    double spacing, origin, tail_bound;
    bool polynomial_coefficients;
    __host__ __device__ int basis_count() const {
        return 1 + 2 * modes;
    }
    __host__ __device__ double chart_period() const {
        return sites * spacing;
    }
    __host__ __device__ double profile_period() const {
        return period_tiles * tile_sites * spacing;
    }
    __host__ __device__ int level_offset(int level) const {
        return 2 * leaves - 2 * (leaves >> level);
    }
};

Axis parse_axis(const std::vector<double> &c) {
    TORCH_CHECK(c.size() == 8, "tile-window axis config requires eight scalars");
    for (double v : c)
        TORCH_CHECK(std::isfinite(v), "axis config must be finite");
    TORCH_CHECK(c[0] == std::floor(c[0]) && c[0] >= 1 && c[0] <= 128,
                "CUDA prototype supports 1..128 sites");
    TORCH_CHECK(c[4] == std::floor(c[4]) && c[4] >= 2 && c[4] <= MAX_MODES,
                "CUDA prototype supports 2..16 Fourier modes");
    Axis a{};
    a.sites = static_cast<int>(c[0]);
    a.spacing = c[1];
    TORCH_CHECK(c[2] == std::floor(c[2]) && c[2] >= 1 && c[2] <= a.sites, "invalid tile_sites");
    a.tile_sites = static_cast<int>(c[2]);
    TORCH_CHECK((a.tile_sites & (a.tile_sites - 1)) == 0 && a.sites % a.tile_sites == 0,
                "tile_sites must be a power-of-two divisor of sites");
    TORCH_CHECK(c[7] == std::floor(c[7]) && c[7] >= 1 && c[7] <= a.sites / a.tile_sites,
                "invalid period_tiles");
    a.period_tiles = static_cast<int>(c[7]);
    TORCH_CHECK(a.period_tiles % 2 == 1 && (a.period_tiles != 1 || a.tile_sites == a.sites),
                "period_tiles must be odd; one tile must cover the chart");
    a.origin = c[3];
    a.modes = static_cast<int>(c[4]);
    a.tail_bound = c[5];
    TORCH_CHECK(c[6] == 0 || c[6] == 1, "invalid coefficient method");
    a.polynomial_coefficients = c[6] == 1;
    TORCH_CHECK((a.sites & (a.sites - 1)) == 0, "site count must be a power of two");
    TORCH_CHECK(a.modes % 2 == 0 && a.spacing > 0, "invalid cell-mass axis");
    TORCH_CHECK(a.tail_bound > 0 && a.tail_bound < 1 &&
                    std::acosh(1 / std::sqrt(a.tail_bound)) / a.modes < std::acosh(2.0),
                "invalid Chebyshev tail bound");
    a.leaf = std::min(8, a.tile_sites);
    a.leaves = a.sites / a.leaf;
    a.nodes = 2 * a.leaves - 1;
    return a;
}

void check_signal(const torch::Tensor &x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2,
                "signal must be a contiguous CUDA matrix");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat64,
                "signal must be float32 or float64");
}

void check_like(const torch::Tensor &t, const torch::Tensor &x) {
    TORCH_CHECK(t.is_contiguous() && t.device() == x.device() && t.scalar_type() == x.scalar_type(),
                "incompatible CUDA tensor");
}

void check_atoms(const torch::Tensor &atoms, const torch::Tensor &rows, const torch::Tensor &x) {
    check_like(atoms, x);
    TORCH_CHECK(atoms.dim() == 2 && atoms.size(1) == 4 && atoms.size(0) <= 1024,
                "prototype physical atoms require shape [A<=1024,4]");
    TORCH_CHECK(rows.device() == x.device() && rows.is_contiguous() &&
                    rows.scalar_type() == torch::kInt64 && rows.dim() == 1 &&
                    rows.numel() == atoms.size(0),
                "invalid sorted atom indices");
    // rows is a private, internally generated argsort permutation. Tensor value
    // validation is performed by the Python entry, not by a synchronizing scan
    // of every intermediate buffer at each native call.
}

void check_z(const torch::Tensor &z, const torch::Tensor &x, Axis a) {
    check_like(z, x);
    TORCH_CHECK(z.dim() == 3 && z.size(0) == x.size(0) && z.size(1) == a.nodes &&
                    z.size(2) == a.basis_count(),
                "invalid Z shape");
}

__device__ double wrap_center(double center, Axis a) {
    double v = fmod(center - a.origin, a.chart_period());
    return v < 0 ? v + a.chart_period() : v;
}

// Each atom evaluates these few scalar quantities in registers. The output
// gather kernel stores only its current chunk's contexts in block shared memory.
// No global atom copy, coefficient table or coefficient-Jacobian table exists.
struct Profile {
    Axis axis;
    double center, beta, beta_sigma, denominator, denominator_sigma;
    double c0, c0_sigma;
    int first_tile;
};

struct ValueDerivative {
    double value, sigma;
};

__device__ ValueDerivative native_density(const Profile &p, double t) {
    double v = cos(PI * t / p.axis.profile_period());
    double argument = p.beta * v;
    double previous = 1, current = argument;
    double dprevious = 0, dcurrent = p.beta_sigma * v;
    for (int j = 2; j <= p.axis.modes; ++j) {
        double next = 2 * argument * current - previous;
        double dnext = 2 * (p.beta_sigma * v * current + argument * dcurrent) - dprevious;
        previous = current;
        current = next;
        dprevious = dcurrent;
        dcurrent = dnext;
    }
    double h = current / p.denominator;
    double dh = (dcurrent - h * p.denominator_sigma) / p.denominator;
    return {h * h, 2 * h * dh};
}

// A degree-M density has exact Fourier coefficients from 2M+1 equispaced
// samples. This small scalar DFT avoids a persistent prepared coefficient table.
__device__ ValueDerivative dft_coefficient(const Profile &p, int harmonic) {
    int samples = 2 * p.axis.modes + 1;
    double sum = 0, derivative = 0;
    for (int j = 0; j < samples; ++j) {
        double angle = 2 * PI * j / samples;
        auto density = native_density(p, p.axis.profile_period() * j / samples);
        double factor = cos(harmonic * angle) / samples;
        sum += density.value * factor;
        derivative += density.sigma * factor;
    }
    return {sum, derivative};
}

// T_M(x)^2 = (1 + T_{2M}(x))/2. Let x=beta^2, t=x-1.
// The k-th complex Fourier coefficient D_k of T_{2M}(beta*cos(theta/2)) is
//   D_M = x^M/2,
//   D_k = M/(M+k)*x^k * sum_{j=1..M-k}
//         binom(M-k-1,j-1)*binom(M+k+j-1,j)*t^j.
// All terms are nonnegative for beta>=1. Advancing adjacent terms avoids both
// a sampling DFT and cancellation in an alternating power expansion.
// The sigma derivative follows the same scalar recurrence. No arrays/tables.
__device__ ValueDerivative polynomial_coefficient(const Profile &p, int harmonic) {
    int m = p.axis.modes;
    int k = harmonic;
    double x = p.beta * p.beta;
    double dx = 2 * p.beta * p.beta_sigma;
    double t = x - 1;
    double x_power = 1;
    for (int j = 0; j < k; ++j)
        x_power *= x;
    double value, derivative;
    if (k == m) {
        value = 0.5 * x_power;
        derivative = value * m * dx / x;
    } else {
        double term = (m + k) * t;
        double dterm = (m + k) * dx;
        double sum = term;
        double dsum = dterm;
        for (int j = 1; j < m - k; ++j) {
            double ratio = static_cast<double>((m - k - j) * (m + k + j)) / (j * (j + 1));
            dterm = ratio * (dterm * t + term * dx);
            term *= ratio * t;
            sum += term;
            dsum += dterm;
        }
        double prefix = static_cast<double>(m) / (m + k) * x_power;
        double dprefix = k == 0 ? 0 : prefix * k * dx / x;
        value = prefix * sum;
        derivative = dprefix * sum + prefix * dsum;
    }
    if (k == 0)
        value += 1;
    double denominator = 2 * p.denominator * p.denominator;
    return {value / denominator,
            (derivative - 2 * value * p.denominator_sigma / p.denominator) / denominator};
}

__device__ ValueDerivative raw_coefficient(const Profile &p, int harmonic) {
    if (p.axis.polynomial_coefficients)
        return polynomial_coefficient(p, harmonic);
    return dft_coefficient(p, harmonic);
}

__device__ Profile make_profile(Axis a, double sigma, double center) {
    Profile p{};
    p.axis = a;
    p.center = wrap_center(center, a);
    // Tile edges are cell edges, not site centres. Keep an unwrapped index so
    // chart-seam tiles can use the correct local Fourier image.
    int owner = static_cast<int>(floor((p.center + a.spacing / 2) / (a.tile_sites * a.spacing)));
    p.first_tile = owner - a.period_tiles / 2;
    double period = a.profile_period(), tile = period / 3;
    double shape = acosh(1 / sqrt(a.tail_bound)) / a.modes;
    double minimum = period / PI * acos(1 / cosh(shape));
    double ratio = minimum / tile;
    double ratio4 = ratio * ratio * ratio * ratio;
    double s = sigma / tile;
    double base = ratio4 + (1 - ratio4) * s * s * s * s;
    double radius = tile * pow(base, 0.25);
    double radius_sigma = (1 - ratio4) * s * s * s * pow(base, -0.75);
    double angle = PI * radius / period;
    p.beta = 1 / cos(angle);
    p.beta_sigma = p.beta * tan(angle) * PI / period * radius_sigma;
    double exponent = a.modes * acosh(p.beta);
    p.denominator = cosh(exponent);
    p.denominator_sigma = sinh(exponent) * a.modes / sqrt(p.beta * p.beta - 1) * p.beta_sigma;
    auto dc = raw_coefficient(p, 0);
    p.c0 = dc.value;
    p.c0_sigma = dc.sigma;
    return p;
}

struct Coefficient {
    double value, center, sigma;
};

__device__ Coefficient coefficient(const Profile &p, int mode, double image_center) {
    if (mode == 0)
        return {1 / p.axis.profile_period(), 0, 0};
    int k = (mode + 1) / 2;
    double omega = 2 * PI * k / p.axis.profile_period();
    auto c = raw_coefficient(p, k);
    double scale = 2 / (p.axis.profile_period() * p.c0);
    double normalized = c.value * scale;
    double ds = (c.sigma - c.value * p.c0_sigma / p.c0) * scale;
    double phase = omega * image_center;
    bool cosine = mode % 2 == 1;
    double f = cosine ? cos(phase) : sin(phase);
    double df = omega * (cosine ? -sin(phase) : cos(phase));
    return {normalized * f, normalized * df, ds * f};
}

__device__ double integrated_basis(Axis a, int mode, int site) {
    if (mode == 0)
        return a.spacing;
    double omega = 2 * PI * ((mode + 1) / 2) / a.profile_period();
    double factor = 2 * sin(omega * a.spacing / 2) / omega;
    double phase = omega * site * a.spacing;
    return factor * (mode % 2 == 1 ? cos(phase) : sin(phase));
}

// Z construction: a tunable team shares a single (batch,mode) sum. Distinct
// teams, leaf tiles and batches are independent. Every thread reaches barriers.
template <typename scalar_t>
__global__ void build_z_direct_kernel(const scalar_t *signal, scalar_t *z, int batches, Axis a,
                                      int team_size) {
    __shared__ double partials[THREADS];
    int team_count = THREADS / team_size;
    int team = threadIdx.x / team_size, lane = threadIdx.x % team_size;
    int node = blockIdx.x, count = BATCH_TILE * a.basis_count();
    int level = 0;
    while ((a.leaves >> (level + 1)) > 0 && node >= a.level_offset(level + 1))
        ++level;
    int width = a.leaf << level;
    int first = (node - a.level_offset(level)) * width;
    for (int wave = 0; wave * team_count < count; ++wave) {
        int q = wave * team_count + team;
        int b = blockIdx.y * BATCH_TILE + q / a.basis_count();
        int mode = q % a.basis_count();
        bool valid = q < count && b < batches;
        double partial = 0;
        if (valid) {
            for (int j = lane; j < width; j += team_size) {
                int site = first + j;
                partial += static_cast<double>(signal[b * a.sites + site]) *
                           integrated_basis(a, mode, site);
            }
        }
        partials[threadIdx.x] = partial;
        __syncthreads();
        for (int step = team_size / 2; step; step /= 2) {
            if (lane < step)
                partials[threadIdx.x] += partials[threadIdx.x + step];
            __syncthreads();
        }
        if (valid && lane == 0)
            z[(b * a.nodes + node) * a.basis_count() + mode] = partials[threadIdx.x];
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void merge_z_level_kernel(scalar_t *z, int batches, Axis a, int level) {
    int parents = a.leaves >> level;
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= batches * parents * a.basis_count())
        return;
    int mode = q % a.basis_count();
    int parent = (q / a.basis_count()) % parents;
    int b = q / (a.basis_count() * parents);
    int left = a.level_offset(level - 1) + 2 * parent;
    int destination = a.level_offset(level) + parent;
    z[(b * a.nodes + destination) * a.basis_count() + mode] =
        z[(b * a.nodes + left) * a.basis_count() + mode] +
        z[(b * a.nodes + left + 1) * a.basis_count() + mode];
}

struct Projection {
    double value[BATCH_TILE], center[BATCH_TILE], sigma[BATCH_TILE];
};

template <bool Derivatives, typename scalar_t>
__device__ Projection project_atom(const scalar_t *signal, const scalar_t *z, int batches,
                                   int batch_begin, const Profile &p) {
    (void)signal; // All input cells are consumed through the common Z basis.
    Axis a = p.axis;
    Projection result{};
    int window_first = p.first_tile * a.tile_sites;
    int window_end = window_first + a.period_tiles * a.tile_sites;
    for (int image = -1; image <= 1; ++image) {
        int first = max(window_first, image * a.sites) - image * a.sites;
        int end = min(window_end, (image + 1) * a.sites) - image * a.sites;
        if (first >= end)
            continue;
        double center = p.center - image * a.chart_period();
        for (int mode = 0; mode < a.basis_count(); ++mode) {
            auto c = coefficient(p, mode, center);
            // All selected cells form whole aligned tiles. Greedily consume
            // the largest available Z nodes; no cell needs a fringe fallback.
            for (int site = first; site < end;) {
                int level = 0, width = a.leaf;
                while (width * 2 <= a.sites && site % (width * 2) == 0 && site + width * 2 <= end) {
                    ++level;
                    width *= 2;
                }
                int node = a.level_offset(level) + site / width;
                for (int local = 0; local < BATCH_TILE && batch_begin + local < batches; ++local) {
                    double moment =
                        z[((batch_begin + local) * a.nodes + node) * a.basis_count() + mode];
                    result.value[local] += c.value * moment;
                    if (Derivatives) {
                        result.center[local] += c.center * moment;
                        result.sigma[local] += c.sigma * moment;
                    }
                }
                site += width;
            }
        }
    }
    return result;
}

// H/G: each thread owns a different sorted slot. Original atom parameters are
// read through rows; all required input tiles complete locally before one store.
template <typename scalar_t>
__global__ void project_atoms_kernel(const scalar_t *signal, const scalar_t *atoms,
                                     const int64_t *rows, const scalar_t *z, scalar_t *h,
                                     int batches, int atom_count, Axis a, int center_column) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= atom_count)
        return;
    int original = rows[slot];
    Profile p = make_profile(a, atoms[4 * original + 1], atoms[4 * original + center_column]);
    int batch_begin = blockIdx.y * BATCH_TILE;
    Projection result = project_atom<false>(signal, z, batches, batch_begin, p);
    for (int local = 0; local < BATCH_TILE && batch_begin + local < batches; ++local)
        h[(batch_begin + local) * atom_count + slot] = result.value[local];
}

// Y/dX: the launch still owns up to four leaves and four batches per block.
// Within that range, C is built once per DISTINCT PROFILE TILE. Its coefficients
// are shared by all leaf/site positions in that tile; only their basis differs.
// A profile tile larger than a block is still recomputed across those blocks.
// A chunk of atoms is prepared once, then all owned C values consume it. Only the current
// chunk's raw harmonic coefficients are shared; no global coefficient table.
// Every thread reaches the barriers, including threads without an output site.
template <typename scalar_t>
__global__ void expand_tiles_kernel(const scalar_t *h, const scalar_t *atoms, const int64_t *rows,
                                    scalar_t *output, int batches, int atom_count, Axis a,
                                    int center_column) {
    __shared__ double shared_center[ATOM_CHUNK];
    __shared__ int shared_first_tile[ATOM_CHUNK];
    __shared__ double shared_amplitude[ATOM_CHUNK];
    __shared__ double shared_h[BATCH_TILE * ATOM_CHUNK];
    __shared__ double shared_harmonic[ATOM_CHUNK * MAX_MODES];
    __shared__ double shared_c[LEAVES_PER_BLOCK * BATCH_TILE * MAX_BASIS];

    int block_width = min(a.sites, min(OUTPUT_TILE, LEAVES_PER_BLOCK * a.leaf));
    int block_first = blockIdx.x * block_width;
    int block_end = min(block_first + block_width, a.sites);
    int first_profile_tile = block_first / a.tile_sites;
    int profile_tile_count = (block_end - 1) / a.tile_sites - first_profile_tile + 1;
    int c_count = profile_tile_count * BATCH_TILE * a.basis_count();
    double local_c[COEFFICIENT_SLOTS] = {0};

    for (int base = 0; base < atom_count; base += ATOM_CHUNK) {
        int count = min(ATOM_CHUNK, atom_count - base);
        if (threadIdx.x < count) {
            int j = threadIdx.x;
            int original = rows[base + j];
            Profile p =
                make_profile(a, atoms[4 * original + 1], atoms[4 * original + center_column]);
            shared_center[j] = p.center;
            shared_first_tile[j] = p.first_tile;
            shared_amplitude[j] = atoms[4 * original];
            for (int k = 1; k <= a.modes; ++k)
                shared_harmonic[j * MAX_MODES + k - 1] =
                    2 * raw_coefficient(p, k).value / (a.profile_period() * p.c0);
        }
        for (int q = threadIdx.x; q < BATCH_TILE * count; q += blockDim.x) {
            int local_batch = q / count, j = q % count;
            int batch = blockIdx.y * BATCH_TILE + local_batch;
            shared_h[local_batch * ATOM_CHUNK + j] =
                batch < batches ? h[batch * atom_count + base + j] : 0;
        }
        __syncthreads(); // The shared atom/H chunk is ready for every C owner.

        for (int v = 0; v < COEFFICIENT_SLOTS; ++v) {
            int q = threadIdx.x + v * blockDim.x;
            if (q >= c_count)
                continue;
            int mode = q % a.basis_count();
            int local_batch = (q / a.basis_count()) % BATCH_TILE;
            int local_profile_tile = q / (a.basis_count() * BATCH_TILE);
            int target_tile = first_profile_tile + local_profile_tile;
            int chart_tiles = a.sites / a.tile_sites;
            for (int j = 0; j < count; ++j) {
                int offset = (target_tile - shared_first_tile[j]) % chart_tiles;
                if (offset < 0)
                    offset += chart_tiles;
                if (offset >= a.period_tiles)
                    continue; // Entire tile is outside this atom's retained window.
                int image = (shared_first_tile[j] + offset - target_tile) / chart_tiles;
                double center = shared_center[j] - image * a.chart_period();
                double value = 1 / a.profile_period();
                if (mode != 0) {
                    int k = (mode + 1) / 2;
                    double phase = (2 * PI * k / a.profile_period()) * center;
                    value = shared_harmonic[j * MAX_MODES + k - 1] *
                            (mode % 2 == 1 ? cos(phase) : sin(phase));
                }
                local_c[v] += shared_amplitude[j] * shared_h[local_batch * ATOM_CHUNK + j] * value;
            }
        }
        __syncthreads(); // Every C owner has finished using this atom/H chunk.
    }

    for (int v = 0; v < COEFFICIENT_SLOTS; ++v) {
        int q = threadIdx.x + v * blockDim.x;
        if (q < c_count)
            shared_c[q] = local_c[v];
    }
    __syncthreads(); // Publish all C values once before the site expansion.

    int local_batch = threadIdx.x / block_width;
    int local_site = threadIdx.x % block_width;
    int site = block_first + local_site;
    int batch = blockIdx.y * BATCH_TILE + local_batch;
    if (local_batch < BATCH_TILE && site < a.sites && batch < batches) {
        // All sites of a profile tile read the SAME C. integrated_basis below
        // supplies the site-dependent value, so no output values are copied.
        int local_profile_tile = site / a.tile_sites - first_profile_tile;
        int c_begin = (local_profile_tile * BATCH_TILE + local_batch) * a.basis_count();
        double accumulator = 0;
        for (int mode = 0; mode < a.basis_count(); ++mode)
            accumulator += shared_c[c_begin + mode] * integrated_basis(a, mode, site);
        output[batch * a.sites + site] = accumulator;
    }
}

template <typename scalar_t>
__global__ void parameter_grad_kernel(const scalar_t *x, const scalar_t *dy, const scalar_t *atoms,
                                      const int64_t *rows, const scalar_t *h, const scalar_t *g,
                                      const scalar_t *zx, const scalar_t *zy, scalar_t *grad,
                                      int batches, int atom_count, Axis input_axis,
                                      Axis output_axis) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= atom_count)
        return;
    int original = rows[slot];
    double amplitude = atoms[4 * original], sigma = atoms[4 * original + 1];
    Profile input = make_profile(input_axis, sigma, atoms[4 * original + 3]);
    Profile output = make_profile(output_axis, sigma, atoms[4 * original + 2]);
    double da = 0, ds = 0, dmu_out = 0, dmu_in = 0;
    for (int begin = 0; begin < batches; begin += BATCH_TILE) {
        Projection input_vjp = project_atom<true>(x, zx, batches, begin, input);
        Projection output_vjp = project_atom<true>(dy, zy, batches, begin, output);
        for (int local = 0; local < BATCH_TILE && begin + local < batches; ++local) {
            double hv = h[(begin + local) * atom_count + slot];
            double gv = g[(begin + local) * atom_count + slot];
            da += hv * gv;
            ds += amplitude * (gv * input_vjp.sigma[local] + hv * output_vjp.sigma[local]);
            dmu_in += amplitude * gv * input_vjp.center[local];
            dmu_out += amplitude * hv * output_vjp.center[local];
        }
    }
    grad[4 * original] = da;
    grad[4 * original + 1] = ds;
    grad[4 * original + 2] = dmu_out;
    grad[4 * original + 3] = dmu_in;
}

torch::Tensor project(torch::Tensor signal, torch::Tensor atoms, torch::Tensor rows,
                      torch::Tensor z, std::vector<double> config, int center_column) {
    check_signal(signal);
    Axis a = parse_axis(config);
    TORCH_CHECK(signal.size(1) == a.sites, "signal shape differs from axis");
    check_atoms(atoms, rows, signal);
    check_z(z, signal, a);
    c10::cuda::CUDAGuard guard(signal.device());
    auto result = torch::empty({signal.size(0), atoms.size(0)}, signal.options());
    if (atoms.size(0) == 0 || signal.size(0) == 0)
        return result;
    dim3 grid((atoms.size(0) + THREADS - 1) / THREADS,
              (signal.size(0) + BATCH_TILE - 1) / BATCH_TILE);
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(signal.scalar_type(), "cell_mass_project", [&] {
        project_atoms_kernel<<<grid, THREADS, 0, stream>>>(
            signal.data_ptr<scalar_t>(), atoms.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(),
            z.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), signal.size(0), atoms.size(0), a,
            center_column);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

torch::Tensor expand(torch::Tensor h, torch::Tensor atoms, torch::Tensor rows,
                     std::vector<double> config, int center_column) {
    check_signal(h);
    Axis a = parse_axis(config);
    check_atoms(atoms, rows, h);
    TORCH_CHECK(h.size(1) == atoms.size(0), "H/G shape differs from atom count");
    c10::cuda::CUDAGuard guard(h.device());
    auto result = torch::empty({h.size(0), a.sites}, h.options());
    if (h.size(0) == 0)
        return result;
    int block_width = std::min(a.sites, std::min(OUTPUT_TILE, LEAVES_PER_BLOCK * a.leaf));
    dim3 grid((a.sites + block_width - 1) / block_width, (h.size(0) + BATCH_TILE - 1) / BATCH_TILE);
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "cell_mass_expand", [&] {
        expand_tiles_kernel<<<grid, THREADS, 0, stream>>>(
            h.data_ptr<scalar_t>(), atoms.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(),
            result.data_ptr<scalar_t>(), h.size(0), atoms.size(0), a, center_column);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
} // namespace

torch::Tensor build_z(torch::Tensor signal, std::vector<double> config, bool hierarchical,
                      int team_size) {
    check_signal(signal);
    Axis a = parse_axis(config);
    TORCH_CHECK(team_size >= 1 && team_size <= THREADS && (team_size & (team_size - 1)) == 0,
                "Z team size must be a power of two in [1,128]");
    TORCH_CHECK(signal.size(1) == a.sites, "signal shape differs from axis");
    c10::cuda::CUDAGuard guard(signal.device());
    auto z = torch::empty({signal.size(0), a.nodes, a.basis_count()}, signal.options());
    if (signal.size(0) == 0)
        return z;
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(signal.scalar_type(), "cell_mass_build_z", [&] {
        dim3 grid(hierarchical ? a.leaves : a.nodes,
                  (signal.size(0) + BATCH_TILE - 1) / BATCH_TILE);
        build_z_direct_kernel<<<grid, THREADS, 0, stream>>>(
            signal.data_ptr<scalar_t>(), z.data_ptr<scalar_t>(), signal.size(0), a, team_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        for (int level = 1; hierarchical && (a.leaves >> level) > 0; ++level) {
            int count = signal.size(0) * (a.leaves >> level) * a.basis_count();
            merge_z_level_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
                z.data_ptr<scalar_t>(), signal.size(0), a, level);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });
    return z;
}

torch::Tensor compute_h(torch::Tensor x, torch::Tensor atoms, torch::Tensor rows, torch::Tensor z,
                        std::vector<double> config) {
    return project(x, atoms, rows, z, config, 3);
}
torch::Tensor compute_g(torch::Tensor dy, torch::Tensor atoms, torch::Tensor rows, torch::Tensor z,
                        std::vector<double> config) {
    return project(dy, atoms, rows, z, config, 2);
}
torch::Tensor compute_y(torch::Tensor h, torch::Tensor atoms, torch::Tensor rows,
                        std::vector<double> config) {
    return expand(h, atoms, rows, config, 2);
}
torch::Tensor compute_dx(torch::Tensor g, torch::Tensor atoms, torch::Tensor rows,
                         std::vector<double> config) {
    return expand(g, atoms, rows, config, 3);
}

torch::Tensor compute_parameter_grad(torch::Tensor x, torch::Tensor dy, torch::Tensor atoms,
                                     torch::Tensor rows, torch::Tensor h, torch::Tensor g,
                                     torch::Tensor zx, torch::Tensor zy,
                                     std::vector<double> input_config,
                                     std::vector<double> output_config) {
    check_signal(x);
    check_signal(dy);
    check_like(dy, x);
    check_atoms(atoms, rows, x);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(x.size(0) == dy.size(0) && x.size(1) == input.sites && dy.size(1) == output.sites,
                "input/output shapes differ from axes");
    check_z(zx, x, input);
    check_z(zy, dy, output);
    check_like(h, x);
    check_like(g, x);
    TORCH_CHECK(h.dim() == 2 && g.sizes() == h.sizes() && h.size(0) == x.size(0) &&
                    h.size(1) == atoms.size(0),
                "invalid H/G shapes");
    c10::cuda::CUDAGuard guard(x.device());
    auto grad = torch::empty_like(atoms);
    if (atoms.size(0) == 0)
        return grad;
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "cell_mass_parameter_grad", [&] {
        parameter_grad_kernel<<<(atoms.size(0) + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(), dy.data_ptr<scalar_t>(), atoms.data_ptr<scalar_t>(),
            rows.data_ptr<int64_t>(), h.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
            zx.data_ptr<scalar_t>(), zy.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), x.size(0),
            atoms.size(0), input, output);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}
