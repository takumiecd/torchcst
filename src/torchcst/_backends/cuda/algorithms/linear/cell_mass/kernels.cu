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
constexpr int ATOM_CHUNK = 32;
constexpr int MAX_MODES = 16;
constexpr int MAX_BASIS = 1 + 2 * MAX_MODES;
constexpr int MAX_PROFILE_TILES_PER_BLOCK = 4;
constexpr int COEFFICIENT_SLOTS =
    (MAX_PROFILE_TILES_PER_BLOCK * BATCH_TILE * MAX_BASIS + THREADS - 1) / THREADS;
constexpr double PI = 3.1415926535897932384626433832795;

// These are mode-only rational factors, not atom coefficients. Build the small
// table at compile time so each atom does not repeat dozens of double divides.
constexpr int POLYNOMIAL_BANKS = MAX_MODES / 2 + 1;
constexpr int MAX_POLYNOMIAL_RATIOS = MAX_MODES * (MAX_MODES - 1) / 2;
struct PolynomialBanks {
    double ratios[POLYNOMIAL_BANKS][MAX_POLYNOMIAL_RATIOS] = {};
    double prefixes[POLYNOMIAL_BANKS][MAX_MODES + 1] = {};
    __host__ __device__ constexpr PolynomialBanks() {
        for (int bank = 1; bank < POLYNOMIAL_BANKS; ++bank) {
            int modes = 2 * bank;
            for (int harmonic = 0; harmonic <= modes; ++harmonic) {
                prefixes[bank][harmonic] = static_cast<double>(modes) / (modes + harmonic);
                int first = harmonic * (2 * modes - harmonic - 1) / 2;
                for (int j = 1; j < modes - harmonic; ++j)
                    ratios[bank][first + j - 1] =
                        static_cast<double>((modes - harmonic - j) * (modes + harmonic + j)) /
                        (j * (j + 1));
            }
        }
    }
};
__device__ __constant__ PolynomialBanks polynomial_banks{};

struct Axis {
    int sites, modes, leaf, leaves, nodes, tile_sites, period_tiles;
    double spacing, origin, tail_bound, radius_ratio4, fundamental_omega;
    bool polynomial_coefficients, polynomial_table, owner_windows, phase_recurrence;
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
    TORCH_CHECK(c[0] == std::floor(c[0]) && c[0] >= 1 && c[0] <= 1024,
                "CUDA prototype supports 1..1024 sites");
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
    TORCH_CHECK(c[6] == 0 || c[6] == 1 || c[6] == 2 || c[6] == 3, "invalid coefficient method");
    a.polynomial_coefficients = c[6] != 0;
    a.polynomial_table = c[6] >= 2;
    a.phase_recurrence = c[6] == 3;
    TORCH_CHECK((a.sites & (a.sites - 1)) == 0, "site count must be a power of two");
    TORCH_CHECK(a.modes % 2 == 0 && a.spacing > 0, "invalid cell-mass axis");
    TORCH_CHECK(a.tail_bound > 0 && a.tail_bound < 1 &&
                    std::acosh(1 / std::sqrt(a.tail_bound)) / a.modes < std::acosh(2.0),
                "invalid Chebyshev tail bound");
    a.leaf = std::min(8, a.tile_sites);
    a.leaves = a.sites / a.leaf;
    a.nodes = 2 * a.leaves - 1;
    // Bank-only constants: compute once on the host, not once per atom.
    double period = a.profile_period(), tile = period / 3;
    a.fundamental_omega = 2 * PI / period;
    double shape = std::acosh(1 / std::sqrt(a.tail_bound)) / a.modes;
    double minimum = period / PI * std::acos(1 / std::cosh(shape));
    double ratio = minimum / tile;
    a.radius_ratio4 = ratio * ratio * ratio * ratio;
    return a;
}

void check_signal(const torch::Tensor &x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2,
                "signal must be a contiguous CUDA matrix");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat64,
                "signal must be float32 or float64");
}

// H/G/parameter gradients have no block-wide reduction: one thread owns one
// atom. Their block size can change independently of cooperative Z and Y/dX.
void check_atom_thread_count(int threads) {
    TORCH_CHECK(threads == 16 || threads == 32 || threads == 64 || threads == 128,
                "atom_threads_per_block must be 16, 32, 64 or 128");
}

void check_like(const torch::Tensor &t, const torch::Tensor &x) {
    TORCH_CHECK(t.is_contiguous() && t.device() == x.device() && t.scalar_type() == x.scalar_type(),
                "incompatible CUDA tensor");
}

void check_atoms(const torch::Tensor &atoms, const torch::Tensor &rows, const torch::Tensor &x) {
    check_like(atoms, x);
    TORCH_CHECK(atoms.dim() == 2 && atoms.size(1) == 4 && atoms.size(0) <= 65536,
                "prototype physical atoms require shape [A<=65536,4]");
    TORCH_CHECK(rows.device() == x.device() && rows.is_contiguous() &&
                    rows.scalar_type() == torch::kInt64 && rows.dim() == 1 &&
                    rows.numel() == atoms.size(0),
                "invalid sorted atom indices");
    // rows is a private, internally generated argsort permutation. Tensor value
    // validation is performed by the Python entry, not by a synchronizing scan
    // of every intermediate buffer at each native call.
}

Axis check_z(const torch::Tensor &z, const torch::Tensor &x, Axis a) {
    TORCH_CHECK(z.is_contiguous() && z.device() == x.device() &&
                    (z.scalar_type() == x.scalar_type() || z.scalar_type() == torch::kFloat64),
                "incompatible hierarchy tensor");
    int owners = a.sites / a.tile_sites;
    TORCH_CHECK(z.dim() == 3 && z.size(0) == x.size(0) &&
                    (z.size(1) == a.nodes || z.size(1) == owners) && z.size(2) == a.basis_count(),
                "invalid Z shape");
    // The single-node case represents the same full-chart window either way.
    a.owner_windows = z.size(1) != a.nodes;
    a.nodes = z.size(1);
    return a;
}

__device__ double mode_frequency(Axis a, int harmonic) {
    if (a.phase_recurrence)
        return harmonic * a.fundamental_omega;
    return 2 * PI * harmonic / a.profile_period();
}

__device__ double wrap_center(double center, Axis a) {
    double v = fmod(center - a.origin, a.chart_period());
    return v < 0 ? v + a.chart_period() : v;
}

// Routing uses the same FP64 cell edges as profile evaluation. The final mod
// maps the seam owner back into the chart's tile range; stable sort resolves ties.
__device__ int chart_tile_owner(double center, Axis a) {
    int owner = static_cast<int>(
        floor((wrap_center(center, a) + a.spacing / 2) / (a.tile_sites * a.spacing)));
    return owner % (a.sites / a.tile_sites);
}

// One traversal replaces separate finite/width reductions and Torch key ops.
// Bits: 1 = nonfinite X, 2 = nonfinite atom, 4 = invalid sigma.
// The error word is zeroed on this stream first. Valid inputs perform no atomics.
// Only keys are written: no packed atom or persistent coefficient copies.
template <typename scalar_t, typename key_t>
__global__ void prepare_inputs_kernel(const scalar_t *x, const scalar_t *atoms, key_t *keys,
                                      int *error_word, int64_t x_elements, int atom_count,
                                      Axis input, Axis output, double maximum_sigma) {
    int64_t q = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int errors = 0;
    if (q < x_elements && !isfinite(static_cast<double>(x[q])))
        errors |= 1;
    if (q < atom_count) {
        bool finite_atom = true;
        for (int field = 0; field < 4; ++field)
            finite_atom &= isfinite(static_cast<double>(atoms[4 * q + field]));
        keys[q] = 0;
        if (!finite_atom) {
            errors |= 2;
        } else if (atoms[4 * q + 1] <= 0 ||
                   atoms[4 * q + 1] > static_cast<scalar_t>(maximum_sigma)) {
            // Torch compares a Python scalar to a Tensor in the Tensor's dtype.
            // Preserve that rule at the upper width boundary, including FP32.
            errors |= 4;
        } else {
            int qi = chart_tile_owner(atoms[4 * q + 3], input);
            int qo = chart_tile_owner(atoms[4 * q + 2], output);
            keys[q] = qo * (input.sites / input.tile_sites) + qi;
        }
    }
    if (errors)
        atomicOr(error_word, errors);
}

// One thread finds one lower bound in the sorted permutation. Empty groups
// naturally have identical begin/end offsets; no sorted-key copy is required.
template <typename key_t>
__global__ void group_offsets_kernel(const key_t *keys, const int64_t *rows, int64_t *offsets,
                                     int atoms, int groups) {
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group > groups)
        return;
    int first = 0, last = atoms;
    while (first < last) {
        int middle = first + (last - first) / 2;
        if (keys[rows[middle]] < group)
            first = middle + 1;
        else
            last = middle;
    }
    offsets[group] = first;
}

// Per-atom scalar context in registers; the C producer shares only one chunk.
// No global atom copy or per-atom coefficient/Jacobian table exists.
struct Profile {
    Axis axis;
    double center, beta, beta_sigma, denominator, denominator_sigma;
    double c0, c0_sigma;
    double log_beta2_sigma, normalized_scale, normalization_sigma;
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
        if (p.axis.polynomial_table)
            derivative = value * m * p.log_beta2_sigma;
        else
            derivative = value * m * dx / x;
    } else {
        double term = (m + k) * t;
        double dterm = (m + k) * dx;
        double sum = term;
        double dsum = dterm;
        for (int j = 1; j < m - k; ++j) {
            double ratio;
            if (p.axis.polynomial_table) {
                int first = k * (2 * m - k - 1) / 2;
                ratio = polynomial_banks.ratios[m / 2][first + j - 1];
            } else {
                ratio = static_cast<double>((m - k - j) * (m + k + j)) / (j * (j + 1));
            }
            dterm = ratio * (dterm * t + term * dx);
            term *= ratio * t;
            sum += term;
            dsum += dterm;
        }
        double prefix, dprefix;
        if (p.axis.polynomial_table) {
            prefix = polynomial_banks.prefixes[m / 2][k] * x_power;
            dprefix = prefix * k * p.log_beta2_sigma;
        } else {
            prefix = static_cast<double>(m) / (m + k) * x_power;
            dprefix = k == 0 ? 0 : prefix * k * dx / x;
        }
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
    double ratio4 = a.radius_ratio4;
    double s = sigma / tile;
    double base = ratio4 + (1 - ratio4) * s * s * s * s;
    double root = sqrt(sqrt(base));
    double radius = tile * root;
    double radius_sigma = (1 - ratio4) * s * s * s / (root * root * root);
    double angle = PI * radius / period;
    p.beta = 1 / cos(angle);
    p.beta_sigma = p.beta * tan(angle) * PI / period * radius_sigma;
    if (a.polynomial_table)
        p.log_beta2_sigma = 2 * p.beta_sigma / p.beta;
    if (a.polynomial_coefficients) {
        // Every raw coefficient and its mass share T_M(beta)^-2. It cancels
        // from the normalized value AND its sigma derivative. DFT keeps D.
        p.denominator = 1;
        p.denominator_sigma = 0;
    } else {
        double exponent = a.modes * acosh(p.beta);
        p.denominator = cosh(exponent);
        p.denominator_sigma = sinh(exponent) * a.modes / sqrt(p.beta * p.beta - 1) * p.beta_sigma;
    }
    auto dc = raw_coefficient(p, 0);
    p.c0 = dc.value;
    p.c0_sigma = dc.sigma;
    if (a.polynomial_table) {
        p.normalized_scale = 2 / (period * p.c0);
        p.normalization_sigma = p.c0_sigma / p.c0;
    }
    return p;
}

struct Coefficient {
    double value, center, sigma;
};

struct CoefficientPair {
    Coefficient cosine, sine;
};

// Paired modes share one raw harmonic, normalization and phase.
template <bool Derivatives>
__device__ CoefficientPair coefficient_pair_from_phase(const Profile &p, int harmonic,
                                                       double cosine, double sine) {
    if (harmonic == 0)
        return {{1 / p.axis.profile_period(), 0, 0}, {0, 0, 0}};
    double omega = mode_frequency(p.axis, harmonic);
    auto raw = raw_coefficient(p, harmonic);
    double scale;
    if (p.axis.polynomial_table)
        scale = p.normalized_scale;
    else
        scale = 2 / (p.axis.profile_period() * p.c0);
    double normalized = raw.value * scale;
    CoefficientPair result{{normalized * cosine, 0, 0}, {normalized * sine, 0, 0}};
    if (Derivatives) {
        double ds;
        if (p.axis.polynomial_table)
            ds = (raw.sigma - raw.value * p.normalization_sigma) * scale;
        else
            ds = (raw.sigma - raw.value * p.c0_sigma / p.c0) * scale;
        result.cosine.center = -omega * normalized * sine;
        result.sine.center = omega * normalized * cosine;
        result.cosine.sigma = ds * cosine;
        result.sine.sigma = ds * sine;
    }
    return result;
}

// Harmonics share one fundamental phase. The recurrence keeps double
// precision and is bounded by MAX_MODES=16; no coefficient array is retained.
struct HarmonicPhase {
    double fundamental, step_sine, step_cosine;
    double sine = 0, cosine = 1;
    bool recurrence;
    __device__ HarmonicPhase(const Profile &profile, double center) {
        fundamental = profile.axis.fundamental_omega * center;
        recurrence = profile.axis.phase_recurrence;
        if (recurrence)
            sincos(fundamental, &step_sine, &step_cosine);
    }
    __device__ void advance(int harmonic) {
        if (harmonic == 0)
            return;
        if (recurrence) {
            double next_cosine = cosine * step_cosine - sine * step_sine;
            sine = sine * step_cosine + cosine * step_sine;
            cosine = next_cosine;
        } else {
            sincos(harmonic * fundamental, &sine, &cosine);
        }
    }
};

// Non-recurrence uses the original omega*center evaluation order.
template <bool Derivatives>
__device__ CoefficientPair coefficient_pair(const Profile &p, int harmonic, double center) {
    double sine = 0, cosine = 1;
    if (harmonic != 0)
        sincos(mode_frequency(p.axis, harmonic) * center, &sine, &cosine);
    return coefficient_pair_from_phase<Derivatives>(p, harmonic, cosine, sine);
}

__device__ double integrated_basis(Axis a, int mode, int site) {
    if (mode == 0)
        return a.spacing;
    double omega = mode_frequency(a, (mode + 1) / 2);
    double factor = 2 * sin(omega * a.spacing / 2) / omega;
    double phase = omega * site * a.spacing;
    return factor * (mode % 2 == 1 ? cos(phase) : sin(phase));
}

// Z construction: a tunable team shares a single (batch,mode) sum. Distinct
// teams, leaf tiles and batches are independent. Every thread reaches barriers.
template <typename scalar_t, typename moment_t>
__global__ void build_z_direct_kernel(const scalar_t *signal, moment_t *z, int batches, Axis a,
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

// All atoms with the same owner use the same K-tile moment window. Aggregate
// it once, including chart-seam basis rotations, instead of walking Z per atom.
template <typename moment_t>
__global__ void aggregate_owner_windows_kernel(const moment_t *z, moment_t *windows, int batches,
                                               Axis a) {
    int owners = a.sites / a.tile_sites;
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= batches * owners * a.basis_count())
        return;
    int mode = q % a.basis_count();
    int owner = (q / a.basis_count()) % owners;
    int batch = q / (a.basis_count() * owners);
    int window_first = (owner - a.period_tiles / 2) * a.tile_sites;
    int window_end = window_first + a.period_tiles * a.tile_sites;
    double total = 0;
    for (int image = -1; image <= 1; ++image) {
        int first = max(window_first, image * a.sites) - image * a.sites;
        int end = min(window_end, (image + 1) * a.sites) - image * a.sites;
        if (first >= end)
            continue;
        double primary = 0, paired = 0;
        int pair_mode = mode % 2 == 1 ? mode + 1 : mode - 1;
        for (int site = first; site < end;) {
            int level = 0, width = a.leaf;
            while (width * 2 <= a.sites && site % (width * 2) == 0 && site + width * 2 <= end) {
                ++level;
                width *= 2;
            }
            int node = a.level_offset(level) + site / width;
            int base = (batch * a.nodes + node) * a.basis_count();
            primary += z[base + mode];
            if (image != 0 && mode != 0)
                paired += z[base + pair_mode];
            site += width;
        }
        if (image == 0 || mode == 0) {
            total += primary;
        } else {
            int harmonic = (mode + 1) / 2;
            double sine, cosine;
            double angle = a.phase_recurrence
                               ? mode_frequency(a, harmonic) * image * a.chart_period()
                               : 2 * PI * harmonic * image * a.chart_period() / a.profile_period();
            sincos(angle, &sine, &cosine);
            // Basis rotation has the opposite sign from the C rotation.
            total +=
                mode % 2 == 1 ? primary * cosine - paired * sine : primary * cosine + paired * sine;
        }
    }
    windows[q] = total;
}

struct Projection {
    double value[BATCH_TILE], center[BATCH_TILE], sigma[BATCH_TILE];
};

template <bool Derivatives, bool CacheExpansion = false, typename scalar_t, typename moment_t>
__device__ Projection project_atom(const scalar_t *signal, const moment_t *z, int batches,
                                   int batch_begin, const Profile &p,
                                   double *cached_coefficients = nullptr) {
    (void)signal; // All input cells are consumed through the common Z basis.
    Axis a = p.axis;
    Projection result{};
    if (a.owner_windows) {
        int owners = a.sites / a.tile_sites;
        int raw_owner = p.first_tile + a.period_tiles / 2;
        int owner = raw_owner % owners;
        double center = p.center - ((raw_owner - owner) / owners) * a.chart_period();
        HarmonicPhase phase(p, center);
        for (int harmonic = 0; harmonic <= a.modes; ++harmonic) {
            int mode = harmonic == 0 ? 0 : 2 * harmonic - 1;
            CoefficientPair pair;
            if (a.phase_recurrence) {
                phase.advance(harmonic);
                pair =
                    coefficient_pair_from_phase<Derivatives>(p, harmonic, phase.cosine, phase.sine);
            } else {
                pair = coefficient_pair<Derivatives>(p, harmonic, center);
            }
            if constexpr (CacheExpansion) {
                // The owner window and owner C use the same unwrapped frame.
                // Export values while they are already in registers for H/dH.
                cached_coefficients[mode] = pair.cosine.value;
                if (harmonic != 0)
                    cached_coefficients[mode + 1] = pair.sine.value;
            }
            for (int local = 0; local < BATCH_TILE && batch_begin + local < batches; ++local) {
                int base = ((batch_begin + local) * owners + owner) * a.basis_count();
                double cosine = z[base + mode];
                result.value[local] += pair.cosine.value * cosine;
                if (Derivatives) {
                    result.center[local] += pair.cosine.center * cosine;
                    result.sigma[local] += pair.cosine.sigma * cosine;
                }
                if (harmonic != 0) {
                    double sine = z[base + mode + 1];
                    result.value[local] += pair.sine.value * sine;
                    if (Derivatives) {
                        result.center[local] += pair.sine.center * sine;
                        result.sigma[local] += pair.sine.sigma * sine;
                    }
                }
            }
        }
        return result;
    }
    int window_first = p.first_tile * a.tile_sites;
    int window_end = window_first + a.period_tiles * a.tile_sites;
    for (int image = -1; image <= 1; ++image) {
        int first = max(window_first, image * a.sites) - image * a.sites;
        int end = min(window_end, (image + 1) * a.sites) - image * a.sites;
        if (first >= end)
            continue;
        double center = p.center - image * a.chart_period();
        HarmonicPhase phase(p, center);
        for (int harmonic = 0; harmonic <= a.modes; ++harmonic) {
            int mode = harmonic == 0 ? 0 : 2 * harmonic - 1;
            CoefficientPair pair;
            if (a.phase_recurrence) {
                phase.advance(harmonic);
                pair =
                    coefficient_pair_from_phase<Derivatives>(p, harmonic, phase.cosine, phase.sine);
            } else {
                pair = coefficient_pair<Derivatives>(p, harmonic, center);
            }
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
                    result.value[local] += pair.cosine.value * moment;
                    if (Derivatives) {
                        result.center[local] += pair.cosine.center * moment;
                        result.sigma[local] += pair.cosine.sigma * moment;
                    }
                    if (harmonic != 0) {
                        double sine_moment =
                            z[((batch_begin + local) * a.nodes + node) * a.basis_count() + mode +
                              1];
                        result.value[local] += pair.sine.value * sine_moment;
                        if (Derivatives) {
                            result.center[local] += pair.sine.center * sine_moment;
                            result.sigma[local] += pair.sine.sigma * sine_moment;
                        }
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
template <bool Derivatives, typename scalar_t, typename moment_t, typename result_t>
__global__ void project_atoms_kernel(const scalar_t *signal, const scalar_t *atoms,
                                     const int64_t *rows, const moment_t *z, result_t *h,
                                     int batches, int atom_count, Axis a, int center_column,
                                     double *sigma_projection, double *center_projection) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= atom_count)
        return;
    int original = rows[slot];
    Profile p = make_profile(a, atoms[4 * original + 1], atoms[4 * original + center_column]);
    int batch_begin = blockIdx.y * BATCH_TILE;
    Projection result = project_atom<Derivatives>(signal, z, batches, batch_begin, p);
    for (int local = 0; local < BATCH_TILE && batch_begin + local < batches; ++local) {
        int offset = (batch_begin + local) * atom_count + slot;
        h[offset] = result.value[local];
        if (Derivatives) {
            sigma_projection[offset] = result.sigma[local];
            center_projection[offset] = result.center[local];
        }
    }
}

// Return the position of a chart tile in an atom's unwrapped retained window.
// The same integer test is used both before heavy preparation and before C.
__device__ int retained_tile_offset(Axis a, int target_tile, int first_tile) {
    int offset = (target_tile - first_tile) % (a.sites / a.tile_sites);
    return offset < 0 ? offset + a.sites / a.tile_sites : offset;
}

__device__ bool block_uses_atom(Axis a, int first_tile, int first_profile_tile,
                                int profile_tile_count) {
    for (int tile = 0; tile < profile_tile_count; ++tile) {
        if (retained_tile_offset(a, first_profile_tile + tile, first_tile) < a.period_tiles)
            return true;
    }
    return false;
}

// One thread prepares one atom into the CURRENT chunk's shared table.
// For a single profile tile, cache both phased components once and reuse them
// across batches. For multiple tiles, cache only raw harmonics: each tile may
// need a different chart image, so its phase must be evaluated separately.
// These two layouts share the same allocation; there is no second coefficient table.
__device__ void cache_expansion_coefficients(const Profile &p, int first_profile_tile,
                                             int profile_tile_count, double *coefficients) {
    Axis a = p.axis;
    coefficients[0] = 1 / a.profile_period();
    int offset = retained_tile_offset(a, first_profile_tile, p.first_tile);
    int image = (p.first_tile + offset - first_profile_tile) / (a.sites / a.tile_sites);
    double center = p.center - image * a.chart_period();
    HarmonicPhase phase(p, center);
    for (int harmonic = 1; harmonic <= a.modes; ++harmonic) {
        double raw = raw_coefficient(p, harmonic).value;
        double normalized;
        if (a.polynomial_table)
            normalized = raw * p.normalized_scale;
        else
            normalized = 2 * raw / (a.profile_period() * p.c0);
        if (profile_tile_count == 1) {
            double sin_value, cos_value;
            if (a.phase_recurrence) {
                phase.advance(harmonic);
                sin_value = phase.sine;
                cos_value = phase.cosine;
            } else {
                sincos(mode_frequency(a, harmonic) * center, &sin_value, &cos_value);
            }
            coefficients[2 * harmonic - 1] = normalized * cos_value;
            coefficients[2 * harmonic] = normalized * sin_value;
        } else {
            coefficients[harmonic] = normalized;
        }
    }
}

// Y/dX ownership has two independent axes: a site range and an atom-chunk range.
// Profile tiles, periods and selected windows are unchanged by either setting.
// Within a split, C lives in registers, then shared; all owned sites use it.
// Splits write disjoint partial outputs, which the next kernel sums without atomics.
// Every thread reaches all barriers, including threads without a C or output site.
template <typename scalar_t, typename moment_t>
__global__ void expand_tiles_kernel(const moment_t *h, const scalar_t *atoms, const int64_t *rows,
                                    scalar_t *output, int batches, int atom_count, Axis a,
                                    int center_column, int block_width, int atom_splits) {
    __shared__ double shared_center[ATOM_CHUNK];
    __shared__ int shared_first_tile[ATOM_CHUNK];
    __shared__ double shared_amplitude[ATOM_CHUNK];
    __shared__ double shared_h[BATCH_TILE * ATOM_CHUNK];
    __shared__ double shared_coefficients[ATOM_CHUNK * MAX_BASIS];
    __shared__ double shared_c[MAX_PROFILE_TILES_PER_BLOCK * BATCH_TILE * MAX_BASIS];

    // The host passes exactly the same width it used to construct the grid.
    int block_first = blockIdx.x * block_width;
    int block_end = min(block_first + block_width, a.sites);
    int first_profile_tile = block_first / a.tile_sites;
    int profile_tile_count = (block_end - 1) / a.tile_sites - first_profile_tile + 1;
    int c_count = profile_tile_count * BATCH_TILE * a.basis_count();
    double local_c[COEFFICIENT_SLOTS] = {0};

    // Split on whole chunks so every atom is owned exactly once for this range.
    int total_chunks = (atom_count + ATOM_CHUNK - 1) / ATOM_CHUNK;
    int first_atom = (total_chunks * blockIdx.z / atom_splits) * ATOM_CHUNK;
    int end_atom = min(atom_count, (total_chunks * (blockIdx.z + 1) / atom_splits) * ATOM_CHUNK);

    for (int base = first_atom; base < end_atom; base += ATOM_CHUNK) {
        int count = min(ATOM_CHUNK, end_atom - base);
        if (threadIdx.x < count) {
            int j = threadIdx.x;
            int original = rows[base + j];
            double center = atoms[4 * original + center_column];
            double canonical_center = wrap_center(center, a);
            int owner = static_cast<int>(
                floor((canonical_center + a.spacing / 2) / (a.tile_sites * a.spacing)));
            int first_tile = owner - a.period_tiles / 2;
            shared_center[j] = canonical_center;
            shared_first_tile[j] = first_tile;
            shared_amplitude[j] = atoms[4 * original];

            // Use only the existing whole-tile window. No support/cell clipping.
            if (block_uses_atom(a, first_tile, first_profile_tile, profile_tile_count)) {
                Profile p = make_profile(a, atoms[4 * original + 1], center);
                cache_expansion_coefficients(p, first_profile_tile, profile_tile_count,
                                             shared_coefficients + j * MAX_BASIS);
            }
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
                int offset = retained_tile_offset(a, target_tile, shared_first_tile[j]);
                if (offset >= a.period_tiles)
                    continue; // Entire tile is outside this atom's retained window.
                int image = (shared_first_tile[j] + offset - target_tile) / chart_tiles;
                double center = shared_center[j] - image * a.chart_period();
                double value;
                if (profile_tile_count == 1) {
                    value = shared_coefficients[j * MAX_BASIS + mode];
                } else if (mode == 0) {
                    value = 1 / a.profile_period();
                } else {
                    int k = (mode + 1) / 2;
                    double phase = mode_frequency(a, k) * center;
                    value = shared_coefficients[j * MAX_BASIS + k] *
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

    // The output tile can exceed the thread count. Each thread expands several
    // independent (batch,site) positions while reusing the same shared C.
    for (int q = threadIdx.x; q < BATCH_TILE * block_width; q += blockDim.x) {
        int local_batch = q / block_width;
        int local_site = q % block_width;
        int site = block_first + local_site;
        int batch = blockIdx.y * BATCH_TILE + local_batch;
        if (site < a.sites && batch < batches) {
            int local_profile_tile = site / a.tile_sites - first_profile_tile;
            int c_begin = (local_profile_tile * BATCH_TILE + local_batch) * a.basis_count();
            double accumulator = 0;
            for (int mode = 0; mode < a.basis_count(); ++mode)
                accumulator += shared_c[c_begin + mode] * integrated_basis(a, mode, site);
            output[(blockIdx.z * batches + batch) * a.sites + site] = accumulator;
        }
    }
}

template <bool WriteG, typename scalar_t, typename moment_t>
__global__ void parameter_grad_kernel(const scalar_t *x, const scalar_t *dy, const scalar_t *atoms,
                                      const int64_t *rows, const moment_t *zx, const moment_t *zy,
                                      scalar_t *g_output, scalar_t *grad, int batches,
                                      int atom_count, Axis input_axis, Axis output_axis) {
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
            // Reprojection already produced these double values. Avoid loading
            // rounded H/G again when combining their parameter pullbacks.
            double hv = input_vjp.value[local];
            double gv = output_vjp.value[local];
            if (WriteG)
                g_output[(begin + local) * atom_count + slot] = gv;
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

// Cached derivatives are projected sums, NOT per-site or per-mode Jacobians.
// One thread combines all batches for its atom; no Z/profile work remains here.
template <typename scalar_t, typename moment_t>
__global__ void cached_parameter_grad_kernel(const scalar_t *atoms, const int64_t *rows,
                                             const moment_t *h, const moment_t *g,
                                             const double *h_sigma, const double *h_center,
                                             const double *g_sigma, const double *g_center,
                                             scalar_t *grad, int batches, int atom_count) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= atom_count)
        return;
    int original = rows[slot];
    double amplitude = atoms[4 * original];
    double da = 0, ds = 0, dmu_in = 0, dmu_out = 0;
    for (int batch = 0; batch < batches; ++batch) {
        int offset = batch * atom_count + slot;
        double hv = h[offset], gv = g[offset];
        da += hv * gv;
        ds += amplitude * (gv * h_sigma[offset] + hv * g_sigma[offset]);
        dmu_in += amplitude * gv * h_center[offset];
        dmu_out += amplitude * hv * g_center[offset];
    }
    grad[4 * original] = da;
    grad[4 * original + 1] = ds;
    grad[4 * original + 2] = dmu_out;
    grad[4 * original + 3] = dmu_in;
}

std::vector<torch::Tensor> project(torch::Tensor signal, torch::Tensor atoms, torch::Tensor rows,
                                   torch::Tensor z, std::vector<double> config, int center_column,
                                   int atom_threads, bool derivatives) {
    check_signal(signal);
    check_atom_thread_count(atom_threads);
    Axis a = parse_axis(config);
    TORCH_CHECK(signal.size(1) == a.sites, "signal shape differs from axis");
    check_atoms(atoms, rows, signal);
    a = check_z(z, signal, a);
    c10::cuda::CUDAGuard guard(signal.device());
    auto value_options = derivatives ? z.options() : signal.options();
    auto result = torch::empty({signal.size(0), atoms.size(0)}, value_options);
    auto sigma = derivatives ? torch::empty(result.sizes(), signal.options().dtype(torch::kFloat64))
                             : torch::Tensor();
    auto center = derivatives ? torch::empty_like(sigma) : torch::Tensor();
    if (atoms.size(0) == 0 || signal.size(0) == 0)
        return {result, sigma, center};
    dim3 grid((atoms.size(0) + atom_threads - 1) / atom_threads,
              (signal.size(0) + BATCH_TILE - 1) / BATCH_TILE);
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(signal.scalar_type(), "cell_mass_project", [&] {
        auto launch = [&](auto *moments, auto *values) {
            if (derivatives) {
                project_atoms_kernel<true><<<grid, atom_threads, 0, stream>>>(
                    signal.data_ptr<scalar_t>(), atoms.data_ptr<scalar_t>(),
                    rows.data_ptr<int64_t>(), moments, values, signal.size(0), atoms.size(0), a,
                    center_column, sigma.data_ptr<double>(), center.data_ptr<double>());
            } else {
                project_atoms_kernel<false><<<grid, atom_threads, 0, stream>>>(
                    signal.data_ptr<scalar_t>(), atoms.data_ptr<scalar_t>(),
                    rows.data_ptr<int64_t>(), moments, values, signal.size(0), atoms.size(0), a,
                    center_column, nullptr, nullptr);
            }
        };
        if (z.scalar_type() == torch::kFloat64) {
            if (derivatives)
                launch(z.data_ptr<double>(), result.data_ptr<double>());
            else
                launch(z.data_ptr<double>(), result.data_ptr<scalar_t>());
        } else
            launch(z.data_ptr<scalar_t>(), result.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {result, sigma, center};
}

// Each thread owns one final (batch,site), after all split producers on this
// stream finish. Only partial OUTPUTS are global; C remains local to its block.
template <typename scalar_t>
__global__ void sum_split_outputs_kernel(const scalar_t *partials, scalar_t *output, int elements,
                                         int atom_splits) {
    int element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= elements)
        return;
    double sum = 0;
    for (int split = 0; split < atom_splits; ++split)
        sum += static_cast<double>(partials[split * elements + element]);
    output[element] = sum;
}

torch::Tensor expand(torch::Tensor h, torch::Tensor atoms, torch::Tensor rows,
                     std::vector<double> config, int center_column, int sites_per_block,
                     int requested_atom_splits) {
    TORCH_CHECK(sites_per_block >= 8 && sites_per_block <= 256 &&
                    (sites_per_block & (sites_per_block - 1)) == 0,
                "sites_per_block must be a power of two in [8,256]");
    TORCH_CHECK(requested_atom_splits >= 1 && requested_atom_splits <= 64 &&
                    (requested_atom_splits & (requested_atom_splits - 1)) == 0,
                "atom_splits must be a power of two in [1,64]");
    check_signal(h);
    Axis a = parse_axis(config);
    check_signal(atoms);
    check_atoms(atoms, rows, atoms);
    TORCH_CHECK(h.device() == atoms.device() &&
                    (h.scalar_type() == atoms.scalar_type() || h.scalar_type() == torch::kFloat64),
                "incompatible projection values");
    TORCH_CHECK(h.size(1) == atoms.size(0), "H/G shape differs from atom count");
    c10::cuda::CUDAGuard guard(h.device());
    auto result = torch::empty({h.size(0), a.sites}, atoms.options());
    if (h.size(0) == 0)
        return result;

    // Avoid empty producers for small/empty atom sets. The one-split path writes
    // directly into result and allocates no partial buffer or reduction kernel.
    int total_chunks = (atoms.size(0) + ATOM_CHUNK - 1) / ATOM_CHUNK;
    int atom_splits = std::max(1, std::min(requested_atom_splits, total_chunks));
    auto partials = atom_splits == 1
                        ? result
                        : torch::empty({atom_splits, h.size(0), a.sites}, atoms.options());
    int block_width =
        std::min(a.sites, std::min(sites_per_block, MAX_PROFILE_TILES_PER_BLOCK * a.tile_sites));
    dim3 grid((a.sites + block_width - 1) / block_width, (h.size(0) + BATCH_TILE - 1) / BATCH_TILE,
              atom_splits);
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "cell_mass_expand", [&] {
        auto launch = [&](auto *values) {
            expand_tiles_kernel<<<grid, THREADS, 0, stream>>>(
                values, atoms.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(),
                partials.data_ptr<scalar_t>(), h.size(0), atoms.size(0), a, center_column,
                block_width, atom_splits);
        };
        if (h.scalar_type() == torch::kFloat64)
            launch(h.data_ptr<double>());
        else
            launch(h.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (atom_splits > 1) {
            int elements = h.size(0) * a.sites;
            sum_split_outputs_kernel<<<(elements + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
                partials.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), elements, atom_splits);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });
    return result;
}
// Group offsets describe lexicographic (output owner,input owner) ranges in
// sorted_rows. Each atom contributes to ONE owner C, rather than rebuilding its
// coefficients separately in every tile touched by its retained window.
template <int CThreads, typename scalar_t, typename moment_t>
__global__ void owner_coefficients_kernel(const moment_t *h, const scalar_t *atoms,
                                          const int64_t *rows, const int64_t *offsets,
                                          double *partials, int batches, int atom_count, Axis a,
                                          int center_column, int input_tiles, int output_tiles,
                                          int atom_splits) {
    // Allocate only this mode count. All C owners share a 32-atom chunk;
    // coefficient preparation assigns one distinct atom to each of 32 threads.
    extern __shared__ double workspace[];
    double *amplitudes = workspace;
    double *values = amplitudes + ATOM_CHUNK;
    double *coefficients = values + BATCH_TILE * ATOM_CHUNK;
    constexpr int slots = (BATCH_TILE * MAX_BASIS + CThreads - 1) / CThreads;
    double local_c[slots] = {0};
    int owner = blockIdx.x;
    int owner_count = a.sites / a.tile_sites;
    int segments = center_column == 2 ? 1 : output_tiles;
    int total_chunks = 0;
    for (int segment = 0; segment < segments; ++segment) {
        int first_group = center_column == 2 ? owner * input_tiles : segment * input_tiles + owner;
        int last_group = center_column == 2 ? (owner + 1) * input_tiles : first_group + 1;
        total_chunks += (offsets[last_group] - offsets[first_group] + ATOM_CHUNK - 1) / ATOM_CHUNK;
    }
    int assigned_first = total_chunks * blockIdx.z / atom_splits;
    int assigned_last = total_chunks * (blockIdx.z + 1) / atom_splits;
    int chunk_prefix = 0;

    for (int segment = 0; segment < segments; ++segment) {
        // Output owners are contiguous. An input owner is a sequence of one
        // contiguous interval per output owner; no second atom copy is needed.
        int first_group = center_column == 2 ? owner * input_tiles : segment * input_tiles + owner;
        int last_group = center_column == 2 ? (owner + 1) * input_tiles : first_group + 1;
        int begin = offsets[first_group];
        int end = offsets[last_group];
        int chunks = (end - begin + ATOM_CHUNK - 1) / ATOM_CHUNK;
        int first_chunk = max(0, assigned_first - chunk_prefix);
        int last_chunk = min(chunks, assigned_last - chunk_prefix);
        chunk_prefix += chunks;
        if (first_chunk >= last_chunk)
            continue; // Uniform across this block; no thread skips a barrier.
        int first = begin + first_chunk * ATOM_CHUNK;
        int last = min(end, begin + last_chunk * ATOM_CHUNK);

        for (int base = first; base < last; base += ATOM_CHUNK) {
            int count = min(ATOM_CHUNK, last - base);
            if (threadIdx.x < count) {
                int j = threadIdx.x;
                int original = rows[base + j];
                Profile profile =
                    make_profile(a, atoms[4 * original + 1], atoms[4 * original + center_column]);
                amplitudes[j] = atoms[4 * original];
                // All atoms in this range have the same owner. The coefficients
                // use that owner's unwrapped chart image and are shared by B.
                cache_expansion_coefficients(profile, owner, 1, coefficients + j * a.basis_count());
            }
            for (int q = threadIdx.x; q < BATCH_TILE * count; q += blockDim.x) {
                int local_batch = q / count;
                int j = q % count;
                int batch = blockIdx.y * BATCH_TILE + local_batch;
                values[local_batch * ATOM_CHUNK + j] =
                    batch < batches ? h[batch * atom_count + base + j] : 0;
            }
            __syncthreads();
            for (int v = 0; v < slots; ++v) {
                int q = threadIdx.x + v * blockDim.x;
                if (q >= BATCH_TILE * a.basis_count())
                    continue;
                int mode = q % a.basis_count();
                int local_batch = q / a.basis_count();
                for (int j = 0; j < count; ++j)
                    local_c[v] += amplitudes[j] * values[local_batch * ATOM_CHUNK + j] *
                                  coefficients[j * a.basis_count() + mode];
            }
            __syncthreads();
        }
    }
    for (int v = 0; v < slots; ++v) {
        int q = threadIdx.x + v * blockDim.x;
        if (q >= BATCH_TILE * a.basis_count())
            continue;
        int mode = q % a.basis_count();
        int batch = blockIdx.y * BATCH_TILE + q / a.basis_count();
        if (batch < batches)
            partials[((blockIdx.z * batches + batch) * owner_count + owner) * a.basis_count() +
                     mode] = local_c[v];
    }
}

// Each block owns an input-owner/split range. For B<=4, every atom appears
// in exactly one block, so parameter gradients have a unique writer.
// The same input coefficients serve H/dH and C_in; no global G is produced.
template <typename scalar_t, typename moment_t>
__global__ void
backward_owner_partials_kernel(const scalar_t *x, const scalar_t *dy, const scalar_t *atoms,
                               const int64_t *rows, const int64_t *offsets, const moment_t *zx,
                               const moment_t *zy, double *partials, scalar_t *grad, int batches,
                               int atom_count, Axis input, Axis output, int atom_splits) {
    int input_owners = input.sites / input.tile_sites;
    int output_owners = output.sites / output.tile_sites;
    int modes = input.basis_count();
    int owner = blockIdx.x;

    extern __shared__ double workspace[];
    double *amplitudes = workspace;
    double *values = amplitudes + ATOM_CHUNK;
    double *coefficients = values + BATCH_TILE * ATOM_CHUNK;
    double *c = coefficients + ATOM_CHUNK * modes;
    // Keep C across chunks in shared, rather than keeping a C register array
    // live through the much larger two-axis parameter projection.
    for (int q = threadIdx.x; q < BATCH_TILE * modes; q += blockDim.x)
        c[q] = 0;
    __syncthreads();

    int total_chunks = 0;
    for (int segment = 0; segment < output_owners; ++segment) {
        int group = segment * input_owners + owner;
        total_chunks += (offsets[group + 1] - offsets[group] + ATOM_CHUNK - 1) / ATOM_CHUNK;
    }
    int assigned_first = total_chunks * blockIdx.z / atom_splits;
    int assigned_last = total_chunks * (blockIdx.z + 1) / atom_splits;
    int chunk_prefix = 0;
    for (int segment = 0; segment < output_owners; ++segment) {
        int group = segment * input_owners + owner;
        int begin = offsets[group], end = offsets[group + 1];
        int chunks = (end - begin + ATOM_CHUNK - 1) / ATOM_CHUNK;
        int first_chunk = max(0, assigned_first - chunk_prefix);
        int last_chunk = min(chunks, assigned_last - chunk_prefix);
        chunk_prefix += chunks;
        if (first_chunk >= last_chunk)
            continue; // Uniform across the block; all barriers remain collective.
        int first = begin + first_chunk * ATOM_CHUNK;
        int last = min(end, begin + last_chunk * ATOM_CHUNK);
        for (int base = first; base < last; base += ATOM_CHUNK) {
            int count = min(ATOM_CHUNK, last - base);
            if (threadIdx.x < count) {
                int j = threadIdx.x;
                int original = rows[base + j];
                double amplitude = atoms[4 * original], sigma = atoms[4 * original + 1];
                amplitudes[j] = amplitude;
                Projection h;
                {
                    Profile profile = make_profile(input, sigma, atoms[4 * original + 3]);
                    h = project_atom<true, true>(x, zx, batches, 0, profile,
                                                 coefficients + j * modes);
                }
                Projection g;
                {
                    Profile profile = make_profile(output, sigma, atoms[4 * original + 2]);
                    g = project_atom<true>(dy, zy, batches, 0, profile);
                }
                double da = 0, ds = 0, dmu_out = 0, dmu_in = 0;
                for (int batch = 0; batch < BATCH_TILE; ++batch) {
                    // Existing dX consumes input-dtype G. Preserve that cast
                    // here, while parameter pullbacks keep the double value.
                    values[batch * ATOM_CHUNK + j] =
                        batch < batches ? static_cast<scalar_t>(g.value[batch]) : 0;
                    if (batch < batches) {
                        double hv = h.value[batch], gv = g.value[batch];
                        da += hv * gv;
                        ds += amplitude * (gv * h.sigma[batch] + hv * g.sigma[batch]);
                        dmu_in += amplitude * gv * h.center[batch];
                        dmu_out += amplitude * hv * g.center[batch];
                    }
                }
                grad[4 * original] = da;
                grad[4 * original + 1] = ds;
                grad[4 * original + 2] = dmu_out;
                grad[4 * original + 3] = dmu_in;
            }
            __syncthreads(); // Publish this chunk's G and input coefficients.
            for (int q = threadIdx.x; q < BATCH_TILE * modes; q += blockDim.x) {
                int mode = q % modes, batch = q / modes;
                double sum = c[q];
                for (int j = 0; j < count; ++j)
                    sum += amplitudes[j] * values[batch * ATOM_CHUNK + j] *
                           coefficients[j * modes + mode];
                c[q] = sum;
            }
            __syncthreads(); // All consumers finish before reusing the chunk.
        }
    }
    for (int q = threadIdx.x; q < batches * modes; q += blockDim.x) {
        int mode = q % modes, batch = q / modes;
        partials[((blockIdx.z * batches + batch) * input_owners + owner) * modes + mode] = c[q];
    }
}

// Unique output owner: combine the neighbouring owner C values, rotate only
// chart-seam images, then expand each site's basis ONCE for all four batches.
template <typename scalar_t>
__global__ void sum_owner_coefficients_kernel(const double *partials, scalar_t *output, int batches,
                                              Axis a, int atom_splits, int block_width) {
    __shared__ double c[BATCH_TILE * MAX_BASIS];
    int first_site = blockIdx.x * block_width;
    int target = first_site / a.tile_sites;
    int owners = a.sites / a.tile_sites;
    int basis_count = a.basis_count();
    int half_window = a.period_tiles / 2;
    for (int q = threadIdx.x; q < BATCH_TILE * basis_count; q += blockDim.x) {
        int mode = q % basis_count;
        int local_batch = q / basis_count;
        int batch = blockIdx.y * BATCH_TILE + local_batch;
        double total = 0;
        if (batch < batches) {
            for (int neighbour = -half_window; neighbour <= half_window; ++neighbour) {
                int owner = (target + neighbour + owners) % owners;
                int offset = (target - (owner - half_window) + owners) % owners;
                int image = (owner - half_window + offset - target) / owners;
                double primary = 0, paired = 0;
                int pair_mode = mode % 2 == 1 ? mode + 1 : mode - 1;
                for (int split = 0; split < atom_splits; ++split) {
                    int base = ((split * batches + batch) * owners + owner) * basis_count;
                    primary += partials[base + mode];
                    if (image != 0 && mode != 0)
                        paired += partials[base + pair_mode];
                }
                if (image == 0 || mode == 0) {
                    total += primary;
                } else {
                    int harmonic = (mode + 1) / 2;
                    double sine, cosine;
                    double angle =
                        a.phase_recurrence
                            ? mode_frequency(a, harmonic) * image * a.chart_period()
                            : 2 * PI * harmonic * image * a.chart_period() / a.profile_period();
                    sincos(angle, &sine, &cosine);
                    total += mode % 2 == 1 ? primary * cosine + paired * sine
                                           : primary * cosine - paired * sine;
                }
            }
        }
        c[q] = total;
    }
    __syncthreads();
    for (int local_site = threadIdx.x; local_site < block_width; local_site += blockDim.x) {
        int site = first_site + local_site;
        double result[BATCH_TILE] = {0};
        for (int mode = 0; mode < basis_count; ++mode) {
            double basis = integrated_basis(a, mode, site);
            for (int local_batch = 0; local_batch < BATCH_TILE; ++local_batch)
                result[local_batch] += c[local_batch * basis_count + mode] * basis;
        }
        for (int local_batch = 0; local_batch < BATCH_TILE; ++local_batch) {
            int batch = blockIdx.y * BATCH_TILE + local_batch;
            if (batch < batches)
                output[batch * a.sites + site] = result[local_batch];
        }
    }
}

// Return bounded [split,batch,owner,mode] scratch. This stage is also a
// reusable boundary for timing and for the backward projection/C fusion.
torch::Tensor make_owner_partials(torch::Tensor h, torch::Tensor atoms, torch::Tensor rows,
                                  torch::Tensor offsets, std::vector<double> input_config,
                                  std::vector<double> output_config, int center_column,
                                  int requested_splits, int c_threads) {
    check_signal(h);
    check_signal(atoms);
    check_atoms(atoms, rows, atoms);
    TORCH_CHECK(h.device() == atoms.device() && h.size(1) == atoms.size(0) &&
                    (h.scalar_type() == atoms.scalar_type() || h.scalar_type() == torch::kFloat64),
                "incompatible owner projection values");
    TORCH_CHECK(center_column == 2 || center_column == 3, "invalid projection axis");
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    Axis a = center_column == 2 ? output : input;
    int input_tiles = input.sites / input.tile_sites;
    int output_tiles = output.sites / output.tile_sites;
    TORCH_CHECK(offsets.device() == atoms.device() && offsets.is_contiguous() &&
                    offsets.scalar_type() == torch::kInt64 && offsets.dim() == 1 &&
                    offsets.numel() == input_tiles * output_tiles + 1,
                "invalid owner group offsets");
    TORCH_CHECK(requested_splits >= 1 && requested_splits <= 128 &&
                    (requested_splits & (requested_splits - 1)) == 0,
                "owner atom_splits must be a power of two in [1,128]");
    TORCH_CHECK(c_threads == 32 || c_threads == 64 || c_threads == 128,
                "C producer threads must be 32, 64 or 128");
    c10::cuda::CUDAGuard guard(atoms.device());
    int chunks = (atoms.size(0) + ATOM_CHUNK - 1) / ATOM_CHUNK;
    int splits = std::max(1, std::min(requested_splits, chunks));
    auto partials = torch::empty({splits, h.size(0), a.sites / a.tile_sites, a.basis_count()},
                                 h.options().dtype(torch::kFloat64));
    if (h.size(0) == 0)
        return partials;
    auto stream = c10::cuda::getCurrentCUDAStream();
    dim3 producer(a.sites / a.tile_sites, (h.size(0) + BATCH_TILE - 1) / BATCH_TILE, splits);
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "cell_mass_owner_coefficients", [&] {
        auto launch = [&](auto *values) {
            size_t shared_bytes =
                (ATOM_CHUNK + BATCH_TILE * ATOM_CHUNK + ATOM_CHUNK * a.basis_count()) *
                sizeof(double);
            if (c_threads == 32)
                owner_coefficients_kernel<32><<<producer, 32, shared_bytes, stream>>>(
                    values, atoms.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(),
                    offsets.data_ptr<int64_t>(), partials.data_ptr<double>(), h.size(0),
                    atoms.size(0), a, center_column, input_tiles, output_tiles, splits);
            else if (c_threads == 64)
                owner_coefficients_kernel<64><<<producer, 64, shared_bytes, stream>>>(
                    values, atoms.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(),
                    offsets.data_ptr<int64_t>(), partials.data_ptr<double>(), h.size(0),
                    atoms.size(0), a, center_column, input_tiles, output_tiles, splits);
            else
                owner_coefficients_kernel<128><<<producer, 128, shared_bytes, stream>>>(
                    values, atoms.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(),
                    offsets.data_ptr<int64_t>(), partials.data_ptr<double>(), h.size(0),
                    atoms.size(0), a, center_column, input_tiles, output_tiles, splits);
        };
        if (h.scalar_type() == torch::kFloat64)
            launch(h.data_ptr<double>());
        else
            launch(h.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return partials;
}

// Combine split/neighbor C values, then expand sites. No atom/profile work is
// performed here. Each output element has a unique writer.
torch::Tensor expand_owner_partials_impl(torch::Tensor partials, torch::Tensor like,
                                         std::vector<double> config, int sites_per_block) {
    check_signal(like);
    Axis a = parse_axis(config);
    TORCH_CHECK(partials.device() == like.device() && partials.is_contiguous() &&
                    partials.scalar_type() == torch::kFloat64 && partials.dim() == 4 &&
                    partials.size(0) >= 1 && partials.size(0) <= 128 &&
                    partials.size(2) == a.sites / a.tile_sites &&
                    partials.size(3) == a.basis_count(),
                "invalid owner C partials");
    TORCH_CHECK(sites_per_block >= 8 && sites_per_block <= 256 &&
                    (sites_per_block & (sites_per_block - 1)) == 0,
                "sites_per_block must be a power of two in [8,256]");
    c10::cuda::CUDAGuard guard(like.device());
    auto result = torch::empty({partials.size(1), a.sites}, like.options());
    if (partials.size(1) == 0)
        return result;
    int width = std::min(a.sites, std::min(sites_per_block, a.tile_sites));
    dim3 grid(a.sites / width, (partials.size(1) + BATCH_TILE - 1) / BATCH_TILE);
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(like.scalar_type(), "cell_mass_expand_owner_partials", [&] {
        sum_owner_coefficients_kernel<<<grid, THREADS, 0, stream>>>(
            partials.data_ptr<double>(), result.data_ptr<scalar_t>(), partials.size(1), a,
            partials.size(0), width);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

torch::Tensor expand_owner_coefficients(torch::Tensor h, torch::Tensor atoms, torch::Tensor rows,
                                        torch::Tensor offsets, std::vector<double> input_config,
                                        std::vector<double> output_config, int center_column,
                                        int sites_per_block, int requested_splits, int c_threads) {
    auto partials = make_owner_partials(h, atoms, rows, offsets, input_config, output_config,
                                        center_column, requested_splits, c_threads);
    return expand_owner_partials_impl(
        partials, atoms, center_column == 2 ? output_config : input_config, sites_per_block);
}

} // namespace

std::vector<torch::Tensor> prepare_inputs(torch::Tensor x, torch::Tensor atoms,
                                          std::vector<double> input_config,
                                          std::vector<double> output_config, bool wide_keys) {
    check_signal(x);
    check_like(atoms, x);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(x.size(1) == input.sites, "signal shape differs from axis");
    TORCH_CHECK(atoms.dim() == 2 && atoms.size(1) == 4 && atoms.size(0) <= 65536,
                "prototype physical atoms require shape [A<=65536,4]");
    c10::cuda::CUDAGuard guard(x.device());
    auto key_dtype = wide_keys ? torch::kInt64 : torch::kInt32;
    auto keys = torch::empty({atoms.size(0)}, x.options().dtype(key_dtype));
    auto errors = torch::zeros({}, x.options().dtype(torch::kInt32));
    // Match the declaration's tile-width multiplication order for the bound.
    double maximum_sigma = std::min(input.period_tiles * (input.tile_sites * input.spacing),
                                    output.period_tiles * (output.tile_sites * output.spacing)) /
                           3;
    int64_t elements = std::max(x.numel(), atoms.size(0));
    int blocks = static_cast<int>(std::max<int64_t>(1, (elements + THREADS - 1) / THREADS));
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "cell_mass_prepare_inputs", [&] {
        auto launch = [&](auto *key_data) {
            prepare_inputs_kernel<<<blocks, THREADS, 0, stream>>>(
                x.data_ptr<scalar_t>(), atoms.data_ptr<scalar_t>(), key_data,
                errors.data_ptr<int>(), x.numel(), atoms.size(0), input, output, maximum_sigma);
        };
        if (wide_keys)
            launch(keys.data_ptr<int64_t>());
        else
            launch(keys.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {keys, errors};
}

torch::Tensor build_z(torch::Tensor signal, std::vector<double> config, bool hierarchical,
                      int team_size, bool double_intermediates, bool owner_windows) {
    check_signal(signal);
    Axis a = parse_axis(config);
    TORCH_CHECK(team_size >= 1 && team_size <= THREADS && (team_size & (team_size - 1)) == 0,
                "Z team size must be a power of two in [1,128]");
    TORCH_CHECK(signal.size(1) == a.sites, "signal shape differs from axis");
    c10::cuda::CUDAGuard guard(signal.device());
    auto options =
        double_intermediates ? signal.options().dtype(torch::kFloat64) : signal.options();
    auto z = torch::empty({signal.size(0), a.nodes, a.basis_count()}, options);
    if (signal.size(0) == 0) {
        if (owner_windows)
            return torch::empty({0, a.sites / a.tile_sites, a.basis_count()}, z.options());
        return z;
    }
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(signal.scalar_type(), "cell_mass_build_z", [&] {
        auto launch = [&](auto *moments) {
            dim3 grid(hierarchical ? a.leaves : a.nodes,
                      (signal.size(0) + BATCH_TILE - 1) / BATCH_TILE);
            build_z_direct_kernel<<<grid, THREADS, 0, stream>>>(
                signal.data_ptr<scalar_t>(), moments, signal.size(0), a, team_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            for (int level = 1; hierarchical && (a.leaves >> level) > 0; ++level) {
                int count = signal.size(0) * (a.leaves >> level) * a.basis_count();
                merge_z_level_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
                    moments, signal.size(0), a, level);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        };
        if (double_intermediates)
            launch(z.data_ptr<double>());
        else
            launch(z.data_ptr<scalar_t>());
    });
    if (owner_windows) {
        auto windows =
            torch::empty({signal.size(0), a.sites / a.tile_sites, a.basis_count()}, z.options());
        int elements = windows.numel();
        AT_DISPATCH_FLOATING_TYPES(z.scalar_type(), "cell_mass_owner_windows", [&] {
            aggregate_owner_windows_kernel<<<(elements + THREADS - 1) / THREADS, THREADS, 0,
                                             stream>>>(
                z.data_ptr<scalar_t>(), windows.data_ptr<scalar_t>(), signal.size(0), a);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return windows;
    }
    return z;
}

torch::Tensor compute_h(torch::Tensor x, torch::Tensor atoms, torch::Tensor rows, torch::Tensor z,
                        std::vector<double> config, int atom_threads) {
    return project(x, atoms, rows, z, config, 3, atom_threads, false)[0];
}
torch::Tensor compute_g(torch::Tensor dy, torch::Tensor atoms, torch::Tensor rows, torch::Tensor z,
                        std::vector<double> config, int atom_threads) {
    return project(dy, atoms, rows, z, config, 2, atom_threads, false)[0];
}

std::vector<torch::Tensor> compute_h_with_derivatives(torch::Tensor x, torch::Tensor atoms,
                                                      torch::Tensor rows, torch::Tensor z,
                                                      std::vector<double> config,
                                                      int atom_threads) {
    return project(x, atoms, rows, z, config, 3, atom_threads, true);
}

std::vector<torch::Tensor> compute_g_with_derivatives(torch::Tensor dy, torch::Tensor atoms,
                                                      torch::Tensor rows, torch::Tensor z,
                                                      std::vector<double> config,
                                                      int atom_threads) {
    return project(dy, atoms, rows, z, config, 2, atom_threads, true);
}

torch::Tensor compute_parameter_grad_from_cache(torch::Tensor atoms, torch::Tensor rows,
                                                torch::Tensor h, torch::Tensor g,
                                                torch::Tensor h_sigma, torch::Tensor h_center,
                                                torch::Tensor g_sigma, torch::Tensor g_center,
                                                int atom_threads) {
    check_signal(h);
    check_like(g, h);
    check_signal(atoms);
    check_atoms(atoms, rows, atoms);
    TORCH_CHECK(h.device() == atoms.device() &&
                    (h.scalar_type() == atoms.scalar_type() || h.scalar_type() == torch::kFloat64),
                "incompatible projection values");
    check_atom_thread_count(atom_threads);
    TORCH_CHECK(h.size(1) == atoms.size(0) && g.sizes() == h.sizes(), "invalid H/G shapes");
    for (const auto &derivative : {h_sigma, h_center, g_sigma, g_center}) {
        TORCH_CHECK(derivative.device() == h.device() && derivative.is_contiguous() &&
                        derivative.scalar_type() == torch::kFloat64 &&
                        derivative.sizes() == h.sizes(),
                    "projection derivatives must be contiguous float64 tensors matching H/G");
    }
    c10::cuda::CUDAGuard guard(h.device());
    auto grad = torch::empty_like(atoms);
    if (atoms.size(0) == 0)
        return grad;
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "cell_mass_cached_parameter_grad", [&] {
        auto launch = [&](auto *hv, auto *gv) {
            cached_parameter_grad_kernel<<<(atoms.size(0) + atom_threads - 1) / atom_threads,
                                           atom_threads, 0, stream>>>(
                atoms.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(), hv, gv,
                h_sigma.data_ptr<double>(), h_center.data_ptr<double>(), g_sigma.data_ptr<double>(),
                g_center.data_ptr<double>(), grad.data_ptr<scalar_t>(), h.size(0), atoms.size(0));
        };
        if (h.scalar_type() == torch::kFloat64)
            launch(h.data_ptr<double>(), g.data_ptr<double>());
        else
            launch(h.data_ptr<scalar_t>(), g.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}
torch::Tensor compute_y(torch::Tensor h, torch::Tensor atoms, torch::Tensor rows,
                        std::vector<double> config, int sites_per_block, int atom_splits) {
    return expand(h, atoms, rows, config, 2, sites_per_block, atom_splits);
}
torch::Tensor compute_dx(torch::Tensor g, torch::Tensor atoms, torch::Tensor rows,
                         std::vector<double> config, int sites_per_block, int atom_splits) {
    return expand(g, atoms, rows, config, 3, sites_per_block, atom_splits);
}

torch::Tensor compute_parameter_grad(torch::Tensor x, torch::Tensor dy, torch::Tensor atoms,
                                     torch::Tensor rows, torch::Tensor h, torch::Tensor g,
                                     torch::Tensor zx, torch::Tensor zy,
                                     std::vector<double> input_config,
                                     std::vector<double> output_config, int atom_threads) {
    check_signal(x);
    check_atom_thread_count(atom_threads);
    check_signal(dy);
    check_like(dy, x);
    check_atoms(atoms, rows, x);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(x.size(0) == dy.size(0) && x.size(1) == input.sites && dy.size(1) == output.sites,
                "input/output shapes differ from axes");
    input = check_z(zx, x, input);
    output = check_z(zy, dy, output);
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
    TORCH_CHECK(zx.scalar_type() == zy.scalar_type(), "hierarchy dtypes must match");
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "cell_mass_parameter_grad", [&] {
        auto launch = [&](auto *input_moments, auto *output_moments) {
            parameter_grad_kernel<false>
                <<<(atoms.size(0) + atom_threads - 1) / atom_threads, atom_threads, 0, stream>>>(
                    x.data_ptr<scalar_t>(), dy.data_ptr<scalar_t>(), atoms.data_ptr<scalar_t>(),
                    rows.data_ptr<int64_t>(), input_moments, output_moments,
                    static_cast<scalar_t *>(nullptr), grad.data_ptr<scalar_t>(), x.size(0),
                    atoms.size(0), input, output);
        };
        if (zx.scalar_type() == torch::kFloat64)
            launch(zx.data_ptr<double>(), zy.data_ptr<double>());
        else
            launch(zx.data_ptr<scalar_t>(), zy.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}

// The same owner representation serves Y (output axis) and dX (input axis).
torch::Tensor compute_y_from_owners(torch::Tensor h, torch::Tensor atoms, torch::Tensor rows,
                                    torch::Tensor offsets, std::vector<double> input_config,
                                    std::vector<double> output_config, int sites_per_block,
                                    int atom_splits, int c_threads) {
    return expand_owner_coefficients(h, atoms, rows, offsets, input_config, output_config, 2,
                                     sites_per_block, atom_splits, c_threads);
}
torch::Tensor compute_dx_from_owners(torch::Tensor g, torch::Tensor atoms, torch::Tensor rows,
                                     torch::Tensor offsets, std::vector<double> input_config,
                                     std::vector<double> output_config, int sites_per_block,
                                     int atom_splits, int c_threads) {
    return expand_owner_coefficients(g, atoms, rows, offsets, input_config, output_config, 3,
                                     sites_per_block, atom_splits, c_threads);
}

// Recomputed parameter pullbacks already contain G. Publish it for dX here,
// avoiding a second output-axis projection and retaining no forward H buffer.
std::vector<torch::Tensor>
compute_g_parameter_grad(torch::Tensor x, torch::Tensor dy, torch::Tensor atoms, torch::Tensor rows,
                         torch::Tensor zx, torch::Tensor zy, std::vector<double> input_config,
                         std::vector<double> output_config, int atom_threads) {
    check_signal(x);
    check_signal(dy);
    check_like(dy, x);
    check_atoms(atoms, rows, x);
    check_atom_thread_count(atom_threads);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(x.size(0) == dy.size(0) && x.size(1) == input.sites && dy.size(1) == output.sites,
                "input/output shapes differ from axes");
    input = check_z(zx, x, input);
    output = check_z(zy, dy, output);
    TORCH_CHECK(zx.scalar_type() == zy.scalar_type(), "hierarchy dtypes must match");
    c10::cuda::CUDAGuard guard(x.device());
    auto g = torch::empty({x.size(0), atoms.size(0)}, x.options());
    auto grad = torch::empty_like(atoms);
    if (atoms.size(0) == 0)
        return {g, grad};
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "cell_mass_fused_g_parameter_grad", [&] {
        auto launch = [&](auto *input_moments, auto *output_moments) {
            parameter_grad_kernel<true>
                <<<(atoms.size(0) + atom_threads - 1) / atom_threads, atom_threads, 0, stream>>>(
                    x.data_ptr<scalar_t>(), dy.data_ptr<scalar_t>(), atoms.data_ptr<scalar_t>(),
                    rows.data_ptr<int64_t>(), input_moments, output_moments, g.data_ptr<scalar_t>(),
                    grad.data_ptr<scalar_t>(), x.size(0), atoms.size(0), input, output);
        };
        if (zx.scalar_type() == torch::kFloat64)
            launch(zx.data_ptr<double>(), zy.data_ptr<double>());
        else
            launch(zx.data_ptr<scalar_t>(), zy.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g, grad};
}

// Integer metadata only: rows remains the sole atom permutation. Int64 keys
// are supported for controlled comparisons; the <=1024-site route fits int32.
torch::Tensor prepare_group_offsets(torch::Tensor keys, torch::Tensor rows, int groups) {
    TORCH_CHECK(keys.is_cuda() && keys.is_contiguous() && keys.dim() == 1 &&
                    (keys.scalar_type() == torch::kInt32 || keys.scalar_type() == torch::kInt64),
                "group keys must be contiguous CUDA int32 or int64");
    TORCH_CHECK(rows.device() == keys.device() && rows.is_contiguous() && rows.dim() == 1 &&
                    rows.scalar_type() == torch::kInt64 && rows.numel() == keys.numel(),
                "invalid sorted group rows");
    TORCH_CHECK(groups >= 1 && groups <= 1024 * 1024, "invalid group count");
    c10::cuda::CUDAGuard guard(keys.device());
    auto offsets = torch::empty({groups + 1}, rows.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    int blocks = (groups + 1 + THREADS - 1) / THREADS;
    if (keys.scalar_type() == torch::kInt32)
        group_offsets_kernel<<<blocks, THREADS, 0, stream>>>(
            keys.data_ptr<int32_t>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
            keys.numel(), groups);
    else
        group_offsets_kernel<<<blocks, THREADS, 0, stream>>>(
            keys.data_ptr<int64_t>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
            keys.numel(), groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return offsets;
}

// Private native boundaries used by split-stage diagnostics and the fused path.
torch::Tensor compute_owner_partials(torch::Tensor h, torch::Tensor atoms, torch::Tensor rows,
                                     torch::Tensor offsets, std::vector<double> input_config,
                                     std::vector<double> output_config, int center_column,
                                     int atom_splits, int c_threads) {
    return make_owner_partials(h, atoms, rows, offsets, input_config, output_config, center_column,
                               atom_splits, c_threads);
}
torch::Tensor expand_owner_partials(torch::Tensor partials, torch::Tensor like,
                                    std::vector<double> config, int sites_per_block) {
    return expand_owner_partials_impl(partials, like, config, sites_per_block);
}

// B<=4 keeps all batch contributions for an atom in its sole owner block.
// Return only C_in partials and original-order atom gradients, never a G table.
std::vector<torch::Tensor> compute_g_parameter_grad_cin(
    torch::Tensor x, torch::Tensor dy, torch::Tensor atoms, torch::Tensor rows, torch::Tensor zx,
    torch::Tensor zy, torch::Tensor offsets, std::vector<double> input_config,
    std::vector<double> output_config, int requested_splits, int c_threads) {
    check_signal(x);
    check_signal(dy);
    check_like(dy, x);
    check_atoms(atoms, rows, x);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(x.size(0) == dy.size(0) && x.size(0) <= BATCH_TILE && x.size(1) == input.sites &&
                    dy.size(1) == output.sites,
                "fused C_in requires matching batches <=4 and axis shapes");
    input = check_z(zx, x, input);
    output = check_z(zy, dy, output);
    int input_owners = input.sites / input.tile_sites;
    int output_owners = output.sites / output.tile_sites;
    TORCH_CHECK(zx.scalar_type() == zy.scalar_type() && zx.size(1) == input_owners &&
                    zy.size(1) == output_owners,
                "fused C_in requires owner window moments");
    // A single hierarchy node and a single full-chart owner window coincide.
    input.owner_windows = output.owner_windows = true;
    TORCH_CHECK(offsets.device() == x.device() && offsets.is_contiguous() &&
                    offsets.scalar_type() == torch::kInt64 && offsets.dim() == 1 &&
                    offsets.numel() == input_owners * output_owners + 1,
                "invalid owner group offsets");
    TORCH_CHECK(requested_splits >= 1 && requested_splits <= 128 &&
                    (requested_splits & (requested_splits - 1)) == 0,
                "owner atom_splits must be a power of two in [1,128]");
    TORCH_CHECK(c_threads == 32 || c_threads == 64 || c_threads == 128,
                "C producer threads must be 32, 64 or 128");
    c10::cuda::CUDAGuard guard(x.device());
    int chunks = (atoms.size(0) + ATOM_CHUNK - 1) / ATOM_CHUNK;
    int splits = std::max(1, std::min(requested_splits, chunks));
    auto partials = torch::empty({splits, x.size(0), input_owners, input.basis_count()},
                                 x.options().dtype(torch::kFloat64));
    auto grad = x.size(0) == 0 ? torch::zeros_like(atoms) : torch::empty_like(atoms);
    if (x.size(0) == 0)
        return {partials, grad};
    size_t shared_bytes = (ATOM_CHUNK + BATCH_TILE * ATOM_CHUNK + ATOM_CHUNK * input.basis_count() +
                           BATCH_TILE * input.basis_count()) *
                          sizeof(double);
    dim3 grid(input_owners, 1, splits);
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "cell_mass_fused_backward_C_in", [&] {
        auto launch = [&](auto *input_moments, auto *output_moments) {
            backward_owner_partials_kernel<<<grid, c_threads, shared_bytes, stream>>>(
                x.data_ptr<scalar_t>(), dy.data_ptr<scalar_t>(), atoms.data_ptr<scalar_t>(),
                rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), input_moments,
                output_moments, partials.data_ptr<double>(), grad.data_ptr<scalar_t>(), x.size(0),
                atoms.size(0), input, output, splits);
        };
        if (zx.scalar_type() == torch::kFloat64)
            launch(zx.data_ptr<double>(), zy.data_ptr<double>());
        else
            launch(zx.data_ptr<scalar_t>(), zy.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {partials, grad};
}
