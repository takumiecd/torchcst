#pragma once
// Shared metadata and existing ChebyshevSquare profile mathematics.
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cmath>
#include <torch/extension.h>
#include <vector>

namespace cell_mass_hierarchy {
constexpr double PI = 3.14159265358979323846;
constexpr int THREADS = 128;
constexpr int BATCH_TILE = 4;
constexpr int MAX_BANKS = 11;

struct Bank {
    int level, tile_sites, tile_count, node_offset;
    double period;
    bool full_chart;
};

struct Axis {
    int sites, min_tile_sites, modes, full_level, bank_count, node_count;
    double spacing, origin, period, tail_bound, radius_ratio4;
    Bank banks[MAX_BANKS];
    __host__ __device__ int basis_count() const {
        return 1 + 2 * modes;
    }
    __host__ __device__ int max_owners() const {
        return sites / min_tile_sites;
    }
};

inline bool power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

inline Axis parse_axis(const std::vector<double> &config) {
    TORCH_CHECK(config.size() == 6, "hierarchy axis needs six scalars");
    for (double value : config)
        TORCH_CHECK(std::isfinite(value), "axis values must be finite");
    TORCH_CHECK(config[0] == std::floor(config[0]) && config[0] >= 1 && config[0] <= 1024,
                "invalid site count");
    TORCH_CHECK(config[2] == std::floor(config[2]) && config[2] >= 1 && config[2] <= config[0],
                "invalid minimum tile size");
    TORCH_CHECK(config[4] == std::floor(config[4]) && config[4] >= 2 && config[4] <= 16,
                "invalid modes");
    Axis axis{};
    axis.sites = static_cast<int>(config[0]);
    axis.spacing = config[1];
    axis.min_tile_sites = static_cast<int>(config[2]);
    axis.origin = config[3];
    axis.modes = static_cast<int>(config[4]);
    axis.tail_bound = config[5];
    TORCH_CHECK(axis.tail_bound > 0 && axis.tail_bound < 1, "invalid tail bound");
    double shape = std::acosh(1 / std::sqrt(axis.tail_bound)) / axis.modes;
    TORCH_CHECK(shape < std::acosh(2.0), "profile modes cannot meet this tail bound");
    double radius_ratio = 3 / PI * std::acos(1 / std::cosh(shape));
    axis.radius_ratio4 = radius_ratio * radius_ratio * radius_ratio * radius_ratio;
    TORCH_CHECK(power_of_two(axis.sites) && power_of_two(axis.min_tile_sites) &&
                    axis.modes % 2 == 0 && axis.spacing > 0,
                "invalid dyadic axis");
    axis.period = axis.sites * axis.spacing;
    TORCH_CHECK(std::isfinite(2 * axis.period) && axis.period > 0, "invalid chart period");
    for (int count = axis.max_owners(); count > 1; count /= 2)
        ++axis.full_level;

    int tile_sites = axis.min_tile_sites;
    int level = 0;
    while (3 * tile_sites <= axis.sites) {
        int count = axis.sites / tile_sites;
        axis.banks[axis.bank_count++] = {
            level, tile_sites, count, axis.node_count, 3 * tile_sites * axis.spacing, false};
        axis.node_count += count;
        tile_sites *= 2;
        ++level;
    }
    axis.banks[axis.bank_count++] = {axis.full_level, axis.sites,  1,
                                     axis.node_count, axis.period, true};
    ++axis.node_count;
    for (int bank = 0; bank < axis.bank_count; ++bank)
        TORCH_CHECK(std::isfinite(2 * PI * axis.modes / axis.banks[bank].period),
                    "bank frequency is not representable");
    return axis;
}

inline void check_cuda_float_matrix(const torch::Tensor &tensor) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.dim() == 2,
                "expected a contiguous CUDA matrix");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat64,
                "expected float32 or float64");
}

struct Membership {
    int level, owner;
};

__device__ inline Membership classify_axis(double sigma, double center, const Axis &axis) {
    int tile_sites = axis.min_tile_sites;
    int level = 0;
    while (sigma > tile_sites * axis.spacing) {
        tile_sites *= 2;
        ++level;
    }
    if (3 * tile_sites > axis.sites)
        return {axis.full_level, 0};

    double relative = fmod(center, axis.period) - fmod(axis.origin, axis.period);
    relative = fmod(relative, axis.period);
    if (relative < 0)
        relative += axis.period;
    int owner =
        static_cast<int>(floor((relative + axis.spacing / 2) / (tile_sites * axis.spacing)));
    return {level, owner % (axis.sites / tile_sites)};
}

__device__ inline double integrated_basis(const Axis &axis, const Bank &bank, int mode, int site) {
    if (mode == 0)
        return axis.spacing;
    int harmonic = (mode + 1) / 2;
    double omega = 2 * PI * harmonic / bank.period;
    double factor = 2 * sin(omega * axis.spacing / 2) / omega;
    // Coordinates are relative to the same chart origin at every level.
    double phase = omega * site * axis.spacing;
    return factor * (mode % 2 == 1 ? cos(phase) : sin(phase));
}

__device__ inline int bank_for_level(const Axis &axis, int level) {
    return level == axis.full_level ? axis.bank_count - 1 : level;
}

__device__ inline double wrapped_center(double center, const Axis &axis) {
    double value = fmod(center, axis.period) - fmod(axis.origin, axis.period);
    value = fmod(value, axis.period);
    return value < 0 ? value + axis.period : value;
}

// Integer coefficients of the positive polynomial, indexed by even M=2..16,
// harmonic and degree. These depend only on the basis, never on atom values.
// The 444 doubles occupy 3552 bytes of constant memory per CUDA translation
// unit; the integer recurrence and its stored values are exact in double.
__host__ __device__ constexpr int horner_bank_offset(int modes) {
    int bank = modes / 2;
    return (bank - 1) * bank * (4 * bank + 1) / 6;
}
constexpr int HORNER_VALUES = horner_bank_offset(16) + 16 * 17 / 2;
struct HornerBanks {
    double coefficients[HORNER_VALUES] = {};
    __host__ __device__ constexpr HornerBanks() {
        for (int modes = 2; modes <= 16; modes += 2) {
            for (int k = 0; k < modes; ++k) {
                int offset = horner_bank_offset(modes) + k * (2 * modes - k + 1) / 2;
                int64_t term = modes + k;
                for (int j = 1; j <= modes - k; ++j) {
                    coefficients[offset + j - 1] = static_cast<double>(term);
                    if (j < modes - k)
                        term = term * (modes - k - j) * (modes + k + j) / (j * (j + 1));
                }
            }
        }
    }
};
static_assert(sizeof(HornerBanks) == 3552, "unexpected Horner constant-bank size");
static __device__ __constant__ HornerBanks horner_banks{};

// The common T_M(beta)^-2 cancels from c_k/(P*c_0). Evaluate the same
// positive polynomial and optionally its beta derivative together. Horner
// avoids runtime coefficient divisions; no per-atom coefficient array exists.
template <bool Derivative>
__device__ inline void raw_horner(double beta, int modes, int harmonic, double &value,
                                  double &derivative) {
    double x = beta * beta, t = x - 1, power = 1;
    for (int j = 0; j < harmonic; ++j)
        power *= x;
    double dpower = Derivative ? harmonic * power * (2 * beta / x) : 0;
    if (harmonic == modes) {
        value = 0.25 * power;
        if (Derivative)
            derivative = 0.25 * dpower;
        return;
    }
    int degree = modes - harmonic;
    int offset = horner_bank_offset(modes) + harmonic * (2 * modes - harmonic + 1) / 2;
    double polynomial = horner_banks.coefficients[offset + degree - 1];
    double slope = 0;
    for (int j = degree - 1; j >= 1; --j) {
        if (Derivative)
            slope = slope * t + polynomial;
        polynomial = polynomial * t + horner_banks.coefficients[offset + j - 1];
    }
    if (Derivative)
        slope = slope * t + polynomial;
    polynomial *= t; // The polynomial has zero constant term.
    double factor = 0.5 * modes / (modes + harmonic);
    value = factor * power * polynomial + (harmonic == 0 ? 0.5 : 0);
    if (Derivative)
        derivative = factor * (dpower * polynomial + power * slope * (2 * beta));
}

__device__ inline double raw_coefficient(double beta, int modes, int harmonic) {
    double value, unused;
    raw_horner<false>(beta, modes, harmonic, value, unused);
    return value;
}

struct Profile {
    double beta, raw_zero, center, period;
};

__device__ inline Profile make_profile(const Axis &axis, const Bank &bank, double sigma,
                                       double center, int owner) {
    // Preserve effective_radius(sigma,P), including its concentration floor.
    // P/3 remains the width scale even for the one-tile full-chart bank.
    double scale = bank.period / 3, s = sigma / scale;
    double base = axis.radius_ratio4 + (1 - axis.radius_ratio4) * s * s * s * s;
    double radius = scale * sqrt(sqrt(base));
    double beta = 1 / cos(PI * radius / bank.period);
    double relative = wrapped_center(center, axis);
    int raw_owner =
        static_cast<int>(floor((relative + axis.spacing / 2) / (bank.tile_sites * axis.spacing)));
    // Shift a centre in the final half-cell together with its owner=0 window.
    int image = (raw_owner - owner) / bank.tile_count;
    return {beta, raw_coefficient(beta, axis.modes, 0), relative - image * axis.period,
            bank.period};
}

__device__ inline void coefficient_pair(const Profile &profile, int modes, int harmonic,
                                        double &cosine, double &sine) {
    if (harmonic == 0) {
        cosine = 1 / profile.period;
        sine = 0;
        return;
    }
    double scale =
        2 * raw_coefficient(profile.beta, modes, harmonic) / (profile.period * profile.raw_zero);
    double phase_sine, phase_cosine;
    sincos(2 * PI * harmonic / profile.period * profile.center, &phase_sine, &phase_cosine);
    cosine = scale * phase_cosine;
    sine = scale * phase_sine;
}
// Differentiate the same positive polynomial as raw_coefficient with respect
// to beta. Carry value/derivative together; this is analytic, not finite differences.
__device__ inline void raw_coefficient_derivative(double beta, int modes, int harmonic,
                                                  double &value, double &derivative) {
    raw_horner<true>(beta, modes, harmonic, value, derivative);
}

// The selected bank, owner and torus image are held fixed in backward. The
// optimizer's next forward will classify again using the updated atom values.
__device__ inline double beta_sigma_derivative(const Axis &axis, const Bank &bank, double sigma,
                                               double beta) {
    double relative = sigma / (bank.period / 3);
    double base =
        axis.radius_ratio4 + (1 - axis.radius_ratio4) * relative * relative * relative * relative;
    double root = sqrt(sqrt(base));
    double radius_sigma =
        (1 - axis.radius_ratio4) * relative * relative * relative / (root * root * root);
    return beta * sqrt(fmax(0.0, beta * beta - 1)) * PI / bank.period * radius_sigma;
}

__device__ inline int positive_mod(int value, int count) {
    int result = value % count;
    return result < 0 ? result + count : result;
}

__device__ inline int group_for_slot(const int64_t *offsets, int count, int slot) {
    int first = 0, last = count;
    while (first + 1 < last) {
        int middle = (first + last) / 2;
        if (offsets[middle] <= slot)
            first = middle;
        else
            last = middle;
    }
    return first;
}

// One atom's derivative contractions against Z. No per-atom coefficient or
// Jacobian array is written to global memory. Each result covers four batches.
__device__ inline void project_derivatives(const double *z, int batches, int batch_begin,
                                           const Axis &axis, const Bank &bank, int owner,
                                           double sigma, double center, double *dcenter,
                                           double *dsigma) {
    Profile profile = make_profile(axis, bank, sigma, center, owner);
    double raw_zero, raw_zero_beta;
    raw_coefficient_derivative(profile.beta, axis.modes, 0, raw_zero, raw_zero_beta);
    double beta_sigma = beta_sigma_derivative(axis, bank, sigma, profile.beta);
    int retained = bank.full_chart ? 1 : 3;
    int first_tile = bank.full_chart ? 0 : owner - 1;
    // The constant coefficient is 1/P, independent of both center and sigma.
    for (int harmonic = 1; harmonic <= axis.modes; ++harmonic) {
        double raw, raw_beta;
        raw_coefficient_derivative(profile.beta, axis.modes, harmonic, raw, raw_beta);
        double omega = 2 * PI * harmonic / bank.period;
        double scale = 2 * raw / (bank.period * raw_zero);
        double scale_sigma =
            2 * beta_sigma / (bank.period * raw_zero) * (raw_beta - raw * raw_zero_beta / raw_zero);
        double ps, pc;
        sincos(omega * profile.center, &ps, &pc);
        for (int offset = 0; offset < retained; ++offset) {
            int raw_tile = first_tile + offset;
            int tile = positive_mod(raw_tile, bank.tile_count);
            int image = (raw_tile - tile) / bank.tile_count;
            double rs = 0, rc = 1;
            if (image != 0)
                sincos(omega * image * axis.period, &rs, &rc);
            int node = bank.node_offset + tile;
            for (int b = 0; b < BATCH_TILE && batch_begin + b < batches; ++b) {
                int base = ((batch_begin + b) * axis.node_count + node) * axis.basis_count();
                double zc = z[base + 2 * harmonic - 1], zs = z[base + 2 * harmonic];
                double rotated_cos = zc * rc - zs * rs;
                double rotated_sin = zs * rc + zc * rs;
                dcenter[b] += omega * scale * (-ps * rotated_cos + pc * rotated_sin);
                dsigma[b] += scale_sigma * (pc * rotated_cos + ps * rotated_sin);
            }
        }
    }
}
inline void check_layout(const torch::Tensor &atoms, const torch::Tensor &rows,
                         const torch::Tensor &groups, const torch::Tensor &offsets) {
    check_cuda_float_matrix(atoms);
    TORCH_CHECK(atoms.size(1) == 4 && atoms.size(0) <= 65536, "invalid atoms");
    TORCH_CHECK(rows.is_contiguous() && rows.device() == atoms.device() && rows.dim() == 1 &&
                    rows.scalar_type() == torch::kInt64 && rows.numel() == atoms.size(0),
                "invalid rows");
    TORCH_CHECK(groups.is_contiguous() && groups.device() == atoms.device() && groups.dim() == 2 &&
                    groups.size(1) == 4 && groups.scalar_type() == torch::kInt32 &&
                    groups.size(0) <= atoms.size(0),
                "invalid groups");
    TORCH_CHECK(offsets.is_contiguous() && offsets.device() == atoms.device() &&
                    offsets.dim() == 1 && offsets.scalar_type() == torch::kInt64 &&
                    offsets.numel() == groups.size(0) + 1,
                "invalid offsets");
}

inline void check_moments(const torch::Tensor &values, const torch::Tensor &like,
                          const Axis &axis) {
    TORCH_CHECK(values.is_contiguous() && values.device() == like.device() && values.dim() == 3 &&
                    values.scalar_type() == torch::kFloat64 && values.size(1) == axis.node_count &&
                    values.size(2) == axis.basis_count(),
                "invalid moments/coefficients");
    TORCH_CHECK(values.numel() <= INT_MAX, "moments exceed prototype index capacity");
}
} // namespace cell_mass_hierarchy
