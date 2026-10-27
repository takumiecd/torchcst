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

// Existing ChebyshevSquare complex coefficient, with the common T_M(beta)^-2
// removed. That factor cancels from c_k/(P*c_0). Same positive polynomial as
// cell_mass/polynomial_coefficient; no per-atom coefficient table is retained.
__device__ inline double raw_coefficient(double beta, int modes, int harmonic) {
    double x = beta * beta, t = x - 1, power = 1;
    for (int j = 0; j < harmonic; ++j)
        power *= x;
    if (harmonic == modes)
        return 0.25 * power;
    double term = (modes + harmonic) * t, sum = term;
    for (int j = 1; j < modes - harmonic; ++j) {
        double ratio =
            static_cast<double>((modes - harmonic - j) * (modes + harmonic + j)) / (j * (j + 1));
        term *= ratio * t;
        sum += term;
    }
    double value = static_cast<double>(modes) / (modes + harmonic) * power * sum;
    if (harmonic == 0)
        value += 1;
    return 0.5 * value;
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
} // namespace cell_mass_hierarchy
