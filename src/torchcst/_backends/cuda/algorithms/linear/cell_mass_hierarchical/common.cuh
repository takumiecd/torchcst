#pragma once
// Shared metadata and existing ChebyshevSquare profile mathematics.
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <torch/extension.h>
#include <type_traits>
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
    double inverse_period, omega;
};

struct Axis {
    int sites, min_tile_sites, modes, full_level, bank_count, node_count;
    double spacing, origin, period, inverse_period, tail_bound, radius_ratio4;
    bool dyadic_period;
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
    int period_exponent;
    axis.inverse_period = 1 / axis.period;
    axis.dyadic_period =
        std::frexp(axis.period, &period_exponent) == 0.5 && std::isfinite(axis.inverse_period);
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
    for (int bank = 0; bank < axis.bank_count; ++bank) {
        TORCH_CHECK(std::isfinite(2 * PI * axis.modes / axis.banks[bank].period),
                    "bank frequency is not representable");
        axis.banks[bank].inverse_period = 1 / axis.banks[bank].period;
        axis.banks[bank].omega = 2 * PI / axis.banks[bank].period;
    }
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

// For a power-of-two period, normal finite scaling, truncation, and scaling
// back are exact. Preserve fmod's signed zero. Outside that safe range, or
// for other periods, retain the general fmod implementation.
__device__ inline double chart_remainder(double value, const Axis &axis) {
    if (axis.dyadic_period) {
        double scaled = value * axis.inverse_period;
        if (isfinite(scaled) && (value == 0 || fabs(scaled) >= DBL_MIN)) {
            double remainder = value - trunc(scaled) * axis.period;
            return remainder == 0 ? copysign(0.0, value) : remainder;
        }
    }
    return fmod(value, axis.period);
}

__device__ inline double wrapped_center(double center, const Axis &axis) {
    double value = chart_remainder(center, axis) - chart_remainder(axis.origin, axis);
    value = chart_remainder(value, axis);
    return value < 0 ? value + axis.period : value;
}

__device__ inline Membership classify_axis(double sigma, double center, const Axis &axis) {
    int tile_sites = axis.min_tile_sites;
    int level = 0;
    while (sigma > tile_sites * axis.spacing) {
        tile_sites *= 2;
        ++level;
    }
    if (3 * tile_sites > axis.sites)
        return {axis.full_level, 0};

    double relative = wrapped_center(center, axis);
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

// Integer coefficients of the positive polynomial, indexed by even M=2..16,
// harmonic and degree. These depend only on the basis, never on atom values.
// The double bank plus M=8 compensated coefficients occupy 4616 bytes of
// constant memory per CUDA translation unit. No atom-dependent values are
// stored here. The FP32 pairs are split from the existing double constants.
__host__ __device__ constexpr int horner_bank_offset(int modes) {
    int bank = modes / 2;
    return (bank - 1) * bank * (4 * bank + 1) / 6;
}
constexpr int HORNER_VALUES = horner_bank_offset(16) + 16 * 17 / 2;
struct DoubleSingle {
    float hi, lo;
};
__host__ __device__ constexpr DoubleSingle split_constant(double value) {
    float hi = static_cast<float>(value);
    return {hi, static_cast<float>(value - static_cast<double>(hi))};
}
struct HornerBanks {
    double coefficients[HORNER_VALUES] = {};
    double factors[72] = {};
    double inverse_integer[17] = {};
    DoubleSingle coefficients_m8[36] = {};
    DoubleSingle factors_m8[8] = {};
    __host__ __device__ constexpr HornerBanks() {
        for (int k = 1; k <= 16; ++k)
            inverse_integer[k] = 1.0 / k;
        for (int modes = 2; modes <= 16; modes += 2) {
            for (int k = 0; k < modes; ++k) {
                int bank = modes / 2;
                factors[bank * (bank - 1) + k] = 0.5 * modes / (modes + k);
                int offset = horner_bank_offset(modes) + k * (2 * modes - k + 1) / 2;
                int64_t term = modes + k;
                for (int j = 1; j <= modes - k; ++j) {
                    coefficients[offset + j - 1] = static_cast<double>(term);
                    if (j < modes - k)
                        term = term * (modes - k - j) * (modes + k + j) / (j * (j + 1));
                }
            }
        }
        for (int j = 0; j < 36; ++j)
            coefficients_m8[j] = split_constant(coefficients[horner_bank_offset(8) + j]);
        for (int k = 0; k < 8; ++k)
            factors_m8[k] = split_constant(factors[12 + k]);
    }
};
static_assert(sizeof(HornerBanks) == 4616, "unexpected Horner constant-bank size");
static __device__ __constant__ HornerBanks horner_banks{};

// Error-compensated FP32 pair arithmetic for the optional M=8 polynomial.
// This is an experimental evaluation route, not an exact FP64 replacement.
__device__ inline DoubleSingle ds_from_double(double value) {
    float hi = static_cast<float>(value);
    return {hi, static_cast<float>(value - static_cast<double>(hi))};
}
__device__ inline DoubleSingle ds_add(DoubleSingle a, DoubleSingle b) {
    float sum = a.hi + b.hi;
    float v = sum - a.hi;
    float error = ((b.hi - v) + (a.hi - (sum - v))) + a.lo + b.lo;
    float hi = sum + error;
    return {hi, error - (hi - sum)};
}
__device__ inline DoubleSingle ds_multiply(DoubleSingle a, DoubleSingle b) {
    float product = a.hi * b.hi;
    float error = fmaf(a.hi, b.hi, -product) + a.hi * b.lo + a.lo * b.hi + a.lo * b.lo;
    float hi = product + error;
    return {hi, error - (hi - product)};
}
__device__ inline double ds_to_double(DoubleSingle value) {
    return static_cast<double>(value.hi) + static_cast<double>(value.lo);
}

template <bool Compensated>
using PolynomialNumber = typename std::conditional<Compensated, DoubleSingle, double>::type;

template <bool Compensated>
__device__ inline PolynomialNumber<Compensated> polynomial_number(double value) {
    if constexpr (Compensated)
        return ds_from_double(value);
    else
        return value;
}

template <bool Compensated>
__device__ inline PolynomialNumber<Compensated>
polynomial_multiply(PolynomialNumber<Compensated> a, PolynomialNumber<Compensated> b) {
    if constexpr (Compensated)
        return ds_multiply(a, b);
    else
        return a * b;
}

// For k > 0, use a precomputed sine when a caller already advances the
// cell-half-phase by recurrence. Multiplying P by 1/pi first avoids overflow.
__device__ inline double integrated_sine_factor(const Bank &bank, int harmonic, double half_sine) {
    constexpr double INV_PI = 0.31830988618379067154;
    return (bank.period * INV_PI) * half_sine * horner_banks.inverse_integer[harmonic];
}

__device__ inline double cell_integral_factor(const Bank &bank, int harmonic, double spacing) {
    if (harmonic == 0)
        return spacing;
    double half_sine = sin((bank.omega * harmonic) * spacing / 2);
    return integrated_sine_factor(bank, harmonic, half_sine);
}

// The common T_M(beta)^-2 cancels from c_k/(P*c_0). Evaluate the same
// positive polynomial and optionally its beta derivative together. Horner
// avoids runtime coefficient divisions; no per-atom coefficient array exists.
template <bool Derivative, bool Compensated = false>
__device__ inline void
raw_horner_power_impl(double beta, int modes, int harmonic, PolynomialNumber<Compensated> power,
                      PolynomialNumber<Compensated> dpower, double &value, double &derivative,
                      PolynomialNumber<Compensated> beta_number = {}) {
    if constexpr (Compensated) {
        // The compensated bank contains only M=8; callers must select it
        // only after checking the axis mode count.
        DoubleSingle x = ds_multiply(beta_number, beta_number);
        DoubleSingle t = ds_add(x, {-1.0f, 0.0f});
        if (harmonic == 8) {
            value = ds_to_double(ds_multiply({0.25f, 0.0f}, power));
            if constexpr (Derivative)
                derivative = ds_to_double(ds_multiply({0.25f, 0.0f}, dpower));
            return;
        }
        int degree = 8 - harmonic;
        int offset = harmonic * (17 - harmonic) / 2;
        DoubleSingle polynomial = horner_banks.coefficients_m8[offset + degree - 1];
        DoubleSingle slope = {0, 0};
        for (int j = degree - 1; j >= 1; --j) {
            if constexpr (Derivative)
                slope = ds_add(ds_multiply(slope, t), polynomial);
            polynomial =
                ds_add(ds_multiply(polynomial, t), horner_banks.coefficients_m8[offset + j - 1]);
        }
        if constexpr (Derivative)
            slope = ds_add(ds_multiply(slope, t), polynomial);
        polynomial = ds_multiply(polynomial, t);
        DoubleSingle factor = horner_banks.factors_m8[harmonic];
        DoubleSingle raw = ds_multiply(ds_multiply(factor, power), polynomial);
        if (harmonic == 0)
            raw = ds_add(raw, {0.5f, 0.0f});
        value = ds_to_double(raw);
        if constexpr (Derivative) {
            DoubleSingle dx = ds_multiply({2.0f, 0.0f}, beta_number);
            DoubleSingle d =
                ds_add(ds_multiply(dpower, polynomial), ds_multiply(ds_multiply(power, slope), dx));
            derivative = ds_to_double(ds_multiply(factor, d));
        }
    } else {
        double x = beta * beta, t = x - 1;
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
        int bank = modes / 2;
        double factor = horner_banks.factors[bank * (bank - 1) + harmonic];
        value = factor * power * polynomial + (harmonic == 0 ? 0.5 : 0);
        if (Derivative)
            derivative = factor * (dpower * polynomial + power * slope * (2 * beta));
    }
}

template <bool Derivative>
__device__ inline void raw_horner_power(double beta, int modes, int harmonic, double power,
                                        double &value, double &derivative) {
    double dpower = Derivative ? harmonic * power * (2 * beta / (beta * beta)) : 0;
    raw_horner_power_impl<Derivative>(beta, modes, harmonic, power, dpower, value, derivative);
}

template <bool Derivative, bool Compensated = false>
__device__ inline void raw_horner(double beta, int modes, int harmonic, double &value,
                                  double &derivative) {
    if constexpr (Compensated) {
        DoubleSingle bn = ds_from_double(beta), x = ds_multiply(bn, bn);
        DoubleSingle power = {1, 0}, dpower = {0, 0};
        for (int j = 0; j < harmonic; ++j) {
            if constexpr (Derivative)
                dpower = ds_add(ds_multiply(dpower, x),
                                ds_multiply(power, ds_multiply({2.0f, 0.0f}, bn)));
            power = ds_multiply(power, x);
        }
        raw_horner_power_impl<Derivative, true>(beta, modes, harmonic, power, dpower, value,
                                                derivative, bn);
    } else {
        double x = beta * beta, power = 1;
        for (int j = 0; j < harmonic; ++j)
            power *= x;
        raw_horner_power<Derivative>(beta, modes, harmonic, power, value, derivative);
    }
}

__device__ inline double raw_coefficient(double beta, int modes, int harmonic) {
    double value, unused;
    raw_horner<false>(beta, modes, harmonic, value, unused);
    return value;
}

struct Profile {
    double beta, raw_zero, center, period, inverse_period, omega, beta_sigma, zero_log_beta;
};

// Keep each axis's chart and final-half-cell image convention independent.
__device__ inline double profile_center(const Axis &axis, const Bank &bank, double center,
                                        int owner) {
    double relative = wrapped_center(center, axis);
    int raw_owner =
        static_cast<int>(floor((relative + axis.spacing / 2) / (bank.tile_sites * axis.spacing)));
    int image = (raw_owner - owner) / bank.tile_count;
    return relative - image * axis.period;
}

template <bool Derivative = false, int FixedModes = 0, bool Compensated = false>
__device__ inline Profile make_profile(const Axis &axis, const Bank &bank, double sigma,
                                       double center, int owner) {
    // Preserve effective_radius(sigma,P), including its concentration floor.
    // P/3 remains the width scale even for the one-tile full-chart bank.
    double s = sigma * (3 * bank.inverse_period);
    double base = axis.radius_ratio4 + (1 - axis.radius_ratio4) * s * s * s * s;
    double root = sqrt(sqrt(base));
    double beta = 1 / cos((PI / 3) * root);
    double beta_sigma = 0;
    double raw_zero, zero_log_beta = 0;
    int modes = FixedModes ? FixedModes : axis.modes;
    if (Derivative) {
        double radius_sigma = (1 - axis.radius_ratio4) * s * s * s / (root * root * root);
        beta_sigma = beta * sqrt(fmax(0.0, beta * beta - 1)) * (0.5 * bank.omega) * radius_sigma;
        double zero_beta;
        raw_horner<true, Compensated>(beta, modes, 0, raw_zero, zero_beta);
        zero_log_beta = zero_beta / raw_zero;
    } else {
        if constexpr (Compensated) {
            double unused;
            raw_horner<false, true>(beta, modes, 0, raw_zero, unused);
        } else {
            raw_zero = raw_coefficient(beta, modes, 0);
        }
    }
    return {beta,
            raw_zero,
            profile_center(axis, bank, center, owner),
            bank.period,
            bank.inverse_period,
            bank.omega,
            beta_sigma,
            zero_log_beta};
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
// to beta. Carry value/derivative together; this is analytic, not finite
// differences.
__device__ inline void raw_coefficient_derivative(double beta, int modes, int harmonic,
                                                  double &value, double &derivative) {
    raw_horner<true>(beta, modes, harmonic, value, derivative);
}

// A thread-local stream over harmonics 0..M, with one phase evaluation and a
// running beta^(2k). Derivatives are with respect to beta at fixed bank,
// owner and center. The caller must request each harmonic in order, once.
template <bool Derivative, bool Compensated = false> struct HarmonicStream {
    double beta, inverse_period, normalized_scale, zero_log_beta;
    PolynomialNumber<Compensated> beta_number, beta_squared, power, power_beta;
    double phase_sine, phase_cosine, step_sine, step_cosine;
    int modes, harmonic;

    __device__ HarmonicStream(const Profile &profile, int mode_count)
        : beta(profile.beta), inverse_period(profile.inverse_period),
          normalized_scale((2 * profile.inverse_period) / profile.raw_zero),
          zero_log_beta(Derivative ? profile.zero_log_beta : 0),
          beta_number(polynomial_number<Compensated>(beta)),
          beta_squared(polynomial_multiply<Compensated>(beta_number, beta_number)),
          power(polynomial_number<Compensated>(1)), power_beta(polynomial_number<Compensated>(0)),
          phase_sine(0), phase_cosine(1), modes(mode_count), harmonic(0) {
        sincos(profile.omega * profile.center, &step_sine, &step_cosine);
    }

    __device__ inline void next(double &cc, double &cs, double &cc_beta, double &cs_beta) {
        if (harmonic == 0) {
            cc = inverse_period;
            cs = cc_beta = cs_beta = 0;
        } else {
            if constexpr (Compensated) {
                if constexpr (Derivative)
                    power_beta = ds_add(ds_multiply(power_beta, beta_squared),
                                        ds_multiply(power, ds_multiply({2.0f, 0.0f}, beta_number)));
                power = ds_multiply(power, beta_squared);
            } else {
                if (Derivative)
                    power_beta = power_beta * beta_squared + power * (2 * beta);
                power *= beta_squared;
            }
            double next_cosine = phase_cosine * step_cosine - phase_sine * step_sine;
            phase_sine = phase_sine * step_cosine + phase_cosine * step_sine;
            phase_cosine = next_cosine;
            double raw, raw_beta = 0;
            raw_horner_power_impl<Derivative, Compensated>(beta, modes, harmonic, power, power_beta,
                                                           raw, raw_beta, beta_number);
            double scale = normalized_scale * raw;
            cc = scale * phase_cosine;
            cs = scale * phase_sine;
            if (Derivative) {
                double scale_beta = normalized_scale * (raw_beta - raw * zero_log_beta);
                cc_beta = scale_beta * phase_cosine;
                cs_beta = scale_beta * phase_sine;
            } else {
                cc_beta = cs_beta = 0;
            }
        }
        ++harmonic;
    }
};

// A local bank spans three dyadic tiles, while the chart spans tile_count
// tiles. Crossing a chart boundary therefore rotates by a cube root of unity.
// Full-chart banks have no nonzero image in their retained window.
__device__ inline void image_rotation(const Bank &bank, int harmonic, int image, double &sine,
                                      double &cosine) {
    if (image == 0 || harmonic == 0 || bank.full_chart) {
        sine = 0;
        cosine = 1;
        return;
    }
    int residue = ((harmonic % 3) * (image % 3) * (bank.tile_count % 3)) % 3;
    if (residue < 0)
        residue += 3;
    cosine = residue == 0 ? 1 : -0.5;
    sine = residue == 0 ? 0 : (residue == 1 ? 0.86602540378443864676 : -0.86602540378443864676);
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
                                           double *dsigma, bool windowed) {
    Profile profile = make_profile<true>(axis, bank, sigma, center, owner);
    HarmonicStream<true> stream(profile, axis.modes);
    double beta_sigma = profile.beta_sigma;
    int retained = windowed || bank.full_chart ? 1 : 3;
    int first_tile = windowed ? owner : (bank.full_chart ? 0 : owner - 1);
    // The constant coefficient is 1/P, independent of both center and sigma.
    for (int harmonic = 0; harmonic <= axis.modes; ++harmonic) {
        double cc, cs, cc_beta, cs_beta;
        stream.next(cc, cs, cc_beta, cs_beta);
        if (harmonic == 0)
            continue;
        double omega = profile.omega * harmonic;
        for (int offset = 0; offset < retained; ++offset) {
            int raw_tile = first_tile + offset;
            int tile = positive_mod(raw_tile, bank.tile_count);
            int image = (raw_tile - tile) / bank.tile_count;
            double rs, rc;
            image_rotation(bank, harmonic, image, rs, rc);
            int node = bank.node_offset + tile;
            for (int b = 0; b < BATCH_TILE && batch_begin + b < batches; ++b) {
                int base = ((batch_begin + b) * axis.node_count + node) * axis.basis_count();
                double zc = z[base + 2 * harmonic - 1], zs = z[base + 2 * harmonic];
                double rotated_cos = zc * rc - zs * rs;
                double rotated_sin = zs * rc + zc * rs;
                dcenter[b] += omega * (-cs * rotated_cos + cc * rotated_sin);
                dsigma[b] += beta_sigma * (cc_beta * rotated_cos + cs_beta * rotated_sin);
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
                    (groups.size(0) <= atoms.size(0) || groups.size(0) <= 4096),
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
