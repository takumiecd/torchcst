#pragma once
#include "limits.cuh"
#include <torch/extension.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <type_traits>
#include <vector>

namespace radius_cell_mass {
constexpr double PI = 3.14159265358979323846;
constexpr int MAX_BANKS = 32;
constexpr int THREADS = 128;

struct Bank {
    int tile_sites, tile_count, node_offset, count;
    double period;
    int level;
};
struct Axis {
    int sites, modes, base_tile_sites, merge_factor, bank_count, node_count;
    double spacing, origin, period;
    bool polar;
    double amplitude_max, w_c, kappa, lower_kappa, upper_decay_power;
    double radius_min, radius_birth, radius_max, upper_floor;
    bool closed_envelope, zero_opening;
    double zero_opening_scale;
    double closed_lower_power, closed_upper_power;
    Bank banks[MAX_BANKS];
    __host__ __device__ int basis_count() const { return 2 * modes + 1; }
};

inline Axis parse_axis(const std::vector<double> &config) {
    TORCH_CHECK(config.size() == 6 || config.size() == 15 || config.size() == 16 || config.size() == 17, "radius axis requires 6, 15, 16 or 17 scalars");
    for (double v : config) TORCH_CHECK(std::isfinite(v), "axis scalars must be finite");
    TORCH_CHECK(config[0] >= 1 && config[0] <= MAX_SITES && config[0] == floor(config[0]), "invalid sites");
    TORCH_CHECK(config[2] >= 1 && config[2] <= config[0] && config[2] == floor(config[2]), "invalid base tile");
    TORCH_CHECK(config[4] >= 1 && config[4] <= 16 && config[4] == floor(config[4]), "invalid modes");
    TORCH_CHECK(config[5] >= 2 && config[5] <= INT_MAX && config[5] == floor(config[5]), "invalid merge factor");
    Axis a{};
    a.sites = int(config[0]); a.spacing = config[1]; a.base_tile_sites = int(config[2]);
    a.origin = config[3]; a.modes = int(config[4]); a.merge_factor = int(config[5]);
    a.period = a.sites * a.spacing;
    TORCH_CHECK(a.spacing > 0 && std::isfinite(3 * a.period) && a.period > 0, "invalid chart period");
    TORCH_CHECK(a.sites % a.base_tile_sites == 0, "base tile must divide chart");
    int ratio = a.sites / a.base_tile_sites;
    while (ratio > 1) {
        TORCH_CHECK(ratio % a.merge_factor == 0, "chart/base tile ratio must be an exact merge power");
        ratio /= a.merge_factor;
    }
    for (int tile = a.base_tile_sites, level = 0;; tile *= a.merge_factor, ++level) {
        TORCH_CHECK(level < 16, "too many hierarchy levels");
        for (int count = 2; count <= 3; ++count) {
            double period = count * tile * a.spacing;
            TORCH_CHECK(std::isfinite(2 * PI * a.modes / period), "bank frequency is not representable");
            a.banks[a.bank_count++] = {tile, a.sites / tile, a.node_count, count, period, level};
            a.node_count += a.sites / tile;
        }
        if (tile == a.sites) break;
    }
    a.polar = config.size() >= 15;
    if (a.polar) {
        a.amplitude_max = config[6]; a.w_c = config[7]; a.kappa = config[8];
        a.lower_kappa = config[9]; a.upper_decay_power = config[10];
        a.radius_min = config[11]; a.radius_birth = config[12]; a.radius_max = config[13];
        a.upper_floor = config[14];
        for (int i = 6; i < 15; ++i) TORCH_CHECK(config[i] > 0, "Polar scalars must be positive");
        TORCH_CHECK(a.kappa > 1 && a.upper_decay_power <= 1 && a.radius_min <= a.radius_birth &&
            a.radius_birth <= a.radius_max && a.radius_max <= a.period / 2 &&
            a.upper_floor >= a.radius_min && a.upper_floor <= a.radius_max, "invalid Polar radius bounds");
        a.zero_opening = config.size() == 16;
        if (a.zero_opening) {
            a.zero_opening_scale = config[15];
            TORCH_CHECK(a.zero_opening_scale > 0, "zero opening scale must be positive");
        }
        a.closed_envelope = config.size() == 17;
        if (a.closed_envelope) {
            a.closed_lower_power = config[15]; a.closed_upper_power = config[16];
            TORCH_CHECK(a.closed_lower_power >= a.closed_upper_power && a.closed_upper_power >= 1 &&
                a.radius_birth == a.radius_max && a.upper_floor == a.radius_min, "invalid closed radius envelope");
        }
    }
    return a;
}

template<class T>
__device__ inline double amplitude(const T *atoms, int atom, const Axis &axis) {
    double s = atoms[4 * atom];
    if (!axis.polar) return s;
    double t = atoms[4 * atom + 1];
    double tiny = std::is_same<T, float>::value ? FLT_MIN : DBL_MIN;
    return axis.amplitude_max * s / sqrt(fmax(s * s + t * t, tiny));
}
template<class T>
__device__ inline double radius(const T *atoms, int atom, const Axis &axis) {
    if (!axis.polar) return atoms[4 * atom + 1];
    double s = atoms[4 * atom], t = atoms[4 * atom + 1];
    double alpha = fmin(1., fmax(0., (s * s + t * t - 1) / 3));
    double ratio = amplitude(atoms, atom, axis) / axis.w_c, x = ratio * ratio;
    double lower = axis.radius_min + (axis.radius_birth - axis.radius_min) / (1 + axis.lower_kappa * x);
    double upper = axis.radius_min + (axis.radius_max - axis.radius_min) * axis.kappa /
        (axis.kappa + pow(x, axis.upper_decay_power));
    if (axis.closed_envelope) {
        double ratio_a = amplitude(atoms, atom, axis) / axis.amplitude_max;
        double base = fmin(1., fmax(0., 1 - ratio_a * ratio_a));
        lower = axis.radius_min + (axis.radius_max - axis.radius_min) * pow(base, axis.closed_lower_power);
        upper = axis.radius_min + (axis.radius_max - axis.radius_min) * pow(base, axis.closed_upper_power);
    }
    upper = fmax(lower, fmax(upper, axis.upper_floor));
    if (axis.zero_opening) {
        double a = amplitude(atoms, atom, axis);
        double h = axis.zero_opening_scale / hypot(a, axis.zero_opening_scale);
        lower = a == 0 ? upper : fmin(upper, lower + h * h * (upper - lower));
    }
    if (axis.closed_envelope || axis.zero_opening) {
        double value = alpha >= 1 ? upper : lower + lower * expm1(alpha * log1p((upper - lower) / lower));
        return fmin(upper, fmax(lower, value));
    }
    return fmin(upper, fmax(lower, exp((1 - alpha) * log(lower) + alpha * log(upper))));
}
template<class T>
__device__ inline double center(const T *atoms, int atom, const Axis &axis, bool output) {
    (void)axis;
    return atoms[4 * atom + (output ? 2 : 3)];
}
__device__ inline double wrapped_center(double center, const Axis &axis) {
    double value = fmod(fmod(center, axis.period) - fmod(axis.origin, axis.period), axis.period);
    return value < 0 ? value + axis.period : value;
}
struct Membership { int bank, first_tile; };
__device__ inline Membership classify(double R, double mu, const Axis &axis) {
    int bank = 0;
    while (bank + 2 < axis.bank_count && R > axis.banks[bank].tile_sites * axis.spacing) bank += 2;
    double T = axis.banks[bank].tile_sites * axis.spacing;
    int K = int(ceil(2 * R / T)) + 1;
    bank += K - 2;
    return {bank, int(floor((wrapped_center(mu, axis) + axis.spacing / 2) / T - (K - 1) / 2.))};
}

// Temporary thread-local samples only. Differentiate with respect to z=cos^2(pi R/P),
// then apply dz/dR after normalization; no secant or overflowing Chebyshev value.
__device__ inline void profile_coefficients_generic(double R, double P, int M, double *c, double *dr) {
    double values[33], slopes[33];
    int samples = 2 * M + 1;
    double angle = PI * (R / P), cb = cos(angle), z = cb * cb;
    double dz = -(2 * PI / P) * cb * sin(angle);
    double scale = 0;
    for (int j = 0; j < samples; ++j) {
        double x = cos(PI * j / samples), previous = 2, current = x;
        double dp = 0, dc = 0;
        for (int n = 2; n <= M; ++n) {
            double next = x * current - z * previous / 4;
            double dn = x * dc - (previous + z * dp) / 4;
            previous = current; current = next; dp = dc; dc = dn;
        }
        values[j] = current; slopes[j] = dc;
        scale = fmax(scale, fabs(current));
    }
    for (int j = 0; j < samples; ++j) {
        double v = values[j] / scale;
        slopes[j] = 2 * v * (slopes[j] / scale);
        values[j] = v * v;
    }
    double zero = 0, dzero = 0;
    for (int j = 0; j < samples; ++j) { zero += values[j]; dzero += slopes[j]; }
    c[0] = 1 / P; dr[0] = 0;
    for (int k = 1; k <= M; ++k) {
        double raw = 0, draw = 0;
        for (int j = 0; j < samples; ++j) {
            double phase = cos(2 * PI * k * j / samples);
            raw += values[j] * phase; draw += slopes[j] * phase;
        }
        c[k] = (raw / zero) / P;
        dr[k] = ((draw - raw * (dzero / zero)) / zero) * (dz / P);
    }
}

// Mathematical constants for z^8 times the raw Chebyshev-square coefficients.
// No atom-dependent coefficient table is stored.
__device__ __constant__ double profile_m8_terms[8][8] = {
    {8, 252, 2520, 11550, 27720, 36036, 24024, 6435},
    {9, 270, 2475, 9900, 19305, 18018, 6435},
    {10, 275, 2200, 7150, 10010, 5005},
    {11, 264, 1716, 4004, 3003},
    {12, 234, 1092, 1365},
    {13, 182, 455},
    {14, 105},
    {15},
};

__device__ __noinline__ void profile_coefficients_m8(double R, double P, double *c, double *dr) {
    double angle = PI * (R / P), cb = cos(angle), sb = sin(angle);
    double z = cb * cb, u = 1 - z;
    double dz = -(2 * PI / P) * cb * sb;
    double zp[9], up[9];
    zp[0] = up[0] = 1;
    for (int j = 1; j <= 8; ++j) { zp[j] = zp[j - 1] * z; up[j] = up[j - 1] * u; }
    // Homogenization cancels the common z^-8 before it can overflow.
    // Differentiate the same polynomial with du/dz=-1, without division by z/u.
    double zero = 0, dzero = 0;
    for (int k = 0; k < 8; ++k) {
        int degree = 8 - k;
        double raw = k == 0 ? 0.5 * zp[8] : 0;
        double draw = k == 0 ? 4 * zp[7] : 0;
        double value = 0, slope = 0;
        for (int j = 1; j <= degree; ++j) {
            int power = degree - j;
            double coefficient = profile_m8_terms[k][j - 1];
            value += coefficient * zp[power] * up[j];
            slope += coefficient * ((power ? power * zp[power - 1] * up[j] : 0) -
                                    j * zp[power] * up[j - 1]);
        }
        double factor = 4.0 / (8 + k);
        raw += factor * value; draw += factor * slope;
        if (k == 0) { zero = raw; dzero = draw; c[0] = 1 / P; dr[0] = 0; }
        else {
            c[k] = (raw / zero) / P;
            dr[k] = ((draw - raw * (dzero / zero)) / zero) * (dz / P);
        }
    }
    c[8] = (0.25 / zero) / P;
    dr[8] = ((-0.25 * (dzero / zero)) / zero) * (dz / P);
}

__device__ inline void profile_coefficients(double R, double P, int M, double *c, double *dr) {
    if (M == 8) profile_coefficients_m8(R, P, c, dr);
    else profile_coefficients_generic(R, P, M, c, dr);
}

template<class T>
__device__ inline void store_gradient(const T *atoms, T *gradient, int atom,
    const Axis &input, double da, double dR, double dmu_input, double dmu_output) {
    if (input.polar) {
        double s = atoms[4 * atom], t = atoms[4 * atom + 1], q = s * s + t * t;
        double tiny = std::is_same<T, float>::value ? FLT_MIN : DBL_MIN;
        double safe = fmax(q, tiny), inverse = 1 / sqrt(safe);
        double ds = input.amplitude_max * inverse * (q >= tiny ? 1 - s * s / safe : 1);
        double dt = q >= tiny ? -input.amplitude_max * inverse * s * t / safe : 0;
        // Derived Polar radii are detached from task gradients.
        gradient[4 * atom] = T(da * ds); gradient[4 * atom + 1] = T(da * dt);
        gradient[4 * atom + 2] = T(dmu_output); gradient[4 * atom + 3] = T(dmu_input);
    } else {
        gradient[4 * atom] = T(da); gradient[4 * atom + 1] = T(dR);
        gradient[4 * atom + 2] = T(dmu_output); gradient[4 * atom + 3] = T(dmu_input);
    }
}
__device__ inline int positive_mod(int value, int count) {
    int result = value % count;
    return result < 0 ? result + count : result;
}

// Explicit FP32 arithmetic for the packed pilot. Double specializations below
// delegate to the original implementation, preserving existing routes.
template<class Calc, class T>
__device__ inline Calc signal_amplitude(const T *p, int atom, const Axis &a) {
    if constexpr (std::is_same<Calc, double>::value) return amplitude(p, atom, a);
    else {
        if (!a.polar) return float(p[4 * atom]);
        float s = p[4 * atom], t = p[4 * atom + 1];
        return float(a.amplitude_max) * s / sqrtf(fmaxf(s*s+t*t, FLT_MIN));
    }
}
template<class Calc> __device__ inline Calc signal_sin(Calc x) {
    if constexpr(std::is_same<Calc,float>::value)return sinf(x);else return sin(x);
}
template<class Calc> __device__ inline Calc signal_cos(Calc x) {
    if constexpr(std::is_same<Calc,float>::value)return cosf(x);else return cos(x);
}
template<class Calc, class T>
__device__ inline Calc signal_radius(const T *p, int atom, const Axis &a) {
    if constexpr (std::is_same<Calc, double>::value) return radius(p, atom, a);
    else {
        if (!a.polar) return float(p[4 * atom + 1]);
        float s=p[4*atom], t=p[4*atom+1], amp=signal_amplitude<float>(p,atom,a);
        float alpha=fminf(1, fmaxf(0,(s*s+t*t-1)/3));
        float ratio=amp/float(a.w_c), x=ratio*ratio;
        float lo=float(a.radius_min)+(float(a.radius_birth)-float(a.radius_min))/(1+float(a.lower_kappa)*x);
        float hi=float(a.radius_min)+(float(a.radius_max)-float(a.radius_min))*float(a.kappa)/(float(a.kappa)+powf(x,float(a.upper_decay_power)));
        if (a.closed_envelope) {
            float base=fminf(1,fmaxf(0,1-(amp/float(a.amplitude_max))*(amp/float(a.amplitude_max))));
            lo=float(a.radius_min)+(float(a.radius_max)-float(a.radius_min))*powf(base,float(a.closed_lower_power));
            hi=float(a.radius_min)+(float(a.radius_max)-float(a.radius_min))*powf(base,float(a.closed_upper_power));
        }
        hi=fmaxf(lo,fmaxf(hi,float(a.upper_floor)));
        if (a.zero_opening) {
            float scale=float(a.zero_opening_scale), h=scale/hypotf(amp,scale);
            lo=amp==0 ? hi : fminf(hi,lo+h*h*(hi-lo));
        }
        float value;
        if (a.closed_envelope || a.zero_opening)
            value=alpha>=1 ? hi : lo+lo*expm1f(alpha*log1pf((hi-lo)/lo));
        else value=expf((1-alpha)*logf(lo)+alpha*logf(hi));
        return fminf(hi,fmaxf(lo,value));
    }
}
template<class Calc>
__device__ inline Calc signal_wrapped(Calc mu, const Axis &a) {
    if constexpr (std::is_same<Calc,double>::value) return wrapped_center(mu,a);
    else {
        float period=float(a.period);
        float v=fmodf(fmodf(mu,period)-fmodf(float(a.origin),period),period);
        // Keep a signed representative. Adding a large period to a small
        // negative FP32 center discards fractional-cell bits; classification
        // and Fourier phases need only a consistent periodic representative.
        // first_tile may be negative: all owner addressing uses positive_mod.
        return v;
    }
}
template<class Calc>
__device__ inline Membership signal_classify(Calc R, Calc mu, const Axis &a) {
    if constexpr (std::is_same<Calc,double>::value) return classify(R,mu,a);
    else {
        int bank=0;
        while(bank+2<a.bank_count && R>float(a.banks[bank].tile_sites)*float(a.spacing))bank+=2;
        float tile=float(a.banks[bank].tile_sites)*float(a.spacing);
        int count=int(ceilf(2*R/tile))+1;bank+=count-2;
        return {bank,int(floorf((signal_wrapped(mu,a)+float(a.spacing)/2)/tile-float(count-1)/2))};
    }
}
template<class Calc, class T>
__device__ inline void signal_store_gradient(const T *p,T *g,int atom,const Axis &a,
    Calc da,Calc dr,Calc dmi,Calc dmo) {
    if constexpr (std::is_same<Calc,double>::value) store_gradient(p,g,atom,a,da,dr,dmi,dmo);
    else {
        if (!a.polar) {
            g[4*atom]=da;g[4*atom+1]=dr;g[4*atom+2]=dmo;g[4*atom+3]=dmi;
            return;
        }
        float s=p[4*atom],t=p[4*atom+1],q=s*s+t*t,safe=fmaxf(q,FLT_MIN),inverse=1/sqrtf(safe);
        float ds=float(a.amplitude_max)*inverse*(q>=FLT_MIN ? 1-s*s/safe : 1);
        float dt=q>=FLT_MIN ? -float(a.amplitude_max)*inverse*s*t/safe : 0;
        g[4*atom]=da*ds;g[4*atom+1]=da*dt;g[4*atom+2]=dmo;g[4*atom+3]=dmi;
    }
}
} // namespace radius_cell_mass

// Definitions above are complete before the value-only helper is included.
#include "profile_values.cuh"
