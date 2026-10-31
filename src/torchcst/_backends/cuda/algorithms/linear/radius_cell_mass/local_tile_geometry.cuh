#pragma once
#include <cstddef>
// Included after hybrid.cuh's phase and compact-index helpers.
namespace radius_cell_mass {
constexpr int LOCAL_TILE_DIRECT = 2;
constexpr int LOCAL_TILE_RETAIN = 5;
constexpr int LOCAL_TILE_BATCH = 32;
constexpr int LOCAL_TILE_FEATURES = 17;
constexpr int LOCAL_TILE_SITES = 16;
constexpr int LOCAL_TILE_THREADS = 32;
constexpr std::size_t local_tile_shared_bytes(int batch) {
    // Complete [B,K,F] moments, unfiltered [T,F] phases, and F cell filters.
    return sizeof(double) * (batch * 3 * LOCAL_TILE_FEATURES +
        LOCAL_TILE_SITES * LOCAL_TILE_FEATURES + LOCAL_TILE_FEATURES);
}

__device__ inline void local_tile_basis_column(double *phi, double *filters,
    int f, const Axis &axis, const Bank &bank) {
    int harmonic = (f + 1) / 2;
    bool sine = f != 0 && f % 2 == 0;
    double filter = axis.spacing;
    if (harmonic) {
        double argument = PI * harmonic * axis.spacing / bank.period;
        filter *= sin(argument) / argument;
    }
    filters[f] = filter;
    HarmonicPhase phase(2 * PI * harmonic * axis.spacing / bank.period);
    for (int local = 0; local < bank.tile_sites; ++local) {
        phi[local * LOCAL_TILE_FEATURES + f] = sine ? phase.sine : phase.cosine;
        double c, s; phase.next(c, s);
    }
}

template<class T>
__device__ inline double local_tile_dot(const T *signal, int b, int first_tile,
    int part, int f, const Axis &axis, const Bank &bank,
    const double *phi, const double *filters) {
    double value = 0;
    for (int local = 0; local < bank.tile_sites; ++local) {
        int site = positive_mod((first_tile + part) * bank.tile_sites + local, axis.sites);
        // Keep direct compact_moment_element's x*filter*phase association.
        value += double(signal[int64_t(b) * axis.sites + site]) * filters[f] *
            phi[local * LOCAL_TILE_FEATURES + f];
    }
    return value;
}

template<bool Used> struct LocalTileRows {};
template<> struct LocalTileRows<true> { double value[LOCAL_TILE_BATCH] = {}; };

// Stage0: H or G only. Stage1: complete G and the output-center gradient.
// Stage2: input-center and amplitude gradients using saved H and completed G;
// it deliberately never constructs another H response.
template<class T, int Stage>
__device__ inline void local_tile_atom(const T *p, const T *signal,
    const double *retained, const double *local_moments, double *response,
    const double *h, const double *g, T *gradient, int atom, int atoms,
    int batch, Axis axis, bool output_side) {
    double R = radius(p, atom, axis), original = center(p, atom, axis, output_side);
    Membership member = classify(R, original, axis);
    Bank bank = axis.banks[member.bank];
    double mu = wrapped_center(original, axis);
    double c[17], unused_dr[17];
    profile_coefficients(R, bank.period, axis.modes, c, unused_dr);
    LocalTileRows<Stage != 2> values;
    LocalTileRows<Stage != 0> centers;
    if (bank.level < LOCAL_TILE_DIRECT) {
        for (int part = 0; part < bank.count; ++part) {
            for (int local = 0; local < bank.tile_sites; ++local) {
                int site = (member.first_tile + part) * bank.tile_sites + local;
                double mass = axis.spacing * c[0], dm = 0;
                HarmonicPhase phase(2 * PI / bank.period * (mu - site * axis.spacing));
                HarmonicPhase cell_phase(PI * axis.spacing / bank.period);
                for (int k = 1; k <= axis.modes; ++k) {
                    double cp, sp, cc, ss; phase.next(cp, sp); cell_phase.next(cc, ss);
                    double filter = axis.spacing * ss / (PI * k * axis.spacing / bank.period);
                    if constexpr (Stage != 2) mass += 2 * c[k] * filter * cp;
                    if constexpr (Stage != 0) dm -= 2 * c[k] * filter * (2 * PI * k / bank.period) * sp;
                }
                #pragma unroll
                for (int b = 0; b < LOCAL_TILE_BATCH; ++b) {
                    if (b >= batch) continue;
                    double x = double(signal[int64_t(b) * axis.sites + positive_mod(site, axis.sites)]);
                    if constexpr (Stage != 2) values.value[b] += x * mass;
                    if constexpr (Stage != 0) centers.value[b] += x * dm;
                }
            }
        }
    } else {
        int F = axis.basis_count(), prefix = prefix_nodes(axis, LOCAL_TILE_RETAIN);
        int nodes = axis.node_count - prefix;
        bool local = bank.level < LOCAL_TILE_RETAIN;
        for (int part = 0; part < bank.count; ++part) {
            int unwrapped_tile = member.first_tile + part;
            int node = bank.node_offset + positive_mod(unwrapped_tile, bank.tile_count);
            double displacement = mu - unwrapped_tile * bank.tile_sites * axis.spacing;
            if constexpr (Stage != 2) {
                #pragma unroll
                for (int b = 0; b < LOCAL_TILE_BATCH; ++b) {
                    if (b >= batch) continue;
                    const double *tile = local ? local_moments + (b * bank.count + part) * F
                        : retained + (int64_t(b) * nodes + node - prefix) * F;
                    values.value[b] += c[0] * tile[0];
                }
            }
            HarmonicPhase phase(2 * PI / bank.period * displacement);
            for (int k = 1; k <= axis.modes; ++k) {
                double cp, sp; phase.next(cp, sp);
                double omega = 2 * PI * k / bank.period;
                #pragma unroll
                for (int b = 0; b < LOCAL_TILE_BATCH; ++b) {
                    if (b >= batch) continue;
                    const double *tile = local ? local_moments + (b * bank.count + part) * F
                        : retained + (int64_t(b) * nodes + node - prefix) * F;
                    if constexpr (Stage != 2) {
                        double moment = cp * tile[2 * k - 1] + sp * tile[2 * k];
                        values.value[b] += 2 * c[k] * moment;
                    }
                    if constexpr (Stage != 0) centers.value[b] += 2 * c[k] * omega *
                        (-sp * tile[2 * k - 1] + cp * tile[2 * k]);
                }
            }
        }
    }
    double da = 0, dm = 0, amp = amplitude(p, atom, axis);
    #pragma unroll
                for (int b = 0; b < LOCAL_TILE_BATCH; ++b) {
                    if (b >= batch) continue;
        int64_t index = int64_t(b) * atoms + atom;
        if constexpr (Stage != 2) response[index] = values.value[b];
        if constexpr (Stage == 1) dm += amp * h[index] * centers.value[b];
        if constexpr (Stage == 2) {
            da += h[index] * g[index];
            dm += amp * g[index] * centers.value[b];
        }
    }
    if constexpr (Stage == 1) gradient[4 * atom + 3] = T(dm);
    if constexpr (Stage == 2) {
        // Same detached-radius Polar pullback as store_gradient. This stage
        // owns cols0/1/2 only; the output-owner stage already finished col3.
        double s = p[4 * atom], t = p[4 * atom + 1], q = s * s + t * t;
        double tiny = std::is_same<T, float>::value ? FLT_MIN : DBL_MIN;
        double safe = fmax(q, tiny), inverse = 1 / sqrt(safe);
        double ds = axis.amplitude_max * inverse * (q >= tiny ? 1 - s * s / safe : 1);
        double dt = q >= tiny ? -axis.amplitude_max * inverse * s * t / safe : 0;
        gradient[4 * atom] = T(da * ds); gradient[4 * atom + 1] = T(da * dt);
        gradient[4 * atom + 2] = T(dm);
    }
}
} // namespace radius_cell_mass
