#pragma once
// Shared atom-local projection and derivative contraction.
#include "common.cuh"

namespace cell_mass_hierarchy {
// The cached stream retains the input center phase; only the normalized
// magnitude and its beta slope are shared with the matching output profile.
// Each thread accesses its own shared-memory column, so no barrier is needed.
struct CachedMagnitudeStream {
    const double *cache;
    int stride, lane, harmonic;
    double inverse_period, phase_sine, phase_cosine, step_sine, step_cosine;

    __device__ CachedMagnitudeStream(const Profile &profile, const double *values, int cache_stride,
                                     int cache_lane)
        : cache(values), stride(cache_stride), lane(cache_lane), harmonic(0),
          inverse_period(profile.inverse_period), phase_sine(0), phase_cosine(1) {
        sincos(profile.omega * profile.center, &step_sine, &step_cosine);
    }

    __device__ inline void next(double &cc, double &cs, double &cc_beta, double &cs_beta) {
        if (harmonic == 0) {
            cc = inverse_period;
            cs = cc_beta = cs_beta = 0;
        } else {
            double next_cosine = phase_cosine * step_cosine - phase_sine * step_sine;
            phase_sine = phase_sine * step_cosine + phase_cosine * step_sine;
            phase_cosine = next_cosine;
            double magnitude = cache[(harmonic - 1) * stride + lane];
            double beta_magnitude = cache[(8 + harmonic - 1) * stride + lane];
            cc = magnitude * phase_cosine;
            cs = magnitude * phase_sine;
            cc_beta = beta_magnitude * phase_cosine;
            cs_beta = beta_magnitude * phase_sine;
        }
        ++harmonic;
    }
};

// The same loop handles regular and cached streams. Magnitudes are stored
// directly before phase multiplication, never recovered by division.
template <bool Value, int FixedModes, bool StoreCoefficients, bool WriteMagnitudes,
          typename projection_t, typename Stream>
__device__ inline void
project_weighted_impl(const double *z, int batches, int batch_begin, const Axis &axis,
                      const Bank &bank, int owner, const Profile &profile, double beta_sigma,
                      const projection_t *h, int atom_count, int slot, double amplitude,
                      double *g_value, double &dcenter, double &dsigma, bool windowed,
                      double *coefficients, int stride, int lane, Stream &stream,
                      double *magnitude_cache, int cache_stride, int cache_lane) {
    // A windowed node already contains the rotated sum of its retained tiles.
    int retained = windowed || bank.full_chart ? 1 : 3;
    int first_tile = windowed ? owner : (bank.full_chart ? 0 : owner - 1);
    int window_node = bank.node_offset + owner;
    int mode_count = FixedModes ? FixedModes : axis.modes;
    // The M=8 instantiation has a compile-time trip count. The generic
    // instantiation retains the runtime-selected profile mode count.
#pragma unroll
    for (int harmonic = 0; harmonic <= mode_count; ++harmonic) {
        double cc, cs, cc_beta, cs_beta;
        double magnitude = 0, beta_magnitude = 0;
        if constexpr (WriteMagnitudes) {
            stream.next(cc, cs, cc_beta, cs_beta, &magnitude, &beta_magnitude);
            if (harmonic != 0) {
                magnitude_cache[(harmonic - 1) * cache_stride + cache_lane] = magnitude;
                magnitude_cache[(8 + harmonic - 1) * cache_stride + cache_lane] = beta_magnitude;
            }
        } else {
            stream.next(cc, cs, cc_beta, cs_beta);
        }
        if (StoreCoefficients) {
            if (harmonic == 0)
                coefficients[lane] = cc;
            else {
                coefficients[(2 * harmonic - 1) * stride + lane] = cc;
                coefficients[2 * harmonic * stride + lane] = cs;
            }
        }
        double omega = profile.omega * harmonic;
        for (int offset = 0; offset < retained; ++offset) {
            double rs, rc;
            int node;
            if (windowed) {
                // Window moments already include the retained tile rotations.
                node = window_node;
                rs = 0;
                rc = 1;
            } else {
                int raw_tile = first_tile + offset;
                int tile = positive_mod(raw_tile, bank.tile_count);
                int image = (raw_tile - tile) / bank.tile_count;
                image_rotation(bank, harmonic, image, rs, rc);
                node = bank.node_offset + tile;
            }
            double weighted_cos = 0, weighted_sin = 0;
            for (int b = 0; b < BATCH_TILE && batch_begin + b < batches; ++b) {
                int base = ((batch_begin + b) * axis.node_count + node) * axis.basis_count();
                if (harmonic == 0) {
                    if (Value)
                        g_value[b] += cc * z[base];
                    continue;
                }
                double zc = z[base + 2 * harmonic - 1], zs = z[base + 2 * harmonic];
                if (Value) {
                    double rotated_cos = zc * rc - zs * rs;
                    double rotated_sin = zs * rc + zc * rs;
                    g_value[b] += cc * rotated_cos + cs * rotated_sin;
                    double weight =
                        amplitude * static_cast<double>(h[(batch_begin + b) * atom_count + slot]);
                    weighted_cos += weight * rotated_cos;
                    weighted_sin += weight * rotated_sin;
                } else {
                    weighted_cos += g_value[b] * zc;
                    weighted_sin += g_value[b] * zs;
                }
            }
            if (harmonic != 0) {
                if (!Value) {
                    // Rotation is shared across batches, so apply it once to
                    // their weighted Z sum. This reassociates FP64 addition.
                    double rotated_cos = weighted_cos * rc - weighted_sin * rs;
                    weighted_sin = weighted_sin * rc + weighted_cos * rs;
                    weighted_cos = rotated_cos;
                }
                dcenter += omega * (-cs * weighted_cos + cc * weighted_sin);
                dsigma += beta_sigma * (cc_beta * weighted_cos + cs_beta * weighted_sin);
            }
        }
    }
}

// Accumulate a weighted projection derivative in scalar registers. On the
// output side, calculate G at the same time; input-side weights are the stored
// rounded G multiplied by amplitude. Gradients reassociate FP64 batch sums.
template <bool Value, int FixedModes, bool StoreCoefficients = false, bool Compensated = false,
          bool WriteMagnitudes = false, typename projection_t>
__device__ inline void project_weighted_derivatives(
    const double *z, int batches, int batch_begin, const Axis &axis, const Bank &bank, int owner,
    const Profile &profile, double beta_sigma, const projection_t *h, int atom_count, int slot,
    double amplitude, double *g_value, double &dcenter, double &dsigma, bool windowed,
    double *coefficients = nullptr, int stride = 0, int lane = 0, double *magnitude_cache = nullptr,
    int cache_stride = 0, int cache_lane = 0) {
    int mode_count = FixedModes ? FixedModes : axis.modes;
    HarmonicStream<true, Compensated> stream(profile, mode_count);
    project_weighted_impl<Value, FixedModes, StoreCoefficients, WriteMagnitudes>(
        z, batches, batch_begin, axis, bank, owner, profile, beta_sigma, h, atom_count, slot,
        amplitude, g_value, dcenter, dsigma, windowed, coefficients, stride, lane, stream,
        magnitude_cache, cache_stride, cache_lane);
}

template <int FixedModes, typename projection_t>
__device__ inline void project_cached_input_derivatives(
    const double *z, int batches, int batch_begin, const Axis &axis, const Bank &bank, int owner,
    const Profile &profile, double beta_sigma, const projection_t *h, int atom_count, int slot,
    double amplitude, double *g_value, double &dcenter, double &dsigma, bool windowed,
    const double *magnitude_cache, int cache_stride, int cache_lane) {
    CachedMagnitudeStream stream(profile, magnitude_cache, cache_stride, cache_lane);
    project_weighted_impl<false, FixedModes, false, false>(
        z, batches, batch_begin, axis, bank, owner, profile, beta_sigma, h, atom_count, slot,
        amplitude, g_value, dcenter, dsigma, windowed, nullptr, 0, 0, stream, nullptr, 0, 0);
}

} // namespace cell_mass_hierarchy
