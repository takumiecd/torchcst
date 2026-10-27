#pragma once
// Shared atom-local projection and derivative contraction.
#include "common.cuh"

namespace cell_mass_hierarchy {
// Accumulate a weighted projection derivative in scalar registers. On the
// output side, calculate G at the same time; input-side weights are the stored
// rounded G multiplied by amplitude. Gradients reassociate FP64 batch sums.
template <bool Value, int FixedModes, bool StoreCoefficients = false, typename projection_t>
__device__ inline void
project_weighted_derivatives(const double *z, int batches, int batch_begin, const Axis &axis,
                             const Bank &bank, int owner, const Profile &profile, double beta_sigma,
                             const projection_t *h, int atom_count, int slot, double amplitude,
                             double *g_value, double &dcenter, double &dsigma, bool windowed,
                             double *coefficients = nullptr, int stride = 0, int lane = 0) {
    // A windowed node already contains the rotated sum of its retained tiles.
    int retained = windowed || bank.full_chart ? 1 : 3;
    int first_tile = windowed ? owner : (bank.full_chart ? 0 : owner - 1);
    int mode_count = FixedModes ? FixedModes : axis.modes;
    HarmonicStream<true> stream(profile, mode_count);
    // The M=8 instantiation has a compile-time trip count. The generic
    // instantiation retains the runtime-selected profile mode count.
#pragma unroll
    for (int harmonic = 0; harmonic <= mode_count; ++harmonic) {
        double cc, cs, cc_beta, cs_beta;
        stream.next(cc, cs, cc_beta, cs_beta);
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
            int raw_tile = first_tile + offset;
            int tile = positive_mod(raw_tile, bank.tile_count);
            int image = (raw_tile - tile) / bank.tile_count;
            double rs, rc;
            image_rotation(bank, harmonic, image, rs, rc);
            int node = bank.node_offset + tile;
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

} // namespace cell_mass_hierarchy
