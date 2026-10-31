#pragma once
#include "common.cuh"

namespace radius_cell_mass {
// Responses/C and detached Polar task gradients need only c. Preserve the
// positive-term arithmetic in profile_coefficients_m8, omitting derivative
// work without rearranging the polynomial or changing normalization.
__device__ __noinline__ void profile_values_m8(double R, double P, double *c) {
    double angle = PI * (R / P), cb = cos(angle);
    double z = cb * cb, u = 1 - z;
    double zp[9], up[9];
    zp[0] = up[0] = 1;
    for (int j = 1; j <= 8; ++j) { zp[j] = zp[j - 1] * z; up[j] = up[j - 1] * u; }
    // Homogenization cancels the common z^-8 before it can overflow.
    double zero = 0;
    for (int k = 0; k < 8; ++k) {
        int degree = 8 - k;
        double raw = k == 0 ? 0.5 * zp[8] : 0;
        double value = 0;
        for (int j = 1; j <= degree; ++j) {
            int power = degree - j;
            double coefficient = profile_m8_terms[k][j - 1];
            value += coefficient * zp[power] * up[j];
        }
        double factor = 4.0 / (8 + k);
        raw += factor * value;
        if (k == 0) { zero = raw; c[0] = 1 / P; }
        else {
            c[k] = (raw / zero) / P;
        }
    }
    c[8] = (0.25 / zero) / P;
}

__device__ inline void profile_values(double R, double P, int M, double *c) {
    if (M == 8) profile_values_m8(R, P, c);
    else {
        double unused_derivatives[17];
        profile_coefficients_generic(R, P, M, c, unused_derivatives);
    }
}
} // namespace radius_cell_mass
