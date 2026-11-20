#pragma once

namespace radius_cell_mass {
// Keep each warp in the source-tile order of the side it differentiates.
// Only two final scalars per Atom cross between the two launches.
template<bool OutputSide>
__global__ void axis_parameter_gradient(
    const float *p, const float *signal, const float *moments,
    const float *h, const float *g, const int64_t *output_order,
    float *output_partial, float *gradient, int atoms, int batch, Axis axis) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= atoms) return;
    int atom = OutputSide ? int(output_order[index]) : index;
    float radius = signal_radius<float>(p, atom, axis);
    float raw_center = float(center(p, atom, axis, OutputSide));
    float mu = signal_wrapped(raw_center, axis);
    Membership membership = signal_classify(radius, raw_center, axis);
    Bank bank = axis.banks[membership.bank];

    float coefficients[17], derivatives[17];
    physical_profile_coefficients(radius, bank.period, axis.modes, coefficients, derivatives);
    float amplitude = signal_amplitude<float>(p, atom, axis);
    float da = 0, dr = 0, dc = 0;
    constexpr int start = 2;
    for (int first = 0; first < batch; first += PARAMETER_BATCH) {
        float radius_rows[PARAMETER_BATCH] = {}, center_rows[PARAMETER_BATCH] = {};
        if (bank.level < start) {
            direct_derivatives_batch4<float, true>(signal, first, batch, axis, membership,
                mu, coefficients, derivatives, radius_rows, center_rows);
        } else {
            compact_derivatives_batch4<true>(moments, first, batch, axis, membership,
                mu, coefficients, derivatives, start, radius_rows, center_rows);
        }
        #pragma unroll
        for (int r = 0; r < PARAMETER_BATCH; ++r) {
            int row = first + r;
            if (row < batch) {
                int64_t offset = int64_t(row) * atoms + atom;
                float response = OutputSide ? h[offset] : g[offset];
                dr += amplitude * (response * radius_rows[r]);
                dc += amplitude * response * center_rows[r];
                if constexpr (!OutputSide) da += h[offset] * response;
            }
        }
    }
    if constexpr (OutputSide) {
        output_partial[atom] = dr;
        output_partial[int64_t(atoms) + atom] = dc;
    } else {
        signal_store_gradient<float>(p, gradient, atom, axis, da, dr + output_partial[atom],
            dc, output_partial[int64_t(atoms) + atom]);
    }
}
} // namespace radius_cell_mass
