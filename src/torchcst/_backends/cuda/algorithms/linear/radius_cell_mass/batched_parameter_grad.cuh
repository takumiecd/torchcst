#pragma once
// Included after hybrid.cuh's scalar projection helpers.
namespace radius_cell_mass {
constexpr int PARAMETER_BATCH = 4;

// Four independent contractions share each tile phase. For each individual
// row, part/harmonic order and the FP64 expressions match compact_project.
__device__ inline void compact_derivatives_batch4(
    const double *z, int first_batch, int batch, const Axis &axis,
    Membership member, double mu, const double *c, const double *dr,
    int start, double *radius_gradient, double *center_gradient) {
    Bank bank = axis.banks[member.bank];
    int features = axis.basis_count();
    int nodes = axis.node_count - prefix_nodes(axis, start);
    for (int part = 0; part < bank.count; ++part) {
        int unwrapped_tile = member.first_tile + part;
        int node = bank.node_offset + positive_mod(unwrapped_tile, bank.tile_count);
        int relative = (node - prefix_nodes(axis, start)) * features;
        double displacement = mu - unwrapped_tile * bank.tile_sites * axis.spacing;
        HarmonicPhase phase(2 * PI / bank.period * displacement);
        for (int k = 1; k <= axis.modes; ++k) {
            double omega = 2 * PI * k / bank.period;
            double cp, sp; phase.next(cp, sp);
            #pragma unroll
            for (int r = 0; r < PARAMETER_BATCH; ++r) {
                int b = first_batch + r;
                if (b < batch) {
                    const double *tile = z + int64_t(b) * nodes * features + relative;
                    double moment = cp * tile[2 * k - 1] + sp * tile[2 * k];
                    radius_gradient[r] += 2 * dr[k] * moment;
                    center_gradient[r] += 2 * c[k] * omega *
                        (-sp * tile[2 * k - 1] + cp * tile[2 * k]);
                }
            }
        }
    }
}

// The cell mass derivatives depend on the atom, not the batch row. Evaluate
// them once per site, then read four signal rows. No partial response is made.
template<class T>
__device__ inline void direct_derivatives_batch4(
    const T *signal, int first_batch, int batch, Axis axis,
    Membership member, double mu, const double *c, const double *dr,
    double *radius_gradient, double *center_gradient) {
    Bank bank = axis.banks[member.bank];
    for (int part = 0; part < bank.count; ++part) {
        for (int local = 0; local < bank.tile_sites; ++local) {
            int site = (member.first_tile + part) * bank.tile_sites + local;
            double mr = 0, mm = 0;
            HarmonicPhase phase(2 * PI / bank.period * (mu - site * axis.spacing));
            double half_cell = PI * axis.spacing / bank.period;
            HarmonicPhase cell_phase(half_cell);
            for (int k = 1; k <= axis.modes; ++k) {
                double omega = 2 * PI * k / bank.period;
                double argument = PI * k * axis.spacing / bank.period;
                double cp, sp, cell_cosine, cell_sine;
                phase.next(cp, sp); cell_phase.next(cell_cosine, cell_sine);
                double filter = axis.spacing * cell_sine / argument;
                mr += 2 * dr[k] * filter * cp;
                mm -= 2 * c[k] * filter * omega * sp;
            }
            #pragma unroll
            for (int r = 0; r < PARAMETER_BATCH; ++r) {
                int b = first_batch + r;
                if (b < batch) {
                    double x = double(signal[int64_t(b) * axis.sites + positive_mod(site, axis.sites)]);
                    radius_gradient[r] += x * mr;
                    center_gradient[r] += x * mm;
                }
            }
        }
    }
}

// One thread still owns the complete atom gradient. Only geometry evaluation
// is shared across rows; final batch reduction remains b=0,1,... in FP64.
// Saved Z/H/G, all side/amplitude conventions and output storage are unchanged.
template<class T>
__global__ void batched_parameter_kernel(
    const T *p, const T *x, const T *dy, const double *zx, const double *zy,
    const double *h, const double *g, T *gradient, int atoms, int batch,
    Axis input, Axis output, int input_start, int output_start) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    double ri = radius(p, atom, input), ro = radius(p, atom, output);
    double mui = wrapped_center(center(p, atom, input, false), input);
    double muo = wrapped_center(center(p, atom, output, true), output);
    Membership mi = classify(ri, center(p, atom, input, false), input);
    Membership mo = classify(ro, center(p, atom, output, true), output);
    double ci[17], di[17], co[17], dout[17];
    profile_coefficients(ri, input.banks[mi.bank].period, input.modes, ci, di);
    profile_coefficients(ro, output.banks[mo.bank].period, output.modes, co, dout);
    double da = 0, dR = 0, dmi = 0, dmo = 0, amp = amplitude(p, atom, input);
    for (int first = 0; first < batch; first += PARAMETER_BATCH) {
        double dri[PARAMETER_BATCH] = {}, dci[PARAMETER_BATCH] = {};
        double dro[PARAMETER_BATCH] = {}, dco[PARAMETER_BATCH] = {};
        if (input.banks[mi.bank].level < input_start)
            direct_derivatives_batch4(x, first, batch, input, mi, mui, ci, di, dri, dci);
        else compact_derivatives_batch4(zx, first, batch, input, mi, mui, ci, di, input_start, dri, dci);
        if (output.banks[mo.bank].level < output_start)
            direct_derivatives_batch4(dy, first, batch, output, mo, muo, co, dout, dro, dco);
        else compact_derivatives_batch4(zy, first, batch, output, mo, muo, co, dout, output_start, dro, dco);
        #pragma unroll
        for (int r = 0; r < PARAMETER_BATCH; ++r) {
            int b = first + r;
            if (b < batch) {
                double H = h[int64_t(b) * atoms + atom], G = g[int64_t(b) * atoms + atom];
                da += H * G;
                dR += amp * (G * dri[r] + H * dro[r]);
                dmi += amp * G * dci[r];
                dmo += amp * H * dco[r];
            }
        }
    }
    store_gradient(p, gradient, atom, input, da, dR, dmi, dmo);
}
} // namespace radius_cell_mass
