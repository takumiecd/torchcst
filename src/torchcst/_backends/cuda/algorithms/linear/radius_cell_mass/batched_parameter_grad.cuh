#pragma once
// Included after hybrid.cuh's scalar projection helpers.
namespace radius_cell_mass {
#ifndef RADIUS_ROW_BATCH
#define RADIUS_ROW_BATCH 4
#endif
static_assert(RADIUS_ROW_BATCH == 4 || RADIUS_ROW_BATCH == 8,
    "only explicit row4 and row8 builds are qualified candidates");
constexpr int PARAMETER_BATCH = RADIUS_ROW_BATCH;

// Empty specializations ensure Polar allocates no radius-derivative scratch.
template<bool LearnRadius> struct RadiusScalar {};
template<> struct RadiusScalar<true> { double value = 0; };
template<bool LearnRadius> struct RadiusCoefficients {};
template<> struct RadiusCoefficients<true> { double input[17], output[17]; };
template<bool LearnRadius> struct RadiusRows {};
template<> struct RadiusRows<true> {
    double input[PARAMETER_BATCH] = {}, output[PARAMETER_BATCH] = {};
};

// A row group shares each tile phase (four normally; eight in the opt-in build).
// For each individual
// row, part/harmonic order and the FP64 expressions match compact_project.
template<bool LearnRadius, class Storage = double>
__device__ inline void compact_derivatives_batch4(
    const Storage *z, int first_batch, int batch, const Axis &axis,
    Membership member, Storage mu, const Storage *c, const Storage *dr,
    int start, Storage *radius_gradient, Storage *center_gradient) {
    Bank bank = axis.banks[member.bank];
    int features = axis.basis_count();
    int nodes = axis.node_count - prefix_nodes(axis, start);
    for (int part = 0; part < bank.count; ++part) {
        int unwrapped_tile = member.first_tile + part;
        int node = bank.node_offset + positive_mod(unwrapped_tile, bank.tile_count);
        int relative = (node - prefix_nodes(axis, start)) * features;
        Storage displacement = mu - unwrapped_tile * bank.tile_sites * Storage(axis.spacing);
        SignalPhase<Storage> phase(2 * Storage(PI) / Storage(bank.period) * displacement);
        for (int k = 1; k <= axis.modes; ++k) {
            Storage omega = 2 * Storage(PI) * k / Storage(bank.period);
            Storage cp, sp; phase.next(cp, sp);
            #pragma unroll
            for (int r = 0; r < PARAMETER_BATCH; ++r) {
                int b = first_batch + r;
                if (b < batch) {
                    const Storage *tile = z + int64_t(b) * nodes * features + relative;
                    if constexpr (LearnRadius) {
                        Storage moment = cp * tile[2 * k - 1] + sp * tile[2 * k];
                        radius_gradient[r] += 2 * dr[k] * moment;
                    }
                    center_gradient[r] += 2 * c[k] * omega *
                        (-sp * tile[2 * k - 1] + cp * tile[2 * k]);
                }
            }
        }
    }
}

// The cell mass derivatives depend on the atom, not the batch row. Evaluate
// them once per site, then read four signal rows. No partial response is made.
template<class T, bool LearnRadius, class Storage = double>
__device__ inline void direct_derivatives_batch4(
    const T *signal, int first_batch, int batch, Axis axis,
    Membership member, Storage mu, const Storage *c, const Storage *dr,
    Storage *radius_gradient, Storage *center_gradient) {
    Bank bank = axis.banks[member.bank];
    for (int part = 0; part < bank.count; ++part) {
        for (int local = 0; local < bank.tile_sites; ++local) {
            int site = (member.first_tile + part) * bank.tile_sites + local;
            RadiusScalar<LearnRadius> mr;
            Storage mm = 0;
            SignalPhase<Storage> phase(2 * Storage(PI) / Storage(bank.period) * (mu - site * Storage(axis.spacing)));
            Storage half_cell = Storage(PI) * Storage(axis.spacing) / Storage(bank.period);
            SignalPhase<Storage> cell_phase(half_cell);
            for (int k = 1; k <= axis.modes; ++k) {
                Storage omega = 2 * Storage(PI) * k / Storage(bank.period);
                Storage argument = Storage(PI) * k * Storage(axis.spacing) / Storage(bank.period);
                Storage cp, sp, cell_cosine, cell_sine;
                phase.next(cp, sp); cell_phase.next(cell_cosine, cell_sine);
                Storage filter = Storage(axis.spacing) * cell_sine / argument;
                if constexpr (LearnRadius) mr.value += 2 * dr[k] * filter * cp;
                mm -= 2 * c[k] * filter * omega * sp;
            }
            #pragma unroll
            for (int r = 0; r < PARAMETER_BATCH; ++r) {
                int b = first_batch + r;
                if (b < batch) {
                    Storage x = Storage(signal[int64_t(b) * axis.sites + positive_mod(site, axis.sites)]);
                    if constexpr (LearnRadius) radius_gradient[r] += x * mr.value;
                    center_gradient[r] += x * mm;
                }
            }
        }
    }
}

// One thread still owns the complete atom gradient. Only geometry evaluation
// is shared across rows; final batch reduction remains b=0,1,... in FP64.
// Saved Z/H/G, all side/amplitude conventions and output storage are unchanged.
template<class T, bool LearnRadius, class Storage = double>
__global__ void batched_parameter_kernel(
    const T *p, const T *x, const T *dy, const Storage *zx, const Storage *zy,
    const Storage *h, const Storage *g, T *gradient, int atoms, int batch,
    Axis input, Axis output, int input_start, int output_start) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    Storage ri = signal_radius<Storage>(p, atom, input), ro = signal_radius<Storage>(p, atom, output);
    Storage mui = signal_wrapped<Storage>(Storage(center(p, atom, input, false)), input);
    Storage muo = signal_wrapped<Storage>(Storage(center(p, atom, output, true)), output);
    Membership mi = signal_classify<Storage>(ri, Storage(center(p, atom, input, false)), input);
    Membership mo = signal_classify<Storage>(ro, Storage(center(p, atom, output, true)), output);
    Storage ci[17], co[17];
    RadiusCoefficients<LearnRadius> derivatives;
    const Storage *di = nullptr, *dout = nullptr;
    if constexpr (LearnRadius) {
        profile_coefficients(ri, input.banks[mi.bank].period, input.modes, ci, derivatives.input);
        profile_coefficients(ro, output.banks[mo.bank].period, output.modes, co, derivatives.output);
        di = derivatives.input; dout = derivatives.output;
    } else {
        profile_values(ri, input.banks[mi.bank].period, input.modes, ci);
        profile_values(ro, output.banks[mo.bank].period, output.modes, co);
    }
    RadiusScalar<LearnRadius> dR;
    Storage da = 0, dmi = 0, dmo = 0, amp = signal_amplitude<Storage>(p, atom, input);
    for (int first = 0; first < batch; first += PARAMETER_BATCH) {
        Storage dci[PARAMETER_BATCH] = {}, dco[PARAMETER_BATCH] = {};
        RadiusRows<LearnRadius> rows;
        Storage *dri = nullptr, *dro = nullptr;
        if constexpr (LearnRadius) { dri = rows.input; dro = rows.output; }
        if (input.banks[mi.bank].level < input_start)
            direct_derivatives_batch4<T, LearnRadius>(x, first, batch, input, mi, mui, ci, di, dri, dci);
        else compact_derivatives_batch4<LearnRadius>(zx, first, batch, input, mi, mui, ci, di, input_start, dri, dci);
        if (output.banks[mo.bank].level < output_start)
            direct_derivatives_batch4<T, LearnRadius>(dy, first, batch, output, mo, muo, co, dout, dro, dco);
        else compact_derivatives_batch4<LearnRadius>(zy, first, batch, output, mo, muo, co, dout, output_start, dro, dco);
        #pragma unroll
        for (int r = 0; r < PARAMETER_BATCH; ++r) {
            int b = first + r;
            if (b < batch) {
                Storage H = h[int64_t(b) * atoms + atom], G = g[int64_t(b) * atoms + atom];
                da += H * G;
                if constexpr (LearnRadius) dR.value += amp * (G * dri[r] + H * dro[r]);
                dmi += amp * G * dci[r];
                dmo += amp * H * dco[r];
            }
        }
    }
    Storage radius_gradient = 0;
    if constexpr (LearnRadius) radius_gradient = dR.value;
    // Torch factors() detaches derived Polar radii; store_gradient uses only
    // da and centers for Polar. Physical dR remains unchanged.
    signal_store_gradient<Storage>(p, gradient, atom, input, da, radius_gradient, dmi, dmo);
}
} // namespace radius_cell_mass
