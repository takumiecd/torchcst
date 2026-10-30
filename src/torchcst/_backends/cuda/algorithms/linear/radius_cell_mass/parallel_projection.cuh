#pragma once

namespace radius_cell_mass {

// One thread owns an atom and up to four complete batch responses. Geometry,
// profile coefficients and Fourier phases are reused across those batch rows.
// Moments are already complete: the preceding launch is the dependency edge.
constexpr int PARALLEL_RESPONSE_THREADS = 32;
constexpr int PARALLEL_RESPONSE_BATCH = 4;

template<class T>
__global__ void parallel_response_kernel(
    const T *signal, const T *p, const double *moments, double *response,
    int atoms, int batch, Axis axis, int start, bool output_side) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    int first_batch = blockIdx.y * PARALLEL_RESPONSE_BATCH;
    int features = axis.basis_count();
    int elements = (axis.node_count - prefix_nodes(axis, start)) * features;
    double R = radius(p, atom, axis);
    double original = center(p, atom, axis, output_side);
    Membership member = classify(R, original, axis);
    Bank bank = axis.banks[member.bank];
    double c[17], dr[17];
    profile_coefficients(R, bank.period, axis.modes, c, dr);
    double mu = wrapped_center(original, axis);
    double values[PARALLEL_RESPONSE_BATCH] = {};

    if (bank.level < start) {
        // Preserve direct_project's site and harmonic summation order for each
        // row. Only the common mass evaluation is shared across batch rows.
        for (int part = 0; part < bank.count; ++part) {
            for (int local = 0; local < bank.tile_sites; ++local) {
                int site = (member.first_tile + part) * bank.tile_sites + local;
                double mass = axis.spacing * c[0];
                HarmonicPhase phase(2 * PI / bank.period * (mu - site * axis.spacing));
                double half_cell = PI * axis.spacing / bank.period;
                HarmonicPhase cell_phase(half_cell);
                for (int k = 1; k <= axis.modes; ++k) {
                    double argument = PI * k * axis.spacing / bank.period;
                    double cp, sp, cell_cosine, cell_sine;
                    phase.next(cp, sp);
                    cell_phase.next(cell_cosine, cell_sine);
                    double filter = axis.spacing * cell_sine / argument;
                    mass += 2 * c[k] * filter * cp;
                }
                #pragma unroll
                for (int local_batch = 0; local_batch < PARALLEL_RESPONSE_BATCH; ++local_batch) {
                    int b = first_batch + local_batch;
                    if (b < batch) {
                        double x = double(signal[int64_t(b) * axis.sites + positive_mod(site, axis.sites)]);
                        values[local_batch] += x * mass;
                    }
                }
            }
        }
    } else {
        for (int part = 0; part < bank.count; ++part) {
            int unwrapped_tile = member.first_tile + part;
            int node = bank.node_offset + positive_mod(unwrapped_tile, bank.tile_count);
            int relative = (node - prefix_nodes(axis, start)) * features;
            double displacement = mu - unwrapped_tile * bank.tile_sites * axis.spacing;
            #pragma unroll
            for (int local_batch = 0; local_batch < PARALLEL_RESPONSE_BATCH; ++local_batch) {
                int b = first_batch + local_batch;
                if (b < batch) values[local_batch] += c[0] * moments[int64_t(b) * elements + relative];
            }
            HarmonicPhase phase(2 * PI / bank.period * displacement);
            for (int k = 1; k <= axis.modes; ++k) {
                double cp, sp;
                phase.next(cp, sp);
                #pragma unroll
                for (int local_batch = 0; local_batch < PARALLEL_RESPONSE_BATCH; ++local_batch) {
                    int b = first_batch + local_batch;
                    if (b < batch) {
                        const double *tile = moments + int64_t(b) * elements + relative;
                        double moment = cp * tile[2 * k - 1] + sp * tile[2 * k];
                        values[local_batch] += 2 * c[k] * moment;
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int local_batch = 0; local_batch < PARALLEL_RESPONSE_BATCH; ++local_batch) {
        int b = first_batch + local_batch;
        if (b < batch) response[int64_t(b) * atoms + atom] = values[local_batch];
    }
}

// General global-moment entry: projection is already complete on this stream.
// Reuse the same profile/phase once for up to four H/G rows, without shared
// projection's N128 budget or any atom-value cache/copy. The caller owns H/G.
inline void parallel_response_into(Tensor signal, Tensor p, Tensor z, Tensor response,
    Config config, bool output_side, int start) {
    check_atoms(p);
    Axis axis = parse_axis(config);
    check_hybrid_signal(signal, p, axis);
    check_compact(z, p, axis, start);
    TORCH_CHECK(z.size(0) == signal.size(0), "parallel response moment batch differs");
    check_response(response, p, signal.size(0));
    TORCH_CHECK(signal.size(0) <= 65535, "parallel response batch exceeds 65535");
    if (!response.numel()) return;
    c10::cuda::CUDAGuard guard(signal.device());
    dim3 grid((p.size(0) + PARALLEL_RESPONSE_THREADS - 1) / PARALLEL_RESPONSE_THREADS,
        (signal.size(0) + PARALLEL_RESPONSE_BATCH - 1) / PARALLEL_RESPONSE_BATCH);
    AT_DISPATCH_FLOATING_TYPES(signal.scalar_type(), "radius_parallel_response", [&] {
        parallel_response_kernel<scalar_t><<<grid, PARALLEL_RESPONSE_THREADS, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            signal.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(), z.data_ptr<double>(),
            response.data_ptr<double>(), p.size(0), signal.size(0), axis, start, output_side);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace radius_cell_mass
