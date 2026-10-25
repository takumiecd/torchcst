// 128 sites・128 atomのCUDA実装。読む順番は次の通り。
//   1. block128_forward_sorted_cuda：実行用の並びとnormを準備する。
//   2. block128：Z → H/G → C → Y/dXを同じblock内で進める。
//   3. block128_backward_sorted_cuda：保存した並びで逆伝播し、勾配を元の行へ戻す。
// 各threadは別atomを担当する。担当外のthreadもblock同期へ必ず到達する。
// Parameterの列は、0=振幅、1=出力中心、2=入力中心、3=幅logit。
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include "block128_math.h"
#include "block128_layout.h"

namespace {
using namespace cst128;

// 128個のkeyを、shared上で比較・交換する。
// 各段階では互いに重ならないペアを扱い、全ペアが終わってから次の段階へ進む。
// Parameterの並べ替えはこの後で行うので、ここで動くのは小さなkeyだけ。
__device__ void sort_atom_keys(AtomSortKey *keys) {
    int slot = threadIdx.x;

    for (int sequence_size = 2; sequence_size <= A; sequence_size *= 2) {
        for (int distance = sequence_size / 2; distance > 0; distance /= 2) {
            int partner = slot ^ distance;

            if (slot < partner) {
                bool ascending = (slot & sequence_size) == 0;
                bool exchange = ascending ? atom_key_less(keys[partner], keys[slot])
                                          : atom_key_less(keys[slot], keys[partner]);

                if (exchange) {
                    AtomSortKey temporary = keys[slot];
                    keys[slot] = keys[partner];
                    keys[partner] = temporary;
                }
            }

            __syncthreads();
        }
    }
}

// forwardごとに現在のPから実行用配置を作る。
// global出力：sorted_P[128,4]、source_rows[128]、atom_groups[128]、
// ranges[2軸,4担当タイル,12階層,2(begin/end)]。
// keyと階層ごとの半幅上限は、このkernel内のsharedだけで使い終える。
__global__ void prepare_atom_layout(const float *parameters, float *sorted_parameters,
                                    int64_t *source_rows, int *atom_groups, int *ranges) {
    __shared__ AtomSortKey keys[A];
    __shared__ GroupSpan groups[HIERARCHY_GROUPS];

    int slot = threadIdx.x;
    keys[slot] = make_atom_sort_key(parameters, slot);
    __syncthreads();

    sort_atom_keys(keys);

    AtomSortKey key = keys[slot];
    source_rows[slot] = key.source_row;
    atom_groups[slot] = key.group;

    for (int field = 0; field < 4; ++field) {
        sorted_parameters[slot * 4 + field] = parameters[key.source_row * 4 + field];
    }

    if (slot < HIERARCHY_GROUPS) {
        groups[slot] = find_group_span(keys, slot);
    }
    __syncthreads();

    // 各担当block・各階層の範囲は一度だけ準備する。
    // 本体ではこのbegin/endを読み、その範囲を追加選別せず計算する。
    for (int range_number = slot; range_number < BLOCK_RANGE_COUNT; range_number += blockDim.x) {
        int group = range_number % HIERARCHY_GROUPS;
        int output_tile = (range_number / HIERARCHY_GROUPS) % OUTPUT_TILES;
        int axis = range_number / (HIERARCHY_GROUPS * OUTPUT_TILES);
        AtomRange range = get_range(keys, groups[group], group, axis, output_tile);

        ranges[range_number * 2] = range.begin;
        ranges[range_number * 2 + 1] = range.end;
    }
}

// 振幅を、混合profileの全域normで割った係数。H/G自体は未スケールで保持する。
__device__ double alpha(const float *p, const double *norm, int a) {
    double raw = ::sqrt(norm[a * 6] * norm[a * 6 + 3]);
    return static_cast<double>(p[a * 4]) / ::fmax(raw, FLOOR);
}

// 各threadが一つのatomの正規化情報を作る。
// norm[atom][出力/入力][3]に、二乗norm・中心微分との内積・幅微分との内積を保存する。
// 混合成分を足してから全128 sitesで測る。タイル単位で正規化し直さない。
__global__ void prepare_norms(const float *p, const double *fit, double *norm) {
    int a = threadIdx.x;
    double width = .1 + 9.9 * sigmoid(p[a * 4 + 3]);
    Component comp[4];
    double co[4][M + 1], ci[4][M + 1];
    for (int c = 0; c < 4; ++c) {
        comp[c] = component(width, c);
        coefficients(fit, width, comp[c], co[c], ci[c]);
    }
    for (int axis = 0; axis < 2; ++axis) {
        double center = p[a * 4 + 1 + axis];
        double n = 0.0;
        double nm = 0.0;
        double ni = 0.0;
        for (int i = 0; i < N; ++i) {
            double f[3] = {};
            for (int c = 0; c < 4; ++c) {
                double term[3];
                series(i, center, comp[c], co[c], ci[c], term);
                for (int k = 0; k < 3; ++k) {
                    f[k] += term[k];
                }
            }
            n += f[0] * f[0];
            nm += f[0] * f[1];
            ni += f[0] * f[2];
        }
        norm[(a * 2 + axis) * 3] = n;
        norm[(a * 2 + axis) * 3 + 1] = nm;
        norm[(a * 2 + axis) * 3 + 2] = ni;
    }
}

// 一つのZタイルを、(batch,basis)の要素ごとにthreadへ分けて作る。
// 親の偶数modeは同じ周波数の子Zを足せる。それ以外は入力値から計算する。
// z[node][batch][basis]を書き、親子間の同期は呼出し側で行う。
__device__ void build_z_tile(const float *source, double *z, int node) {
    int bank = node_bank(node);
    int first = node_start(node);
    int size = bank_size(bank);
    for (int q = threadIdx.x; q < BATCH * R; q += blockDim.x) {
        int b = q / R;
        int r = q % R;
        int mode = r <= M ? r : r - M;
        double value = 0.;
        if (bank >= 4 && mode % 2 == 0) {
            int cb = bank - 4, child = bank_offset(cb) + first / bank_size(cb);
            int cr = mode / 2 + (r <= M ? 0 : M);
            value = z[(child * BATCH + b) * R + cr] + z[((child + 1) * BATCH + b) * R + cr];
        } else {
            for (int i = first; i < first + size; ++i) {
                value += static_cast<double>(source[b * N + i]) * basis(bank, r, i);
            }
        }
        z[node * BATCH * R + q] = value;
    }
}

// 一つのatom成分について、入力とprofileの内積をresultへ「加算」する。
// result[field][batch]のfieldは値・中心微分・幅微分の3項目。
// 支持内の完全なタイルはZを使い、端のsiteはsourceから直接足す。
// 各atomを担当する一つのthreadが、この関数を必要な成分数だけ呼ぶ。
__device__ void contract_component(const float *source, const double *z, double center,
                                   Component comp, const double *co, const double *ci,
                                   bool derivatives, double result[3][BATCH]) {
    int lo, hi;
    bounds(center, comp.length, lo, hi);
    int left = lo, right = lo;
    if (comp.bank >= 0) {
        int size = bank_size(comp.bank);
        int first = (lo + size - 1) / size, last = hi / size;
        if (last > first) {
            for (int t = first; t < last; ++t) {
                int node = bank_offset(comp.bank) + t;
                for (int r = 0; r < R; ++r) {
                    int mode = r <= M ? r : r - M;
                    double beta_fields[3];
                    beta(comp, center, r, co[mode], ci[mode], beta_fields);
                    for (int b = 0; b < BATCH; ++b) {
                        for (int k = 0; k < (derivatives ? 3 : 1); ++k) {
                            result[k][b] += z[(node * BATCH + b) * R + r] * beta_fields[k];
                        }
                    }
                }
            }
            left = first * size;
            right = last * size;
        }
    }
    for (int i = lo; i < hi; ++i) {
        if (i >= left && i < right) {
            continue;
        }
        double f[3];
        series(i, center, comp, co, ci, f);
        for (int b = 0; b < BATCH; ++b) {
            for (int k = 0; k < (derivatives ? 3 : 1); ++k) {
                result[k][b] += static_cast<double>(source[b * N + i]) * f[k];
            }
        }
    }
}

// 現在のthreadが担当する一つのatomについて、混合成分をH/Gへ集約する。
// HはXから、GはdYから作る。各値は全支持の縮約が終わってから他threadへ渡す。
// この関数には同期を入れない。呼出し側がsharedへ保存し、block全員で同期する。
template <bool BACKWARD>
__device__ __forceinline__ void
compute_atom_responses(const float *p, const double *fit, const float *sx, const float *sy,
                       const double *zx, const double *zg, bool owner, bool needs_response,
                       double h[3][BATCH], double g[3][BATCH]) {
    int a = threadIdx.x;
    // dP担当blockだけは、dXの担当範囲に関係なく全atomのH/Gを作る。
    // dPはatomの全支持を使うため、出力32 sitesへ切り詰めてはいけない。
    bool compute_h = owner || (!BACKWARD && needs_response);
    bool compute_g = BACKWARD && (owner || needs_response);

    if (!compute_h && !compute_g) {
        return;
    }
    double width = .1 + 9.9 * sigmoid(p[a * 4 + 3]);
    for (int k = 0; k < 4; ++k) {
        Component cp = component(width, k);
        if (!(cp.weight > 0. || cp.width_derivative != 0.)) {
            continue;
        }
        double co[M + 1], ci[M + 1];
        coefficients(fit, width, cp, co, ci);
        if (compute_h) {
            contract_component(sx, zx, p[a * 4 + 2], cp, co, ci, owner, h);
        }
        if (compute_g) {
            contract_component(sy, zg, p[a * 4 + 1], cp, co, ci, owner, g);
        }
    }
}

// HとGを使って、一つの小バッチ分のatom勾配を保存する。
// 幅・中心にはnormの微分も含める。振幅0でも振幅勾配を計算する。
// 出力はdp[batch_group][atom][parameter_field]で、後のkernelでbatch群を合算する。
__device__ void write_parameter_partial(const float *p, const double *norm, double h[3][BATCH],
                                        double g[3][BATCH], double *dp, int batch_group) {
    int a = threadIdx.x;
    double nu = norm[a * 6];
    double nv = norm[a * 6 + 3];
    double raw = ::sqrt(nu * nv);
    double denom = ::fmax(raw, FLOOR);
    double al = static_cast<double>(p[a * 4]) / denom;
    double s = 0.0;
    double om = 0.0;
    double im = 0.0;
    double w = 0.0;
    for (int b = 0; b < BATCH; ++b) {
        s += h[0][b] * g[0][b];
        om += h[0][b] * g[1][b];
        im += h[1][b] * g[0][b];
        w += h[2][b] * g[0][b] + h[0][b] * g[2][b];
    }
    double lmout = 0., lmin = 0., lw = 0.;
    if (raw > FLOOR && nu > 0. && nv > 0.) {
        lmout = norm[a * 6 + 1] / nu;
        lmin = norm[a * 6 + 4] / nv;
        lw = norm[a * 6 + 2] / nu + norm[a * 6 + 5] / nv;
    }
    double sig = sigmoid(p[a * 4 + 3]);
    int offset = (batch_group * A + a) * 4;
    dp[offset] = s / denom;
    dp[offset + 1] = al * (om - s * lmout);
    dp[offset + 2] = al * (im - s * lmin);
    dp[offset + 3] = al * (w - s * lw) * 9.9 * sig * (1. - sig);
}

// 完成したH/Gから、担当出力タイルで使うC係数を作る。
// C[bank][batch][basis]の一要素を一threadが担当し、対象atomの寄与を足す。
// 同じ要素への複数writerを作らないのでatomicAddは不要。
// タイル全体が支持内に入る成分だけCへ入れ、残りはstore_outputで直接扱う。
template <bool SORTED>
__device__ void build_c(const float *p, const double *fit, const double *norm, const double *values,
                        double *cbuf, int axis, int output_begin, const int *ranges) {
    for (int q = threadIdx.x; q < BANKS * BATCH * R; q += blockDim.x) {
        // qはC[bank][batch][basis]を一列にした番号。
        int bank = q / (BATCH * R);
        int b = q / R % BATCH;
        int r = q % R;
        int size = bank_size(bank);
        int start = output_begin / size * size;
        int mode = r <= M ? r : r - M;
        double sum = 0.;
        int group_count = SORTED ? HIERARCHY_GROUPS : 1;
        for (int group = 0; group < group_count; ++group) {
            AtomRange range{0, A};
            if constexpr (SORTED) {
                range = read_block_range(ranges, axis, output_begin / OUTPUT_TILE_SITES, group);
            }
            for (int a = range.begin; a < range.end; ++a) {
                double width = .1 + 9.9 * sigmoid(p[a * 4 + 3]);
                double center = p[a * 4 + 1 + axis];
                double scaled = alpha(p, norm, a) * values[a * BATCH + b];
                for (int k = 0; k < 4; ++k) {
                    Component cp = component(width, k);
                    if (cp.bank != bank) {
                        continue;
                    }
                    int lo, hi;
                    bounds(center, cp.length, lo, hi);
                    if (lo > start || hi < start + size) {
                        continue;
                    }
                    double co, ci, bt[3];
                    coefficient(fit, width, cp, mode, co, ci);
                    beta(cp, center, r, co, ci, bt);
                    sum += scaled * bt[0];
                }
            }
        }
        cbuf[q] = sum; // assigns every element, including unused/empty nodes
    }
}

// Cを基底へ展開し、直接成分・支持端の寄与を足してY/dXを保存する。
// 各threadの保存先は一つの(batch,site)。同じ結果を別threadが上書きしない。
template <bool SORTED>
__device__ void store_output(const float *p, const double *fit, const double *norm,
                             const double *values, const double *cbuf, float *output,
                             int batch_count, int batch_begin, int output_begin, int axis,
                             const int *ranges) {
    int b = threadIdx.x / 32;
    int i = output_begin + threadIdx.x % 32;
    double acc = 0.;
    for (int bank = 0; bank < BANKS; ++bank) {
        for (int r = 0; r < R; ++r) {
            acc += cbuf[(bank * BATCH + b) * R + r] * basis(bank, r, i);
        }
    }
    int group_count = SORTED ? HIERARCHY_GROUPS : 1;
    for (int group = 0; group < group_count; ++group) {
        AtomRange range{0, A};
        if constexpr (SORTED) {
            range = read_block_range(ranges, axis, output_begin / OUTPUT_TILE_SITES, group);
        }
        for (int a = range.begin; a < range.end; ++a) {
            double width = .1 + 9.9 * sigmoid(p[a * 4 + 3]);
            double center = p[a * 4 + 1 + axis];
            double direct = 0.;
            for (int k = 0; k < 4; ++k) {
                Component cp = component(width, k);
                if (!(cp.weight > 0. || cp.width_derivative != 0.)) {
                    continue;
                }
                int lo, hi;
                bounds(center, cp.length, lo, hi);
                if (i < lo || i >= hi) {
                    continue;
                }
                if (cp.bank >= 0) {
                    int size = bank_size(cp.bank), start = i / size * size;
                    if (lo <= start && hi >= start + size) {
                        continue; // already in C
                    }
                }
                double co[M + 1], ci[M + 1], f[3];
                coefficients(fit, width, cp, co, ci);
                series(i, center, cp, co, ci, f);
                direct += f[0];
            }
            acc += alpha(p, norm, a) * values[a * BATCH + b] * direct;
        }
    }
    if (batch_begin + b < batch_count) {
        output[(batch_begin + b) * N + i] = static_cast<float>(acc);
    }
}

// 一つのblockは「4バッチ × 出力32 sites」を最後まで担当する。
// phaseごとに同じthreadsの役割が変わる：Z要素 → atom → C要素 → 出力要素。
// SORTED=falseは全atomを処理する比較基準、trueは準備済みbegin/endを使う。
template <bool BACKWARD, bool SORTED = false>
__global__ void block128(const float *x, const float *p, const float *dy, const double *fit,
                         const double *norm, float *output, double *dp, int batch_count,
                         int64_t *clocks, const int *atom_groups, const int *ranges) {
    extern __shared__ __align__(16) unsigned char arena[];
    int a = threadIdx.x;                  // このthreadが担当するatomのslot
    int tile = blockIdx.x % OUTPUT_TILES; // このblockの出力担当タイル
    int batch_group = blockIdx.x / OUTPUT_TILES;
    int batch_begin = batch_group * BATCH;
    bool owner = BACKWARD && tile == 0; // 各batch群で、dPを書くblockは一つだけ
    // shared arenaのview。Zを使い終わった後、その領域をCとして再利用する。
    // ownerはHとGの両方を作るため、X側とdY側の領域を別々に持つ。
    float *sx = reinterpret_cast<float *>(arena);
    float *sy = reinterpret_cast<float *>(arena + (owner ? 2048 : 0));
    double *zx = reinterpret_cast<double *>(arena + (owner ? 4096 : 2048));
    double *zg = owner ? reinterpret_cast<double *>(arena + 19328) : zx;
    double *values = reinterpret_cast<double *>(arena + (owner ? 34560 : 17280));
    // 128版では、ソート済みの全Parameter 2 KiBを一度sharedへ読む。
    // H/GとC/直接出力が同じ表を再利用する。モデル正本をsharedへ移すわけではない。
    int parameter_offset = BACKWARD ? BACKWARD_SHARED : FORWARD_SHARED;
    float *shared_parameters = reinterpret_cast<float *>(arena + parameter_offset);
    if constexpr (SORTED) {
        for (int field_index = a; field_index < A * 4; field_index += blockDim.x) {
            shared_parameters[field_index] = p[field_index];
        }
        p = shared_parameters;
    }
    // [init] Each thread reads four source elements per required side.
    for (int q = a; q < BATCH * N; q += blockDim.x) {
        int b = batch_begin + q / N;
        int i = q % N;
        if (!BACKWARD || owner) {
            sx[q] = b < batch_count ? x[b * N + i] : 0.f;
        }
        if (BACKWARD) {
            sy[q] = b < batch_count ? dy[b * N + i] : 0.f;
        }
    }
    __syncthreads();
    if (clocks && a == 0) {
        clocks[blockIdx.x * 5] = clock64();
    }
    // [Z] Input tiles -> shared basis moments, with child-to-parent dependencies.
    for (int node = 0; node < NODES; ++node) {
        if (!BACKWARD || owner) {
            build_z_tile(sx, zx, node);
        }
        if (BACKWARD) {
            build_z_tile(sy, zg, node);
        }
        __syncthreads();
    }
    if (clocks && a == 0) {
        clocks[blockIdx.x * 5 + 1] = clock64();
    }
    // [H/G] Shared Z -> one complete response per atom, owned by this thread.
    double h[3][BATCH] = {}; // thread専用：Xからの縮約
    double g[3][BATCH] = {}; // thread専用：dYからの縮約
    bool needs_response = true;
    if constexpr (SORTED) {
        int axis = BACKWARD ? INPUT_AXIS : OUTPUT_AXIS;
        AtomRange range = read_block_range(ranges, axis, tile, atom_groups[a]);
        needs_response = range.begin <= a && a < range.end;
    }
    compute_atom_responses<BACKWARD>(p, fit, sx, sy, zx, zg, owner, needs_response, h, g);
    for (int b = 0; b < BATCH; ++b) {
        values[a * BATCH + b] = BACKWARD ? g[0][b] : h[0][b];
    }
    if (owner) {
        write_parameter_partial(p, norm, h, g, dp, batch_group);
    }
    __syncthreads(); // no thread can read Z after this point
    if (clocks && a == 0) {
        clocks[blockIdx.x * 5 + 2] = clock64();
    }
    // [C] Shared H/G -> output-tile coefficients, with one writer per C element.
    // Reuse Z_X (or Z_dY for dX-only blocks) only after the above barrier.
    build_c<SORTED>(p, fit, norm, values, zx, BACKWARD ? 1 : 0, tile * 32, ranges);
    __syncthreads();
    if (clocks && a == 0) {
        clocks[blockIdx.x * 5 + 3] = clock64();
    }
    // [Y/dX] Expand C and add direct/boundary terms; one writer per result element.
    store_output<SORTED>(p, fit, norm, values, zx, output, batch_count, batch_begin, tile * 32,
                         BACKWARD ? 1 : 0, ranges);
    if (clocks) {
        __syncthreads(); // Diagnostic only: account for the last output writer.
        if (a == 0) {
            clocks[blockIdx.x * 5 + 4] = clock64();
        }
    }
}

// 小バッチ群ごとのdPを合算し、最終gradientへ一度だけ書く。
// ソート版はsource_rowsで元の行へ戻す。行対応は順列なのでatomicAddは不要。
__global__ void reduce_parameters(const double *partial, float *dp, int groups,
                                  const int64_t *source_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A * 4) {
        return;
    }
    double sum = 0.;
    for (int g = 0; g < groups; ++g) {
        sum += partial[g * A * 4 + i];
    }
    int sorted_slot = i / 4;
    int parameter_field = i % 4;
    int source_row = source_rows ? static_cast<int>(source_rows[sorted_slot]) : sorted_slot;
    dp[source_row * 4 + parameter_field] = static_cast<float>(sum);
}
} // namespace

using namespace cst128;

std::vector<at::Tensor> block128_forward_cuda(at::Tensor x, at::Tensor p, at::Tensor fit,
                                              bool profile) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    auto y = at::empty_like(x);
    auto norm = at::empty({128, 2, 3}, x.options().dtype(at::kDouble));
    auto clocks = profile ? at::empty({4 * ((x.size(0) + 3) / 4), 5}, x.options().dtype(at::kLong))
                          : at::Tensor();
    prepare_norms<<<1, 128, 0, stream>>>(p.data_ptr<float>(), fit.data_ptr<double>(),
                                         norm.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    block128<false><<<4 * ((x.size(0) + 3) / 4), 128, cst128::FORWARD_SHARED, stream>>>(
        x.data_ptr<float>(), p.data_ptr<float>(), nullptr, fit.data_ptr<double>(),
        norm.data_ptr<double>(), y.data_ptr<float>(), nullptr, x.size(0),
        profile ? clocks.data_ptr<int64_t>() : nullptr, nullptr, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (profile) {
        return {y, norm, clocks};
    }
    return {y, norm};
}

std::vector<at::Tensor> block128_backward_cuda(at::Tensor x, at::Tensor p, at::Tensor dy,
                                               at::Tensor fit, at::Tensor norm, bool profile) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    int groups = (x.size(0) + 3) / 4;
    auto dx = at::empty_like(x);
    auto dp = at::empty_like(p);
    auto partial = at::empty({groups, 128, 4}, x.options().dtype(at::kDouble));
    auto clocks = profile ? at::empty({4 * groups, 5}, x.options().dtype(at::kLong)) : at::Tensor();
    block128<true><<<4 * groups, 128, cst128::BACKWARD_SHARED, stream>>>(
        x.data_ptr<float>(), p.data_ptr<float>(), dy.data_ptr<float>(), fit.data_ptr<double>(),
        norm.data_ptr<double>(), dx.data_ptr<float>(), partial.data_ptr<double>(), x.size(0),
        profile ? clocks.data_ptr<int64_t>() : nullptr, nullptr, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    reduce_parameters<<<4, 128, 0, stream>>>(partial.data_ptr<double>(), dp.data_ptr<float>(),
                                             groups, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (profile) {
        return {dx, dp, clocks};
    }
    return {dx, dp};
}

std::vector<int64_t> block128_resources_cuda(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "resource query requires a CUDA tensor");
    c10::cuda::CUDAGuard guard(x.device());
    cudaFuncAttributes prep{}, forward{}, backward{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&prep, prepare_norms));
    C10_CUDA_CHECK(cudaFuncGetAttributes(&forward, block128<false>));
    C10_CUDA_CHECK(cudaFuncGetAttributes(&backward, block128<true>));
    std::vector<int64_t> result;
    for (auto attr : {prep, forward, backward}) {
        result.push_back(attr.numRegs);
        result.push_back(attr.localSizeBytes);
        result.push_back(attr.sharedSizeBytes);
        result.push_back(attr.maxThreadsPerBlock);
    }
    return result;
}

// Pythonへ返す順番：sorted_P, source_rows, atom_groups, ranges。
// 正本Pは書き換えない。実行用コピーを毎forwardで作るので、optimizer更新後も古いkeyを使わない。
std::vector<at::Tensor> block128_prepare_layout_cuda(at::Tensor parameters) {
    c10::cuda::CUDAGuard guard(parameters.device());
    auto stream = at::cuda::getCurrentCUDAStream(parameters.get_device());

    auto sorted_parameters = at::empty_like(parameters);
    auto source_rows = at::empty({A}, parameters.options().dtype(at::kLong));
    auto atom_groups = at::empty({A}, parameters.options().dtype(at::kInt));
    auto ranges = at::empty({RANGE_AXES, OUTPUT_TILES, HIERARCHY_GROUPS, 2},
                            parameters.options().dtype(at::kInt));

    prepare_atom_layout<<<1, A, 0, stream>>>(
        parameters.data_ptr<float>(), sorted_parameters.data_ptr<float>(),
        source_rows.data_ptr<int64_t>(), atom_groups.data_ptr<int>(), ranges.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {sorted_parameters, source_rows, atom_groups, ranges};
}

// ソート版forward。layout準備と本体は同じstreamで順に起動する。
// 戻り値のlayoutは、このforwardに対応するbackwardまで保存する。
std::vector<at::Tensor> block128_forward_sorted_cuda(at::Tensor x, at::Tensor parameters,
                                                     at::Tensor fit, bool profile) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

    auto layout = block128_prepare_layout_cuda(parameters);
    auto sorted_parameters = layout[0];
    auto source_rows = layout[1];
    auto atom_groups = layout[2];
    auto ranges = layout[3];

    int batch_groups = (x.size(0) + BATCH - 1) / BATCH;
    int block_count = OUTPUT_TILES * batch_groups;
    auto output = at::empty_like(x);
    auto norm = at::empty({A, 2, 3}, x.options().dtype(at::kDouble));
    auto clocks =
        profile ? at::empty({block_count, 5}, x.options().dtype(at::kLong)) : at::Tensor();

    // normもsorted_Pと同じslot順。H/C側ではそのまま同じ行を参照できる。
    prepare_norms<<<1, A, 0, stream>>>(sorted_parameters.data_ptr<float>(), fit.data_ptr<double>(),
                                       norm.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    block128<false, true><<<block_count, A, FORWARD_SHARED + SORTED_PARAMETERS_BYTES, stream>>>(
        x.data_ptr<float>(), sorted_parameters.data_ptr<float>(), nullptr, fit.data_ptr<double>(),
        norm.data_ptr<double>(), output.data_ptr<float>(), nullptr, x.size(0),
        profile ? clocks.data_ptr<int64_t>() : nullptr, atom_groups.data_ptr<int>(),
        ranges.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (profile) {
        return {output, norm, sorted_parameters, source_rows, atom_groups, ranges, clocks};
    }
    return {output, norm, sorted_parameters, source_rows, atom_groups, ranges};
}

// Gはsorted slot順で計算するが、最終dPはsource_rowsを使って元のParameter行へ戻す。
// optimizerのmomentを動かす必要はなく、モデル正本の並びも変わらない。
std::vector<at::Tensor> block128_backward_sorted_cuda(at::Tensor x, at::Tensor sorted_parameters,
                                                      at::Tensor dy, at::Tensor fit,
                                                      at::Tensor norm, at::Tensor source_rows,
                                                      at::Tensor atom_groups, at::Tensor ranges,
                                                      bool profile) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

    int batch_groups = (x.size(0) + BATCH - 1) / BATCH;
    int block_count = OUTPUT_TILES * batch_groups;
    auto dx = at::empty_like(x);
    auto dp = at::empty_like(sorted_parameters);
    auto parameter_partials = at::empty({batch_groups, A, 4}, x.options().dtype(at::kDouble));
    auto clocks =
        profile ? at::empty({block_count, 5}, x.options().dtype(at::kLong)) : at::Tensor();

    block128<true, true><<<block_count, A, BACKWARD_SHARED + SORTED_PARAMETERS_BYTES, stream>>>(
        x.data_ptr<float>(), sorted_parameters.data_ptr<float>(), dy.data_ptr<float>(),
        fit.data_ptr<double>(), norm.data_ptr<double>(), dx.data_ptr<float>(),
        parameter_partials.data_ptr<double>(), x.size(0),
        profile ? clocks.data_ptr<int64_t>() : nullptr, atom_groups.data_ptr<int>(),
        ranges.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    reduce_parameters<<<4, A, 0, stream>>>(parameter_partials.data_ptr<double>(),
                                           dp.data_ptr<float>(), batch_groups,
                                           source_rows.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (profile) {
        return {dx, dp, clocks};
    }
    return {dx, dp};
}

std::vector<int64_t> block128_sorted_resources_cuda(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "resource query requires a CUDA tensor");
    c10::cuda::CUDAGuard guard(x.device());

    cudaFuncAttributes layout{};
    cudaFuncAttributes norm{};
    cudaFuncAttributes forward{};
    cudaFuncAttributes backward{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&layout, prepare_atom_layout));
    C10_CUDA_CHECK(cudaFuncGetAttributes(&norm, prepare_norms));
    C10_CUDA_CHECK(cudaFuncGetAttributes(&forward, block128<false, true>));
    C10_CUDA_CHECK(cudaFuncGetAttributes(&backward, block128<true, true>));

    std::vector<int64_t> result;
    for (auto attributes : {layout, norm, forward, backward}) {
        result.push_back(attributes.numRegs);
        result.push_back(attributes.localSizeBytes);
        result.push_back(attributes.sharedSizeBytes);
        result.push_back(attributes.maxThreadsPerBlock);
    }
    return result;
}
