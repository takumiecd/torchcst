// output_local の読み合わせ用 CUDA C++ 版。
// 実行経路には未接続。nvcc compile / GPU correctness / 性能は未検証。
//
// 数学は local_kernels.py::output_local と同じだが、thread配置とsharedへの
// profile配置をここで明示している。Tritonの生成コードの再現ではない。
// atom方向の加算順序も違うため、bitwise一致は要求しない。
//
// 入力の契約（準備処理は既存のもの）:
//   X:      float  [B,N]
//   P:      float  [A,4]             振幅・出力中心・入力中心・幅logit
//   Norm:   double [A,2,3]           field0は混合後profileの二乗和
//   Native: double [A,2,3,NC]        axis0=出力、axis1=入力、field0=値
//   Bounds: int    [A,2,2]           各軸の[支持始点,支持site数]
//   Order:  int64  [2*A]             出力tile所属をsortした「枠番号」
//   Begin:  int64  [ceil(N/32)+1]     tileごとのOrderの開始位置
//   Y:      float  [B,N]             このkernelは局所寄与だけを書き込む
//
// Orderの有効範囲にはlocal atomだけを入れる。両軸の支持site数は0..32、
// Nativeにはそのprofileが保存済み。NC>=32、blocks>0、blockは必ず128 threads。
// 分類・正規化・全atomのbackwardはこのファイルの範囲外。
//
// CUDA環境でのcompile例（このコマンド自体はまだ実行していない）:
// nvcc -std=c++17 --fmad=false -c output_local_walkthrough.cu

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// 32 threadが持つdoubleをwarp内で合計し、全laneへ同じ合計を渡す。
// 必ずwarpの32 thread全員から呼ぶ。b>=Bでも途中returnしてはいけない。
__device__ __forceinline__ double sum_input_sites(double value) {
    constexpr unsigned full_warp = 0xffffffffu;
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(full_warp, value, offset);
    }
    return __shfl_sync(full_warp, value, 0);
}

__global__ void output_local_walkthrough(
    const float* X, const float* P, const double* Norm,
    const double* Native, const int* Bounds,
    const std::int64_t* Order, const std::int64_t* Begin,
    float* Y, int B, int N, int NC) {

    constexpr int atom_group = 4;
    constexpr int batch_group = 4;
    constexpr int tile_sites = 32;

    // [1] threadと担当出力の対応。
    const int lane = threadIdx.x % 32;  // 0..31: site方向
    const int warp = threadIdx.x / 32;  // 0..3: batch方向
    const int tiles = (N + tile_sites - 1) / tile_sites;
    const int tasks = tiles * ((B + batch_group - 1) / batch_group);

    // [2] このCUDA版で明示的に作るshared領域。
    // 値のprofileだけ。中心/幅微分やX全体、global Hはここには置かない。
    __shared__ double input_profile[atom_group][tile_sites];
    __shared__ double output_profile[atom_group][tile_sites];
    __shared__ int input_begin[atom_group];
    __shared__ int input_count[atom_group];
    __shared__ double alpha[atom_group];

    // Triton: range(program_id(0), tasks, WORKERS)
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int tile = task % tiles;
        const int batch = task / tiles;
        const int b = batch * batch_group + warp;
        const int o = tile * tile_sites + lane;

        // [3] 各threadは、自分の出力1要素の累積値を持つ。
        double acc = 0.0;  // registerを意図したthreadローカル変数
        const int first = static_cast<int>(Begin[tile]);
        const int last = static_cast<int>(Begin[tile + 1]);

        for (int base = first; base < last; base += atom_group) {
            // [4] 読み込み段階では、warp番号をatomのslot番号として使う。
            // warp0がslot0、warp1がslot1...のprofileを32 laneで協力して読む。
            const int slot = warp;
            const bool member = base + slot < last;
            const int a = member ? static_cast<int>(Order[base + slot] / 2) : 0;
            const int il = member ? Bounds[a * 4 + 2] : 0;
            const int ni = member ? Bounds[a * 4 + 3] : 0;
            const int ol = member ? Bounds[a * 4] : 0;
            const int no = member ? Bounds[a * 4 + 1] : 0;

            input_profile[slot][lane] = member && lane < ni
                ? Native[((a * 2 + 1) * 3) * NC + lane] : 0.0;
            const int index = o - ol;
            output_profile[slot][lane] = member && o < N && index >= 0 && index < no
                ? Native[((a * 2) * 3) * NC + index] : 0.0;

            if (lane == 0) {
                input_begin[slot] = il;
                input_count[slot] = ni;
                alpha[slot] = member
                    ? static_cast<double>(P[a * 4]) /
                        fmax(sqrt(Norm[a * 6] * Norm[a * 6 + 3]), 1.0e-6)
                    : 0.0;
            }
            __syncthreads();  // 全4 warpのsharedへの書き込み完了を待つ。

            // [5] 計算段階では、warp番号はbatch行の番号に戻る。
            for (int j = 0; j < atom_group && base + j < last; ++j) {
                const int s = input_begin[j] + lane;
                const double x = b < B && lane < input_count[j]
                    ? static_cast<double>(X[b * N + s]) : 0.0;
                const double product = x * input_profile[j][lane];

                // Triton: h = sum(xx * fi, 入力site軸) * alpha
                const double h = sum_input_sites(product) * alpha[j];

                // Triton: acc += sum(h * fo, atom軸)
                // この版では4 atomをforで順に足す。
                acc += h * output_profile[j][lane];
            }
            __syncthreads();  // 全員の利用が終わってから次の4 atomで上書き。
        }

        // [6] タイルへの全局所寄与を足した後、各threadが1要素だけstore。
        // atomリストが空ならゼロをstoreする。atomicやglobal Hへのstoreはない。
        if (b < B && o < N) {
            Y[b * N + o] = static_cast<float>(acc);
        }
    }
}

// 全pointerはdevice上。Yを消費する別kernelとの依存は呼出側で管理する。
void launch_output_local_walkthrough(
    const float* X, const float* P, const double* Norm,
    const double* Native, const int* Bounds,
    const std::int64_t* Order, const std::int64_t* Begin,
    float* Y, int B, int N, int NC, int blocks, cudaStream_t stream) {
    output_local_walkthrough<<<blocks, 128, 0, stream>>>(
        X, P, Norm, Native, Bounds, Order, Begin, Y, B, N, NC);
}
