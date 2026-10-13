# 有限 Fourier profile の共有タイル Triton 実装

2026-10-14。数学的定義は [profile の定義](fourier-profile-product.ja.md)に従う。
新しい計算方式を benchmark Registry に登録する研究実装であり、公開 dispatcher への採用は別判断とする。

## 演算と共有量

入力・出力はそれぞれ一つの周期座標。`tile=10`、局所周期 `P=30`、
`spacing=0.1`、一つの tile は `S=100` sites。
global 周期が30なら3 tiles、70なら7 tilesで、各 atom は owner と左右の計3 tilesを使う。
global 周期と profile の局所周期を区別する。
対象 profile は ChebyshevSquare/KaiserSquare、最高正周波数 `M=32,64`。
幅は `[0.1,10]` の live Polar 幅で、task VJP では stop-gradient。

\[
K(d;\sigma)=H(d;\sigma)^2
=\sum_{k=-M}^{M}c_k(\sigma)e^{ik\omega d},\qquad \omega=2\pi/P.
\]

実基底 `phi(t)=(1, cos(k omega t), sin(k omega t))` の feature 数は `F=2M+1`。
atom ごとに全 sites を読む代わりに、tile ごとの
`Z[b,t,f]=sum_{j in tile t} X[b,j] phi_f(t_j)` を一度生成する。
左右の tile の基底を owner 座標へ回転し、owner に属する全 atom がその集計を共有する。
出力側は owner ごとに atom 寄与を feature 空間で集約し、回転の随伴を使って各 tile に配り、
sites の Y を復元する。入力側と出力側の owner ソートは毎 invocation 再生成する。
バックワードの dY→G と G→dX にも同じ共有・随伴を使う。

一様な完全周期の `3S>2M` に限定すると、離散直交性から中心非依存の

\[
\|K_a\|_2^2=3S\sum_{k=-M}^{M}c_{a,k}^2
\]

を使える。入力・出力ノルムの積に一度だけ floor を適用し、
`D_a=max(norm_out*norm_in,epsilon)`、両側を `sqrt(D_a)` で割る。
幅・係数・L2 の準備も完全 step に含む。
係数 H と FFT 畳み込みによる c は GPU 上の Torch FP64、Triton 集計は IEEE FP32。
TF32 と autocast は使わない。

## H の保存と比較対象

共有 route は `[B,A]` の H と入力中心微分 Hprime を backward まで保存する。
全 site×atom factors は作らず、静的基底と回転だけを AlgorithmState に保存する。
live coefficients、routing、Parameter snapshots、正規化は invocation の所有とする。
retained backward は forward 時点の状態を使う。二階微分は対象外。

native direct route は独立な H² のサイト評価を Triton で行い、入力・出力 factors と
中心微分 factors を保存する。共有用の位相生成、ソート、静的基底は作らない。
この baseline は global sites 全体を materialize するため、主測定は N=300 とする。
N=700 の比較から compact support baseline に対する性能を主張しない。
Torch factored と dense AdamW は追加の比較対象。dense は別のパラメータ化の工学的対照である。

## 検証と再現

実装 ID は `research_cuda_fourier_tiles_triton/v1`。
recipe は `strategy=shared|direct, block_atoms=32, block_batch=16`。
研究 catalog は `benchmarks/cuda/linear/plans-fourier-tiles.json`。
既存 `periodic_comparison.py` の明示的な Fourier rev4 case 経路を使い、新 runner は作らない。
この研究 JSON は公開提出・DB schema ではない。

```bash
python -m pytest -q tests/test_fourier_tiles_cuda.py
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/fourier-chebyshev-300-a1024-b32-m32.json \
  --plans fourier-shared fourier-direct torch-factored dense \
  --output output/fourier/cohort.json
```

独立 CPU FP64 oracle は各 global site の H² から Y・dX・全 atom 勾配を計算する。
GPU の feature preparation や tile moments は再利用しない。
max_abs と relative_L2 の両方を `4e-4` 以下に固定し、初期状態と24回の実更新後に確認する。
Graph/public eager 更新の Parameter・AdamW moments・clock も照合する。
完全 step は forward、loss、backward、fused AdamW、Polar projection、周期中心更新を含む。
各独立 worker は同じ X、target、dy、初期 atom、Case、source hash を使う。
21 replay samples は一つの run のサンプルである。
メモリは warmup/capture/replay を含む allocated/reserved peak。独立 oracle scratch は除外し、
GPU process 全体の使用量は未測定とする。

実機結果と PR 判定は、検証完了後にこの文書へ記録する。
