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

## 実機結果と採否

NVIDIA L4、Torch `2.11.0+cu130`、CUDA `13.0`、Triton `3.6.0`。
実測 source は `52393e7997a26ef4a2f1ad0c1064cc2f29d5ca54`。
CPU 最終 suite は **1707 passed / 2724 GPU・DB skips / 18 existing warnings**。
実 L4 の回帰は **22 passed / 0 skipped**。
wheel/sdist build、隔離 wheel の metadata import、catalog の JSON 往復、Ruff、PR CPU CI も合格。

以下は N=300、B=32、live sigma `[0.1,10]` の結果。
時間は二つの独立 subprocess cohort の各中央値の範囲、メモリは両 run で同一の値。
全て L2・backward・AdamW・Polar 更新を含む完全 step。
peak は **MiB** (`2^20` bytes)、direct → shared の順。

| profile | M | atoms | direct ms | shared ms | allocated MiB | reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chebyshev | 32 | 1024 | 1.146–1.147 | 0.869–0.874 | 38.13 → 3.55 | 108 → 28 |
| chebyshev | 64 | 1024 | 2.510–2.513 | 1.153–1.154 | 39.92 → 6.85 | 108 → 32 |
| kaiser | 32 | 1024 | 1.028–1.028 | 0.753–0.753 | 38.13 → 3.55 | 108 → 30 |
| kaiser | 64 | 1024 | 2.242–2.243 | 0.885–0.886 | 39.92 → 6.85 | 108 → 30 |
| chebyshev | 64 | 64 | 0.722–0.723 | 0.696–0.697 | 33.16 → 0.64 | 46 → 6 |
| chebyshev | 64 | 256 | 1.090–1.091 | 0.825–0.827 | 34.28 → 1.88 | 46 → 6 |

1024 atoms の4条件は direct より約24–60%短縮、allocated peak は約83–91%減。
再利用確認の Chebyshev M64 は、64 atoms で約3.6%短縮、256 atoms で約24%短縮。
全6条件が二つの run とも事前の時間・allocated 判定に合格した。
これらの条件に限る研究 route として保持し、公開 default selector への採用は行わない。
N700・別 batch・別 GPU・DPSS・多次元側への速度の外挿はしない。

各 cohort の shared/direct/Torch-factored に対する初期・24更新後の独立 FP64 gate、
計 **72 operator/VJP 比較**が合格。
全 route を通した最大絶対誤差は Y `5.36e-6`、dX `5.21e-6`、all-dP `4.01e-5` 以下。
最大 relative L2 は順に `6.76e-7 / 6.28e-7 / 9.83e-7` 以下。
両 run は逆の worker 順で、初期値・X/target/dy・metadata・source hashes の一致を確認した。
live width は bounds に固定された一つを除き更新され、全 worker の実 clock は24。

同サイズの dense は `0.063–0.065ms`、allocated `17.85MiB`、reserved `26MiB`。
時間では dense が速い。これは90,000個の自由な行列要素と4096個の atom parameters の
異なるモデルの比較であり、今回の採否は同じ profile 演算の direct 比較で行う。
Torch-factored の1024 atom 完全 step は profile/M ごとに約5.15–26.20ms、
allocated は344.59–951.15MiBだった。
allocator peak には各 route が保持する library workspace も含む。
共有 Tensor の byte 予算や process VRAM の削減率として読み替えない。

## 証拠の保存

raw evidence は worktree の ignored `output/fourier-shared-tiles-20261014/` に保存。
`jobs/` に全8 job の frozen source tar、spec、receipt、回収 archive、raw results を保持した。
最初の2失敗は4 Graph テストの eager reference 設定と CPU constant upload の問題であり、
修正前の失敗ログも残している。
`measurement-summary.json` と `artifact-manifest.json` は全 case・source・回収結果を確認する入口。
元データは host-wide pool の同じ job ID ディレクトリにも保存される。

| job ID | 内容 |
| --- | --- |
| `l4job-04db29fb30a944b2824bb2769bfe9d88` | 初回 regression、18 pass / 4 fail |
| `l4job-35de3d77284c4dcd8723ce063bab9390` | 2回目 regression、18 pass / 4 fail |
| `l4job-72fc0f7c174f4333b243877034b3667e` | 最終22 regression、Chebyshev M32 A1024、2 cohorts |
| `l4job-96d51b1b2bb0447284924cc48c45c04b` | Chebyshev M64 A1024、2 cohorts |
| `l4job-f64ad439880b4c04b4c989bb5c56787d` | Kaiser M32 A1024、2 cohorts |
| `l4job-c26a8d7cfeaa455393789aaa66d25e45` | Kaiser M64 A1024、2 cohorts |
| `l4job-2abd4be053b64d46a3ed652ec1ef7ec9` | Chebyshev M64 A64、2 cohorts |
| `l4job-2f284aa05e5245c59bb4532bb29704a7` | Chebyshev M64 A256、2 cohorts |

`measurement-summary.json` SHA256: `436805a9e3d17bd3361ced45719ba65687db194dbd39bd514c681569cc1644fd`。
`artifact-manifest.json` SHA256: `0caba2e949a4b7c5d7146862a33058db3758fcbe05986254a5d5181de7df25b5`。

独立監査で成功6 job の回収99ファイル・各 source 327ファイルと現在の Python source の一致を確認した。
全 job の回収後、pool supervisor の終了と全 slots の停止を確認し、`gpu-shutdown.json` を保存した。
