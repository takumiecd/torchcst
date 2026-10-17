# Quadratic B-splineの支持点列挙とatom集約

2026-10-17。PR #109、branch `kernel/quadratic-bspline-prefix`。
[前回のstage診断](quadratic-bspline-bottleneck-20261017.ja.md)で残った
smallのW構築・atom VJP、mixedのH/G query・Y/dX injectを対象とする。
profile、離散L2、Polarの幅則、入力勾配・全atom勾配の契約は同じである。

## 数学と保存状態

入力軸のprofileを \(F_{ia}\)、出力軸を \(F_{ja}\)、
振幅と離散L2を含む係数を \(s_a\) とする。

\[
H_{ba}=\sum_i X_{bi}F_{ia},\qquad
Y_{bj}=\sum_a s_a H_{ba}F_{ja}.
\]

hybridはHと入力中心微分Hpをforwardからbackwardまで保存する。
backwardのG/GpはdYに同じqueryを適用して作る。今回も全site×atomの
profile表や支持点索引表を追加せず、固定の基底・有限和moment表だけを再利用する。
live sigma・中心・norm・その微分は毎forwardで再計算し、そのforward専用Metaに保存する。

小さい支持は軸別のprofile評価、大きい支持は3個の区分二次式とprefix差分を使う。
native/prefixの境目は支持点数32で、step中のsigma更新に追従する。
どの経路でも支持を切り捨てない。FP32のX/H/W/Yと、FP64の座標・profile演算・
L2有限和・prefix・勾配縮約を維持し、TF32も使わない。

## atomとbatchをまとめるhybrid

従来の1 atom×16 batchのCTAを、4×16、8×8、16×4へ変える。
queryの結果は同じH[B,A]へ保存するが、atom方向の読み書きをまとめる。
nativeでは[batch,atom,site]をCTA内で縮約し、wideでは[batch,atom]ごとに
3区間のprefix差分を評価する。支持が異なるatomのgroupではnative支持点数の最大まで
ループし、各atomの有効範囲をmaskする。sigmaごとのsortや固定分類表は作らない。

injectも同じgroupでHを読み、nativeはYへ、wideは差分配列の区間端点へatomic加算する。
空区間の同一点への+c/-cの組は省く。非空区間の寄与は変更しない。
atom VJPもH/Hp/G/Gpをatom方向にまとめて読み、batch軸でFP64縮約する。
計算量の次数を変える最適化ではなく、CTA数・配置・読み書きの変更である。

## 支持内の組だけを列挙するmatrix

atom aの出力支持点数をSo、入力をSiとし、連続した番号pで

\[
j=l_o+\lfloor p/S_i\rfloor,\qquad
i=l_i+(p\bmod S_i),\qquad 0\le p<S_oS_i
\]

を列挙する。実装はSi=0の除算を避け、空支持では全要素をmaskする。
16/32/64組ごとにWの加算またはdWからのatom VJPを行う。
番号はCTA内で生成し、globalの支持点索引配列を保存しない。
広い支持も同じループで全組を処理するため、上限を超える支持を捨てる方式ではない。
この研究matrix経路は従来どおり各軸1024 siteまでに制限する。

## 準備と互換性

prepareのwarp数1/4を同じ演算で比較する。source Parameterの余分なcloneは除くが、
prepareが独立したMetaへ数値を保存し、backwardはそのMetaを使う。
origin/spacingのsnapshotは保つ。forward後にParameterを変更したretained backwardも検査する。

v1の5 field、v2の9 fieldのRecipeは保持する。追加設定はv3の
BSplineParallelRecipeの13 fieldへ分ける。v1/v2のlaunch・snapshot既定値を変えない。
新候補はbenchmark-local Registryに登録し、公開dispatcherへの採用は含めない。

## 検証と証拠

原本は共有poolとignored `output/bspline-parallel-20261017/verified-jobs/`に保持する。
archive SHA256をreceiptと照合し、manifestの全ファイルも個別照合する。

最初のqualification `l4job-2477b9059a414373a48264d3106a0980`は20件通過後、
Tritonのloop-carried maskのrank不一致で停止した。2次元/3次元maskの名前を分けた。
速度・精度を示す成功結果として扱わず、失敗原本も保存した。
修正source `c0d3c51b`、`l4job-a2824869e43847e389cabbd630a8b787`は147件PASS。
このうち1件はCPU metadata、146件は実CUDAである。
result archive SHA256は`f0245e4688e5dac939a9292a672f3c2a9b7cd951a5e5a4e07728e41bcb50eede`。

CPU全suiteは1968 passed / 2923 skipped (41.69 s)。Ruff checkもPASS。
wheel/sdistをbuildし、全331 runtimeファイルがcheckoutとbyte一致した。
GPU timing、独立oracle、capture/replay込みpeakはCPU検証とは別に記録する。

## 第1回の完全step比較

source `a6913ece`、job `l4job-ee5bafda3b914f4d859b1eaa40abbd6d`。
NVIDIA L4 / Torch 2.11.0+cu130 / CUDA 13.0 / Triton 3.6.0。
1024×1024、A=52428、B=32、spacing=.1、AdamWとfused Polar更新を含む。
small=(.1,.14,.18)、mixed=(.1,.4,1.6,6.4,12)をshuffleする。
前回と同じ未計装Graph完全step、各21 sample、候補ごと独立process 1回。
L2のnorm_replayは追加の別計装診断であり、主測定には含めない。

| small候補 | 完全step ms | prepare ms | W構築 ms | atom VJP ms |
| --- | ---: | ---: | ---: | ---: |
| v2 separable4 / 1 warp | 1.7767 | .3308 | .4260 | .9298 |
| v3 同じ演算、prepare 1 warp | **1.6050** | **.1618** | .4260 | .9288 |
| v3 同じ演算、prepare 4 warps | 1.7918 | .3308 | .4260 | .9277 |
| flat 16 pairs / 8 atoms | 2.2305 | .1618 | .7864 | 1.1919 |
| flat 32 pairs / 8 atoms | 2.2423 | .1618 | .7997 | 1.1940 |
| flat 64 pairs / 8 atoms | 2.8253 | .1618 | 1.0342 | 1.5391 |
| flat 32 pairs / 16 atoms | 2.2471 | .1618 | .8059 | 1.1991 |

Torch=441.7435 ms、Dense=.0821 ms。allocatedはmatrix全候補49.83 MiB、Dense49.00。
reservedはv2=114、v3=112、Dense=106 MiB。clone除去でallocated peakが減ったとはいえない。
prepareのwarp削減は約10%の完全step短縮につながった。prepare内部の演算は同じである。
flatは全配置で負の結果。indexの除算/剰余と、組ごとの軸profile再評価が加わる方式で、
有効な組だけを列挙しても今回の小patchより速くならなかった。
flatのcompiler spillは0なので、この負の結果をspillだけで説明しない。

| mixed候補 (atom×batch / warps) | 完全step ms | H query ms | Y inject ms | atom VJP ms | query registers / spill |
| --- | ---: | ---: | ---: | ---: | --- |
| v2 1×16 / 1 | 9.1258 | 2.6921 | 1.4787 | .3062 | 80 / 0 |
| v3 同配置、prepare 1 warp | **8.9580** | 2.6911 | 1.4776 | .3062 | 80 / 0 |
| grouped 4×16 / 1 | 11.6036 | 3.3997 | 2.0890 | .3082 | 251 / 0 |
| grouped 8×8 / 1 | 17.5342 | 5.2060 | 3.3178 | .1884 | 255 / 2 |
| grouped 16×4 / 1 | 27.5056 | 7.9534 | 5.5706 | .1393 | 255 / 56 |

Torch=446.7155 ms、Dense=.0825 ms。hybrid全候補のallocated/reservedは35.68/98 MiB、
Dense49.00/106 MiB。大きなgroupはquery/injectの時間を増やし、queryのregistersとspillも増えた。
atom VJPだけは8/16 atom groupで短縮した。したがって追加比較を、groupを小さくする、
warpを増やす、query/injectを従来どおりにしてVJPだけをgroupにする3点へ絞った。
occupancyやDRAM/cache/atomic counterは未測定であり、compilerの数値と物理原因を区別する。

GPUテスト148件 (metadata 1、実CUDA 147)、全候補のfull-axis Y/dX/全atom FP64 oracle、
動的sigma、optimizer、両caseの提出前検査がPASS。全52428 atomのsigma更新を確認した。
result archive SHA256は`350bf7075005384acfb6e36dda99d5b019377bfd065ab12d88e7c3a4eda63322`。
manifest全58ファイルを照合済み。負の候補も含めて原本を保存している。

## 絞り込んだ配置と独立再測定

source `9d963f57`、job `l4job-35f9fcf97f0c42528df0dd28d143765b`。
新しく確保したL4で160件 (metadata 1、実CUDA 159) がPASSした。
既存gateは最大絶対誤差と相対L2の両方が4e-4未満で、今回も変更していない。
新しいquery/inject配置と、`hybrid_vjp_grouped`の24回Graph更新も検証した。
全候補のfull-axis Y/dX/全atom勾配、全52428 atomのsigma更新、提出前検査がPASS。

smallのv2=1.7824 ms、prepare 1 warp=1.6120 msで、第1回1.6050 msと近い値だった。
allocated/reservedも前回と同じで、完全stepの独立runは計2 jobとなった。
smallのgrouped 4×4 / 1 warpと4×16 / 4 warpsは17.6534、17.6023 msで、
35.68/98 MiBへ保存量を減らせるが、matrixより大幅に遅くなる負の結果だった。

| mixed配置 | 完全step ms | H query ms | Y inject ms | atom VJP ms |
| --- | ---: | ---: | ---: | ---: |
| v2 hybrid / 1 warp | 9.1285 | 2.6921 | 1.4787 | .3062 |
| v3 prepareのみ1 warp | 8.9599 | 2.6921 | 1.4776 | .3062 |
| grouped 4×4 / 1 warp | 34.6102 | 11.7596 | 5.2296 | .3092 |
| grouped 4×16 / 4 warps | 25.8160 | 7.5162 | 5.0842 | .2990 |
| grouped 8×8 / 4 warps | 30.2130 | 8.8945 | 5.9177 | .2785 |
| 従来query/inject + 16 atom VJP | **8.7860** | 2.6921 | 1.4776 | **.1393** |

mixedのTorch=443.2286、Dense=.0824 ms。全hybrid候補のallocated/reservedは35.68/98 MiB。
小さくしたgroupもwarp増加もquery/injectを改善しなかった。
4×16 / 4 warpsではqueryのcompiler register数は96、spill=0へ下がったが、時間は悪化した。
レジスタ数だけでは全配置の速さを説明できない。

有効だった組合せは、prepare 1 warpと、従来query/injectを保ったVJPだけの16 atom集約である。
後者は同じT/Uのbatch縮約を行い、加算順序以外の演算を変えない。
query/injectの大きなgroup化は採用候補から外し、研究経路と原本は比較用に残す。
残るmixedの主要費用はH/G query各2.69、Y/dX inject各1.48 msであり、
L2準備.16 ms、atom VJP.14 msをさらに縮めるだけではDenseとの差を解消できない。

このsourceのCPU全suiteは1968 passed / 2935 skipped (39.37 s)。Ruff check PASS。
wheel/sdist build成功、全331 runtimeファイルがcheckoutとbyte一致。
source archive SHA256は`e3dd941544f3ddb682170683c24fa27baabda8e8cea71f2b1ee2d49f86a093e9`、
result archive SHA256は`d69ccf66461b6303046c878765c456dd5f19c3a20cb0e1c63a002021e871434b`。
manifest全52ファイルを照合済み。

再現する選択候補と対照をtracked Plan/Caseに保存した。

```bash
python -m tools.kernel_dev check --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-small-parallel.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-mixed-parallel.json
python -m pytest tests/test_quadratic_bspline_cuda.py -q
python -m benchmarks.cuda.linear.run --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-mixed-parallel.json \
  --polar-update fused --kernel-diagnostics --phase-diagnostics --output output/bspline-mixed-parallel.json
python -m benchmarks.submissions check output/bspline-mixed-parallel.json
```

8192²の完全step性能、周期・複数座標軸、実DB登録は今回も未実施。
Denseとの差は残り、公開dispatcherへ自動採用する結果ではない。
