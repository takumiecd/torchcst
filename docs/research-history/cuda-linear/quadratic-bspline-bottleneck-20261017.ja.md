# Quadratic B-spline GPUの内訳とlaunch配置の比較

2026-10-17。PR #109、branch `kernel/quadratic-bspline-prefix`。
[初版の数学・測定](../../quadratic-bspline-gpu.ja.md)を維持して処理を切り分ける。
独立FP64 oracle、global離散L2 floor、全atom勾配、live sigma、Polar更新を変更しない。

## 比較条件と計測境界

NVIDIA L4 / Torch 2.11.0+cu130 / CUDA 13.0 / Triton 3.6.0。
1024×1024、A=52428、B=32、spacing=.1、FP32 X/source、FP64 chart/geometry/norm。
small sigma=(.1,.14,.18)、mixed=(.1,.4,1.6,6.4,12)、等比率に近くshuffleする。
更新で全atomのsigmaが変わる。AdamWとfused Polar更新を含む。

既存Linear runnerの未計装Graph完全stepを主測定とし、capture/replay込みallocated/reservedを測る。
21 sampleは一つの独立processの標本である。別Graphのexternal CUDA eventで各stageを計測する。
診断Graphは追加でnorm_replayを実行する。これはlive geometryをdecodeした後、同じ_norm helperで
Q/S、逆norm、norm微分を別scratchへ再評価する診断専用kernelで、Y/dX/勾配には使わない。
そのため診断forwardには追加のL2評価とeventの費用が入る。主測定には入らない。
norm_replayの時間をfused prepareから引いたり、各stageを足して主測定の代わりにしない。
レジスタ数・spillはTriton/コンパイラの報告であり、DRAM/cache/atomic counterは未測定。

## 初版の実測内訳

source `8e833b93`、job `l4job-757e4f69fae84127a21b10f53496824b`。
GPU検証45件を通過後、両caseのfull-axis Y/dX/全atom勾配、更新と提出前検査もPASS。
主測定はsmall matrix=4.7113 ms、mixed hybrid=24.7876 ms。

| Stage (ms) | small matrix | mixed hybrid |
| --- | ---: | ---: |
| forward状態snapshot | .0061 | .0061 |
| 準備全体 (decode・L2・norm微分) | .3308 | .3308 |
| 別kernelでのnorm再評価 | .2130 | .2130 |
| W構築 | 1.3609 | — |
| Y GEMM | .0154 | — |
| dX GEMM | .0143 | — |
| dW GEMM | .0123 | — |
| atom勾配 | 2.9297 | 1.1827 |
| 入力prefix構築 | — | .0287 |
| dY prefix構築 | — | .0287 |
| H/Hp query | — | 6.8680 |
| G/Gp query | — | 6.8680 |
| 出力へのinject | — | 4.6930 |
| dXへのinject | — | 4.6930 |
| 出力scan | — | .0236 |
| dX scan | — | .0236 |
| 出力restore | — | .0051 |
| dX restore | — | .0051 |

smallの主要費用はW構築とatom勾配である。matrix_vjpはregisters=255、spill_slots=14、
shared=32 bytesを報告。W構築は128 registers、spill=0。
混在sigmaのqueryは48 registers、injectは40、atom VJPは32で、いずれもspill=0。
累積scanそのものより、atomごとのquery/injectが重い。
prepareは80 registers / spill=0、norm_replayは64 / spill=0。

L2込みの準備全体はsmall完全stepの約7%に相当し、現状の主要費用ではない。
normだけの診断再評価は約.21 msだが、fused準備内の厳密な支配率とは区別する。
L2の定義を変える検討は、今回の実装改善とは別の数学的契約として評価する必要がある。

## 比較する配置

smallでは数学的支持を変更せず、matrixのsite演算枠を16から8/4へ変える。
atom群の大きさとwarp数も比較して、演算枠の無駄・レジスタ退避・並列性を見る。
mixedでは元の1 atom×16 batchの処理を保ち、query/inject/atom VJPのwarp数を4から1/2へ変える。
物理的なボトルネックをwarp数だけで確定せず、完全stepとコンパイラ報告で採否を決める。

既存v1 Recipeのserialized fieldsと固定launchを保ち、追加設定はAlgorithm revision v2と
BSplineTuningRecipeへ分ける。古い結果の提出前検査が引き続き通ることを確認した。
途中の2 jobはRecipeの必須field不一致で計測前に失敗したため、速度の証拠に使わない。
raw log/sourceは保存し、修正後のcatalogをCPUでdecodeしてから再提出した。

## 証拠保全

原本は共有poolのjob directoryとignored `output/bspline-bottleneck-20261017/verified-jobs/`へ保存する。
回収archiveとmanifestの全ファイルをSHA256で照合する。
初版診断result archive SHA256:
`d5c101cc8e337a0b1cf1db1197fb365832d9992ce9f7c3f56b93d96f3e38ceee` (27 files verified)。

## 演算枠・warpの比較結果

以下も全て未計装の完全step。各cell独立process 1回、21 sample。
matrix系は同じjobでTorch=440.5532 ms、Dense=.0809 ms。

| small matrix配置 (sites/atoms/warps) | 完全step ms | W構築 ms | atom勾配 ms | VJP registers / spill |
| --- | ---: | ---: | ---: | --- |
| 16/8/4 (v1) | 4.7113 | 1.3619 | 2.9297 | 255 / 14 |
| 8/8/4 | 3.7252 | 1.0086 | 2.2958 | 124 / 0 |
| 4/8/4 | 3.2499 | .9380 | 1.8893 | 69 / 0 |
| 8/32/4 | 3.3458 | .8192 | 2.1053 | 212 / 0 |
| 16/32/8 | 4.7859 | 1.3742 | 2.9921 | 255 / 74 |

全配置でallocated/reserved=49.83/114 MiB、Dense=49.00/106 MiB。
4/8/4はv1から約31%短縮し、8/32/4はW構築が速い一方でVJPが遅いため完全stepでは次点。
16/32/8は退避が増え、負の結果となった。FP64演算と数学的支持は同じ。
GPU検証69件と全候補のfull-axis全atom検査、提出前検査を通過した。
job `l4job-da5c9d41f6e144a7b968d715255bc299`、source `1b547949`。
result archive SHA256 `b849cc8a736c9c7a9a114e2a11aadaea7a0b394f4dbf74cc029e4b2b53100fb8`。

mixed系は同じjobでTorch=443.1642 ms、Dense=.0815 ms。

| query/inject/VJPのwarp数 | 完全step ms | H query ms | G query ms | Y inject ms | dX inject ms | atom勾配 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 (v1) | 24.7923 | 6.8680 | 6.8669 | 4.6920 | 4.6920 | 1.1827 |
| 2 | 14.2895 | 4.0540 | 4.0540 | 2.5508 | 2.5508 | .5980 |
| 1 | 9.1276 | 2.6911 | 2.6921 | 1.4787 | 1.4787 | .3062 |

全配置でallocated/reserved=35.68/98 MiB、Dense=49.00/106 MiB。spillは全て0。
warp=1はv1から約63%短縮した。処理当たりのwarpを減らした効果は実測したが、
occupancy・cache hit・atomic競合率などの物理counterは未測定なので支配要因を細分化しない。
GPU検証69件と各候補のfull-axis全atom検査、提出前検査を通過した。
job `l4job-9215e0323d6e48b687e28d13fbc25e83`、source `1b547949`。
result archive SHA256 `b1585371dc4168b97b9bb5d1e1d7a422cae4f0d86f41745059f1c8d5329e30ab`。

## 軸別に縮約するatom pullback

追加候補 `matrix_separable` は、dWの入力軸を先に縮約する。

\[
u_{aj}=\sum_i dW_{ji}F_{ia},\qquad
v_{aj}=\sum_i dW_{ji}F'_{ia},
\]
\[
T_a=\sum_j F_{ja}u_{aj},\quad
U_{oa}=\sum_j F'_{ja}u_{aj},\quad
U_{ia}=\sum_j F_{ja}v_{aj}.
\]

T/Uを既存のnorm補正・Polar pullbackへ渡す。入力側積をTとUoで共有し、
3種類の2次元積を作る代わりに、2種類の入力軸縮約と3種類の出力軸積を使う。
同じforwardのmetadataを使い、FP64を維持する。u/vは小patch内だけで保持し、
全site×atomの中間Tensorは追加しない。加算順序が変わるので独立oracleとGraph更新で検証する。

## 小patchのwarp数と最終候補

source `be76ad05`、job `l4job-86e53d1f74cc41e7ad8947a462c17b35`。
GPU検証112件 (metadata 1、実CUDA 111)、各候補のfull-axis/全atom oracleと提出前検査がPASS。
Graph検証は各候補で24更新し、eager公開optimizerのParameter/moments/stepと比較した。
TF32や内部FP64、L2の定義は変更していない。

| small候補 | 完全step ms | W構築 ms | atom勾配 ms | VJP registers / spill |
| --- | ---: | ---: | ---: | --- |
| v1 matrix | 4.7112 | 1.3609 | 2.9297 | 255 / 14 |
| matrix4 / 4 warps | 3.2467 | .9380 | 1.8862 | 69 / 0 |
| separable4 / 4 warps | 3.1775 | .9380 | 1.8166 | 64 / 0 |
| separable4 / 2 warps | 2.2256 | .6001 | 1.2042 | 72 / 2 |
| separable4 / 1 warp | **1.7747** | **.4260** | **.9288** | 92 / 0 |

全候補でallocated/reserved=49.83/114 MiB。Dense=.0813 ms、49.00/106 MiB。
同値な軸別縮約だけの差は別jobでも3.2481→3.1769 msで、約2%の改善を再現した。
1 warpの候補はv1から約62%短縮したが、Denseの約22倍の時間が必要で、メモリ条件も未達。
次の主要費用はatom VJP約.93 ms、W構築約.43 ms、L2を含む準備約.33 ms。
L2込みprepareの比率は速度改善後には約19%となり、初版の約7%より相対的に大きくなる。

同時sourceのCPU全suiteは1968 passed / 2887 skipped (37.98 s)。
wheel/sdist build成功、全330 runtimeファイルがcheckoutとbyte一致。

最終候補jobのresult archive SHA256: `db3b0eb2346d6e6816653aece9301eba6d8079a2af8aaf185dfd9b250d4a6923`。

## 最終ソースでの独立再測定

source `9213933d`、job `l4job-df5c54bf463d48d1a3d1335c6a81de21`。
新しく確保したL4でGPU検証112件を再実行し、両caseのfull-axis Y/dX/全atom oracle、
optimizer/動的幅、提出前検査もPASSした。各run 21 sample。候補の独立計測はそれぞれ計2 jobとなった。

| 条件 | Torch ms | v1 ms | v2最終候補 ms | Dense ms | v2 allocated/reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 441.3304 | 4.7139 | **1.7802** | .0828 | 49.83 / 114 |
| mixed | 445.0876 | 24.7902 | **9.1279** | .0824 | 35.68 / 98 |

初回のsmall=1.7747、mixed=9.1276 msから大きな変動はなかった。
両条件で全52428 atomのsigma更新を確認した。全候補を含む独立FP64検査の最大絶対誤差は、
smallでY=2.55e-6、dX=3.20e-6、全atom勾配=5.84e-5、
mixedでY=5.98e-6、dX=6.87e-6、全atom勾配=5.12e-5。
いずれも最大絶対誤差・相対L2誤差の両方に対する既存4e-4 gateを維持している。

初版からsmallは約62%、mixedは約63%短縮したが、Denseとの差はそれぞれ約22倍/111倍残る。
smallの残る費用はatom VJP=.9308、W構築=.4270、prepare=.3308 ms。
mixedではH/G queryが各約2.69、Y/dX injectが各約1.48 ms。
CST全体や他の幅分布に共通する最適dispatcherを確立したという結果ではない。

再現コマンド (tuning Caseは検証済みの対照も含む):

```bash
python -m pytest tests/test_quadratic_bspline_cuda.py -q
python -m tools.kernel_dev check --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-small-tuning.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-mixed-tuning.json
python -m benchmarks.cuda.linear.run --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-small-tuning.json \
  --polar-update fused --kernel-diagnostics --output output/bspline-small-tuning.json
python -m benchmarks.submissions check output/bspline-small-tuning.json
```

source archive SHA256: `2e66214d62474c91e55fd6e177e6ffbf6f369777acd82bd7e751ab7e64b0ebef`。
result archive SHA256: `9f9e9e9306aac663a8041dceb15bf4bc37ea297715ee7a848cef472a3a566cd4`。manifest 33ファイルを照合済み。
