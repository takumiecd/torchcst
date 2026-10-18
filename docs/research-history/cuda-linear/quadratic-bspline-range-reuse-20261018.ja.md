# B-spline係数面の寿命とblock scan


B64 mixedは従来W版4.307→3.254 ms、別jobで4.310→3.249 ms（約24.5%短縮）。
peak allocated/reservedは86.840/188→78.840/156 MiBへ減った。
B32 mixed/境界付近も改善したが、小sigmaだけは従来の直接集約が速い。
Denseとの差はB64約31倍、B32約39倍残り、Dense以下のメモリ目標も満たさない。
形・離散L2・全source勾配を保った研究候補としてPR110に保持し、既定dispatcherへは採用しない。

既存のv8 W区間集約のW構築・全atom VJPを細分化し、同じB-spline・離散L2・
Polar幅stop-gradient・全source勾配を保ったまま、作業領域の再利用を比較する。
公開dispatcher・Torch参照・kernel宣言は変更しない。PR110上の研究候補。

## 候補と寿命

- v8 `range-fused-c02`: 比較基準。各出力次数pで3係数面と1中間面を確保する。
- v9 `range-reuse-c02`: 3係数面を一度だけ確保し、各pの前にゼロクリアする。
  input scanは担当行の全係数を読み込んでから、その行の第0面へ合成結果を保存する。
  行の所有はCTA間で重ならない。backwardでも第0面を出力軸scanの中間領域に使い、
  その行を読み終えてから3面の入力軸随伴を保存する。
- v10 `range-block{64,128}c{16,32}`: v9に加え、出力軸scanをBR行×BC列へ分割する。
  連続した列の読み書きと短いscanを使い、block間のcarryを小さい配列に保持する。

再利用するのはbufferの領域である。atomごとのsigma・多項式係数・切替site・L2は
forwardごとの現在値で計算し、保存済みforwardのMeta/Wを後続forwardで上書きしない。
作業領域はautograd呼び出しの局所変数で、共有mutable cacheへは保存しない。
retained backwardは新しい局所作業領域で再計算する。

1024²のFP64 1面は8 MiB。v8のscan作業領域は3+1面で32 MiB、v9は3面の24 MiB。
これとは別にFP64 W合成8 MiB、FP32 W/dW、Meta、勾配、optimizer/Graph状態がある。
v10 BR128ではcarry 8×1024×8 byte = 64 KiBが加わる。
このtensor予算と、capture/replay込みの実測peak allocated/reservedは区別する。

## block scanの数式

出力軸のblockをb、block内行をrとする。forwardでは

\[
 U_{b,r,i}=\sum_{s=0}^{r}T_{bBR+s,i},\quad
 S_{b,i}=U_{b,BR-1,i},\quad
 E_{b,i}=\sum_{k<b}S_{k,i}.
\]
\[
 (P_oT)_{bBR+r,i}=U_{b,r,i}+E_{b,i}.
\]

末尾のdomain外行はゼロで埋める。随伴はblock内もblock間も逆順にし、
`E_b = sum_{k>b} S_k`とする。exclusive carryはinclusive sumから自分のblockを
引かず、入力を1blockずらしてscanする。全計算はFP64、FP fusion無効。
加算順の変更はあるためbitwise同一とはせず、独立FP64 oracleの既存閾値を適用する。

## 計測規約

既存Linear runnerのみ使用。1024²、A52428、spacing .1、seed41、AdamW/fused Polar、
warmup5/21時間サンプル、no TF32、sigmaはatom別・stepで更新する。
全shape・全atomの独立FP64 Y/dX/DP（3 live状態、maxabsとrelative L2の両方4e-4未満）を
主測定の前に通す。入力・初期P/targetのhash一致を確認する。
主指標は構築/L2/optimizer込みの未計装Graph完全stepとcapture/replayのpeak。
細分化のeventは別Graphで、各p別のzero/scatter/input scan/output scan/随伴scan/query、
FP64初期化・FP32書戻し・partial初期化・pullbackを記録する。
これらの中央値を足して主測定時間と同一視しない。

## 証拠保全

raw source/result archive、検証manifest、pytest/CPUログはignored
`output/bspline-range-reuse-20261018/`と共有poolのjob directoryへ保存する。
v9初回source commitは90387df5、v10探索sourceはaed1c310。
主測定・再測定と採否を末尾に記録した。

## v9初回の切り分け（独立repeat前）

L4 B64 mixedの完全stepはv8 4.309892、v9 4.239705、既存H版7.390236、
Dense .104460 ms。capture/replay込みpeak allocated/reservedはv8 86.840/188、
v9 78.840/156 MiB（Dense49.502/106）。領域共有だけでは約1.6%短縮に留まる。

別計装Graphのv8内訳は、各pの時間を診断目的で集約すると、W側scatter .991232、
入力scan/係数合成 .623616、出力scan/係数合成 .285696、zero .146432 ms。
VJP側はquery/収縮 .927744、入力随伴scan .525312、出力随伴scan .284672 ms。
prepare .162816、native W .057344、native DP .111616 ms。
外部eventの費用・中央値の性質により、主測定の完全step時間へ足し引きしない。
v9の主な変化はzeroが合計 .112640 msへ減ったことで、query/入力scanはほぼ同じ。
これに基づき、係数scatter/queryのatom数・warp数も既存recipeで6構成比較する。

job `l4job-12721c4c8ce54b61b4445a2b12cdfdb9`、source 90387df5、GPUテスト333件通過。
独立FP64 Y/dX/DPは3 live状態・全AでPASS。v9の最大絶対誤差は
Y2.036e-5、dX2.107e-5、DP1.234e-4。relative L2も4e-4未満。
source archive `3894fcddf55d095cf4719182cf017f7ff61aac60fb765c169532d32a89e6d233`、
result archive `f157e4c63d9ed3811c04bbb3dae664278ee340f8baa2f25e869728f2bb06e3bd`。
17 manifest filesを再hash検証し、returncode 0を確認。

## block scanとlaunch構成の探索

v10は精度・GPUテスト362件を通過したが、B64 mixedで改善しなかった。

| Plan | 完全step ms | allocated / reserved MiB |
|---|---:|---:|
| v8 | 4.308205 | 86.840 / 188 |
| v9 reuse | 4.233513 | 78.840 / 156 |
| block128c32 | 4.341168 | 78.903 / 156 |
| block64c32 | 4.293657 | 78.965 / 156 |
| block128c16 | 4.347764 | 78.903 / 156 |

block合計/carry/書戻しが追加され、単にscanを短くしても完全stepは速くならない。
この負の結果を保持し、selected routeには使わない。
job `l4job-88105f1542824326bf58fc4e5422bd50`、source aed1c310、
source archive `749122d7b539eceaa87dbef115dc08852fc37663c7af562615538456a1cc1855`、
result archive `da21523836fa335c0dde4dcb8dcb960d2249e013d8a53257593fbf80e82bab84`、
23 manifest files・returncode 0を検証。

v9の既存recipeだけでscatter/queryの実行粒度を比較した。batch・sigma・正規化・
係数面は同じで、range_atoms/range_warpsだけを変える。

| atom / warp | 完全step ms | scatter合計 ms | query合計 ms |
|---|---:|---:|---:|
| 16 / 4 基準 | 4.238091 | .976896 | .926720 |
| 8 / 1 | 3.577863 | .656384 | .598016 |
| 8 / 2 | 4.238051 | .974848 | .928768 |
| 16 / 1 | 3.253430 | .457728 | .462848 |
| 16 / 2 | 3.598263 | .650240 | .617472 |
| 32 / 2 | 3.274323 | .472064 | .470016 |
| 32 / 4 | 3.588135 | .647168 | .610304 |

右2列は別Graphの各pの診断中央値を集約したもので、主時間の代用ではない。
全候補でallocated/reservedは78.840/156 MiB。
全A・3 live状態のY/dX/DP gateを通過。16/1の最大絶対誤差は
Y2.151e-5、dX2.075e-5、DP1.848e-4。相対L2も4e-4未満。
job `l4job-c5a1d57dc1244c548c234276637ba34d`、source d4094cd3、
source archive `3eeb3a791cf1eb27e1007bca56152748aae052a4d0e6ef5aa846d05772981a3f`、
result archive `a37cbce65f062204ffebc81f2e30cc6ea9d2194dfef2b9270d1af4a4a7f5c3e5`、
29 manifest files・returncode 0を検証。

v11ではbroad atomだけを、支持幅のbucketと左端切替siteの空間tileで並べる。
これは物理的な実行順序だけで、sigma/support/係数を量子化しない。source Parameterや
optimizer momentsの行は動かさず、同じorder snapshotでDPを書き戻す。
各forwardでkeyをdevice上で検査し、幅bucket・tile・native/range分類が変われば
histogram/scan/scatterで配置を再構築する。keyが不変でも数値Metaは毎回更新する。
native数はnative histogram末尾で取り、broad bucketが複数あっても分類を混ぜない。
この配置と16/1を組み合わせる候補も比較する。

## 空間順の負の結果

v11の16/32/64 site tileは16atom/4warpで4.257490/4.253470/4.248479 ms、
v9の4.233572 msを上回れなかった。query診断は各候補とも合計 .926720 msで、
期待した読み出し・収縮の短縮は観測されなかった。全52 forwardで配置を再構築し、
再利用は0回だった。単一のkey変更で全配置をinvalidにする今回の方式では、
atom数が多くstepで中心/支持境界が動くと、継続配置の利点を失う。
空間順そのものの一般的不可能性を示す結果ではない。

16atom/1warpとの組合せも、空間tile16/32/64で3.266855/3.264662/3.264760 ms、
空間順なしの3.248515 msを改善しなかった。メモリはtile16で79.262/156、
tile32で78.946/156、tile64で78.867/156 MiB、空間順なしは78.840/156 MiB。
追加のhistogramと配置更新を払う利点がないためselected routeに使わない。

- 空間順のみ: `l4job-806ecb9c611d4613884fcdfe4e26dfc9`、source 1e19a99b、GPU382件PASS。
  source `3b26f60f2bb5d9c4e70d1fe8c9e1cef5278b2f93ba00de5392d9033fd1f47a62`、
  result `aa8c76e661e8c7a42376224a3616006b09c4d7d3eb82f6a0e3480e91be5278ba`。
- 空間順と1warp: `l4job-c9c5a78c306c41c29f807865540941a8`、source 6f2f0c2d、GPU394件PASS。
  source `3277fb60bcfbddc814d8dc7fa3982288336e444e72b3a684af2980922649541c`、
  result `00446223a562382610dcdafa97b027ab6fb4ef108d9fc057b16d6a8dcf1125b6`。

manifest20/23 files・returncode 0を検証。全A・3 live状態の独立FP64 gateも全候補PASS。

## 最終主測定と独立repeat

source commit `2f4c8bfb`。N1024²/A52428、上記と同じ学習完全step。
B32 switchの初期幅は旧cutoff .5の近傍であり、選択した新cutoffは.2。
主時間中にσが変わることと、分類が実際に跨ぐ追加Graph gateは別々に検証する。

| 条件 | 旧最速H/直接 ms | 従来W v8 ms | reuse 16/1 ms | Dense ms |
|---|---:|---:|---:|---:|
| B32 mixed | 4.316166 | 4.318032 | 3.222893 | .082312 |
| B32 small | 1.206467 | 3.230827 | 3.176751 | .082783 |
| B32 switch | 4.633588 | 4.324029 | 3.221315 | .083269 |
| B64 mixed | 7.386624 | 4.307322 | 3.253650 | .103893 |

別jobのrepeatはB32 mixedでv8 4.322980 / reuse3.227034 / Dense.082747、
B32 switchで4.323792 / 3.224066 / .082663、B64 mixedで4.310140 / 3.248508 / .104309 ms。
21時間サンプルを21独立runとは数えず、同一L4上の独立process/job 2回として扱う。

| 条件 | 従来W allocated/reserved MiB | reuse allocated/reserved MiB | Dense allocated/reserved MiB |
|---|---:|---:|---:|
| B32（3幅条件） | 86.590 / 186 | 78.590 / 154 | 49.002 / 106 |
| B64 mixed | 86.840 / 188 | 78.840 / 156 | 49.502 / 106 |

ピークは独立repeatでも一致。B64でallocated9.2%、reserved17.0%減。
旧H版B64は54.194/114 MiBなので、新W版は依然として旧H版よりメモリを使う。
小sigma-onlyはrange atomが0でも全行列scanの固定費を払い、直接集約に勝てないため
全条件の万能routeとしては使わない。今回はruntime dispatcherを変更していない。

別計装GraphのB64細分化（各pの中央値を診断目的で集約、ms）：

| 部品 | 従来W v8 | reuse 16/1 |
|---|---:|---:|
| 係数scatter | .992256 | .457728 |
| 入力scanと係数合成 | .623616 | .623616 |
| 出力scanと係数合成 | .285696 | .274432 |
| 係数面zero | .146432 | .112640 |
| 出力側随伴scan | .284672 | .284672 |
| 入力側随伴scan | .526336 | .525312 |
| atom query/収縮 | .927744 | .463872 |

range_buildの診断spanは2.104320→1.524736、range_vjpは1.779712→1.313792 ms。
native W .057344、native DP .109568、prepare（L2込み）.162816 msは別に存在する。
`norm_replay=.214016`は診断Graphだけの追加L2再評価であり、実prepareのL2費用として
足したり引いたりしない。主測定にこのnorm_replayも外部eventsも含めない。
入力scanとその随伴の合計約1.15 msが、最適化後も残る大きな処理である。

選択版のcompiler報告はscatter58/query80 registers、spill/local ld/stなし。
これはDRAM transaction、cache hit、occupancy、stallの実測ではない。
warp変更の効果は完全stepと上の部品時間で確認し、未測定の物理trafficを推定値で補わない。

## 最終精度とsource整合性

主測定+repeatの選択版における独立FP64・全shape・全A・3 live状態の最大値：

| Tensor | max absolute | relative L2 |
|---|---:|---:|
| Y | 1.921555e-5 | 1.225093e-6 |
| dX | 2.121292e-5 | 1.208360e-6 |
| 全P勾配 | 1.933782e-4 | 4.144062e-7 |

maxabsとrelative L2の両方4e-4未満という従来基準を維持する。bitwise同一の主張ではない。
各case・plan・jobで初期P/X/targetのhashが一致し、全52428 atomでinitial→finalのsigma変更を確認。
各timed stepの全atomで変化するという別の強い主張には読み替えない。
幅bucket配置は52calls中1rebuild/51reuse（warmup/eager/capture/replayを含む範囲）で、
これは係数値やL2値の再利用率ではない。再利用中も数値Metaは毎forward更新する。

主測定時のCUDA対象スイート394件（metadata4、実GPU390）PASS、CPU全体1971 PASS /
3166 skip / 既存warning18、変更PythonファイルのRuff PASS。
後述の追加σ跨ぎtest以外は主測定後にruntime/benchmark/testを変更していない。
runtime338 filesとbenchmark/catalog/4case等11 filesは両測定jobのsnapshotと現在checkoutで
byte一致を確認した。追加testを含む最終testsファイルは別のGPU gate jobと照合する。

- main `l4job-9b19d97b80d64bc0b36649661659d236`:
  source `bd4488ebf3d1c149ac141c3c71a921e25abbde8c950567c0c4830a21cf5db866`、
  result `4740a61cf6b84fe9f0821db3b86dd3b8bcec216742832f07bb66a99c2c244acd`、56 manifest files。
- repeat `l4job-da5c9621dcab4fdcb6a58a72d56cdd6e`:
  source `f070be7ada29cc58a929cb0128301b62911676f06ee1045ea77043361dfa496f`、
  result `f4bfa8e203099749831e7d9a2efcc725b8108e806fe55e97382eb77940a6a37a`、42 manifest files。

全archiveとmanifestのSHA256、returncode0、timeoutなしを確認。
NVIDIA L4 UUID `563a56c0-9fe7-3b85-7fb7-a2ba10906369`、driver580.82.07、23034 MiB、
Python3.13.15、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0。
GPU process usageは未測定。DB取り込みは接続環境変数が未設定のため実施せず、
検証済みartifactを保全した。DB保存/公開dispatcher採用/PR mergeとは区別する。

再現は既存runnerを使用する：

```sh
python -m pytest tests/test_quadratic_bspline_cuda.py -q
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-b64-mixed-range-reuse-final.json \
  --polar-update fused --phase-diagnostics --kernel-diagnostics \
  --source-commit 2f4c8bfb32c20bfd8d0f39e4f190dcfaff493482 --output output/reuse.json
```

再実装・変更したcheckoutでは`--source-commit`にその実際の完全SHAを渡す。

追加のselected-route gateはsource `9bdcb691`、
job `l4job-0fa455b110f8478d9a267fdfda99c894`で実行。
16atom/1warp・cutoff .2のまま、全33atomの幅を .1/.19/.21/1.6 の間でGraph replayごとに
巡回させ、native/rangeの実count・Y/dX/全P勾配を毎回独立FP64と照合した。
最終スイート395件（metadata4・実GPU391）がPASS。主測定後の変更はこのtest追加だけで、
runtime338 files・benchmark等11 filesのbyteは両測定jobと一致している。
最終testsを含む350 filesはこのgate jobのsnapshotとbyte一致。
source archive `56f74f659abff25edfe539ddcbdcec89272c5933f6b01c407cb03c5de1579a3d`、
result archive `697f22644f8f2f97743724065883669fecd4fc53c781ed6e929c8b3404c72a30`、
4 manifest files・returncode0・timeoutなしを確認した。

配布wheel/sdistも同じ338 runtime filesのbyte一致を検証する。
最終testを含むCPU metadata実行は4 PASS / GPU391 skip。全CPUスイートの1971 PASSは
このGPU-only test追加前のsourceで、追加はCPUでskipされることを確認済み。
