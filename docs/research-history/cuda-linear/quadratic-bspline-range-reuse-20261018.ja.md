# B-spline係数面の寿命とblock scan

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
性能結果と採否は回収・検証後に追記する。

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
