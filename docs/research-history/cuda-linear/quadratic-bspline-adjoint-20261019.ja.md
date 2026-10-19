# Quadratic B-spline: G/Gμの読み取り共有とdX集約

2026-10-19。[forward H/Hμ保存](quadratic-bspline-h-save-20261019.ja.md)を対照に、数学を変更せずbackwardの実行を比較する。研究用revision v15。公開dispatcherの採用とは区別する。

## 数学と精度の契約

生profileをfi/fo、正確な全site二乗和をQi/Qoとし、

\[
D_a=\max(\sqrt{Q_{i,a}Q_{o,a}},\epsilon),\quad
H_{ab}=\sum_i f_{i,a}(i)X_{bi},\quad
G_{ab}=\sum_j f_{o,a}(j)dY_{bj},\quad
dX_{bi}=\sum_a(A_a/D_a)f_{i,a}(i)G_{ab}.
\]

H/Hμは同じforwardに属するFP32 snapshotを使い、G/Gμも既存のFP32丸めを保つ。係数・prefix moment・縮約・差分集約はFP64。支持、正確なsampled product L2と一回だけのfloor、中心微分、振幅0の勾配、既存Polar pullbackを維持する。σはatomごと・stepごとにlive値をdecodeし、task VJPの幅stop-gradientという既存契約も維持する。

各profileの4個の切替点をκ、そこで切り替わる多項式係数をcとすると、GはdYのprefix moment Pから求められる。

\[
G_{ab}=-\sum_{k=0}^{3}\sum_{p=0}^{2}c^o_{a,k,p}P^o_{p,\kappa^o_{a,k},b}.
\]

Gμは同じPの0/1次planeと中心微分係数を使う。G側で共有するのは出力側の切替点への読み取りである。dX側では入力側切替点への差分書き込みをまとめる。

\[
\Delta^i_{p,s,b}=\sum_{a,k:\kappa^i_{a,k}=s}(A_a/D_a)G_{ab}c^i_{a,k,p}.
\]

最後のscanと多項式復元は既存方式と同じ。この和をatom群rに分け、\(\Delta=\sum_r\Delta^{(r)}\)としても数学は変わらない。FP64の加算順序だけは変わるため、独立FP64 oracleでY/dX/全atom4成分を照合する。maxabsとrelative L2はともに既存の4e-4未満を要求し、閾値を緩和しない。

## 比較する実行方式

- backwardのatom tileを8/16/32に変更。forwardは同じ8atom設定を維持する。
- 端点0だけを16個のFP64 bufferへ分散し、最後に合計する。主deltaは従来のまま。atomicの数自体は変えず、同じaddressへの集中を分散する候補。
- delta全体を4/16個へ分散してから合計する。追加容量は `S * 3 * (Ni+1) * B * 8` byteで、peakは別に実測する。
- 入力側の外側2切替点をkeyにbackwardだけのindex順序を作り、隣接して同じ切替点へ書くatomを局所FP64 scanで合算する。canonical Parameter、optimizer状態、forward保存値の並びは変えない。
- 出力側の外側2切替点で並べ、同じ切替点へのprefix読み取りをblock内のleaderだけが行いgatherする。並べ替えなしの明示共有も対照とする。
- 端点0への4個の切替点の寄与をregister内でまとめ、1 blockあたり各plane最大1回だけatomicを行う。端点分散と組み合わせる。
- radix sortの代わりに、左切替点と支持site数のlog2階級による整数histogram/scan/placeで区間群を作る候補も比較する。この階級は実行順序だけに使い、profileの幅や係数は元の連続値を使う。

並べ替えは毎forwardの現在の切替点から作り、対応するbackwardまで保存する。古いσに基づく順序を無検証で再利用しない。index順序は性能のためのもので、支持を近似・切捨て・量子化しない。順序作成・buffer初期化・最終合算を完全stepに含める。

## 検証範囲と測定条件

L4一台、1024²、52,428 atom、spacing .1、FP32入力/Parameter、TF32 off、AdamW + fused Polar。混合σの初期値は `.1/.4/1.6/6.4/12` 各20%。幅の上限は12.8。warmup 5、21 replay median。主測定のX/targetは同じ固定fixtureとし、Xも変わる正しさはeager/Graphテストで別に確認する。

完全stepにはrouting・配置検証・必要時の再配置、係数構築、L2、forward/backward、更新、今回の並べ替えと最終合算を含む。allocated/reservedはCUDA Graph capture/replay込みのpeak。診断は別Graphで、stage時間の和から完全stepを再構成しない。

新規GPUテストはσ・中心・Xの変化、cutoffをまたぐGraph replay、retained forward snapshot、zero amplitude、dXのみ/dPのみ、floor枝、空atom、B64を含む。性能と精度の結果は回収archive・manifest・source・cleanupを確認して以下に記録する。

## 切替点の共有可能性

job `l4job-6dc51a857da24c76941b0c931f5db1db`、初期混合fixtureの実device切替点をGPU測定外で回収して数えた。広いσ側41,942 atom、16atom tile。以下は論理的な読み取り行数とatomic vector数で、DRAM transactionやcache hit率の測定ではない。

| backwardの順序 | G prefix行の共有による要求削減 | 内部端点の局所合算によるatomic vector削減 |
|---|---:|---:|
| 元の順序 | 4.51% | .089% |
| 入力側区間でsort | 5.15% | **77.51%** |
| 出力側区間でsort | **81.83%** | .103% |

G側のprefix行要求は元167,768行に対して出力側sortで30,491行、dX側の内部atomic vectorは元461,640本に対して入力側sortと局所合算で103,818本となる。各atomic vectorはB32の32個のFP64要素。端点0の書き込みはこの内部数と分けた。入力側と出力側の両方を同じ順序で最大限共有できるという結果ではない。

## 検証の途中で除外した実装

初回 `l4job-fb0cd5321b3d4e8fa792c08ed614aaf6` は端点分散とedge集約無効の組合せで12件のdX検証に失敗した。分散側へ端点0を常に流すよう修正した。このjobは回収後のcleanup接続も失い、poolがVM停止とserver assignment消失を確認した。速度比較には使わない。

修正後 `l4job-873f0a417bc1470a854ccc333681de87` は70件（新規対象61件と既存のadjoint関連9件）と全12候補の3状態の独立FP64 gateを通過した。しかし完全step診断に未記録のイベントを含めており、baseline測定で失敗したため速度・peak比較から除外した。診断イベントを新revisionのみに限定して修正した。

追加のfold候補は `l4job-d9c971df343543d983fc5dafcdbe78fc` / `l4job-dce78fd4df014029be9f25647d66e4d5` でTriton SSAのaddress変数のshape衝突を検出した（18件compile失敗）。変数名を分けて修正した。これらも速度比較には使わない。新しいrecipe fieldを追加した際のcatalog宣言も完全なfield集合に揃え、凍結した最終比較を別sourceで行う。

これらは実装・診断の失敗として保持する。数学や誤差閾値を変更して通した結果ではない。

CPU全体は1,975 passed / 3,415 skipped、39.80s、18件の既存warning。最終候補source `d7647d19` のwheel/sdist内347 runtimeファイルをcheckoutとbyte照合した。旧sourceの全回帰 `l4job-f21d77b9c52744e79a16d4686656ef3a` も同じ18件のfold compile失敗（609 passed）を含むため、最終版の全回帰合格とは扱わない。

## 全候補のB32探索比較

job `l4job-054761eb479b40b0b008f465b34b6a99`、source `d7647d19`。archive/manifest/source/cleanupを照合し、全recordがPASSであることを確認した。各候補1 process。選択した110件のテスト（新規対象101件 + 既存adjoint関連9件）は117.50sで通過した。

| 候補 | 完全step ms | allocated / reserved MiB |
|---|---:|---:|
| 対照: v14 H/Hμ保存 | **1.7087** | 81.288 / 172 |
| v15同条件・変更なし | 1.7077 | 81.288 / 172 |
| 16atom tile | 1.7295 | 81.288 / 172 |
| 32atom tile | 1.7628 | 81.288 / 172 |
| 端点0だけ16分散・8atom | 1.7095 | 81.288 / 172 |
| 端点0だけ16分散・16atom | 1.7293 | 81.288 / 172 |
| 端点0だけ16分散・32atom | 1.7717 | 81.288 / 172 |
| delta全体4分散 | 1.7087 | 82.741 / 172 |
| delta全体16分散 | 1.7312 | 91.553 / 200 |
| 入力sort + 局所合算 | 2.1457 | 81.688 / 172 |
| 出力sort + 読み取り共有 | 1.8475 | 81.688 / 172 |
| sortなし・読み取り共有 | 1.7855 | 81.288 / 172 |
| 端点0の寄与をfold・8atom | 1.8833 | 81.288 / 172 |
| fold・16atom | 1.9559 | 81.288 / 172 |
| fold・32atom | 1.8955 | 81.288 / 172 |
| 入力histogram + 局所合算 + fold・16atom | 2.3526 | 81.688 / 172 |
| 入力histogram + 局所合算 + fold・32atom | 2.4142 | 81.688 / 172 |
| 出力histogram + 読み取り共有 | 1.7925 | 81.688 / 172 |
| Dense | .0831 | 49.002 / 106 |

元のv14と同じ条件のv15は同じ時間帯であり、改善とは呼ばない。B32混合で明確に元の最速版を上回る候補はなかった。

### 準備と融合backwardの診断

| 候補 | 順序作成 ms | 融合backward区間 ms | その内側の最終合算 ms | backward register/thread |
|---|---:|---:|---:|---:|
| v14対照 | skip | .5919 | skip | 48 |
| v15同条件 | skip | .5980 | skip | 48 |
| 端点0だけ16分散 | skip | .6031 | .0092 | 48 |
| delta全体4分散 | skip | .6083 | .0113 | 56 |
| 入力sort + 局所合算 | .0809 | .9636 | .0102 | 128 |
| 出力sort + 読み取り共有 | .0809 | .6646 | .0092 | 128 |
| 出力histogram + 読み取り共有 | .0215 | .6636 | .0102 | 128 |
| fold・8atom | skip | .7782 | .0092 | 92 |

これらは別Graphのexternal event診断で、主測定の完全stepを置き換えない。最終合算は融合backward区間の内側なので二重に足さない。v15は追加eventも含み、空markerはskipとした。表のB32 compiler spillはすべて0だった。

入力sortではatomic vectorを約77.5%減らせても、順序作成だけでなくblock内合算を含むbackward自体が長くなった。出力histogramはsortの約.081msを約.022msへ短縮できたが、共有gatherを含むbackwardは対照より長く、完全stepでは約5%遅い。foldもregisterの保持量とkernel時間が増えた。論理的な要求削減率から高速化を直接推定しない。帯域、cache、atomic競合、occupancyの物理的な内訳はcounter未測定であり、この時間とcompiler情報だけから確定しない。

全候補の独立FP64最大誤差はY `1.218e-6`、dX `1.336e-6`、全atom勾配 `2.089e-5`、relative L2は `1.020e-7` 以下。全52,428 atomのσが各CST測定で変化した。同じ初期Pと、Denseも含む同じ初期X/targetをhashで確認した。

元の最速H保存版と代表候補のB32/B64・切替付近・小σだけの最終比較、および修正済みsourceのGPU全回帰を別jobで進める。
