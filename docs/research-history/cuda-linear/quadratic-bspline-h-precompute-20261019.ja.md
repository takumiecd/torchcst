# H方式の係数前計算・入力問い合わせ・出力集約

同じ二次B-spline、全観測siteの離散product L2、Polar更新、task幅stop-gradientを保ち、
H方式の入力累積bank・係数・H生成・Y集約とbackwardを分けて比較する。
Tensor Core向けの密なprofile因子展開は導入しない。研究用v12で、本番登録は変更しない。

## 数学と共有単位

正規化index `z_d(n)=2n/(N_d-1)-1`。軸profileを4切替点のjoined係数で表す：

\[
f_{d,a}(n)=\sum_{r=0}^{3}\mathbf1[n\ge t_{d,a,r}]\sum_{k=0}^{2}J_{d,a,r,k}z_d(n)^k.
\]

入力bank `P_k(t)=sum_{i<t} X_i z_i^k` を使うと、

\[
H_{ba}=-\sum_{r,k}J_{i,a,r,k}P_{b,k}(t_{i,a,r}).
\]

Yへの寄与は `c_a=amp_a/max(sqrt(Q_o Q_i),floor)` として、

\[
\Delta_{k,b,t_{o,a,r}}\mathrel{+}=c_a H_{ba}J_{o,a,r,k},\qquad
Y_{bj}\mathrel{+}=\sum_k z_o(j)^k\sum_{t\le j}\Delta_{k,b,t}.
\]

入力側も出力側も現在のsigma/中心を使う。前処理は各forwardで更新し、forward時点の
Meta/cuts/Jをctxへ保存する。同じsnapshotをbackwardとretained backwardで使い、
後続forwardやoptimizer更新で上書きしない。norm/floorは従来prepareで一度だけ適用。
H/Gとその中心微分は従来のFP32保存相当への丸めを維持し、係数と収縮はFP64。

## 候補

- uncached split: 従来と同じ幾何計算をH問い合わせとY注入に残して、HをFP32で保存し別々のkernelで測る。
- cache fused: 各atom・各軸の4cutと3係数/2中心微分を一度準備し、register内Hから直接注入する。
- cache split: 前計算後のH queryとY injectionを実際に分離し、保存/読み戻し込みの完全stepを測る。
- compact cache: 3係数だけ保存し、固定sigmaに対する平行移動の恒等式
  `J'_0=-J_1/half, J'_1=-2J_2/half`で中心微分を得る。同じ数式だが丸め順の差は独立gateで判定。
- edge reduction: 終端Nへの未使用差分書込を省き、atom群内でcut0の寄与を足してからatomicする。
- segmented injection: 4cut eventを出力site別にinteger histogram/scan/placeで並べ、
  固定長のevent block内で同じsiteの寄与をsegmented sumしてからatomicする。
  入力側の同じ配置はbackwardのdX注入に使う。site×atomの全接続表は作らない。

segmented版のH/Gはforward/backwardの局所workspaceで、backwardへH自体を保存しない。
同じ形のworkspaceをallocatorが再利用でき、retained backwardで保存Hを破壊しない。
出力event orderはforwardで使い終われば解放し、入力event orderだけbackwardまで保存する。
4cut/axisは現在の値で毎forward構築するので、integer histogram/scan/place費用も主測定に含む。
clipped supportが2点以下の軸はnativeへ回し、norm-floorで増幅する係数相殺を避ける。

## 計測規約

既存Linear runner、L4、1024²/A52428、B64 mixedから開始。
基準は既存H `stream-reuse8`、最速W `range-reuse-a16w1`、Dense。
atom別sigmaがoptimizer stepで変わる同一fixture・初期P/X/target・AdamW/fused Polarを使う。
独立FP64の全shape・全atom・3 live状態でY/dX/全P勾配のmaxabsとrelative L2を両方4e-4未満に保つ。
可変sigmaのcutoff跨ぎ、切り出し/floor、保存snapshot、Graph/optimizerを別途検証。
主指標は未計装の構築・L2・optimizer込み完全stepとcapture/replayのpeak allocated/reserved。
別計装Graphのcoef_prepare/H_query/Y_inject、bank/scan/restore、back_query/dX_injectを記録する。
fused版のh_query/y_injectイベントは空で、stream_forward/backward spanに両処理が入る。
この空イベントをHが無料になった証拠として扱わない。部品時間は主時間から足し引きしない。

raw source/results/logsとhash検証はignored `output/bspline-h-precompute-20261019/`に保全。
性能と採否は実機結果の回収・検証後に記録する。

初回 `l4job-15b6cb5104204de3927a664e0f5e1536`（7325d2f2）は、GPUの係数準備kernelで
unrolled loop内のconstexpr再定義がコンパイルエラーになった。数値・性能結果は採用しない。
修正ad0954d0で再提出。初回のsource/result/log/manifestは保全している。


## 最初のB64 mixed比較

L4、Torch 2.11.0+cu130 / CUDA 13.0 / Triton 3.6.0、全52428 atom、B64、1024×1024。
各sigma群は `.1/.4/1.6/6.4/12`、各20%。以下は未計装完全stepの中央値。
peakはMiB、capture/replay込みのallocated / reservedであり、GPUプロセス総量ではない。

| 方式 | ms | allocated / reserved MiB |
| --- | ---: | ---: |
| 既存H stream-reuse8 | 7.388 | 54.194 / 114 |
| 最速W range-reuse-a16w1 | 3.258 | 78.840 / 156 |
| uncached split | 6.826 | 66.994 / 128 |
| cache fused | 4.673 | 71.191 / 132 |
| cache split | 5.046 | 83.991 / 160 |
| compact cache fused | 5.915 | 65.591 / 140 |
| cache fused + edge reduction | 4.461 | 71.191 / 132 |
| cache segmented | 5.377 | 84.795 / 160 |
| compact segmented | 6.621 | 79.392 / 156 |
| Dense | 0.108 | 49.502 / 106 |

分離したcache splitの別計装Graphでは、係数前計算0.060ms、H query 0.598ms、
Y injection 0.879ms。back_query（G/G'/H/H'と全atom VJPを含む）は1.265ms、
dX injection 0.855ms。input bankは約0.05ms、出力/input scanは各約0.024ms。
これらは主時間に足し引きしない。fused版のH query/injectionは一つのkernel内であり、
個別時間は分離版の測定値として読む。

係数をforwardで一度準備してbackwardまで共有する効果が大きい。
compact版は容量を減らしたが、中心微分の繰り返し除算などを含むbackwardが遅くなった。
segmented版はevent構築が出力/入力各約0.042msだが、H/Gのglobal保存・event参照・
segment reduction込みの完全stepで改善しなかった。atomic削減だけから高速化とは判断しない。
compiler reportではcache fused backwardのspill_slots=132、edge版は0だが、
物理DRAM転送・stall counterは測っておらず、この差だけで原因を断定しない。

この時点ではH方式をWより速いとはいえず、Denseの速度/メモリ条件も満たさない。
本番dispatcherへ採用しない。引き続きlaunchとcompact微分を別snapshotで検証する。

実機検証：独立FP64全shape gateは全候補でY/dX/全atom勾配の両基準を通過。
GPUの対象suiteは437 passed。可変sigma、cutoff跨ぎ、floor、保存snapshotを含む。
source ad0954d0、job `l4job-98d29ac329a045dca008bb30998a35e3`。
source archive SHA256 `4567df922efbf8aaab6b4c67f701a6c06f9a056fae91e4b9f6dfc84edda0f768`、
result archive SHA256 `efeb2d2a6a9cae9f5fb39569f9b4bd1c1c206a38911acf40a9a631623058db4c`。
36 manifest filesを照合し、returncode=0・timeoutなし・POOL_CLEANEDを確認した。


実行粒度の比較（source 870f004a、job `l4job-65b2f94b86e94cc581c4cf5b9bf245e1`）では、
full-cache fusedのatom4/warp1、atom8/warp1、atom8/warp2は各4.748/4.691/4.599msで、
既定atom8/warp4の4.514msを上回らなかった。event16/warp1は4.684msに改善したが、
allocated/reserved 84.795/160MiBと大きい。compact fusedは6.385〜7.745msで不採用。
追加のゼロ振幅・片側だけの勾配・save_w=Falseを含む41 tests passed、全shape gateも通過。
このsnapshotのcompact微分には除算が残っており、既存逆数の共有は次の8b404e77で検証する。
source archive SHA256 `aa58ca1229ead0cd998ad95124808007f7c394b655f8f7f926942f8091c159b6`、
result archive SHA256 `dff7d7ef607b94b9e8aee7595ff0a2122a215ed3529dd1b60540fdbc0f11f28c`。
38 manifest filesを照合済み。


## compact微分の逆数共有

8b404e77では、Meta内の逆spacingを使い、`inverse_half=inverse_spacing*2/(N-1)`、
`J'_0=-J_1*inverse_half`、`J'_1=-2J_2*inverse_half`とした。
同じ平行移動の恒等式で、繰り返しFP64除算を避ける。
cut0を持たないatom blockはedge reduction atomic自体を発行しない。
full/compact両版で形・L2・task幅stop-gradientを維持し、41 GPU testsと全shape gateを通過。

B64 mixedでは、compact atom4/warp1が7.310→4.529ms、allocated/reservedは65.591/140MiB。
full cache既定版は4.483ms・71.191/132MiB。Wは3.260ms・78.840/156MiB、Denseは0.109ms。
fullの実行粒度4/1・8/1・8/2は4.602/4.474/4.470msで、差は小さく新たな優位とは判断しない。
最終比較には既定full `h-cache-edges` とcompact `h-compact-edge-a4w1`を残す。
compactはallocatedを減らしたが、reservedも減るとはいえない。

full fusedの別計装では係数0.065ms、H→Y 1.125ms、backwardのbroad H/G・VJP・dX集約1.724ms。
小支持側W構築0.298ms、native atom VJP 0.624msも残る。
compactはH→Y 1.135ms、同backward1.787ms。これは完全stepの差分ではない。
full backwardは128 registers / spill_slots=0、compactは255 / 92。
このcompiler情報は実機のDRAM転送量・stallの計測ではない。

job `l4job-744c9411145f4f4fa39a0324337e16d6`、38 manifest files照合済み。
source archive SHA256 `aa233c5de83d582802eb9296daa3f030cc77317c84625d1baa4295ed12cf4216`、
result archive SHA256 `a1380c0f6853a9e93cc397fd50620ad3f01c8a4184591827ccb45fd79c2cdac8`。


## 通信中断と再実行

最終比較の最初のjob `l4job-a175399434aa481c8b9fc78120687fe9` は結果archiveを回収できなかった。
ホストが2026-10-19 12:50:50〜13:04:28 JSTにclamshell/maintenance sleepへ入り、
再接続時はpool設定のen7が存在せず、現在のdefault/Wi-Fiはen0だった。
応答待ちのローカルCLIへSIGTERMを送り、poolはquarantineした。stopの初回は通信経路の不在、
en0へ合わせた後は既に存在しないendpointの404となった。
poolの排他下でserver sessionsを読み、stale CLI cacheがpruneされ、server assignmentがゼロと確認。
その後の公式recoverで全slot停止と中断jobのfailed化を確認した。DB/slot記録を手動変更していない。
GPU計算の遅さや数値失敗とは扱わず、このjobは性能・GPUテストの成功根拠から全て除外する。
source/spec/transport logをignored evidenceの`interrupted-job/`に保全した。

以後は比較だけのjobとGPU testsだけのjobを分け、新規VMで実行する。
実験中のidle sleepを避けるため、supervisorの存続中だけ`caffeinate -i`を使う。
ネットワークの変更はpool CLI用interfaceだけで、システム全体のVPN/経路は変更しない。


## 固定したv12の完全step・独立再測定

runtime/test/runner/caseを固定した33a2a74bで、新しいL4 VMの独立processから再測定した。
GPU UUID `GPU-a5fbfdef-7b9d-04fc-2a35-f3b1411b5194`、driver 580.82.07、23034MiB。
Torch 2.11.0+cu130 / CUDA 13.0 / Triton 3.6.0 / Python 3.13.15。
B32/B64 mixedとswitchは2 run（同一VM、異なるprocess）、smallは1 run。
以下の時間範囲は各runの未計装Graph medianの最小〜最大、MiBはallocated / reserved。
Denseは同じ入出力shape・batch・dtype・精度設定・optimizerの対照で、同一atom Parameter化ではない。

| 条件 | 方式 | 完全step ms | allocated / reserved MiB |
| --- | --- | ---: | ---: |
| B32 mixed | stream-reuse8 | 4.320–4.321 | 51.688 / 116 |
| B32 mixed | range-reuse-a16w1 | 3.223–3.225 | 78.590 / 154 |
| B32 mixed | h-cache-edges | 2.775–2.775 | 68.488 / 132 |
| B32 mixed | h-compact-edge-a4w1 | 2.812–2.813 | 62.888 / 140 |
| B32 mixed | dense_linear | 0.082–0.083 | 49.002 / 106 |
| B64 mixed | stream-reuse8 | 7.396–7.396 | 54.194 / 114 |
| B64 mixed | range-reuse-a16w1 | 3.246–3.249 | 78.840 / 156 |
| B64 mixed | h-cache-edges | 4.342–4.349 | 71.191 / 132 |
| B64 mixed | h-compact-edge-a4w1 | 4.414–4.436 | 65.591 / 140 |
| B64 mixed | dense_linear | 0.103–0.104 | 49.502 / 106 |
| B32 switch | stream-reuse8c02 | 4.631–4.639 | 51.688 / 116 |
| B32 switch | range-reuse-a16w1 | 3.220–3.223 | 78.590 / 154 |
| B32 switch | h-cache-edges | 4.746–4.748 | 68.488 / 132 |
| B32 switch | h-compact-edge-a4w1 | 4.760–4.763 | 62.888 / 140 |
| B32 switch | dense_linear | 0.082–0.082 | 49.002 / 106 |
| B32 small | matrix-support4a16 | 1.214 | 50.231 / 114 |
| B32 small | range-reuse-a16w1 | 3.176 | 78.590 / 154 |
| B32 small | h-cache-edges | 1.468 | 68.488 / 132 |
| B32 small | h-compact-edge-a4w1 | 1.477 | 62.888 / 140 |
| B32 small | dense_linear | 0.082 | 49.002 / 106 |

B32 mixedでのH（cutoff=.5）は最速Wより約14%速く、旧Hより約36%速い。
B64 mixedは旧H比約41%短縮だがW比では約34%遅い。Denseの時間・総peak目標には届かない。
smallは全atomが小支持経路となり、cache容量とH側の固定処理が余分になる。
既存小支持専用版1.214ms・50.231/114MiBが新Hの1.468ms・68.488/132MiBより有利。
smallの結果から広いsigmaや逆方向の混在分布の採用判断はしない。

switchではsigma-only診断でcutoff=.5を跨ぐatomが20971個、native countが20971→41942へ増えた。
最終device countも41942。sigma-only診断とtiny-support guard込みdevice countは別に記録する。
placementは52呼出し中3 rebuild・49 reuseで、sigmaが変わるたびに全sortしていない。
遅さの主因は別計装で小支持W構築1.024msと小支持atom VJP 2.735msに集中し、
broad H→Y 0.212ms・broad backward 0.314msではない。
単にH自体が遅いとせず、係数共有で変わった費用に合わせてcutoff=.2を追加比較する。
この追加はcatalogへの2 Plan追記のみで、340 runtime filesとtests/runner/既存Caseは変えない。

全atom sigmaは初期→最終で52428/52428が変化し、初期P/X/targetのhashも全方式・run間で一致した。
全shapeの独立FP64 gateは3 live状態全てPASS。
この2 runでのH候補maxabsはY 2.694e-6、dX 2.758e-6、全P勾配3.879e-5、
relative L2は各1.708e-7 / 1.724e-7 / 1.469e-7以下。基準は両指標4e-4未満のまま。

- job `l4job-112941f281d9421c9a10ffbf337ccde8`: 51 manifest files、source archive SHA256
  `06a7809b2e53bb76f2eefb05257f487fcffc1bbf3ac26248ccfc0c53578b4610`、result archive SHA256
  `e40c3b11267dad62034d9c75f017306a6b57acb0dcd12b96ea16af18b1b6128b`。
- job `l4job-c49a75e2e948486c942b8aeece868f7f`: 67 manifest files、source archive SHA256
  `f8e0d48b9a8863e12c7f76e8d39c01cdf93e2a6c48e409781074154f1607e091`、result archive SHA256
  `f39cde6d69bc30eacbf1990c3a8e072d7f581eeac54263235ffdeea4b99e82c8`。

両jobともreturncode=0、timeoutなし、archive/全manifest SHA256とPOOL_CLEANEDを確認。
raw result・source・width全配列・logsはignored evidenceに保存。


### 保存容量と対応範囲

係数snapshotの固定容量は、fullで`2軸*4cut*(5*8+4)*A = 352A byte`、
compactで`2*4*(3*8+4)*A = 224A byte`。A52428では17.600 / 11.200MiB。
native atomの未使用slotも含む。毎stepの幅と経路が変わるGraphで、現在のnative countを
ホストへ読み戻してallocate量を変える方式にはしていない。
これは係数/cutだけの予算で、表の総peakとは異なる。Parameter・optimizer・Meta・
native W/dW・bank・保存snapshot・allocator/Graph poolも総peakに入る。

split/segmentedのHまたはGはFP32で`4*B*A`、B64なら1buffer 12.800MiB。
H/Gをbackwardまで全保存せず、forward/backward局所workspaceとして使う。
fused版はこのglobal H/Gを確保せず、atom群のregisterから直ちに消費する。
segmentedの出力event orderはforward後に解放し、入力event orderだけbackwardまで保存する。

対応は既存研究routeと同じ、Euclidean RegularGridの各論理軸1座標、
Ni/Noそれぞれ2〜1024、batch1〜64、Polar [A,4]、CUDA FP32、TF32/autocastなし。
chart座標はFP64固定buffer、chart勾配と二階微分は対象外。
一般の多座標側profile-productや周期geometryへの対応をこの結果から主張しない。


## 最終GPU回帰テスト

job `l4job-d12cffa9f75d498fae37bf6545915c2e`、source 33a2a74bで対象suiteは
**444 passed**（CPUで動く5 metadata testsと439 CUDA tests）、516.81秒。
一つの既存stream testでAccumulateGradのstream警告が出たが、失敗・skip・gate緩和はない。
新v12の可変幅/shape/floor、Graph cutoff跨ぎ、zero amplitude、保存したforward状態、
片側だけのgrad、save_w=Falseを含めて完了した。
source archive SHA256 `aac542ee9605ad5955ad2c0b4458e045424e09593e57a6f3a060a6a15a600fd8`、
result archive SHA256 `97e612e1c648914f65e84ddb7c15145dca47bb9e7190f84e73ffe1f8c2c521f4`。
4 manifest files・両archiveのSHA256、returncode=0、timeoutなし、POOL_CLEANEDを確認。

ローカルCPU suiteは1972 passed / 3215 skipped（CUDA/外部サービス）で、18件の既存JIT deprecation警告。
変更ファイルのRuff check/format、各Caseのdeclarations checkを通過。
wheel/sdistは340 runtime filesをソースとbyte照合した。
wheel SHA256 `ded8e25f9dc18f1d645632fcf3646dcabb81ff9cc9c3823482d1741743c53b6e`、
sdist SHA256 `2154dd25b8948add051a80360cc1d59200bb5c1a559cdeb5d8547affc41637d2`。
CPU CIもa50fa682で通過。packageのruntime proofとGPU proofのruntimeは同じ内容。
追加cutoff比較は既存宣言を変えずcatalogへ2 Planを追記するだけで、既存GPU testsと340 runtime filesに差分はない。


## 係数共有後のsigma cutoff再評価と最終判断

小支持側へ回す閾値を.5から.2へ下げた2 Planをa50fa682で追記した。
sigma自体や数学上の支持は変えず、計算経路だけを変える。同じruntime/test/runnerを使い、
元のCaseと同じP/X/target hashで全atom sigma更新・全shape gateを確認した。
各条件で独立process 2 run（同一L4 VM）。各runの未計装完全step medianの範囲を示す。

| 条件 | 方式 | 完全step ms | allocated / reserved MiB |
| --- | --- | ---: | ---: |
| B32 mixed | range-reuse-a16w1 | 3.221–3.221 | 78.590 / 154 |
| B32 mixed | h-cache-edges | 2.775–2.776 | 68.488 / 132 |
| B32 mixed | h-cache-edges-c02 | 2.572–2.588 | 68.488 / 132 |
| B32 mixed | h-compact-edge-a4w1-c02 | 2.620–2.654 | 62.888 / 140 |
| B32 mixed | dense_linear | 0.082–0.083 | 49.002 / 106 |
| B32 switch | range-reuse-a16w1 | 3.219–3.221 | 78.590 / 154 |
| B32 switch | h-cache-edges | 4.745–4.746 | 68.488 / 132 |
| B32 switch | h-cache-edges-c02 | 2.626–2.674 | 68.488 / 132 |
| B32 switch | h-compact-edge-a4w1-c02 | 2.625–2.628 | 62.888 / 140 |
| B32 switch | dense_linear | 0.082–0.083 | 49.002 / 106 |
| B64 mixed | range-reuse-a16w1 | 3.245–3.248 | 78.840 / 156 |
| B64 mixed | h-cache-edges | 4.346–4.350 | 71.191 / 132 |
| B64 mixed | h-cache-edges-c02 | 4.832–4.840 | 71.191 / 132 |
| B64 mixed | h-compact-edge-a4w1-c02 | 4.897–4.972 | 65.591 / 140 |
| B64 mixed | dense_linear | 0.103–0.104 | 49.502 / 106 |

B32 mixedはcutoff=.2のfull Hが今回比較したCST候補では最速で、Wより約20%短縮。
B32 switchもcutoff=.2で旧H/W/新H(.5)を上回る。
compactはB32 mixedでfull Hより少し遅いがallocatedを減らし、switchでは同じ速度帯。
reservedはcompactの方が大きいので、単一の「省メモリ」勝者とはしない。
B64 mixedはcutoff=.2でHを増やすと遅くなり、H内では.5が有利、全CST候補ではWが有利。
all-smallは既存グループ直接版が有利。今回の条件での候補選択であり、任意分布への自動dispatcher採用ではない。

係数共有でHの費用が下がると、従来のsigma cutoffも最適とは限らない。
Hは原理的に不利とも、どのbatchでもWを上回るともいえない。
H queryとY/dXへの注入はatom×batchごとに必要であり、Wの構築はbatch非依存で再利用できる。
B32/B64の差はこの費用構造と整合するが、物理DRAM/atomic/stall counterを測った因果証明ではない。

Denseは依然大幅に速く、allocated/reservedの双方も新Hより小さい。
Denseと同等の速度・総peakという採用目標は未達。既定登録を変えず、研究用Draft PR111で保全する。
変更は係数snapshotの共有、compactの既存逆数利用、未使用終端書込の省略、cut0局所加算であり、
新しいKernelSpec・形状近似・L2積分近似・sigma量子化・勾配基準緩和は導入していない。
segmented event版は総stepと総peakで採らない。再現用のPlanと負の結果は残す。

- job `l4job-e3f9a86bb8e74185aa1503ccfb05427a`、source archive SHA256 `1bd18f519f41f336f719016f50b2891322685f9c87b4a844e7a52a621dd0ea4d`、result archive SHA256 `2a838d13658abfcc2f91bd96e62da10cf5ad698137e37ffb04ee159ba2988240`、54 manifest files照合済み。

- job `l4job-0dbccdfe8b6c49088e1e0952f2fba48b`、source archive SHA256 `1c629216c444e593935e21e1e1322753f29326942c4c2b5729708a1af5d506e4`、result archive SHA256 `b365fa1525505def76d7ef280d927600f205aee26fae1537a45bdfcc5841ff30`、54 manifest files照合済み。

両runのreturncode=0・timeoutなし・POOL_CLEANED、340 runtime filesとtests/runner/既存Caseのbyte一致を確認。
前の比較と異なるcatalogは末尾追記だけで、全既存Plan宣言が同一であることも照合した。

全final/cutoff比較を合わせたH候補の独立FP64誤差上限：

| 対象 | maxabs | relative L2 |
| --- | ---: | ---: |
| y | 2.6934e-06 | 1.70768e-07 |
| dx | 3.55559e-06 | 1.72302e-07 |
| dp | 3.8789e-05 | 1.46885e-07 |

各判定は両指標4e-4未満。全52428 sigmaの初期→最終変化を確認し、入力/P/target hashも分布別に全方式・全runで一致。
物理counter、別GPU世代、一般多次元/周期grid、実DB投稿、本番dispatcherの採用は未実施。


## 再現と保全

選択した閾値比較のcatalog/Caseはa50fa682にあり、最終文書追加後もruntimeは同一。

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-mixed-h-cutoff.json \
  --polar-update fused --phase-diagnostics --kernel-diagnostics \
  --source-commit "$(git rev-parse HEAD)" --output output/h-cutoff.json
python -m benchmarks.submissions check output/h-cutoff.json
python -m pytest tests/test_quadratic_bspline_cuda.py -q
```

B64は`b64-mixed-h-cutoff.json`、切替は`switch-h-cutoff.json`、smallは`small-h-final.json`。
回収archiveと全manifestのSHA256検証、全shape gate、初期条件一致、全幅の変化、
sourceのbyte照合、package runtime照合をignored evidenceの検証JSONへ残した。
終了時はqueueにrunning/queuedがなく、poolの全slot stoppedとserver停止確認を取得済み。
PR111はPR110をbaseにした研究用draftとして維持する。mainの変更・mergeはしていない。
