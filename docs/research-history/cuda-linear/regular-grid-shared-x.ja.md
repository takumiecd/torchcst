# RegularGridのH/dH生成で重なるXを共有する比較

2026-10-11。input-owner backward融合の負の結果を受け、G16の共有・寿命を
現行のままにし、H/dH生成で重なるXを8 atom内で共有する研究候補。
公開Registry/default、RegularGrid/FlatTorus、profile-product、operator-sites L2と
一回floor、singleton、task width stop-gradient、AdamW/Polar更新は変更しない。

## 実装と固定条件

`research_cuda_regular_grid_shared_x_h` v1、`SharedXHRecipe(panel_sites=32/64)`。
`input-shared-x32-bm16-g16-h32` / `input-shared-x64-bm16-g16-h32`を明示的に比較する。
forward、prepared source snapshot、input routing/sort、hot fields、G16/slab、BM16、
dX ownerとparameter partial/source VJPは既存BM16/G16へ戻して共用する。
G/dGはatomごとに一度計算し、Gをglobal Tensorへ保存してdXへ共有する。
H/dHの支持のunionを入力座標順の8 atomで計算し、32/64 site内に収まる場合は
[BM16,panel_sites]のXを一度loadし、register gatherから各atomのH/dHを作る。
unionの外側はloadしない。周期siteへ戻すのはXのload時で、支持のunwrapped
low/highと各atomのBK8 accumulation順は維持する。空支持・無効atomをunionに含めない。
大きなunionは既存のatom別gatherへ戻す。新たなsort・全量Xコピー・global panelは作らない。
source-only/dX-onlyは現行経路を使う。live Parameterはbackwardから読まない。

まず独立FP64/floor/empty/singleton/周期境界/全axis支持/strides/端数/
retained snapshot/20capturedとpublic更新をGPU検証する。producer単体も既存のGと
parameter partial、canaryを比較する。資格確認失敗時は時間比較へ進まない。
全GPU suiteは1000秒以内。数値許容値4e-4を変更しない。

既存N2048/B32/A209715、初期rho3/8、seed41、IEEE FP32/TF32off、MSE、24更新を固定。
各rhoにprimaryと独立逆順の2回、計4cohortを実行する。各cohortはfresh processの
matrix-torch、factor、onchip-h、現行BM16/G16、共有32/64、denseの7workers、2700秒上限。
全6CSTのinitial/post24全shape/all atom FP64 Y/dX/all four dPを照合してから、
支持union/共有panelの準備も含む未計装の完全step21samplesとcapture/replayの
allocated/reserved peakを測る。phaseは別Graphで測定しprimaryへ足し引きしない。
遅い候補も全結果を保存し、結果に応じた反復・構成の追加をしない。

比較後、固定post24 snapshotの実routeを別Graphで計装して、input routing、hot fields、
G/H/dH＋parameter partial producer、dX集約、partial accumulation、source VJPを分ける。
保存済みprimary・未計装routeとY/dX/all dPの一致とParameter不変を確認する。
compiler register/spill/shared bytesと論理的X load要求量はhardware cache/DRAM trafficとは
区別する。L2/DRAM/stall counterを測らずにcache bottleneckを確定しない。

採否は現行に対する両独立runの完全stepと総peak、およびdenseの両peakで決める。
raw source/results/tensorsはignored outputとremote runに保存する。これは研究schemaで、
公開benchmark submission/DB schemaの対応を意味しない。

## 初回コンパイルの修正

source `7d77b724fc6fbf1ea750a9aeb1b1f330a4b9a6ae`のGPU資格確認は81PASS/56FAIL。
Tritonで条件分岐内だけに定義した`panel`を後続loopで参照し、SSA値が未定義として
コンパイルに失敗した。性能cohortは0で、数学gateや時間・メモリの結果には扱わない。
両分岐のSSA値を定義するゼロ初期値を追加する。global Xのloadは共有分岐だけで維持する。
失敗source/137testsのlogs/XMLをremote `cst-shared-x-ada-7d77b724-20261011`と
localのsource archive/operating notesに保持し、修正commitは別directoryで全資格確認を行う。
数値許容値・比較条件・候補構成は変更しない。
