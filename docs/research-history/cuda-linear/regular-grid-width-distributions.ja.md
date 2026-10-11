# RegularGridの支持幅分布別比較

2026-10-11。速度の限界をrho3/8だけで判断せず、狭い・中間・広い・混在の条件で
H/Gの生成・集約・微分の費用を確認する。公開dispatcherの変更は行わない。

## 固定した比較条件

RTX 6000 Ada、D2 FlatTorus、RegularGrid `((2048,), (2048,))`、spacing=1、
B32、209715 atoms、seed41、FP32、TF32無効、全積のdiscrete L2正規化とfloor=1e-6。
新しい全caseで共通の幅境界はminimum/birth/upper_floor=0.125、maximum=16。
旧rho3/8 caseの下限1とは異なる研究条件として区別し、旧結果との速度差を改善率にしない。

| case | 初期rho | atom比率 |
| --- | --- | --- |
| rho0p5 / rho0p75 / rho1 | 0.5 / 0.75 / 1 | 各単一幅 |
| rho1p25 / rho3 | 1.25 / 3 | 各単一幅 |
| rho8 / rho15 | 8 / 15 | 各単一幅 |
| mix-narrow | 0.5, 3, 15 | 8:1:1 |
| mix-balanced | 0.5, 3, 15 | 1:1:1 |
| mix-wide | 0.5, 3, 15 | 1:1:8 |

幅のquotaは整数切り捨て後の余りを最後の群に入れ、seed+7919でatomへ割り当てる。
中心と振幅の乱数列は従来fixtureと共通。各Planに同じ初期Parameter、入力、target、dyを渡す。
幅はPolar Parameterから生成し、24更新中に変化する。固定幅に置き換えない。
現行Polarは共有alphaを使い、この研究の同じ側別boundsでは入力・出力幅が等しい。
各atomで独立した入力・出力幅の分布を与えた実験ではない。

Triweightの支持は `abs(delta) < sigma`。rho<=1でも中心の位置によって空支持・単一・複数siteとなる。
一般にone-hotと仮定せず、両軸の支持site数、空product atom数、幅分位点、実際の幅変化を記録する。
空支持を削除したり、floorを軸別に変更したりしない。

## 測定と採否

既存 `benchmarks.cuda.linear.periodic_comparison` に新しいfrozen caseを追加し、別runnerは作らない。
比較Planはmatrix-torch、factor、onchip-h、現行tile-owner-late-bm16、dense。
全Planでoptimizer state preallocate、Graph開始前warmup gradient解放、native allocatorを揃える。
準備・集約・全backward・AdamW/Polar更新を含む未計装の完全stepを主指標とする。
別Graphのforward/loss・backward・optimizer診断は主計測と非加算。
capture/warmup/replay全体のpeak allocatedとreservedを報告する。GPU process全体は未測定。

幅分布ごとにCPUと小さいCUDA回帰を通し、全shape・全atom・Y/dX/all_dPの独立FP64 gateを
initial/post24で確認する。許容誤差は従来のmax_abs/relative L2とも4e-4を変更しない。
失敗caseの時間を有効な速度比較として使わず、raw snapshotと失敗を保管する。
各有効caseは独立2cohort、2回目はPlan順を反転。21時間sampleを21独立runとは数えない。

再現例:

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:native python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-width-2048-mix-balanced.json \
  --plans matrix-torch factor onchip-h tile-owner-late-bm16 dense \
  --isolated-oracle --phases --optimizer-state-policy preallocate \
  --graph-pool-policy release-warmup --output output/width-distribution/comparison.json
```

測定結果とsource/result hashは実験完了後に追記する。
