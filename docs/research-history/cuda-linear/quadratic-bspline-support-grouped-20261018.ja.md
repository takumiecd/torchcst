# Quadratic B-spline: live支持点数によるatomのグループ化

2026-10-18。`kernel/quadratic-bspline-prefix` / Draft PR #109。
基準は[前回の選択結果](quadratic-bspline-parallel-20261017.ja.md)。
数学的なprofile、Polar幅則、全観測siteの厳密離散L2とその中心微分を変えず、
行列構築とatom VJPの支持ループだけをまとめる。

## 処理と保存

`matrix_support_grouped`はv3 Recipeの追加strategy。既存の5/9/13個の
serialized fieldや既存Planの意味は変えない。公開dispatcherには登録しない。

各forwardで現在のatomからFP64 Metaを構築し、各軸の支持区間を格子端でclipする。
半開区間の境界は既存matrix kernelと同じ`ceil`とreciprocal設定を使う。
支持点数を`Sᵒ_a,Sⁱ_a`、patch幅を`P`として、

\[
t^o_a=\lceil S^o_a/P\rceil,\quad t^i_a=\lceil S^i_a/P\rceil,\quad
k_a=t^o_a(\lceil N_i/P\rceil+1)+t^i_a.
\]

GPUでint32 keyを生成し、`torch.argsort`のint64 atom ID列を得る。
ソート後の連続`AB`個を一つのCTAで処理する。同じkeyのatom群の端では
一つのCTAに異なる支持点数が混在し得るが、全atomを一度ずつ扱う。
支持を捨てず、σの上限による固定小支持の仮定もしない。

\[
E_g=AB\,P^2\max_{a\in g}t^o_a\max_{a\in g}t^i_a,
\qquad U=\sum_a S^o_aS^i_a.
\]

`E`の減少は実行patch枠の減少であり、速度向上率ではない。
並べ替え・間接参照・atomic加算・正規化・optimizerを含めて実測する。
実際の最終不完全CTAの無効laneも既存kernelと同じmaskで除外する。

ParameterとAdam状態は元の順序を保持する。W構築はIDでMetaを参照し、
separable VJPは元のatom番号にDPを書き戻す。forwardで得たID列を
Meta/X/W/格子snapshotとともにbackwardまで保存し、更新後のPで再分類しない。
次のforward/CUDA Graph replayでは現在の幅・中心からkeyとIDを再生成する。

\[
W=\sum_{a=0}^{A-1}W_a=\sum_{r=0}^{A-1}W_{\pi(r)}.
\]

支持点数やsort順序は離散的に切り替わるが、全atomの同じ寄与を合計する実行順序の変更である。
routingそのものを微分せず、同じprofileとL2のpullbackを元のatomへ返す。
加算順序の浮動小数点差は独立oracleの固定基準で検証する。

ID保存量は`8A` byte、keyは`4A` byte。A=52,428ではそれぞれ0.4000MiB、
0.2000MiB。このほかGPU sortの作業領域・Graph poolがあるため、合計peakは
allocatorのcapture/replay込みallocated/reservedで測定する。
site×atomの支持index表は作らない。全W/DW各4MiBは比較元のmatrix経路と共通。

## 検証と比較条件

L4、N=1024、A=52,428、B=32、spacing=0.1。FP32 source/input、
FP64幾何・profile・L2・atom reduction。IEEE FP32、TF32無効。
同じAdamW+fused Polar更新を毎step実行する。

- small: 初期σ=0.1/0.14/0.18、各約1/3、atom順をshuffle。
- mixed: 初期σ=0.1/0.4/1.6/6.4/12、各約1/5、shuffle。
- 幅の許容上限12.8、atomごと・stepごとにlive σを評価する。
- small対照は`matrix-separable4p1`、mixed対照は`hybrid-vjp16`。各条件でDenseも測る。
- 既存runnerによる独立FP64全軸Y/dX/全atom勾配のmaxabsとrel-L2両方<4e-4。
- 支持の空/単一/広い範囲、異なる軸spacing・origin、floor作動、
  支持点数の切替、retained backward、24回Graph更新のoptimizer状態を別途検証。

CPU全suiteは1968 passed / 2955 skipped / 18 warnings（38.11s）。Ruff、
新旧catalog/Caseの宣言往復チェックが通過した。wheel/sdist内の332 runtime fileは
checkoutとbyte一致。最初の`build --no-isolation`は手元venvにhatchlingがなく失敗し、
標準の隔離buildで成功した。

L4 qualificationは180 passed（metadata 1、CUDA 179、173.84s）。
`matrix_atoms=4/8/16`、不完全最終group、空のatom集合も含む。
source commit `ddcb46f15b4333de834959ae89113d84a10e039e`。

| 実行 | job | source archive SHA256 | result archive SHA256 |
| --- | --- | --- | --- |
| 接続/CUDA動作確認 | `c0283395c36d435ca7a85ae256d7a89e` | `23301eab839c83c480c337e93fe25fc7f2955c101846def3c3182ef324d8034e` | `0681b2c13ee2ba43f46017db6da7fc231fb15e1031f9bc29497d81caef2e148d` |
| 全GPU試験 | `a7dd1a25de884dc4bd5f0758b0437b96` | `02195ca30b2f13c73b8f2723ceeb10bd7387adf3504d9bfb717d0cb32b6ad741` | `7ca0171bc96f2b942f6746c1c6c9d8996a45291d43126fec4b22829d21200ee6` |

両archiveとmanifest全項目を照合し、source/result/receiptを
`output/bspline-support-grouped-20261018/verified-jobs/`へ保存した。

再現は既存runnerを使う。

```bash
python -m pytest tests/test_quadratic_bspline_cuda.py -q
python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-small-support-grouped.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-mixed-support-grouped.json
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-small-support-grouped.json \
  --polar-update fused --kernel-diagnostics --phase-diagnostics \
  --output output/bspline-support-small.json
python -m benchmarks.submissions check output/bspline-support-small.json
```

mixedはCase名を`mixed-support-grouped`に置き換える。

## 完全stepの初回比較

source/runtimeは上記qualificationと同じ。GPUはNVIDIA L4、Torch 2.11.0+cu130、
CUDA 13.0、Triton 3.6.0、Python 3.13.15。
各候補は独立process、21 samplesの中央値。CST候補は同じ初期P/X/target hashで開始した
（Denseは同shapeの通常の学習可能行列を使う比較であり、atomの再parameterizationではない）。

| 条件 | 今回測り直した従来最速 ms | 支持点group AB=8 ms | Dense ms | 従来allocated/reserved MiB | group allocated/reserved MiB | Dense allocated/reserved MiB |
| --- | ---: | ---: | ---: | --- | --- | --- |
| small | 1.606197 | 1.226771 | 0.083062 | 49.831/112 | 50.231/114 | 49.002/106 |
| mixed | 8.793847 | 444.667121 | 0.083174 | 35.679/98 | 50.231/114 | 49.002/106 |

smallは23.62%短縮、Denseの14.77倍。mixedは従来H経路の50.57倍の時間で、
このmatrix方式をmixedへ適用する理由はない。両条件とも全52,428 atomのσが更新された。
初回groupの幅変化最大はsmall約0.000213、mixed約0.4546。
小さいσだけの結果を混合条件のdispatcher採用へ一般化しない。
allocated/reservedはcapture/replayを含む。総GPU process memoryは未計測。
smallでもDenseよりallocated約1.23MiB、reserved8MiB多く、時間・メモリ両方での
Dense到達とは判定しない。

### stageと支持の量

別の診断Graph内のCUDA event中央値。norm-only replayは診断にだけ追加する計算で、
以下のprepareには既に厳密L2とその微分が含まれる。主測定からnorm-only時間を引かない。
stageの和を完全step時間として扱わない。

| smallのstage | 従来 ms | group AB=8 ms |
| --- | ---: | ---: |
| prepare（厳密L2込み） | 0.161792 | 0.161792 |
| key構築+sort | — | 0.047104 |
| W構築 | 0.427008 | 0.310272 |
| 全atom VJP | 0.929792 | 0.618496 |
| Y/dX/dWの3 GEMM合計 | 0.041984 | 0.041984 |

smallのW構築は27.34%、atom VJPは33.48%短縮。分類の費用を上回った。
prepare register数128、group W59、group VJP96。compilerのspill slotと
PTX local load/storeはすべて0/false。cache hitやatomic stall等の物理counterは測っていない。

mixedではgroupのWが99.406ms、VJPが335.724ms、sortは0.049ms。
ソートが遅いという結果ではなく、広い支持をすべて対で評価するmatrix経路の仕事量が大きい。
初期状態をCPUで数えると、smallの実支持対数は2,592,254、mixedは3,593,625,196。
AB=8の推定patch枠数は以下のとおり。

| 条件 | 元順序 | 支持点順序 | 枠の減少 |
| --- | ---: | ---: | ---: |
| small | 7,181,056 | 4,331,136 | 39.69% |
| mixed | 13,505,859,584 | 3,690,268,416 | 72.68% |

mixedで枠が73%減っても、累積集計の3多項式区間へ置き換えられるわけではない。
この負の結果はW方式の適用範囲を示すもので、H側の別のgrouping方式を検証したものではない。
上の枠数は初期stateの計算量モデルで、GPU instructionやspeedupの実測値ではない。

### 初回の全軸FP64照合

3 live stateのworst値。基準4e-4はmaxabs/rel-L2の双方へ適用する。

| group AB=8 | Y maxabs / rel-L2 | dX maxabs / rel-L2 | 全DP maxabs / rel-L2 |
| --- | --- | --- | --- |
| small | 2.344e-6 / 1.700e-7 | 2.594e-6 / 1.702e-7 | 1.286e-5 / 1.058e-7 |
| mixed | 1.814e-5 / 1.463e-6 | 1.593e-5 / 1.430e-6 | 1.452e-5 / 1.051e-7 |

benchmark result JSONのsubmission整合チェックも両条件で通過した。

初回完全step job `d85dfda243f04d51b3669f2d2fb8dcdb`：
source archive `2857983dc271e3f89a6946bfe9d0628aa05564b122dea8c0da72616ac0846d18`、
result archive `f0c0539fb8c9d11e11f15471ccc986f969206d34923ef556ba35f37dbc490a33`。
29 manifest項目を照合。sourceは`ddcb46f1`に当ノートの当時の未commit版を加えたsnapshot。

## 小支持のgroupサイズ比較と独立再測定

別のL4 VMでsmallの従来最速、AB=4/8/16、Torch、Denseを独立processで比較した。
21 samples、同じ初期P/X/target、全52,428 σの更新、各候補の独立FP64全勾配検査を維持。

| small | 完全step ms | W ms | atom VJP ms | allocated/reserved MiB |
| --- | ---: | ---: | ---: | --- |
| 従来最速 | 1.608939 | — | — | 49.831/112 |
| AB=4 | 1.551591 | 0.437248 | 0.816128 | 50.231/114 |
| AB=8 | 1.226322 | 0.310272 | 0.620544 | 50.231/114 |
| AB=16 | 1.207850 | 0.300032 | 0.614400 | 50.231/114 |
| Dense | 0.083060 | — | — | 49.002/106 |

key構築+sortはいずれも0.047104ms、prepareは0.161792ms。
AB=4へ小分けするとW/VJPとも遅くなった。AB=16のW/VJPのregister数は96/168、
spill slot 0、PTX local load/storeなし。AB=8の59/96より増える。
このcompiler報告だけから実際のcache/occupancyの原因は確定しない。

各AB=4/8/16のY maxabsはそれぞれ2.344e-6/2.344e-6/2.398e-6、
dXは2.344e-6/2.378e-6/2.746e-6、全DPは1.504e-5/1.441e-5/1.402e-5。
全rel-L2も1.728e-7以下で固定基準を通過した。

job `7e09d5f494f744b7a02295fe9ba5faec`：
source archive `5a8a0259beb62d0ee0a75d68e21c54904a99a5f7ad79505ad014d6f11817cd17`、
result archive `0da958bce8af477c0ed6ebc79dc2b77f0f5a007590e60548d62672383a0b6811`。
23 manifest項目、result JSON整合チェック、returncode=0を確認。
既存runnerへ渡したsweep用Case JSONもartifactとして保存した。
最終small Caseは同じ候補一覧を追跡し、単独の測定runnerは追加しない。

### AB=8/16の最終再確認と判断

さらに別のL4 VMでAB=8/16と対照を再測定した。

| small、独立job | 従来最速 ms | AB=8 ms | AB=16 ms | Dense ms |
| --- | ---: | ---: | ---: | ---: |
| 初回 | 1.606197 | 1.226771 | 未測定 | 0.083062 |
| groupサイズ比較 | 1.608939 | 1.226322 | 1.207850 | 0.083060 |
| 最終再確認 | 1.603437 | 1.222572 | 1.207326 | 0.082046 |

AB=8の改善は3独立jobで再現した。AB=16は2独立jobで再現し、各jobでAB=8より
約1.25–1.51%短かった。AB=16は従来比24.70–24.93%短縮、最終比較ではDenseの14.72倍。
メモリはAB=8と同じ50.231/114MiBである。今回のsmall条件の研究候補にはAB=16を選ぶ。
小さい支持での約25%短縮を、mixedや任意の将来幅分布へ一般化しない。

最終AB=16のstageはprepare 0.161792ms、分類+sort 0.047104ms、W 0.301056ms、
全atom VJP 0.611328ms。主な残りの費用は全atom VJPとW構築で、
ソートだけを消してDenseへ届く状態ではない。
最終のY/dX/全DP maxabsは2.398e-6/2.410e-6/1.455e-5、
rel-L2は1.701e-7/1.717e-7/1.073e-7。全52,428 σが更新され、
初期P/X/target hashも各CST候補で一致した。両jobとも全軸FP64照合は省略していない。

最終job `6bd34cf27c25454d9f274200acaee14f`：
source archive `fc0c775936b21917e06ac7121c7190594d798d415926b73e426f0e263d374260`、
result archive `6be17948bf004efeca5caf7342043297c9a80b6f033c32ed830ef228358b565f`。
20 manifest項目、result JSON整合チェック、returncode=0を確認。
runtime/GPU test sourceは全試行で`ddcb46f1`と同じで、後続差分はCaseの候補一覧と文書のみ。
wheel/sdistのruntime 332 fileも同じ内容である。

全jobの結果を回収・照合し、poolの全slotは停止、queued/running jobなしを確認した。
研究Draft PR #109へまとめ、公開dispatcherは変更しない。
8192²完全step、周期・複数座標CUDA、総process memory、実DB登録は未検証。
