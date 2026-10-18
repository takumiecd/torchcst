# Quadratic B-spline: atomごと・stepごとのsigma振り分け

2026-10-18。研究branch `kernel/quadratic-bspline-prefix` / Draft PR #109。
[支持点数grouping](quadratic-bspline-support-grouped-20261018.ja.md)の小支持W方式と、
既存の大支持H/prefix方式を同一forwardへ組み合わせる。数学的なprofile、L2、Polar則は共通。

## 数式と実行

時刻tの現在のsigmaと閾値cから、全atomを重複なく二群へ分ける。

\[
\mathcal S_t=\{a:\sigma_a(t)\le c\},\quad
\mathcal B_t=\{a:\sigma_a(t)>c\}.
\]

既存の生profileを`fᵒ,fⁱ`として、全観測siteのnormを一度だけ準備する。

\[
D_a=\max\left(\sqrt{\sum_j f^o_a(j)^2\sum_i f^i_a(i)^2},\epsilon\right),
\quad W_a(j,i)=\frac{A_a}{D_a}f^o_a(j)f^i_a(i).
\]

\[
W_S=\sum_{a\in\mathcal S_t}W_a,\quad
H_{ba}=\sum_i X_{bi}f^i_a(i),\quad
Y=XW_S^T+\sum_{a\in\mathcal B_t}\frac{A_a}{D_a}H_{ba}f^o_a(j).
\]

smallは4点patchごとの両軸支持点数でIDをまとめ、16 atom/CTAでW構築する。
broadは既存の軸別native/3区間prefix queryと差分注入を使う。
broad内の軸が32点以下なら既存のnative軸別処理になる。σによる二群化と、
各軸のnative/prefix切替は別の実行判断であり、どちらも支持を切り捨てない。

小さい側のbackwardは`dW=dYᵀX`とseparable VJP、大きい側はH/HP/G/GPの既存pullback。
両者のdXを合計し、DPは元のatom番号へ書く。DPの二群は互いに排他的で加算不要。
norm floorは全profile積の後へ一度だけ適用し、中心についてのnorm微分も両群で保持する。
Polar task VJPのsigma stop-gradient契約、幅を変えるPolar更新は既存Torch参照と同じ。

分類が離散的に変わっても、同じ全atom和の計算方法を変えるだけである。
routeの指示関数を新しいカーネル窓として掛けたり、切替でprofileを変更したりしない。
浮動小数点の加算順序の差は独立FP64 oracleで検証する。

## GPU上の分類と寿命

`research_cuda_quadratic_bspline` revision v4 / `BSplineSplitRecipe`を追加する。
v1/v2/v3の5/9/13 serialized fieldは維持。v4は既存13 fieldに`sigma_cutoff`を加えた14 field。
公開dispatcherの既定には採用しない。matrix側があるため各軸1024 siteまで。

1. live PからFP64 Meta[A,12]を準備。厳密離散L2は既存の有限和moment式で共通に計算。
2. 各軸のclip済み支持patch数をkeyにし、large flagを上位へ加える。
   GPU sortでsmall IDを前半、broad IDを後半へ集める。small数はGPU上のint32 reduction。
3. 固定容量のlaunchとdevice countで非担当CTAをskip。hostへのcount読み戻し、
   CPU同期、動的shapeの`nonzero`、site×atom支持表を使わない。
4. W小支持和、GEMM、broad query/inject/scan/restoreを実行してYを合算する。
5. Meta/X/W/H/HP/格子snapshot/ID/countをbackwardまで保存。
   backwardはforward時の分類を使い、更新後Pで再分類しない。
   次のforward/Graph replayは現在値から分類・sort・countを再構築する。

ParameterとAdam状態の順序は動かさない。H/HP/G/GPは各[B,A]の固定容量で、
broadの有効slotだけを前詰めして読み書きする。空いたslotを容量から除く動的確保ではない。
各bufferのFP32 payloadは`4BA` byte、W/dWは各`4NoNi` byte、IDは`8A` byte。
backwardのsmall VJPが終わるとdWを解放してからbroad scratchを作る。
このTensor見積りと、実測するGraph capture/replay込みallocated/reserved peakを区別する。

## 検証と条件

source commit `00e1054354a5281144210a3f1db90c5cee7bc9df`。
CPU全suite: 1968 passed / 2977 skipped / 18 warnings（38.61s）。
変更範囲のRuff、宣言/JSON往復チェック通過。wheel/sdistの333 runtime fileはcheckoutとbyte一致。
repo全体のRuffは既存33件を13ファイルで検出した。全対象ファイルが開始点`3d0ad3c7`とbyte一致することを
照合し、今回の変更に混ぜて修正していない。CPU CIの定義は全repo Ruffを必須にはしていない。

L4の新経路smokeは22 passed / 180 deselected（33.52s）。
全small/全broad、空atom集合、格子端、floor、可変幅の全勾配、retained backward、
24更新のoptimizer状態、Graph中のcount/ID変更、閾値を両方向にまたぐ全勾配を検証した。
smokeでcuBLASがbackward threadのprimary contextを設定する警告1件が出たが、全検査は通過。

比較はL4、N=1024²、A=52,428、B=32、spacing=.1。
FP32 source/input/中間値、FP64幾何/profile/L2/atom reduction、TF32無効。
AdamW+fused Polar更新を含む完全step、21 samples/独立process。

- small: 初期σ=.1/.14/.18。今回のgroup AB=16と既存H経路を対照にする。
- mixed: .1/.4/1.6/6.4/12。c=.2/.5/2.0を比較し、既存HとDenseを対照にする。
- switch: .1/.4998/.5002/.501/1.6。c=.5近傍の分類移動も見る。
- 各幅群はshuffleし、幅の最大12.8。全atomを毎step更新する。
- 各候補は計測前に独立FP64全軸Y/dX/全DPのmaxabsとrel-L2両方<4e-4を必須とする。
- 主要時間/peakの後に別Graphでstageを診断。norm-only replayは主要測定に含まない。

## 証拠と再現

| 実行 | job | source SHA256 | result SHA256 |
| --- | --- | --- | --- |
| 新経路smoke | `b586e9672e0f4ff7867dd486e6b7727e` | `a20c674c7b0b40800cc5787ae3571193cd29eb9673630fa189b38fcac56d5990` | `6511c943eee2c48e679fdc4a83de72e959715cf47b53f15346b55455276b9785` |
| 全GPU試験 | `a8e7e7649f4d42d5ac444114d55744dd` | `25a68082fa99df800b9d1d82c720611066d8c7c54deae58983d84d22c275e729` | `578dc56d941053b78839ac3e9e01a682b379efe73f8db3834d7bf4eaafbd7849` |

全GPU試験は202 passed（metadata 1 / CUDA 201、162.62s）。
過去のv1/v2/v3の結果と数学を維持する回帰試験も含め、固定誤差基準は緩めていない。

archiveとmanifest全項目を照合して、source/result/receiptを
`output/bspline-sigma-split-20261018/verified-jobs/`へ保存する。

```bash
python -m pytest tests/test_quadratic_bspline_cuda.py -q
python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-mixed-sigma-split.json
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-mixed-sigma-split.json \
  --polar-update fused --phase-diagnostics --kernel-diagnostics \
  --output output/bspline-split-mixed.json
python -m benchmarks.submissions check output/bspline-split-mixed.json
```

## 初回の完全step比較

NVIDIA L4、Torch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0、Python 3.13.15。
各候補は独立process、21 samplesの中央値。各CaseでCSTの初期P/X/target hashが一致した。
全候補で全52,428 atomのsigma更新を確認した。profile/L2/全勾配の基準は変更していない。

| mixedの候補 | 完全step ms | 初期small / broad atom数 | small支持対数のCPU見積り |
| --- | ---: | --- | ---: |
| 従来H `hybrid-vjp16` | 8.788789 | — | — |
| sigma cutoff .2 | 7.609090 | 10,486 / 41,942 | 251,173 |
| sigma cutoff .5 | **5.828915** | 20,971 / 31,457 | 4,240,623 |
| sigma cutoff 2.0 | 12.806470 | 31,457 / 20,971 | 66,231,100 |
| Dense | 0.082571 | — | — |

この混合条件ではc=.5が従来より33.68%短い。cを2.0まで上げると、
H側の仕事が減ってもW側の支持対数が増えて全体は遅くなった。
sigmaで分けることと、その閾値をどこに置くかは別に計測する必要がある。
支持対数は初期状態のCPU列挙で、測定時間やGPU instruction数ではない。
今回のmixed初期幅は離散群なので、c=.5に鋭い最適点があるという証拠ではない。
σ≈.4をWへ、σ≈1.6以上をHへ渡す区分が、比較した三候補でよかったと解釈する。
任意のspacing・batch・連続幅分布へ同じ数値閾値を普遍化しない。

| 他のCase | 従来の適切な経路 ms | c=.5 split ms | Dense ms |
| --- | ---: | ---: | ---: |
| 全small（従来group AB=16） | 1.207884 | 1.615194 | 0.082643 |
| 閾値近傍（従来H） | 10.634445 | 5.196014 | 0.082858 |

全smallでは、不要なH側の固定容量と空CTA/bank/scan処理を残す現実装が約33.72%遅い。
既存Hを全smallへ適用した値は7.481004msなので、splitでもH単独よりは短いが、
純groupを上回る結果ではない。全smallを含む万能の既定routeとは判定しない。

閾値近傍Caseでは、small atom数が20,971→41,942、broadが31,457→10,486となり、
20,971 atomが経路を変更した。これはTorch decodeによる主要測定前後の診断で、
warmup/eager/capture/replay全体を含む移動数。timed replayの移動数だけを数えた値ではない。
別のGraph試験ではcapture後に幅を大きく変更し、両方向の分類変更と全勾配を直接確認した。

### 時間の内訳

別Graph内のCUDA event中央値。主要時間へ足したり、norm-only replayを主要時間から引いたりしない。

| mixed、c=.5 | ms |
| --- | ---: |
| 分類+count+sort | 0.059392 |
| 共通prepare（厳密L2込み） | 0.161792 |
| small W構築 | 0.251904 |
| small全atom VJP | 0.578560 |
| broad input/adjoint query | 1.269760 / 1.268736 |
| broad output/input注入 | 0.990208 / 0.987136 |
| broad全atom VJP | 0.091136 |

H側のquery/注入が残りの主な費用であり、分類sortの0.059msが支配しているわけではない。
全smallでも四つの空query/注入launchが各0.074752ms、bank/scan/restoreも実行される。
H担当がゼロでもこの固定launchと保存容量を消さないことが純groupとの差になる。

### メモリ

| 経路 | capture/replay peak allocated / reserved MiB |
| --- | --- |
| 従来H | 35.679 / 98 |
| 純group AB=16 | 50.231 / 114 |
| split（全Case・全cutoff） | **74.181 / 140** |
| Dense | 49.002 / 106 |

H/HP/G/GPの各容量はB*A=1,677,696要素（FP32約6.4MiB）で、broad数に応じて確保を縮めていない。
さらにWとGEMMの作業領域を併用する。具体的なallocationごとのpeak寄与・物理counterや
総GPU process使用量は計測していない。容量モデルと実測allocator peakを混同しない。
混在の速度改善とメモリ増加は別々の結果であり、Denseを時間・メモリ両方で満たした候補ではない。

### 全軸FP64照合

各候補3 live statesのworst。Y/dX/全DPのmaxabsとrel-L2の双方<4e-4を通過。

| split候補 | Y maxabs / rel-L2 | dX maxabs / rel-L2 | 全DP maxabs / rel-L2 |
| --- | --- | --- | --- |
| mixed c=.2 | 5.659e-6 / 2.209e-7 | 5.344e-6 / 2.162e-7 | 1.602e-5 / 1.008e-7 |
| mixed c=.5 | 1.790e-6 / 1.233e-7 | 1.646e-6 / 1.204e-7 | 1.271e-5 / 1.032e-7 |
| mixed c=2.0 | 2.543e-6 / 2.095e-7 | 2.785e-6 / 2.021e-7 | 1.518e-5 / 1.032e-7 |
| small c=.5 | 2.398e-6 / 1.700e-7 | 2.344e-6 / 1.683e-7 | 1.624e-5 / 1.061e-7 |
| switch c=.5 | 1.351e-5 / 5.145e-7 | 1.961e-5 / 5.080e-7 | 1.431e-5 / 1.022e-7 |

初回比較job `438ec2f0a89647d19befaa0810f734a4`：
source archive `d80dd8dc75d32f29cc8e7e8868918f640f2a826625a02fe0babac6202db5a3aa`、
result archive `161e4eb555743ad2468e95575e362c5bb909bc421ddd04f9023f4851a6dc777f`。
51 manifest項目を照合し、3完成result JSONのsubmission整合検査を確認した。

## 独立再測定と判断

別のL4 VMでc=.5と対照を再測定した。source、初期P/X/target、optimizer、
主要測定・全軸FP64 gate・peakの境界を維持した。runtime/GPU test/runner/Registryは
smoke・全試験・初回比較・再測定の全snapshotで同じファイルと照合済み。

| 再測定Case | 従来の適切な経路 ms | split .5 ms | Dense ms |
| --- | ---: | ---: | ---: |
| mixed | 8.791423 | **5.835184** | 0.082496 |
| small | 1.210140 | 1.618549 | 0.086685 |

mixedの短縮は33.63%、smallの増加は33.75%で、初回の方向と大きさが再現した。
peakは再度split 74.181/140MiB、H 35.679/98、group 50.231/114、Dense49.002/106。
全52,428 sigmaが更新された。mixed/smallの経路変更数は0だが、分類・sortは各stepで再実行する。
幅の群が閾値をまたがないことと、幅を固定していることは別である。

再測定splitの全軸FP64 worst maxabs Y/dX/DP：
mixed 1.834e-6/1.900e-6/1.579e-5、small 2.398e-6/2.658e-6/1.449e-5。
全rel-L2も固定基準を通過した。202回帰試験と合わせ、数式・微分の意味を変えていない。

job `c026faefe3eb4d3ea92b312c47b768e1`：
source archive `dd2a0a225e5ca488530b253aa800df787a8f9dffe68de33e2e97bebed8ce80ef`、
result archive `891abbaac59e9e353c6742abee8b22595105e9262206ca2f1376c7ad8ad29915`。
34 manifest項目と2完成result JSONの整合検査を確認した。
両完全step jobとも3 live statesの独立FP64全勾配照合を省略していない。

結論は「sigmaによるatom単位の分割はこの混合条件で有効、現在の併用実装は固定容量と空処理が残る」。
small-onlyは既存の純groupを維持し、mixedではsplitを研究候補として保持する。
自動splitを公開既定へ採用したり、メモリ条件を満たしたと扱ったりしない。
次に改善する対象は全atom分のH/HP/G/GP容量と、担当がゼロのH側launch/bank/scanである。
支持を切り捨てる代替や、近似L2による速度改善ではない。

全result/source/receipt/manifestを保持し、pool全slot停止、queued/running jobなしを確認した。
未実施：8192²完全step、周期・複数座標CUDA、総GPU process peak、実DB登録、公開dispatcher採用。
