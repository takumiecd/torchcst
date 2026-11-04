# #130基準のint32一時sort key比較

## 事前に固定する変更と判定基準

基準はPR #130の `7771bdde1807b8ca1bb87c7e0a9221a93ad5b3c5`。
候補は `GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32` の明示選択だけで有効にする。
公開dispatcherは変更しない。浮動小数点kernel、geometry payload、保存keyのint64、
rows/offsetsのint64、数学、optimizer則は変えない。

各軸4094 ownerのため、完全複合keyの上限は
`4094²−1 = 16,760,835 < 2²⁴`。出力ownerだけではなく入力ownerも含む
複合key全体をstable sortする。同一key内は元Parameter行順を維持する。
forwardはint64値をdevice上で検査してint32一時keyへ変換し、backwardは向きの変換と
検査・castを融合する。積と境界はint64で計算する。構造的上限がint32に収まらなければ
従来のint64 ATenへ戻る（現在のMAX_OWNER_NODESではこのfallbackに到達しない）。
負値・範囲外値を切り詰めて受理しない。D2Hの値検査は追加しない。

先に両方向のrows/offsetsをint64基準・CPU stable sortと完全一致させる。
A=0/1/65536、同一key、多数tie、入力owner逆順、最大key、動くkeyを含む。
invalid keyのdevice assertは子processに隔離する。saved hooksのoffload/復元、
metadata、retained backward、Parameter更新後の分類、Graph replayの現在値も検査する。
Y/dX/全dP/P/m/v/clock/全Parameter勾配はbyte一致必須。
独立FP64 oracleの既存 `3e-5 + 3e-4 abs(ref)` は維持する。

性能条件はL4、N1024/A52428/真batch4、seed41、TF32 off、CPU math threads=1、
native CSTAdamW、3重み付き層（中央のみCST）の完全Graph学習step。
初期状態は同一初期値から**基準#130のFP32で8回実更新**した状態。
学習後状態はFP32自身の45,000更新checkpoint（#129/#130最終状態一致確認済み）を固定する。
各状態256更新、順序を反転させたfresh processの6比較対、同等のdense Graph対照を使う。

両状態でpaired幾何平均の短縮率5%以上、6/6勝、全状態byte一致、
captureを含むpeak allocated増分2MiB以下・reserved増分8MiB以下、unexpected fallback=0を要求する。
別processの3epoch短期smokeでeval/save/train/certificate再準備/recaptureと状態復元、
epoch末メモリが増え続けないことも確認する。profileは別の16更新processで取得し、
計装時間を採否に使わない。上記予算はこの研究の判断基準であり、一般のユーザ上限ではない。

全短期基準を満たした場合だけ、順序を反転した2対の各45,000実更新へ進む。
全3epochで42数値状態+5勾配をbyte比較し、capture/復元/再準備/旧Graph破棄とGC/
最終破棄込みの時間で5%以上短縮を要求する。同じメモリ増分基準とepoch末の安定性を確認する。
denseも同等のGraph条件で測る。短期基準を満たさなければ負結果を保存し、長期比較は実施しない。

## keyと比較式

入力ownerを `i`、出力ownerを `o`、owner数を `Ni/No` とすると、
`k=o Ni+i`、逆方向は `k'=(k mod Ni) No+floor(k/Ni)`。
offset境界は方向に応じた `owner × stride` のlower boundである。
cast前の値と積、逆方向の計算、境界はint64を維持する。

短期対 `j` の基準時間を `b_j`、候補時間を `c_j` とすると、採否に使う短縮率は
`1−exp(mean_j(log(c_j/b_j)))`。表の平均ms/stepから計算した比率とは区別する。
長期は `1−候補のadoption_seconds/基準のadoption_seconds` で順序対ごとに判断する。

## 再実行の入口

入力は#130と同じhash固定の初期値、MNIST、FP32自身とdenseの学習後checkpoint。
raw driverとpayloadはignored `output/sort32/driver.py` と共有poolのsource archiveに保全する。
driverは整数/Graph契約、48更新lifecycle、36短期cell、4 profile cell、全短期gateの判定、
条件付き長期比較の順に実行する。各cellはfresh processである。

```bash
PYTHONPATH=src:. python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-radius-cell-mass.json \
  --case benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-window-fp32-geometry-sort32-initial.json
PYTHONPATH=src:. python -m tools.kernel_dev test --suite cpu

# INPUTSはmanifestで固定した7入力を含むdirectory。REPEATは0..5。
PYTHONPATH=src:. python -m benchmarks.training.full_step_graph_comparison \
  --inputs "$INPUTS" --output "$OUTPUT" --kind cst --mode graph \
  --state initial --precision fp32_geometry_sort32 --repeat "$REPEAT" \
  --steps 256 --bootstrap-fp32 --geometry-baseline
# 基準は--precision fp32_geometry。学習後状態は--state final。
# 全短期gateを満たした場合だけ実行。基準は--sort32を省略。
PYTHONPATH=src:. python -m benchmarks.training.full_step_graph_learning \
  --inputs "$INPUTS" --output "$LONG_OUTPUT" --kind cst --mode graph \
  --geometry-shared --sort32
```

## 実測結果

初回job `l4job-63a1bb3e7ed9405ba528615695ee5801` は333 GPU検証を通過した。
初期/学習後の短期完全stepは各9.3398%/9.3787%短縮、各6/6勝、状態・全勾配・集計は
byte一致、両peakの増分0。ところがdriverのlifecycle判定が「全3epoch末のreservedが
完全に同じ」という、事前の「継続増加なし」より強い条件になっており、長期へ進まず終了した。
基準reservedは444/444/444MiB、候補は442/444/444MiB、allocatedは両者とも一定。
両者の最終peakは同じ444MiBで、数値も全epoch一致している。

この初回decisionは保全し、合格へ上書きしない。追加の独立jobで同じCUDA実装を使い、
各16更新×8epochのeval/save/train/reprepare/recaptureを対照・候補の順に実行する。
初期2epochを含む全体peakの既存2/8MiB増分予算は維持し、最後6epochのallocatedと
reservedがすべて同じ、全8epochの42状態+5勾配・集計がbyte一致、fallback=0の場合だけ
長期へ進む。速度・数値・peak予算は一切緩めない。前回同様3epochが長期の既定であり、
`--epochs 8 --steps-per-epoch 16` はlifecycleの継続増加を識別する追加診断にだけ使う。
追加jobの結果と採否は以下に追記する。
