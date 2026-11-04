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

## 実測結果

未実測。結果取得後にsource/result hash、環境、数値・時間・メモリ、採否を追記する。
