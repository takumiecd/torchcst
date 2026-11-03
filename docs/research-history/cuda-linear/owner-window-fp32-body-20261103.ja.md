# Owner-window FP32 body: 数値成立性の研究

## 測定前の固定条件

#128 (`ac4c8980cb65c7a64aefe37cf20cccabd8d65c3d`) に積んだ別候補。
main/既定dispatcherは変更しない。まだ採用・高速化を主張しない。
まず内部native入口だけで成立性を検査し、合格した場合だけPlan/Caseと
全モデル測定へ進む。内部入口は公開hooks/live設定の代替APIではない。

N1024/M8/spacing1、Polar、radius bounds [.5,50] のみ。
Parameter/公開入出力はFP32。decode・wrap・分類、local displacementの減算、
最後のPolar pullbackは既存FP64を保つ。その他の係数、phase、cell積分、
raw/window、H/Gと中心微分、C縮約、da/dmu集約をFP32にする。
atom/chunk/level順、kernel分解、正規化、学習する中心・半径、支持、全dP、
dX、radius task-gradient detach、optimizer則は維持する。
fixedDFT/C配置変更/tight mask/cooperative raw/fast-math/TF32は追加しない。

## 事前ゲート

- 独立CPU FP64 Fourierおよび小集合の独立quadratureに対し、Y/dX/全dPの
  **各要素**で `abs(error) <= 3e-5 + 3e-4*abs(reference)`、全値finite。
  平均/RMSが良くても最大要素の不合格を隠さない。
- 参照勾配をFP32へ一度丸めたものと候補勾配を、同一Adam初期状態に入力。
  `u=mhat/(sqrt(vhat)+eps)` のmax差 <= .01、各Parameter列RMS <= .001。
  post-Polar更新差はs/tで `1e-6 + 2 ULP(FP32)`、中心は周期距離で比較。
  同一勾配を入れるoptimizer対照はbyte一致。
- constant/one-hot/alternating/random、owner集中、tiny amplitude、極端なs/t、
  zero/clamp、周期・分類境界の隣接FP32値、wide radius、小勾配(.1eps..10eps)
  を含める。通常checkpointの全52,428 atomも検査する。
- eager/Graphの同じ精度同士はbyte一致。異精度には上記数値基準を使う。
- 不合格は、係数c/q/dq、active window、H/G・中心微分・pullback前で診断。
  未初期化のinactive raw/window領域は比較しない。

数値ゲートに不合格なら、実装不具合か丸めかを切り分けて保全し、
この候補の速度測定・長期学習は行わない。結果を見て許容差を緩めない。

合格した場合のみ同じL4上で初期/学習後checkpoint、256 minibatches×6反復の
fresh processでFP64/FP32のeager/Graphと同等条件のdenseを比較する。
未計装完全step、guard/入力供給込み、capture込みpeak allocated/reservedを記録。
継続目安は両状態10%以上短縮・paired 95%下限5%、eagerの明瞭な回帰なし。
これは研究の継続目安であり、ユーザー指定の性能・メモリ上限ではない。
長期学習は局所数値・性能ゲートを通った場合だけ行う。

#128 harnessのrestoreは符号付きzeroも含むbyte比較に修正し、capture後の
実際のlive開始状態を計時直前に再取得して元の開始状態と照合する。
過去#128の原本・測定結果は変更しない。
