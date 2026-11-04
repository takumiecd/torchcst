# FP32 owner-windowの半径・振幅共有

## 測定前に固定する条件

#129のFP32主体＋専有全step Graphを対照とし、分類でFP64の半径・振幅を保存して
forward/backwardで再利用する候補だけを追加する。Tensor Core化、係数保存、加算順変更は併用しない。
ユーザーはProレビュー後、この候補の実装と短期比較、通過時のみ長期比較を指示した。
#129は基準として固定し、本研究を別のkernel branch/PRで扱う。

- 元のParameter行順でFP64 `[2,A]` を各forwardで生成し、対応するbackwardまで保存する。
  A52428では838,848 bytes（約.80MiB）。Graphでも分類kernelが毎replay全要素を上書きする。
- 分類と両contractのradius decodeをsource上4A→Aにする。radius/amplitude関数と使用時のcastは維持。
  分類内のコンパイラ共通化は未確認で、source呼出数から速度倍率を推定しない。
- 最新P・live設定を用い、input/output同一軸の既存制約を維持する。
  membership、wrap、局所変位、profile/q/dq、H/DH再計算、走査・加算順、Polar pullback、optimizer則は維持。
- 保存物はinvocationごとに所有し、saved-variable/version保護を維持。
  新payloadのdevice/dtype/shape/contiguousをbackwardのsortより前に検証する。
- 初期状態はFP32で8実更新した状態。学習後状態は#129のFP32自身の45,000更新後P/m/v/clock。
  共通FP64 warmupや古いFP64学習後checkpointへ置き換えない。
- L4を共有poolから1台、N1024/A52428/真B4、seed41、CPU math1、TF32 off、前後Linearを含む完全step。
- 各状態で同じ256 minibatch、6組のbalanced-order・fresh processで対照/候補を比較。
  dense Graphも同条件の参照として添える。開始・終了状態、全Parameter勾配、指標、data/order hashを比較。
- 短い16更新の別profileでCPU/CUDA timeline、kernel別時間、Python関数費用を取得する。
  profile時間は採用性能へ使用しない。profileの観測だけからdecodeを律速と断定しない。
- 短期timeは未計装・区間境界同期。gather/loss/backward/実optimizer/metrics/guard/index/version通知を含む。
  準備とGraph退役・GCを別記し、45,000回へ償却した推計は長期実測と区別する。
  長期runnerでも評価境界の退役・GCと最終退役を明示的に計時する。
- peak allocated/reservedはoptimizer生成前からcapture/復元/replayまで連続計測し、capture後resetしない。

## 継続・中止基準

1. Y/dX/全dP、P/m/v/clock、全Parameter勾配、指標が現行FP32とbyte一致。独立参照の従来誤差基準も維持。
2. 小振幅、constant/alternating、周期/owner境界、wide、集中、空atoms/batch、部分勾配、
   multiple forward、retained backward、saved-tensor offload/不正payload、P変更を検査する。
3. 初期・学習後の両状態でpaired時間比の幾何平均による完全step短縮が5%以上、各6/6組で短縮。
4. 現行FP32 Graph比のcapture込みpeak増分がallocated≤2MiB、reserved≤8MiB。
   これは今回の工学的研究予算であり、ユーザー一般の上限ではない。
5. 想定外fallback0。hooks/settings/storage/callback/finite-checkのguardを維持。
   短い3 epochの評価・保存・再準備・再captureで状態不変、継続的なメモリ増加なし。

byte不一致なら保存R/amplitude・keysから原因を追い、途中で許容誤差へ基準を緩めない。
全基準を通った場合だけ、45,000更新の順序反転比較へ進む。
長期も各epoch47 Tensor・指標一致、退役GCを含む準備込み5%以上短縮、メモリ予算内を要求する。
未達なら結果を保全して候補探索を終了する。既定採用・mergeは別判断。
