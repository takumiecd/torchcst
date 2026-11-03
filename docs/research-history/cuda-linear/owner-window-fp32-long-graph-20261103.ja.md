# FP32自身の45,000更新: eager / 全step Graph

## 目的と測定前プロトコル

ユーザーはProレビュー後、FP32 eager / Graphで45,000更新の比較を指示した。
#129の候補CUDA算術を固定し、長期学習の実行互換性・品質・時間・メモリを確認する。
既定dispatcherの変更、main merge、新たな高速化候補や数学・座標・spacing変更は行わない。
Proレビューの提案は外部の助言として扱い、以下を今回の研究条件として結果取得前に固定した。

- seed41、同じ凍結初期値・初期effective W・MNIST全train/test・データ順序。
- N1024/A52428/真B4、Linear(784,1024)→ReLU→CST→ReLU→Linear(1024,10)。
  重み付き層3つ、CSTは中央1つ。FP32主体版（decode/classify/local subtraction/Polar pullbackはFP64）。
- TF32 off、CPU math threads1、native CSTAdamWの凍結係数、dX・全dP・全層更新。
- 同じL4 allocationでfresh processの eager→Graph と Graph→eager の2組。
  dense Graphも45,000更新の参照枠として同じallocationで実行する。
- 最初の8 minibatchもFP32の実更新として45,000回に数える。FP64による学習warmupは使わない。
  Graph capture中の仮更新だけ、P/Adamと既に数えたloss・正解数をbyte一致で復元する。
- 各epochの評価・checkpoint保存・trainへの復帰を実行。guardは迂回しない。
  評価後に資格が失われた場合のみ公開入口で無更新の再準備と再captureを行い、費用を計上する。
  想定外のstep内fallbackは失敗として止める。
- 未計装の学習区間はepoch境界（初epochは実8更新と残り）で同期する。
  gather/loss/全backward/optimizer/metrics/guard/index供給/version通知を含む。
- 採用用時間 = optimizer生成 + 学習全45,000実更新 + capture/復元/必要再準備。
  通常評価・checkpoint保存・data/model配置は別記し、学習時間には混ぜない。
  共有の初回compile・smokeはpool phase logsに別記する。
- peak allocated/reservedはoptimizer生成前から測り、capture後にresetしない。
  初期評価・全学習・capture/復元・epoch評価を含み、epoch末の使用量も記録する。
  最終VJP診断は別processで実行し、このmemory scopeには混ぜない。

## 合否と解釈

- 同精度eager/Graphの各epoch末の全model/Adam/step、全Parameter勾配、loss/正解数集計、
  評価結果、data order/initial hashがbyte/値一致すること。想定外fallback0、全値finite。
- FP32自身の最終checkpointで、実際の学習済み前後層から得る中間X/dYを使用する。
  最初のtrain minibatchとtest先頭4件を測定前に選ぶ。
  全52,428 atomの独立CPU FP64 Fourier VJPと、半径で層化した8 atomの独立48点quadratureを比較。
  Y/dX/全dP各要素で `3e-5 + 3e-4*abs(reference)` を維持する。
- 保存された実m/v/stepから候補勾配/参照勾配（FP32へ一度丸め）による次の実CSTAdamW更新を比較。
  同勾配対照も実行し、Parameterとmoment/stepがbyte一致することを確認する。
  既知のcold最大方向差gateを合格へ変更せず、実momentの感度は診断として別記する。
- 品質の継続目安は既存FP64 baselineに対し精度悪化.2 percentage points以内、loss悪化.005以内。
- 今回の工学上の継続目安として、両実行順序で準備込み30%以上短縮、
  eager比allocated増分8 MiB以内・reserved増分32 MiB以内、epochをまたぐ継続的な増加なしを置く。
  ユーザー一般のメモリ上限ではなく、この専有Graph試作の研究予算である。
- 不一致・非有限・独立参照超過・想定外fallbackは採用を止めて診断する。
  速度・メモリ目安を超えた場合も結果を残し、結果を見て基準を緩めない。

## 実装と入口

`benchmarks.training.full_step_graph_learning` は既存PreparedStepと公開学習入口を使う。
最初の8実更新を重複/欠落なく数え、captureは `preserve_totals=True` で累積指標を復元する。
通常の短区間比較は従来どおりcapture後に指標0へ戻す。CUDA kernelやoptimizer算術は変更しない。

`benchmarks.training.full_step_graph_learning_diagnostics` は、学習後のP/m/v/stepを公開load経路で読み、
実中間X/dYによるVJPと、同じ実状態からの次の更新を計時外に検査する。
新FP32公開入口そのもののcapture拒否も直接テストする。
