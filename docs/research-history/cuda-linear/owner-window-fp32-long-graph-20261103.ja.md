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

## 短いlifecycle検証で見つかった再準備の不足

初回source `215a31a9` / job `l4job-a409292a8c4d4bc99fc58b3a578d1ecf` は
212件の検証とeagerの3×16更新を完了したが、Graphはepoch1の評価後、
epoch2の再準備で `initial whole-step guard refused the cohort` となり停止した。
45,000更新には進んでいない。失敗job/source/log/checkpointは保全する。

評価は共有のfixed metadata証明を更新し得る。training forward/backwardだけでは、
Linearの証明は作り直されても、Polar更新の証明は作り直されない。
再準備は無更新のforward/backwardに加え、通常Dispatcherの`prepare`でlive bindingを検証し、
`prepare_optimizer_update`でoptional Polar programを再構築する。guardの書換え・無効化はしない。
P/Adamは変えず、45,000回の実更新数と初回8更新の数え方も変えない。
再準備費用は採用用時間へ、追加workspace/captureは同じ連続memory scopeへ含める。
評価→train→両証明の再準備→captureでP/Adamを変更しないことを直接GPUテストする。

## 記録する量

学習時間を \(T_{\rm train}\)、optimizer生成・capture・復元・必要再準備の和を
\(T_{\rm prep}\) として、採用用時間を
\(T_{\rm adoption}=T_{\rm train}+T_{\rm prep}\) とする。
各実行順序での短縮率は
\(1-T_{\rm adoption,Graph}/T_{\rm adoption,eager}\)。
評価・保存・入力転送と共有初回compileは別の費用として残す。

最終checkpointの実際のmoment \(m_t,v_t\) とclock \(t\) を固定し、
候補勾配 \(g\) と一度FP32へ丸めた独立参照勾配 \(g_{\rm ref}\) を使って

\[
m^+=\beta_1m_t+(1-\beta_1)g,\quad
v^+=\beta_2v_t+(1-\beta_2)g^2,\qquad
u^+(g)=\frac{m^+/(1-\beta_1^{t+1})}
{\sqrt{v^+/(1-\beta_2^{t+1})}+\epsilon}
\]

の差を診断する。これはcold momentの最大方向差gateの再判定ではない。
別に同一checkpointから実native CSTAdamWを1回実行し、P・moment・clockの差を残す。
中心の差はperiod1024で最短距離を取り、差分はFP64へ変換してから計算する。
参照勾配を2回与える同勾配対照は更新の再現性の検査であり、
候補勾配と参照勾配で更新が一致したという意味にはしない。

## 結果転送上限による中断

source `a49a907c` / job `l4job-013d1eb581294028b9a7926d623d7039` は
`POOL_RESULT_READY` まで到達したが、checkpointを含む結果archiveのdownloadが
pool既定の180秒で中断された。poolは所有VMを停止し、serverにassignmentがないことも確認した。
終了したVMの結果は回収できず、数値・性能の結果として使わない。
source/spec/transport/lifecycleを保全し、通常の`recover`で中断状態を解消する。

再試行は明示的に新jobとして提出する。計算・測定・入力は同じに保ち、共有poolの
supervisorを使ったまま、downloadだけlocal timeoutを900秒に延ばすwrapperを使用する。
共有queue、account確認、排他、所有権、manifest検証、停止処理、自動retryなしは変更しない。
mainやインストール済みskillを変更せず、wrapperもignored evidenceに保全する。

## 完了結果: FP32自身の45,000更新

**両実行順で準備込み61.64–62.08%短縮し、各epochの数値状態・勾配・指標がbyte一致した。**
評価後の再準備と再captureも含め、今回の速度・品質・メモリ継続目安を通過した。
この限定したGraph prototypeとの併用を継続してレビューする根拠になる。
通常APIの汎用Graph対応や既定dispatcherへの採用は、この結果とは別の判断である。

成功jobは `l4job-b077b97a10a340feb1e85e0f83f2e4af`、sourceは
`cd37de6af0697e6924f9b628cb62307c3a043527`。
`a49a907c` からの変更は研究文書だけで、CUDA・benchmark・testの実行コードは同じ。
NVIDIA L4 UUID `bae0762b-d9e6-a033-d049-15d081a22de5`、driver580.82.07、
Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、Python3.13.16。
CPUはIntel Xeon2.20GHz/12 logical cores、math threads1。
全セルをこの一つのallocationで、fresh processで順次実行した。

### 時間・品質

| 実行順 | FP32 eager 学習ms/step | FP32 Graph 学習ms/step | eager 準備込みms/step | Graph 準備込みms/step | 準備込み短縮率 |
|---|---:|---:|---:|---:|---:|
| eager→Graph | 2.452393 | .898471 | 2.472740 | .937774 | 62.08% |
| Graph→eager | 2.423019 | .898020 | 2.443449 | .937333 | 61.64% |

各45,000実更新。学習総時間はeager110.357666/109.035854秒、Graph40.431216/40.410920秒。
optimizer生成・capture/復元・再準備を加えた総時間は
eager111.273311/109.955209秒、Graph42.199842/42.179963秒。
Graphはupdate8でcaptureし、評価後のupdate15,000/30,000で再準備・再captureした。
各回のcapture/再準備は約.28–.30秒。総準備費用はeager約.916/.919秒、Graph約1.769秒。
最初の8実更新を含む全45,000回を数え、Graph replayは44,992回、想定外fallbackは0。

FP32 eager/Graphの4セルすべてで、最終test精度 **97.37%**、loss **.083672396839**。
epoch1/2の精度も94.88%/96.43%、loss .167267531157/.109340876341で同じ。
既存FP64 eager基準の97.37%/.082855619490に対し、精度差0pp、loss増分.000816777349。
FP64基準は前研究の品質参照であり、今回の同時測定によるFP64/FP32速度比較ではない。

同じ全step Graphで測ったdense45,000更新は、学習 **.198231ms/step**、
準備込み **.223139ms/step**、最終97.87%/.070017516613。
両者Graph同士のCST/dense時間比は、学習4.530–4.532倍、準備込み4.201–4.203倍。
今回のCSTの改善後もdenseの方が速い。CPUの特定関数やFP64 ALUを原因と断定するprofileは実施していない。

### epoch状態と評価・保存・train復帰

- 3 epochとも、4セル間でmodel/Adam/clockの **42 Tensor**、全Parameter勾配の **5 Tensor**、
  loss/正解数、test値、初期値/data orderのhashが一致。保存状態と勾配はすべてfinite。
- 回収後、各epochの実checkpointをCPUへloadし、基準eager対残り3セルの計9比較を実行。
  各 **47 Tensor** がbyte一致した。全stepの全中間Tensorを比較したという意味ではない。
- 前研究の別L4で得たFP32 eager最終checkpointとも、modelの27 Tensorとoptimizerの15 Tensorがbyte一致。
  これは最終数値状態の一致であり、別GPUで長期Graph速度を再現したという主張ではない。
- Graphは各epoch評価後にguard資格を失った。両実行証明を通常の準備入口で再構築し、
  capture/復元時のP/Adamと既に数えた指標の一致を確認してtrainへ復帰した。
- 各epochのcheckpoint保存を通した。最終診断では別processへloadした実状態から次のnative更新も確認した。
  任意のcheckpointからprocessを再起動し、全モデル学習を長期間再開する一般APIの検証ではない。

### captureと再準備を含むメモリ

| 条件 | peak allocated MiB | peak reserved MiB | epoch1/2/3末 allocated MiB | epoch1/2/3末 reserved MiB |
|---|---:|---:|---|---|
| FP32 eager（両セル共通） | 247.037109 | 418 | 244.588379 / 244.588379 / 244.588379 | 416 / 418 / 418 |
| FP32 Graph（両セル共通） | 251.992188 | 442 | 245.407227 / 245.407227 / 245.407227 | 442 / 442 / 442 |
| dense Graph | 264.597656 | 432 | 256.595703 / 256.595215 / 256.595215 | 432 / 432 / 432 |

FP32 eager→Graphのpeak増分はallocated **4.955078MiB**、reserved **24MiB**。
事前目安の8/32MiB以内。Graphは再captureを挟んでもepoch末の両使用量が一定で、
今回3 epochの範囲で継続的な増加を観測しなかった。プロセス全体のGPU使用量は未測定。
このscopeは初期評価・cold optimizer・全学習・capture/復元/再準備・epoch評価を含む。
前研究の短区間比較やoptimizer/lifecycleの異なるmemory値と混ぜない。

### FP32自身の最終Pと実X/dY、学習済みmomentの診断

事前に選んだtrain最初のminibatchとtest先頭4件を使い、学習済み前後層からX/dYを生成した。
全52,428 atomの独立CPU FP64 Fourier VJPを、従来の各要素
`3e-5 + 3e-4*abs(reference)` で照合した。

| 実minibatch | Y 最大絶対差 | dX 最大絶対差 | 全dP 最大絶対差 | Y 相対L2 | dP 相対L2 |
|---|---:|---:|---:|---:|---:|
| train | 6.6134e-7 | 1.3419e-10 | 8.0213e-10 | 2.4489e-7 | 2.4469e-7 |
| test | 8.4076e-7 | 6.7940e-11 | 8.1639e-10 | 2.5706e-7 | 2.5827e-7 |

両minibatchのY/dX/全209,712 dP要素で超過0。
半径で層化した8 atomの独立48点quadratureにもGPU候補のY/dX/dPが通過した。

保存された実m/v/step45,000を共通にした方向差は、両minibatchとも最大 **1.1921e-7**。
列別RMS最大はtrain6.8667e-9、test4.5415e-9。
同じ学習済み状態から実native CSTAdamWを1回実行したPの最大差は、
両minibatchとも列別 **[3.7253e-9, 0, 0, 0]**。中心2列のperiodic差は0。
moment最大差はtrain exp_avg1.8626e-9 / exp_avg_sq1.1369e-13、
test exp_avg4.6566e-10 / exp_avg_sq0。両clockは45,001で同じ。
参照勾配を2回与える対照では、Pとmoment/clockがbyte一致した。
既知のcold最大方向差gateの負結果はそのまま保持し、今回の実moment診断と区別する。

### 検証・保全・再現

- L4環境で **213件通過**。新FP32公開入口のcapture拒否、指標込みcapture復元、
  評価後の無更新再準備を含む。短い3×16更新のeager/Graph比較と実moment診断も通過。
- ローカルCPU suiteは1,710 passed / 2,151 skipped（追加lifecycle GPUテスト前）。
  [測定sourceのCPU CI](https://github.com/takumiecd/torchcst/actions/runs/41668291751)も
  1,709 passed / 2,153 skippedで成功し、宣言検査とwheel/sdist buildも通過。
  CUDA算術とoptimizer則は本追加研究では変更していない。
- source archive SHA256: `de4d46353f31f2b6d088f2fe2c3e96009ac7088dc9e2a15bc172551d220deff2`
- result archive SHA256: `a779556ffda8bdf0c5bcb0a85662c5fd7c938b32349a70e28f74cd5bf0056f2c`
- driver SHA256: `d64cf819c3357289cd6fa1baf66f79328c5bfba4e70bf61b58d6d01f896729b4`
- 回収manifest **61 files** とarchiveを再hash照合し、
  `output/owner-window-fp32/long-graph-results/` にログ・全epoch checkpoint・独立診断を保全。
  ローカル直接Tensor比較は`local-checkpoint-verification.json`へ保存。
- wrapperとそのtimeout検査、全driver、失敗試行metadataは`output/owner-window-fp32/`で保全。
  pool supervisor正常終了・所有全slot停止を確認し、`final-pool-status.json`へ保存。
- [公開用の集計・状態hash・診断JSON](owner-window-fp32-long-graph-results-20261103.json) は
  raw tensor/logではなく、回収済み結果からの集計。

凍結初期値 `initialization.pt` とMNIST gzip4ファイルを入力directoryへhash照合して用意する。
初期値SHA256は `b49e95e722dab4e42d935a9ac2b239212e93660af55bf6d6a9f12f437a9f2ed6`。
前研究のbaseline/dense最終checkpointは今回の学習入力には不要。
同じGPU allocationで下記をfresh processでeager→Graph、Graph→eagerの順に実行し、
dense Graphも1回実行する。出力directoryは毎回新規にする。

```bash
PYTHONPATH=src python -m benchmarks.training.full_step_graph_learning \
  --inputs FROZEN_INPUTS --output output/fp32-eager --kind cst --mode eager
PYTHONPATH=src python -m benchmarks.training.full_step_graph_learning \
  --inputs FROZEN_INPUTS --output output/fp32-graph --kind cst --mode graph
PYTHONPATH=src python -m benchmarks.training.full_step_graph_learning_diagnostics \
  --inputs FROZEN_INPUTS --checkpoint output/fp32-graph/epoch-3.pt \
  --output output/fp32-final-diagnostics
```

一つのmodel・seed・成功した一つのL4 allocationでの結果である。
追加seed/別model/別GPUでの長期Graph学習、汎用Graph入口、process全体メモリは未検証。
[#129](https://github.com/takumiecd/torchcst/pull/129) はDraftを維持し、merge/既定採用はしていない。
