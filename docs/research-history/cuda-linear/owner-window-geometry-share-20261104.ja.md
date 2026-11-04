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

## 初回GPU検証の停止と修正

source `b26c1b86` / job `l4job-739c9a71eb684716959be5ea9bf746a9` は
11件通過後、Parameterのinplace変更を拒否する例外の文言がテストの正規表現と異なり停止した。
実際にはReadLeaseによるview/inplace保護が意図どおりRuntimeErrorを返していた。
検査は拒否を要求したまま`inplace`へ文言一致を修正し、CUDA算術は変更しない。
通常/constant/alternating/小振幅/境界wide/owner集中のY/dX/全dP byte比較と独立参照、
保存geometryのdtype/shape/stride/device拒否は通過済み。性能測定には到達していない。
source/spec/log/result/manifestは`output/geometry-share/attempt-1/`とpool jobに保全する。

次のsource `88eb0957` / job `l4job-a7bc0730cc8b44119701b903e2e4dee3` は
12件通過後、空atom fixtureが整数`atoms=0`を渡し、公開APIの正しい拒否で停止した。
空Parameterを渡す既存`_reference_model`経路にfixtureを修正し、全4境界fixtureの構築と
metadata supportをCPUで確認した。CUDA算術は変更していない。
原本は`output/geometry-share/attempt-2/`とpool jobに保全する。

## 完了結果

**短期で約17.2–17.3%、45,000更新では準備・Graph退役・GC込みで約16.0–16.1%短縮した。**
事前の数値・速度・メモリ・lifecycle基準はすべて通過した。
sourceは `ed945156d21632ba5400d43467f9e6633fe37e9d`、
成功jobは `l4job-cd17f5dba2f64fb2a73e703fcf93176c`。
`b26c1b86`以降のCUDA算術変更はなく、途中の修正はテストfixtureと研究文書だけ。
以後の結果記録も実行コードを変更しない。

L4 UUID `832dc16f-875a-d06b-7759-a4a77fd9112e`、driver580.82.07、
Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、Python3.13.16。
Intel Xeon2.20GHz/12 logical cores、CPU math1、TF32 off。
すべての成功セルを同じ一つのallocationで、fresh processにより順次実行した。

### 短期・完全Graph step

| 開始状態 | 現行FP32 Graph mean ms/step | 共有候補 mean ms/step | paired比の幾何平均による短縮 | 勝敗 |
|---|---:|---:|---:|---:|
| FP32で8実更新後 | .863182 | .713694 | 17.3183% | 6/6 |
| FP32自身の45,000更新後 | .876085 | .725085 | 17.2358% | 6/6 |

各256 minibatch、6組balanced order。各状態のCST12セルについて、開始/終了状態・
全Parameter勾配・指標・入力/data order hashが一致した。dense参照の各6セルも数値状態が一致。
同条件dense Graphは初期.203794/学習後.205841ms/step。
短期の準備＋最終Graph退役GCは約3.09–3.22秒で別記し、45,000更新への償却値は推計として保存した。
下記の長期結果はその推計ではなく、条件通過後に実行した別の45,000更新の実測である。

短期のcapture込みpeak allocatedは両方式とも251.534180MiB。
reservedは対照440MiB、共有442MiBで、allocated増分0、reserved増分2MiB。

### 45,000更新・評価/保存/train復帰

短期全基準通過後、対照→共有、共有→対照の2組を追加実行した。
初回8更新も各方式の実更新として45,000回に数え、Graphは44,992 replay。
update8でcapture、評価後の15,000/30,000で再準備・再captureし、想定外fallback0。

| 順序 | 対照 学習ms/step | 共有 学習ms/step | 対照 準備・退役込みms/step | 共有 準備・退役込みms/step | 同短縮率 |
|---|---:|---:|---:|---:|---:|
| 対照→共有 | .898391 | .746754 | .945179 | .793260 | 16.0731% |
| 共有→対照 | .897883 | .746669 | .944079 | .792931 | 16.0101% |

学習総時間は対照40.427594/40.404724秒、共有33.603947/33.600125秒。
準備・退役込み総時間は対照42.533066/42.483537秒、共有35.696682/35.681874秒。
Graph退役・GCの3回分は各セル約.305–.320秒で、今回はすべて計上した。
総準備・退役費用は対照約2.079–2.105秒、共有約2.082–2.093秒。
通常評価・checkpoint保存・data/model配置・epoch order生成/転送は別費用で、process全体の壁時計ではない。
#129の退役GCを除外した準備込み時間と混ぜず、この対称な対照/候補の結果を使う。

3 epochともmodel/Adam/clock42 Tensorと全Parameter勾配5 Tensor、loss/正解数、test指標が一致。
回収checkpointをCPUへloadして、対照1セル対残り3セル×3epochの9比較でも各47 Tensorがbyte一致した。
#129の以前の最終checkpointとも47 Tensorがbyte一致。各保存値はfinite。
これはepoch境界の比較で、全stepの全中間Tensor比較ではない。

最終test精度は全4セル **97.37%**、loss **.083672396839**。
同条件dense Graphの45,000更新は学習.196892、準備・退役込み.223719ms/step、
最終97.87%/.070017516613。CST/dense比は共有候補で学習3.792倍、準備・退役込み3.544–3.546倍。

### メモリ

| 長期条件 | peak allocated MiB | peak reserved MiB | epoch末allocated MiB（3回共通） | epoch末reserved MiB（3回共通） |
|---|---:|---:|---:|---:|
| 現行FP32 Graph | 251.992188 | 442 | 245.407227 | 442 |
| 共有候補 | 251.992188 | 444 | 245.407227 | 444 |

約.80MiBの保存Tensorは存在するが、最大peakを決める時点の使用量は両方式で同じだった。
全体のpeak allocated削減や「cacheが無償」という主張ではない。
allocated増分0、reserved増分2MiBで事前2/8MiB予算内。3 epochの継続的増加なし。
capture・復元・全学習・評価・再準備を含む連続scopeで、capture後resetなし。
GPU process全体の使用量は未測定。

### 独立参照・実moment

共有Planで実forwardを実行し、学習済み前後層から得た事前選択train/test minibatchのX/dYを使った。
全52,428 atomの独立CPU FP64 Fourier Y/dX/全dP、および半径で層化した8 atomの48点quadratureが
従来の `3e-5 + 3e-4*abs(reference)` を通過した。
両minibatch中の最大絶対差はY8.4077e-7、dX1.3420e-10、dP8.1640e-10。
実m/v/clock45,000を共通にした最大Adam方向差は1.1921e-7。
次の実native更新後のP差は両minibatchとも列別[3.7253e-9,0,0,0]、clockは両者45,001。
同参照勾配の繰り返し対照はP/moment/clockがbyte一致した。
既存FP32のcold最大方向差gateの負結果は解消したと扱わない。

### 診断profile（採用時間とは別）

torch.profiler CPU/CUDA traceとcProfileを各16更新で取得した。traceのkernel categoryは
対照・共有とも1,776 events/16更新＝111 kernels/step。GPU memcpy等の別categoryは含まない。
過去のeager90 kernels/stepと同じ計測条件ではなく、今回の対照/候補内で同数という観測である。

| 状態・GPU kernel群 | 対照 us/step | 共有 us/step |
|---|---:|---:|
| 初期・分類 | 171.591 | 110.942 |
| 初期・両contract合計 | 345.577 | 254.681 |
| 学習後・分類 | 171.525 | 111.270 |
| 学習後・両contract合計 | 355.527 | 263.887 |

対象kernel群の短縮約152us/stepは、未計装の短期差約149–151usと整合する。
他の主要GPU群は大きく変わらず、共有による対象処理削減が全stepへ届いたという解釈を支持する。
ただし、profile時間を採用性能へ足したり差し引いたりしない。
CPU側では`_signature`/`_checked_roots`等のguard処理が観測されるが、計装負荷があるため
未計装時のCPU律速や残り時間の内訳をこのprofileだけで確定しない。
nsys/ncuは環境上利用可能だったが、今回はtorch.profiler/cProfileを使用し、
Nsightでのregister/spill/occupancyや命令別費用の測定は実施していない。

profile終了時（各開始状態から16診断更新後）のowner分布は、初期は各軸256非空ownerで全52,428 atomがrank12。
学習後はinput262/output261非空owner、rank6が6 atom、rank12が52,422 atom。
最大owner内atom数はinput初期255/学習後257、output205。両方式の分布は同じ。
計時loop内で毎stepCPUへ分布を回収していない。

### 検証・保全と判断

- L4環境で **248件通過**。新保存payloadの拒否・offload・retained/multiple forward・最新P・
  空atoms/batch/部分勾配・Graph replay・hooks/settings/storage/callback/finite-checkの契約を含む。
- ローカルCPUは1,711 passed/2,186 skipped。
  [測定sourceのCPU CI](https://github.com/takumiecd/torchcst/actions/runs/41759332747)は
  Linuxで1,710 passed/2,187 skipped、宣言検査とwheel/sdist buildも成功。
- source archive SHA256: `d44a22bc5fca5f1a05062f9e4e52a5821e1f56ea87708499d102c1f962a2bac7`
- result archive SHA256: `55715fc1153e8a671523eaa069b2fd4d2519378fcc7ef41b16c2fdd2e82456e9`
- driver SHA256: `284af551e61c76fb8f5757721eef3e4659ad3fff7aef1b3cf43a099a10cfeda0`
- 回収manifest **158 files** とarchiveを再hash照合し、`output/geometry-share/results/`へ保全。
  全epoch checkpoint、全36短期セル、4 profile、独立診断、入力manifest、phase logsを含む。
  `checkpoint-checks.json`に回収後の直接Tensor比較を保存した。
- 両失敗試行、全driver、source snapshot、解析scriptはignored evidenceに保全した。
  supervisor正常終了・所有全slotのstoppedを確認し、`final-pool-status.json`/`pool-lifecycle.log`へ保存。
- [公開集計JSON](owner-window-geometry-share-results-20261104.json)はraw tensor/logではなく、
  sample時間・状態hash・profile集計・独立診断をまとめたもの。

今回の一つのMNISTモデル・seed41・成功した一つのL4 allocationでは、候補継続基準をすべて通過した。
追加seed/モデル/GPU、汎用Graph API、既定dispatcherへの採用、mergeはこの研究では行わない。
[#130](https://github.com/takumiecd/torchcst/pull/130)をDraftとして結果ごとレビューに残す。

### 再現入口

#129と同じ凍結初期値/MNIST gzipに、FP32自身の最終checkpointを`fp32_final.pt`として加える。
dense用には同じ初期値から学習した`dense_final.pt`も用意する。各hashは公開集計の入力manifestを参照。
下記をfresh process・balanced orderで各6組実行する。`--profile`は別runで16更新だけに使う。

```bash
PYTHONPATH=src python -m benchmarks.training.full_step_graph_comparison \
  --inputs FROZEN_INPUTS --output output/base.json --kind cst --mode graph \
  --state initial --precision fp32 --bootstrap-fp32 --steps 256 --repeat 0
PYTHONPATH=src python -m benchmarks.training.full_step_graph_comparison \
  --inputs FROZEN_INPUTS --output output/shared.json --kind cst --mode graph \
  --state initial --precision fp32_geometry --bootstrap-fp32 --steps 256 --repeat 0
```

学習後は`--state final`。長期は`benchmarks.training.full_step_graph_learning --mode graph`の
対照と、同じ引数に`--geometry-shared`を追加した候補を比較する。
最終診断にも`full_step_graph_learning_diagnostics --geometry-shared`を指定する。
