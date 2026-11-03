# 全学習stepのCUDA Graph検証

最終結果：guard付き全step Graph化と数値・状態保護の検証は成立した。
CSTの256-step完全step比較は初期13.063%、学習後5.547%短縮したが、
denseは両状態で約84%短縮し、CST/denseの時間比は広がった。
capture込みのCSTメモリも増えたため、大幅高速化を狙う今回の試作は採用保留とし、
45,000 step実学習や追加GPU探索には進まない。

基準はmain/#125の `7aa7c89e42a4dd515630f8358e4ade880ddea482`。
#126/#127は閉鎖済みのまま保全し、本branch `kernel/full-step-graph` とは混ぜない。
目的は真B4の全モデルforward/loss/backward/実optimizer/指標集計を1 replayにまとめ、
外側のguard・入力供給・mutation通知込みの利益を検証すること。

## 限定した試作

最初に同一凍結初期値、N1024/A52428/FP32公開・FP64内部、TF32 off、native CSTAdamW、
同じowner-window-recompute算術と既存stable sortでcapture成功とbyte状態を確認する。
専用native入口はこの研究branchだけに追加し、通常のcapture拒否やdispatcherを変更しない。
専用入口もshape/dtype/軸/launch等の検査を維持する。既存sort/optimizerの大規模な再実装が
必要なら、この最小実験では止めて記録する。

速度比較へ進む条件は、通常eager・準備済みeager・Graph間でのlogits/loss/全grad/全model・
Adam状態のbyte一致、毎replayの正しいstep進行、最新Parameterからの分類である。
単なるcapture成功や最終P一致だけを対応完了とはしない。hooks、live設定、storage、
retained reader、version通知、有限値検査、入力・状態の寿命も対応範囲と検査対象にする。
対応外はそのstep全体をeagerへ戻し、guard費用を計時へ含める。

成立した場合はCST/dense各3実行方式、初期8更新後・基準学習後、256 minibatch×6 paired反復。
同じallocation内で条件順を釣り合わせ、各条件は別process。区間前後だけ同期する。
メモリはoptimizer準備・warmup・capture前からreplayまでのallocated/reserved実peakを測る。
準備時間とbreak-evenも別記する。短区間を45,000 stepの実学習結果とは呼ばない。

10%短縮は継続判断の目安で、予測値やユーザー指定ではない。
メモリ増加ゼロもユーザーの確定条件ではないため、増減を隠さず速度と併記する。
raw driver・結果・checkpoint・source hashはignored `output/full-step-graph/` とdefault poolへ保全する。
準備時点ではcapture・正しさ・速度は未測定だった。以下に段階的な実測を追記する。採用・mergeは行わない。

## 段階的な成立確認

最初の `f01097a6` / `l4job-29e83fcc352c4b7cb6afbd6f84c88f8b` では、準備済みeagerが
通常eagerと4 step連続で49 Tensor項目一致した。一方、Graphのbackward captureは
`cudaErrorStreamCaptureImplicit` で失敗した。通常streamで先に作ったautograd状態を
保持したまま、別streamでcaptureした構成だった。

モデルの最初の学習呼び出しから、同じ専用non-default streamを使う限定構成へ直した。
このstream条件は全対照にも同じように適用する。NVIDIAの
[capture失敗の説明](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/capture-failures.html)
も、default streamに由来するAccumulateGradの依存を注意点として挙げている。

`69c8adce` / `l4job-f8549883165741c2b494efe83675738e` では、CSTのcaptureと4 replayが成功し、
通常eagerに対してlogits/loss/全Parameter勾配/model・Adam状態の49項目が毎step byte一致。
同jobはその後、dense構築のexecution指定漏れで終了したため、全比較成功とは数えない。

`43ad10b3` / `l4job-2273baddae0142fabb1fa0397fb3e07e` ではCST/dense両方のcaptureが成功した。
各々、通常eager/準備済みeager/Graphの4 step比較はbyte一致（CST49項目、dense27項目）。
続く契約検査ではmodule/Tensor/optimizer hooks、saved-tensor hooks、lr、live設定、storageの
変更を検知し、通常eagerへ戻った。retained readerでもGraphを使わず、更新後の古いbackwardは
PyTorchが拒否した。検査側が想定した例外文言と実際のReadLease例外が違ったため、
その段階で検査を止めて文言条件を修正した。速度測定には進んでいない。

中間の `l4job-fb5201b59d11417d9db82336bfe903f9` はdriverのindent、
`l4job-b9ae31d2b8584d78aa2a7ba4d1cf2a68` はCST Planとdenseの構築指定の混同で停止した。
いずれもアルゴリズムの否定結果ではなく、性能標本に含めない。修正後、両モデルの
構築をローカルCPUでも確認した。これらの失敗もpoolのsource/results/receiptに残している。

## 比較の解釈

| 条件 | CST | dense |
| --- | --- | --- |
| 既存モデル・optimizerのeager完全step | C0 | D0 |
| 専用入口・固定snapshot scratchを使う準備済みeager | CP | DP |
| 同じ準備済みbodyのGraph replay | CG | DG |

CP/DPとCG/DGは同じ数値bodyを使い、captureの有無を変える。C0/D0との差には準備・
実行入口の変更も含む。CGとD0だけでCST/denseの優劣を主張しない。
GraphでGPU kernelが1つになるわけではなく、既存の演算・分類・sortは毎step走る。

初期費用の回収step数は、同じ測定範囲の秒単位の値を使って

\[
K_{\rm BE}=
\frac{\max(0,T_{\rm setup,G}-T_{\rm setup,0})}{t_0-t_G}
\]

とする。分母が正でなければ回収不能。startupはoptimizer構築/ロード、warmup、captureと
同一storageへの状態復元を含む。共有したsourceの初回compile等は別の共通準備として記録する。
45,000 step相当の総時間を計算する場合は推定と明記し、全学習の実測と混ぜない。

各状態の6 run-pairを単位として、`log(tG/t0)`の平均と標本標準偏差から、df5の
片側95% Student-t上限を計算し、`1-exp(上限)`を短縮率の探索的な下限とする。
係数は2.0150483733330233。これは独立性・log比の分布を仮定した同一allocation内の
小標本の指標であり、256 stepを独立標本とした検定でも、他環境への保証でもない。
計算方法は結果の取得前に固定した。

## 再実行の入口

hash照合済みinputsには`initialization.pt`、`baseline_final.pt`、`dense_final.pt`、
`data/`以下のMNIST4ファイルを置く。元のsource/data/checkpointを変更しない。

```bash
PYTHONPATH=src python -m benchmarks.training.full_step_graph \
  --inputs /absolute/frozen-inputs --output output/graph-probe
CST_GRAPH_INPUTS=/absolute/frozen-inputs PYTHONPATH=src \
  python -m pytest -q tests/test_full_step_graph.py tests/test_radius_owner_window.py
PYTHONPATH=src python -m benchmarks.training.full_step_graph_comparison \
  --inputs /absolute/frozen-inputs --output output/graph-cell.json \
  --kind cst --mode graph --state initial --repeat 0 --steps 256
```

`--kind`は`cst/dense`、`--mode`は`eager/prepared/graph`、`--state`は`initial/final`。
各cellは新しいprocessで起動する。ローカルのpool driverは6反復の奇数回で条件順を反転し、
正しさ検査が通過した場合だけ計72 cellを実行する。成立しない状態をeagerへ戻した測定値を
Graphの速度と誤認しないよう、測定中のfallbackを検出したcellは失敗として止める。

## 最終L4比較

`l4job-f4d04952a73f44269559660294efc8c5` は正常終了し、全結果をSHA256照合して回収した。
測定sourceは `eed517c8f52017250408ed54910eb09d0b3f4574`。
L4 UUID `e251bffd-6042-23af-82e5-739c5d599de6`、driver580.82.07、
Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、Python3.13.16。
CPUはIntel Xeon @2.20GHz、12論理core、cgroup `max 100000`、math threads1。

### 正しさと契約

最初にCST/denseの4-step対照が再度byte一致した。GPU環境で188件がPASSした。
内訳は新規の全step契約検査16件と、既存window検査172件である。
既存検査には独立FP64 VJP、広い半径・境界、FP32/64、B0/1/3/4、
retained/offload/hook等を含む。Graph試作そのものの対応はFP32公開・真B4に限定される。

新規検査はcapture後のmodule/Tensor/optimizer/saved-tensor hooks、lr、live scalar、
storage、grad storage、optimizerのparams順、step/zero_grad callback、finite-check設定の
変更をreplay前に検知し、既存eagerへ戻ることを確認した。
retained readerがある時はGraphを使わず、更新後の古いbackwardもPyTorchの保護が働く。
Graph replay後のParameter version通知、最新の周期境界/wide atomからの分類、
logits/loss/中間dX/全Parameter勾配/全model・Adam状態の3 step byte一致も通った。
通常native入口のcapture拒否も維持した。

全72 cellで予期しないfallbackはなかった。各状態・各モデルの18 cellについて、
開始/終了Tensor状態、最終勾配、loss/正解数集計、初期値・データ・順序hashが一致した。
CSTは42状態Tensor＋5 Parameter勾配、denseは20状態Tensor＋5 Parameter勾配。
これは256 step後の一致と非計時の4-step/境界3-step検査であり、
全256 stepの全中間Tensorを保存比較したという意味ではない。

### 完全stepの未計装wall time

各数値は256 step×6新規processの平均ms/step。同じallocation内で条件順を反転した。

| 開始状態 | モデル | 現行eager | 準備済みeager | Graph | 現行→Graph短縮 |
| --- | --- | ---: | ---: | ---: | ---: |
| 初期8更新後 | CST | 2.455995 | 2.413110 | 2.135178 | 13.063% |
| 学習後 | CST | 2.437795 | 2.434076 | 2.302579 | 5.547% |
| 初期8更新後 | dense | 1.281189 | 1.457595 | 0.203985 | 84.078% |
| 学習後 | dense | 1.274478 | 1.440289 | 0.202212 | 84.134% |

CSTは両状態で6/6 paired反復が短縮した。探索的な片側95%下限は初期12.227%、学習後4.402%。
準備済みeager→Graphだけの短縮はCST初期11.518%、学習後5.402%だった。
denseの準備済みeagerはguard等で現行より遅く、準備の費用も無視できない。
Graphの本測定はそのguard・入力indexコピー・host version通知を含む。

同条件のCST/dense時間比は、eagerで初期1.917・学習後1.913だったのに対し、
Graphでは初期10.467・学習後11.387だった。GraphはCSTも短縮したが、
この構成ではdenseがはるかに大きく短縮し、denseとの差を縮める設計にはならなかった。
この結果からCPU関数別の律速や、残存時間のGPU/guard内訳を確定はしない。

### capture・startupを含むメモリ

| 開始状態 | モデル | eager allocated MiB | prepared allocated MiB | Graph allocated MiB | eager/prepared reserved MiB | Graph reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 初期8更新後 | CST | 245.402344 | 245.450195 | 251.532715 | 418 | 442 |
| 学習後 | CST | 250.713379 | 250.713379 | 251.532715 | 418 | 442 |
| 初期8更新後 | dense | 256.606934 | 256.638672 | 264.596191 | 410 | 432 |
| 学習後 | dense | 264.577148 | 264.577148 | 265.158691 | 410 | 432 |

各条件の6反復でpeak値は同じだった。CSTは初期allocatedが6.130371 MiB、学習後0.819336 MiB増え、
reservedは両状態とも24 MiB増えた。Graphでメモリを維持できた、とは言わない。
学習後側はcheckpointロード・warmup後の復元もstartup内に含み、初期側とpeakが異なる。
この測定範囲は過去の全epoch結果と同じではないため、過去の246.09/392 MiBと直接比較しない。
プロセス全体のGPU使用量は測っていない。

### 準備費用と判定

CSTの平均startup秒は、初期eager2.738771→Graph2.858072、
学習後eager2.841659→Graph2.949813。上記式による回収stepの目安は約372/800 stepだった。
いずれも既存extension等を共通に準備した状態のprocess内startupであり、
sourceの初回compile等は前段probeの215.519秒に含まれる。probe全体がcompileだけという意味ではない。
2つの固定checkpointでの短区間から45,000 step全体の速度は確定しない。

この仮説では「完全step Graph化は成立し、CSTにも短縮効果がある」ことを確認できた。
ただし学習後は継続目安の10%短縮に届かず、探索的下限も5%未満で、メモリも増えた。
denseとの差を縮める成果にもならなかったため、今回の限定試作は採用保留とし、
追加GPU探索・45,000 step学習へは進まない。これは大幅な高速化を狙う研究の継続判断であり、
5.5%の短縮を効果なしとしたり、他設計での改善を否定するものではない。

汎用Graph対応・本採用とは区別する。標準モデル/optimizerを専有する研究用実行器であり、
任意のPython callbackや並行stream競合まで認証したものではない。無効化後は再captureせずeagerへ戻り、
Graph poolは実行器の寿命まで保持する。通常dispatcherと公開APIは変更せず、mainにも統合しない。
本モデルの45,000 step学習、別GPU・別seed、Graph終点の新規全atom quadratureは未実施。

- source archive SHA256: `70891458b14f1f487a056bef6452b1ca0842347ab3aa14bd8719d75564fed157`
- result archive SHA256: `f15379794a16254db34b333ec1ec2cc73c82fe68f775c20f045fd4ba56737e81`
- 保全先: `output/full-step-graph/matched-results/`、`matched-spec.json`、`matched-receipt.json`。
- 原本と中間失敗: default poolの各job directory。source/driver/input payloadも保全済み。
- CPU標準suiteは1,704 passed / 2,130 skipped（`43ad10b3`）。その後のGPU専用追加検査3件は
  ローカルではskip、上記GPU環境では通過した。対象Pythonのlint、compile、`git diff --check`もPASS。
- 回収後、supervisorの正常終了と全slotの`stopped`を確認した。
  `output/full-step-graph/final-pool-status.json`へ記録済み。
