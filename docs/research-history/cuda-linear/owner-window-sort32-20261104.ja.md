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

### 初回短期比較（未計装）

| 開始状態 | #130平均ms/step | int32平均ms/step | paired幾何平均短縮 | 勝数 | dense Graph平均ms/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| #130 FP32で8実更新後 | 0.713087829 | 0.646487014 | 9.3397787% | 6/6 | 0.206735482 |
| FP32自身の45,000更新後 | 0.725764667 | 0.657697164 | 9.3786981% | 6/6 | 0.208835405 |

各状態のCST全12cellで開始/終了42状態、全5勾配、loss/正解数、入力・データ順序hashが一致。
各状態のdense 6cellも同じ状態・勾配・集計である。fallbackは全cellで0。
全CST cellのcapture込みpeak allocatedは251.5341797MiB、reservedは442MiBで同じ。

### 別processの16更新profile

| 開始状態 | #130 Radix µs/step | int32 Radix µs/step | #130 reverse µs/step | int32 forward+reverse cast µs/step |
| --- | ---: | ---: | ---: | ---: |
| 初期 | 119.584500 | 61.450375 | 1.450063 | 2.828000 |
| 学習後 | 119.470000 | 61.428000 | 1.456062 | 2.824000 |

両状態でkernel-category eventは111→96/step、Radix/radixを名前に含むeventは20→12/step。
castは両方向合わせて2回/stepであり、従来reverseの1回を置換し、forwardの1回を追加する。
offset生成は約6.85→6.55–6.57µs/step。分類・浮動小数点contractの演算は変更していない。
Radix群の約58µs減と、その他のsort関連kernel数・時間の減少は、未計装の約67–68µs短縮と
整合する。ただしprofileを採用throughputへ換算せず、host CPU律速やDRAM/L2量、
正確なradix pass数、register/occupancyは未測定とする。

### 初回の環境と保全

source `ee675f29cb44b94bc9ce8b522313fbe71bd0adbb`、L4 UUID
`5c19e97a-9ece-ac64-ebb3-bde118f79b42`、driver580.82.07、
Torch2.11.0+cu130/CUDA13/Triton3.6/Python3.13.16。
source archive SHA256 `db4a56c380d86570462da4da66964021b72dbf11977185ae2c30f4d732ceb9cb`、
driver SHA256 `ac10aad6b801ca4b3fbb8dc452f8782ad88613484b1d1703b12825f8b2cbd929`、
result archive SHA256 `2570e259e3a06bf7b9255379eaacedb062bf7034447f289db954ef993b3f9799`。
122 manifest fileとarchive hashをローカルで再検証した。
GPU333検証、ローカルCPU1712 pass/2270 skip、Linux CI1711 pass/2271 skipと
宣言検査・wheel/sdist buildが通過。初回の3epoch smoke実checkpointも各47 Tensorをbyte比較した。

結果回収後のremote cleanupが180秒timeoutになりsupervisorは非0終了したが、
job自体はreturncode0/timed_out=falseで検証済み結果を回収済み。poolはそのVMを停止し、
serverから消えたことを確認した。停止状態を確認しrecoverを完了してから追加jobを投入した。
結果成功、remote cleanup失敗、VM停止確認を別の事実として保全する。

raw evidenceはworktreeの `output/sort32/results` とpoolの当該job directoryに保存する。
新しいjobは別directoryへ保存し、初回decision・ログ・checkpointを上書きしない。

### 追加jobの転送障害と再実行

`l4job-5915513af2f94fc595254a22ac54c89f` はremoteの `POOL_RESULT_READY` まで進んだが、
結果downloadが25分以上応答せず、ローカルに結果byteを取得できなかった。
当該jobのdownload CLI processだけを中断し、poolに停止・回復を任せた。
停止APIはendpoint Not Foundを返し、recoverがserverに当該assignmentがないことを確認した。
raw結果・checkpointは未回収のため、このjobの長期成績を成功・失敗いずれにも数えない。
source/spec/driver/transport・中断理由・lifecycleはローカルに保全している。

同じruntimeと長期条件で明示的に再実行する。今度は計測・数値比較・checkpoint保存が
すべて終わった後に、checkpoint ZIP memberのpayloadとmetadata断片をSHA256で重複排除し、
gzip圧縮する。各元ファイルを全byte復元してSHA256一致を確認した後だけ、その元ファイルを
転送archive内の復元manifestと圧縮blobへ置換する。学習・計時・数値状態は変更しない。
ローカルの実checkpoint6個で104,367,930→31,270,879 bytesへの梱包と、元ファイル全6個の
完全復元を検証した。受信後も元の47 Tensorに加えて元ファイルのbyte/hashを再検証する。
ローカルの実行・転送中だけidle sleepを抑制し、終了時に解除する。

### 回収できた追加8 lifecycleと45,000更新の結果

再実行 `l4job-9829fe58160946658cd40c87b3fdf6f5` は正常終了・結果回収・remote cleanup・VM停止を完了した。
全runtime 242ファイルとtestsのhashは初回成功sourceと同じ。
`--epochs` の導入と文書・private driverだけが変わり、長期の既定は3epochのままである。

16更新×8epochのlifecycleは全8epochで42状態+5勾配・集計が一致した。
基準のepoch末reservedは8回とも444MiB、候補は442/444/444/444/444/444/444/444MiB。
allocatedは両者とも245.4072266MiBで8回一定。peakの増分も0で、事前に追加した後半6回の
一定条件を満たしたため、45,000更新へ進んだ。

| 実行順序 | #130学習ms/step | int32学習ms/step | #130準備・退役込みms/step | int32準備・退役込みms/step | 準備・退役込み短縮 |
| --- | ---: | ---: | ---: | ---: | ---: |
| #130 → int32 | 0.745009972 | 0.675453250 | 0.794161642 | 0.724684450 | 8.7484950% |
| int32 → #130 | 0.744782231 | 0.675552930 | 0.793863962 | 0.724819701 | 8.6972410% |

各cellは全train60,000/test10,000、3epoch、45,000**実更新**、44,992 replay。
初期8更新も計時・学習集計へ含む。captureは8/15,000/30,000更新後で、復元時の42状態と
計数済みmetricsが一致。全層backward・全atom勾配・dX・実optimizer・gather/loss/集計・
replay guard/index/version通知を含む。準備・退役費用は2.2087–2.2170秒で、
うち旧Graph解放・GCと最終退役は0.3705–0.3774秒。両者を対称に加算した。
評価・checkpoint保存・data/model配置とepoch順序の生成/転送は別計時で、process全体wall timeではない。
転送用梱包もcheckpoint保存後の処理であり、adoption_secondsには含めない。

4cell全3epochの42数値状態・全5Parameter勾配・loss/正解数・data orderが一致し、fallbackは0。
回収・復元した実checkpointも、9個のepoch間比較と#130旧最終checkpointとの比較を含め、
すべて47 Tensorでbyte一致した。最終精度97.3699987%、test loss0.08367239683866501。
全saved値はfinite。独立のFP64参照基準は緩めていない。

同条件dense Graphは学習0.199588680ms/step、準備・退役込み0.227641692ms/step。
int32 CST/dense比は学習3.3842–3.3847、準備・退役込み3.1834–3.1840。
一つのMNIST構成・seed41における結果であり、一般のモデル・hardware・batchには外挿しない。

### メモリ判定を分離する

長期のcapture込みpeak allocatedは両者251.9921875MiB、reservedも両者444MiBで、増分は0。
epoch末allocatedは全cellで245.4072266MiB一定。基準reservedは両順序で444/444/444MiB、
候補は両順序で442/444/444MiBだった。8 lifecycleの追加確認も初回増加後は一定で、
この観測範囲では継続的な増加は見られない。allocatorの物理的原因をcounterで特定したとはしない。

**長期の速度・数値・peak予算は通過したが、raw driverの `long_gates_pass` はfalseのまま残す。**
理由は候補の「全3epoch末reservedが完全同値」という追加の厳格な条件だけがfalseだからである。
これは速度/数値の不合格やpeak予算超過ではない。事前の「継続増加なし」と、より強い
「初回epochから完全同値」を混同しないよう、初回停止・追加8回の合格・長期の各flagを
processed JSONにすべて保存する。結果を合格表示にするための再warmupやpeak resetは行わない。

今回の依頼である実装・短期比較・条件付き45,000更新は完了。研究用Draftとして成果を保全し、
このboundedなreserved推移を正式採用時にどう判定するかは明示してレビューに残す。
公開dispatcherの変更やmergeは行わない。この候補のGPU再探索はここで終了する。

### 最終checkpointの独立診断

候補自身の45,000更新後で、固定train/test minibatch各1個の実際の学習済みX/dYを使い、
全52,428 atomの独立CPU FP64 Fourier Y/dX/dPと、stratified8 atomの48点quadratureを検査した。
既存 `3e-5 + 3e-4 abs(ref)` ですべて通過。最大絶対差はY8.4077e-7、dX1.3420e-10、dP8.1639e-10。
実際のm/v/clock45,000を使ったAdam方向差は最大1.1921e-7、次のnative更新のP差は
振幅列3.7253e-9以下・他列0、clock45,001。同じ勾配を渡すcontrolはP/optimizer状態ともbyte一致。
以前のcold-Adam direction gateの負結果を合格へ読み替えてはいない。

### 最終provenanceと復元

長期source `b9deb0843062581ce283c07e3dcc867f745a0e5a`、L4 UUID
`d39a769f-16ec-34a4-f0dc-697635b6ad13`。runtimeは初回と同じTorch/CUDA/Triton、
CPUはXeon2.20GHz/12logical、math threads1。
source archive SHA256 `13da633b244d8d14832bd8b3cbdd2501560296dea0220aff174faa502527a656`、
driver SHA256 `73a44fdde94620eb3a1e2b15736edaad1d2a03e36022243b98df7d5d1f01ef34`、
result archive SHA256 `940d9d2f754bce57560fbd0454df634af401ecd55c5de640b4aa8a724e62fd8a`。
archiveと1,029 manifest fileをローカルで再検証し、checkpoint/診断Tensorの33個の元PTファイルを
全byte/hash一致で復元した。最終ローカル検証は計21比較×47 Tensorでbyte一致。

raw archive/manifestはpoolのjob directoryに原形で保存し、復元先は
`output/sort32/compact-results/artifacts`。初回結果は `output/sort32/results` に別保存。
復元manifestは `checkpoint-packing.json`、元PTのSHA256はprocessed JSONにも収録した。
`output/sort32/checkpoint_pack.py unpack <artifacts>` で、未展開のcopyへ全ファイルを復元できる。
梱包blobのSHA256と元ファイルのSHA256を検査してから保存する。

短期source CI、epoch数指定導入後CI、長期source CIはすべて宣言・CPU・wheel/sdistを通過した。
raw数値・各gate・個別sample・profile・hashは
[`owner-window-sort32-results-20261104.json`](owner-window-sort32-results-20261104.json) に保存する。
回収後のfresh pool statusで全slot stopped、supervisorと一時的なsleep抑制も正常終了を確認した。

## 公開判定v2と限定採用（追加GPU実験なし）

レビュー後、#131をN1024/A52428/真B4の次の研究基準と明示opt-inとして採用する。
根拠は速度・状態一致・対照以下のendpoint・等しいpeakと、追加8回のplateauである。
2MiB増加の物理的原因を特定したとはしない。mergeの依存整理と公開既定化は別の判断である。

公開判定は `benchmarks.training.full_step_graph_memory.evaluate_lifecycle`、
versionは `graph-lifecycle-memory-v2`。8回ちょうど・各16実更新、同じ測定位相、
cycle3–8のallocated/reserved各byte一定、全cycleの42状態/5勾配の記録hashと集計/順序一致、
fallback0、全scope peak増分allocated≤2MiB/reserved≤8MiBを固定する。
8回未満は判定不能、8回超から都合のよいsuffixを選ぶことは拒否する。
endpointは評価→train復帰→同期後、checkpoint保存・旧Graph退役前であり、
最終退役後の全pool解放やPyTorch外の割当を証明するものではない。

旧raw結果JSONとfalse判定を変更せず、
[`owner-window-sort32-memory-v2-20261104.json`](owner-window-sort32-memory-v2-20261104.json)
へ元JSONのSHA256、run ID、`evidence_scope=retrospective_supplemental` と新判定を別保存した。
定性的な条件を初回結果後・追加run前に具体化した履歴も保持する。
判定関数は記録hash/metadataの比較であり、元Tensorをloadした検証とは区別する。

CPUテストはplateau/一定を合格、後半reserved/allocated増加、途中peak超過、
状態/勾配/順序/start差、fallback、測定位相不一致、更新数不一致を不合格として確認する。
情報欠落は判定不能とし、旧結果のfalseと元file hashを変更しないことも検証する。
#131のreservedだけを理由とした追加45,000更新は行わない。
