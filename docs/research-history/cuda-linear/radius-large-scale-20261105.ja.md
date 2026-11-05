# Radius CSTの大規模対応と速度・メモリの確認

## 測定前の方針

幅2048で正しさを確認した後、4096・8192・16384の完全学習stepを比較する。
本命は8192。atom数は`floor(N*N/20)`、真batch4、FP32 body＋既存FP64 geometry/Polar、
TF32 off、CPU math threads1。radius[.5,50]、spacing1、M8、base1、merge2と更新則は維持する。
これはatom密度5%であり、1atom4変数なので学習Parameter数はdense中央層の約20%となる。
domainに対する相対radiusを固定した別の広支持条件やbatch拡張は今回含めない。

実装は最終G1の`9ee4642c`を親とする研究ブランチで行う。公開defaultや他の旧owner経路へ
大規模対応を一般化しない。大規模metadataは最終FlatCと比較用Sort32の2経路だけに許可する。

## 添字と容量の契約

- `MAX_SITES=16384`、`MAX_ATOMS=2**24`。旧shared128/4096制約は維持する。
- N16384のowner数は65534、複合key数は4294705156でsigned32を超える。
  int64で範囲検査してからuint32で復号する。全有効keyはuint32に収まり、復号nodeはintに収まる。
  既存のstable int64 sort fallbackを維持し、短いblock sortのpackingを拡張しない。
- N16384は15level/30bankで既存32bank/uint32 mask容量内。
  B4・A13421772では4A/BAもsigned32内。nativeのwork overflow検査も維持する。
- 同一input/output site数を明示検査する。研究Graph入口は承認した幅・正確な5%atoms・B4のみ。
  通常の公開入口がcaptureを拒否する契約は維持する。

## 初期値と独立参照

旧128/1024初期値生成はそのまま残す。新しい幅ではpublic CPU Fourierセル積分の局所支持だけを
構成し、atom順のFP32 scatterでdense Wを生成する。巨大な`eye(N)`を全atomへ適用しない。
全atom寄与・周期画像・正規化を保存するが、旧GEMM exportとは丸めの順序が異なるため、
新しいexport policyとして明示し、旧1024からの学習軌道byte一致は主張しない。
これは測定前のdense初期値生成であり、CST実行中にdense W/dWを保持する方法ではない。

L4は共有pool1台で、以下を正しさの入口とする。

1. 小atomの2048/4096/8192/16384でY/dX/全dP、境界・安定sort・高位keyを確認。
   16384では通常の狭いradiusでもsigned32を超える実keyを検査する。
2. 2048/A209715でpublic CPU局所Fourier参照により全Y/dX/dPを確認し、
   小atomの独立48点求積と既存optimizer参照を添える。
3. 2048の3学習層モデル（中央のみCST）でeager/Graphを同じ初期値・データ順序から実行。
   CST/dense各2epoch×16更新。capture復元・再準備・全state/grad/clock/metricsを比較し、PTを保全する。
4. 既存FlatC/Graph契約テストを維持する。CPU・GPUを区別し、skipをGPU成功とは数えない。

## 大規模測定

L4正しさの原本回収後、orchestratorがG4を明示的に1台選択し、固定した3幅だけを測る。
各幅でA順CST→dense、B順dense→CSTの4 fresh process。同allocation内の2順序であり、
独立GPU再現や信頼区間とは解釈しない。各3epoch×256更新（合計768実更新）、最初の8更新も数える。
初期状態と更新後のepoch別時間を全て報告し、有利な区間だけを選ばない。

既存`full_step_graph_learning`を使い、large studyではevaluationを同じ先頭64test例に制限する。
学習集合とshuffleは元の60,000例を維持する。この短期runのtest精度は学習性能の評価に使わない。
eval後のcapture/再準備/退役は自然な契約どおり行い、trainingと準備・退役込みの時計を別々に残す。
evaluation/checkpoint/import/data配置は従来の時計外。cold compile時間は独立計時できないため未分離。

大きい幅では全state/gradのdtype/shape/SHA256を各epochで保存し、同kindの順序間で一致を要求する。
巨大なdense optimizer PTを全epochで出力することはせず、raw全状態の直接比較は2048で行う。
大規模のSHA比較と小規模PTの直接byte比較は区別する。初期値・source・driverと実loaded binaryは保全する。
大規模の初期値は、実際のatom・通常層・buffer・生成policyを保存する。派生するdense Wは
content SHAを記録し、元のexporterで再構成してSHAを照合する。巨大な派生Wの全ファイルを
archiveへ重複保存せず、異なるTorch/CPUでのbyte再構成は保証しない。

capture前から評価・replay・再captureを含む累積peak allocated/reservedを測る。
GPU process全使用量と混同しない。速度・メモリ比は記述的に報告し、dense勝利や5%短縮を合格条件にしない。
正しさ・資格・非finite・OOM・時間予算で停止した場合は原因と未完了の幅を明示する。
同じ結果を改善するためのkernel候補sweepや条件変更は行わない。

L4 qualificationは1200秒、G4の固定batchは2400秒を上限とし、結果回収後に所有runtimeを停止する。
この文書作成時点は準備段階であり、GPU結果ではない。

## 入口検証の途中記録

最初のL4 job `l4job-def6a53dbfb940d499fd23cac05768b2`（source`5e021e00`）では
139件中138件がpassし、retained-readerテストの入力が1024固定であったため1件が停止した。
runtimeの形状検査が2048モデルへの1024入力を正しく拒否している。全atom参照や短期学習には未到達。
失敗原本を保全し、fixtureの幅だけを修正した`33b840ef`で、その1件と未到達の数値・更新検証を続行する。
先に通過した16384の高位キー・VJP等の結果と、修正後の追加結果は実行単位を区別して記録する。
runtimeとbenchmarkの数値処理はこのfixture修正では変えていない。


## 計画変更と最終結果

追加L4 job `l4job-b5d8dc25d86c44e7a0ef098156af9dc6` はColab CLIの初期`chdir`接続で応答せず、
通信上限で中断された。実験script dispatch前で、数値結果・速度結果はない。poolで所有runtime停止を
確認し、recover後にG4を1台だけ明示選択した。修正した2048入口をG4で先に完了させてから、
大規模測定へ進んだ。失敗を成功へ読み替えたり、L4とG4の未paired時間を接合していない。

測定sourceは`31e1ad1acdc1a4ae00107f655f52a9fc39eff95e`。初回からruntime/benchmark算術は同じで、
fixture修正・native復号のCPU境界テスト・記録を追加している。G4はRTX PRO 6000 Blackwell Server Edition、
SM120、Torch2.11.0+cu130/CUDA13.0/Triton3.6.0、Python3.13.16、driver580.82.07。
全12測定cell・各768実更新、合計9216実更新を完了した。

**16384まで実行でき、今回の範囲ではdense比の学習時間は約3〜4倍、
allocated peakはdenseの約45〜67%だった。8192がdenseより速いという結果ではない。**

以下の幅は同じG4 allocation内で比較した。A/Bは順序を完全に反転したfresh processであり、
二順序の範囲を全て示す。

| N | atoms | CST 学習ms/step | dense 学習ms/step | CST/dense |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 838,860 | 2.297–2.467 | 0.593–0.594 | 3.870–4.151 |
| 8192 | 3,355,443 | 6.319–6.550 | 2.126–2.129 | 2.972–3.077 |
| 16384 | 13,421,772 | 26.018–26.206 | 7.774–7.775 | 3.347–3.371 |

初回epochには実eager bootstrap8更新を含み、後の2epochは全てGraph replayである。
全epochの個別値はJSONへ保存し、初期値・更新後のどちらかを選び取っていない。

\[
 t_{train}=\frac{\sum_{e=1}^{3}T_{train,e}}{768},\qquad
 t_{with\ prep}=\frac{T_{preparation}+\sum_{e=1}^{3}T_{train,e}}{768}.
\]

準備・capture/復元・再準備・退役/GC込みの短期平均も次に示す。evaluation/checkpointは時計外。
これは256更新ごとに境界を通る768更新の値であり、45,000更新へ外挿しない。

| N | CST 準備等込みms/update | dense 同ms/update | CST/dense |
| --- | ---: | ---: | ---: |
| 4096 | 5.358–5.665 | 1.890–1.906 | 2.834–2.972 |
| 8192 | 17.639–17.864 | 6.460–6.499 | 2.731–2.749 |
| 16384 | 70.580–70.785 | 23.992–24.085 | 2.939–2.942 |

| N | CST allocated MiB | dense allocated MiB | CST reserved MiB | dense reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 402.972 | 597.909 | 472 | 632 |
| 8192 | 833.599 | 1607.659 | 1162 | 1720 |
| 16384 | 2478.623 | 5550.159 | 3586 | 5640 |

両順序のpeakは一致した。8192のCST/denseはallocated約52%・reserved約68%、
16384では約45%・約64%。capture・評価・replayを含む累積allocator peakで、GPU process全使用量ではない。
16384では既存int64 stable sort fallbackが使われる。追加profileを実行しておらず、
時間差の原因をこの切り替えだけに帰属していない。

## 正しさ・原本回収の確認

- 修正したretained-readerテストはG4でpass。2048/A209715の全Y/dX/dPはpublic CPU局所Fourier
  autogradと照合し、31atom独立48点求積・optimizer参照もpassした。全dPのmax absは約2.067e-6、
  relative L2は約3.410e-7。Adam方向max差0.009863は事前0.01条件内、RMS各列条件も通過した。
  同一勾配によるoptimizer state/controlはbyte一致し、全atomのpost-Pも既存ULP条件を通過した。
- 2048のCST/denseそれぞれで32更新（Graph側24replay）を実行。2epoch末の4直接比較、
  延べ144 Tensor項目についてeager/Graphがbyte一致し、clock16/32も直接確認した。
- G4では追加の形状検証9件（GPU5件、CPU4件）も全てpass。初回L4の138passと
  G4の修正1件の結果は別実行として保持する。native復号の追加CPU境界テストもpassした。
- 大規模の36epoch記録について全state/gradを保存時にhash化し、18組・延べ648項目の
  順序間hash一致を回収後に再確認した。大規模optimizer raw PTの直接byte比較ではない。
- 原本478ファイルのsize/SHAを検証し、13 PTを元の全file bytesへ復元した。
  2048の保存実Y/dX/dPから誤差を再計算し、状態・勾配・clockと実loaded libraryのSHAも照合した。
  CPU readerはTorch2.13で、workerの2.11を装っていない。
- source`5e021e00`のCPU一式は2092pass/2275skip。追加CPU decoderテスト1件pass。
  各修正headのGitHub CPU CIも確認し、最終文書更新後のCIは別途記録する。

## 保全先・再現・限界

rawは研究worktreeのignored `output/large-scale-20261105/` にある。
`g4-source.tar.gz`、`g4-qualification.py`、`g4-scaling.py`、`g4-launch.py`、
`frozen-2048.pt`、`g4-upload-manifest.json`、`g4-receipt.json`、`g4-results.tar.gz`、
`g4-results/artifacts/`、`verify-g4.py`、`g4-verification.json`を保全する。
結果archiveは471,537,153bytes、SHA256
`b4950f289085c97664be082b56e0d84b7da6457879dd7e82257be86d1c0588d8`。
公開集計は`radius-large-scale-results-20261105.json`。

再実行はsource/runtimeと初期値SHAを照合し、既存runnerへ次の条件を渡す。
CST/denseを別fresh process、反転2順序で実行し、出力先を使い回さない。

```bash
python -m benchmarks.training.full_step_graph_learning \
  --inputs /absolute/inputs --output /absolute/new-output \
  --kind cst --mode graph --flat-c --guard flat \
  --epochs 3 --steps-per-epoch 256 --evaluation-limit 64 --checkpoint-policy hashes
```

大規模初期値のdense Wは省略し、実際のatom・通常層・policyとWのcontent SHAを保存した。
保存済み`scaling/reconstruct-initialization.py`でWを再生成してSHA照合後に使える。
別CPU/Torchでのbyte再構成は保証しない。大規模の全optimizer値はraw保存しておらず、
hashだけから数値を復元できない。768更新・seed41・同じ半径分布での確認であり、
長期学習、広支持分布、batch拡張、別モデルでの学習品質や速度保証は今回の結果に含めない。

所有G4`cst-radius-scale-20261105`は原本SHA/size確認後に停止し、serverのactiveなしを確認した。
L4の所有slotも停止済み。追加の最適化・GPU測定・main統合は行わず、PR #139へ研究結果を保全する。
