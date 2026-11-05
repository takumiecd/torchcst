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
