# M8のAtom予算タスクpilot

[English](m8-atom-budget-20261110.md) | 日本語

以下は初期準備と、その後の実測検証の記録。所有者は未mergeのM16実装PR159〜162をcloseし、M8のAtom数探索を指示した。
確保GPUはRTX4090へ移行。閉じたbranchと全結果は保全しており、closeはM16がタスク差の原因だという証明ではない。
旧結果では同じAtom数のM8/M16の損失は近かった。
M8のPR163と証拠はopenで保全。この研究branchはその上に積むため、履歴にはM16が残るが新実行では選択しない。
採用時にはM8の変更を停止したstackから取り出して統合する必要がある。

別physical/envelope予算Algorithmは512↔2048・FP32・1..256行・非接続M8/base1/merge2/origin0/spacing1・
A16384/24576/32768/49152/65536/98304/131072に限定する。
追加は条件付きhost guardとprivate Plan/extension宣言のみ。
RADIUS_LM_BUDGET blockを除けば既存headerとbyte一致する。
数値kernel・旧c8/c16 guard・初期化・optimizerは不変。公開/既定Registryは変更しない。
中央benchmarkのfixture認定やGraph対応ではない。

学習前にmetadata/拒否/roundtrip、nativeのretained・複数forward・非連続入力を検査する。
各段階で全AのFP64 Y/dX/raw各field（maxabs≤3e-4かつrelativeL2≤1e-4を個別要求）、
保存/observer後の次更新完全一致、checkpoint/moment/時計/有効数を監査する。
consumerはsource/data/protocolを固定し、実機4090/CC8.9を記録する。
完全step時間とallocated/reserved peakを区別する。merge/既定採用は行わない。

固定前CPU検証は2288合格・2677件の任意要件skip。Ruffとdiff検査に合格。


## 完了した検証とタスクpilot

固定library0bed48655b1d988e0c59b4442005aaec8c463a31、consumer
9b9e2eb1e4579fe035ca2442b36bad99f1df029dで、実機RTX4090（CC8.9・Torch2.9.1+cu128・CUDA12.8）の
新17条件を全完了した。native123件、17条件の保存/observer後の次更新完全一致、34 checkpoint監査、
32明示的Atom/マスク1数監査に合格。初期/終端・全AのFP64は96 tensor/raw-field対を検証し、
maxabs最大1.9908421e-5、relativeL2最大5.9737507e-7、旧component違反0。
maxabs<=3e-4かつrelativeL2<=1e-4は不変。有限FP32 moment、時計CST976/DST973、終端validation再現にも合格。
タスク検証済みはc16/32/64/128で、中間24/48/96はmetadataのみで学習していない。

DEVはvalidationを意味し、TRAINとstory単位で分離した8192 targetをmodel.eval()/no_grad()で評価する。
記録したminibatch TRAIN CEとは別。validationはhyperparameter選択にも使い、confirmation32768 targetは性能未評価。
置換は全8FFN mapで、attention/embedding/共有語彙はDense。終端validation CEは以下。

| A/map | M8 validation CE | DEV選択dense-mask DST CE |
| --- | ---: | ---: |
|16384|3.350850|3.373828|
|32768|3.344251|3.358979|
|65536|3.331671|3.323396|
|131072|3.330237|3.315642|

新Denseは3.308981。固定pilot基準のCE+.01/Top1-.5ポイント以内を満たすCSTはなく、中間点は実行せず、
selected_coefficientはnullのまま。65536は記述的な次の折衷候補に限る。
Top1 37.280%対Dense37.451%、CE差.022690、step459.236ms・allocated1622.90MiB・reserved2114MiB。
131072への増加はCEを.001435下げる一方、観測Top1は36.780%。収束・正式な同等性・最適値・唯一原因の主張はしない。
同じ初期化方針でもAで半径分布が変わるため、純粋な表現容量の分離ではない。
この小型LMではDense（18.194ms・allocated948.74MiB）への速度/allocatedメモリ優位はない。
DSTは密重み+boolマスクを維持しCOO/CSRは使わない。

[consumer PR40](https://github.com/takumiecd/cst-research/pull/40)に英日validation/TRAIN図・全12DST LR候補・
幾何・実測コスト・派生来歴を保存した。source archive
 e77a8b6ed48d24a8dc44600f3f6459d44774391194b44b858665c27d5abd5a07、
final compact60f83fa3f78f9135c5c8d9612cca02d6479e1021860d02f80c2327936e952ffa。
195 payload/2237 source hashを全取得検証。所有33commandは全成功、GPU idle、raw checkpointはremoteに保全しHubは維持。
API403で一時監視できなかったが、所有者token更新で復旧し重複jobは起動していない。
merge/既定採用は行わない。M16 PR159〜162はcloseしコード/証拠を保全。統合時は停止した履歴からM8を取り出す必要がある。
