# M8のAtom予算タスクpilot

[English](m8-atom-budget-20261110.md) | 日本語

準備段階で未測定。所有者は未mergeのM16実装PR159〜162をcloseし、M8のAtom数探索を指示した。
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
