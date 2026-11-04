# G4での最終G1実行器の確認

目的は、L4で検証済みの最終版をG4でも測り、denseとの差と安全検査変更の効果を確認すること。
追加のkernel最適化、幅・batchの探索、公開defaultへの採用やmain統合は行わない。

## 測定前に固定する条件

- sourceは`9ee4642cad1beda9646101aa94e2bf88ca9bdc75`。既存L4測定の303 sourceファイル、
  runtime242ファイルとguard核心4ファイルをbyte単位で維持する。
- G4を1台・1 allocation。実hardware、Torch/CUDA/Triton、CPUを記録し、所有sessionを結果回収後に停止する。
  Torch2.11.0+cu130/CUDA13.0と既存資格境界を要求する。異なる場合は測定せず記録する。
- MNIST full train60,000/test10,000、seed41、N1024/A52428、真B4、中央1層CSTの3学習層。
  FP32 body＋既存FP64 geometry/Polar、TF32 off、math threads1、既存optimizer契約。
  初期値・データ・順序はL4と同じhashの入力を使う。
- 最大8 fresh process。AはCST qualified→CST G1→dense G1→dense qualified、Bは完全逆順。
  各3epoch/45,000実更新、Graph replay44,992回。候補とdenseに同じ実行方式を与える。
- 新GPUで既存FlatC kernel/whole-step契約テストを実行する。小規模31atomでは独立48点求積で
  Y/dX/全atom勾配とoptimizer参照を確認してから学習を測定する。
- 同一G4内では各epoch末のG0/G1・二順序の状態・勾配・指標・clockを直接byte比較する。
  L4との学習軌道のbyte一致は異世代間の合格条件にしない。モデル・入力・契約・データ順序は維持する。
- 既存runnerのtraining時間と準備・capture/再capture・復元・退役/GC込み時間を両方報告する。
  evaluation/checkpoint/import/data配置等は従来の時計外。cold compile時間は未分離。
- capture/評価込み累積peak allocated/reservedを記録し、G1増分は従来どおり2MiB/8MiB以内。
  L4 memory-v2.1合格をG4の合格として流用せず、今回は3epochの実測peakを報告する。
- G4の速度は記述的比較。5%以上というL4採用閾値をG4へ課して都合よく打ち切らず、
  正しさ・資格・メモリ違反または予算超過で停止する。新しい候補・救済再測定は行わない。
- 全体1200秒、回収用90秒を予約。GPUテスト最大500秒、最初の学習process最大500秒、以降180秒。
  source・入力・結果archiveのSHA256、実loaded binary、raw log/PTをignored outputへ保全する。

この文書作成時点では準備段階であり、G4結果ではない。
