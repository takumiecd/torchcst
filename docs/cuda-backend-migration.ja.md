# CUDA backend の整理

作業 branch は `codex/cuda-dispatch-registry-v1`。基準は `7b17408`。
数式、正規化、支持域、入力勾配・atom 全成分の勾配を維持して段階的に移す。

## 配置の checkpoint

`nn/_backends/` の実装を root の `_backends/` に移した。

- `_backends/linear.py`: 公開 Module の backend 選択と遅延接続。
- `_backends/torch/operators/`: Torch の linear と Strip + Torus の参照計算・固定座標計画。
- `_backends/cuda/algorithms/normalized_euclidean_strip/`: 正規化した Euclidean Strip の full / window と実際に共用する `_shared/`。
- `_backends/cuda/algorithms/strip_torus/fused/`: Strip + Torus の融合準備・forward / backward・schedule。
- `docs/backend-history/`: 以前の設計・GPU 計測ノート。本文中の旧 source path は当時の記録。

normalized Euclidean Strip と Strip + Torus は数学契約が異なるため分ける。
Torch の tiled 実行は CUDA package に依存しない。Triton は実行時に読み込む。
未採用方式は `experiments/` に残す。

ローカル検証: 全テスト **482 passed / 196 skipped**。GPU は未実施。
移動した 26 Python ファイルは import を除いた AST が移動前と一致する。
分割した `_preparation.py` の全 function / class 本体も一致する。
生の比較用 source は ignored `benchmarks/cuda/linear/evidence/cuda-backend-migration/` に保存。

最初の全テストでは宣言の import 境界テストが 2 件失敗した。原因は root の
linear facade から Torch evaluator を先行 import したこと。実行時の遅延接続へ
修正し、境界テストを維持したまま全件通過した。

次の checkpoint で CUDA 専用の意味の判定を Algorithm 内に移し、共通 OperatorSpec
を Registry に渡す。GPU・wheel の検証はその完成 source で実施する。
