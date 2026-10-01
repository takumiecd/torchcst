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

配置 checkpoint: `76abff7`。

## 共通契約への接続

`operators/normalized_strip.py` と CUDA schema の別 OperatorSpec / alias を削除。
Kernel の固定 preset は `kernels/presets.py`、Chart の組立は公開 Module の設定境界、
CUDA が対応する宣言の判定は Algorithm グループ内 `contract.py` に置く。
Registry / Context は共通 OperatorSpec と実測した parameter dimension を受け取り、
normalized Strip の 5 列という条件は Algorithm 側だけで確認する。
異なる Profile・norm・domain・floor・width を forced plan に渡しても拒否する。

full / window の Algorithm・Recipe・executor・Triton kernels を各ディレクトリに整理。
full の autograd と window の provider は per-forward の state を保持する。
旧 full の 8 定義、旧 window の 15 定義、Torch reference の本体は AST が一致する。
Torch reference も `nn/` から Torch backend に移した。数学の変更はない。

ローカル全テスト **483 passed / 196 skipped**（最終 GPU source の検証は以下に追記）。
CPU-only fake Algorithm に 7 列の atom を渡すテストで Registry の 5 列固定を除いた
ことを確認。正規化の変更に対する forced-plan 拒否も追加した。

Strip + Torus は CUDA の正しい配置へ移したが、既存 live Module を使う接続を維持。
共通 Spec の Registry への登録は次の段階。normalized Euclidean Strip は registry
経由、Strip + Torus は linear facade から fused executor へ接続する。これは配置を
統一しても数学契約が同一でないことと、設定 snapshot を forward に入れないための
現時点の移行境界である。性能による既定選択の変更・承認制度は追加していない。

local wheel は **140 Python modules** 全てが source と SHA256 一致。
旧 `nn/_backends/` と `operators/normalized_strip.py` は wheel に含まれない。
wheel 単体の import path で全テスト **483 passed / 196 skipped**。
wheel SHA256: `5b5e891284e2e636b3abfdfdad243915281fb9be7c53708f487593b1495ea937`。
