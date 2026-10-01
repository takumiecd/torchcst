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

ローカル全テスト **483 passed / 196 skipped**（最終 GPU source の検証は以下）。
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
wheel SHA256: `9b2d2fd6e390866323f37849abcfcff13582dcc98206354542a43a1e64e2a767`。

## Installed-wheel L4 検証

source checkpoint は `3294bac`。Job は
`l4job-f6d7928b69a443bc8f2c339925cb8092`。
配布 wheel 上の全テストは **743 passed / 2 skipped**、490.982 秒。
Strip + Torus の Triton Linear 74 件、CUDA dispatch 31 件、schedule 21 件、
Operator 22 件、CSTOptimizer 42 件は全件成功。
正規化 Strip の独立 FP64 mixed fixture、更新後の再照合、plain AdamW の Graph
capture / replay も成功。full / window の二つの同時 forward と必要な勾配の各組合せ、
checkpoint、chart 設定変更後の plan invalidation も全テストに含む。

GPU: NVIDIA L4、23034 MiB、driver 580.82.07。
Python 3.13.15、Torch 2.11.0+cu128、CUDA 12.8、Triton 3.6.0。
wheel と実際の installed package の **140 Python modules** 全てが frozen source に
SHA256 一致。source archive SHA256 は
`b2fa7d3e80c6d92e23c6902a3e915802679e8abbccd24d086f51a85cd9292086`、
result receipt の archive SHA256 は
`1c4dc0fbabc1e7467970af48cb5033373bdf768add791a30c6557ca5c68d2766`。

### 学習一 step の時間・メモリ

N=K=1024、batch=128、atoms=52,429、FP32、TF32 off。
固定の Euclidean site、Triweight、global L2 norm、幅範囲 `.03–3.25`。
plain fused/capturable AdamW（lr=1e-4、weight_decay=.01）。
性能検証の optimizer は CSTOptimizer wrapper ではない。
各方式は独立プロセス。値は 7 回の中央値で、forward・backward・optimizer を含む。
peak は warmup 後の CUDA Graph capture / replay を含む実測値。

| fixture | 実装 | eager ms | Graph ms | peak allocated MiB | peak reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 通常 sigma=3 | full | 1.587 | 0.615 | 51.848 | 106 |
| 通常 sigma=3 | window | 2.937 | 1.507 | 46.162 | 86 |
| 通常 sigma=3 | dense reference | 0.736 | 0.125 | 50.502 | 106 |
| sharp 専用 sigma=.199 | full | 1.575 | 0.536 | 51.848 | 106 |
| sharp 専用 sigma=.199 | window | 2.953 | 0.304 | 46.162 | 86 |
| sharp 専用 sigma=.199 | dense reference | 0.721 | 0.124 | 50.502 | 106 |

前の `1df491e` job と initial atom hash・batch・optimizer 条件が一致する。
各方式の peak allocated / reserved も変更前と一致。単発の時間差から実装の性能改善は
主張しない。dense は別の重み表現の性能参照であり、数学的な correctness baseline
ではない。sharp fixture は通常 sigma=3 の性能目標と別に扱う。
GPU process 全体の使用量は未測定。full-shape の独立全 atom 勾配 oracle は未実施で、
mixed fixture の独立基準と既存 GPU suite で正しさを検証した。

詳細な数値・前回値・誤差・hash は
[機械可読の検証記録](../benchmarks/cuda/linear/results/cuda-backend-migration-20261001.json)。
生の source、wheel、ログ、receipt は ignored
`benchmarks/cuda/linear/evidence/cuda-backend-migration/l4job-f6d7928b69a443bc8f2c339925cb8092/`
に保存。queue 原本も保持。supervisor が正常終了し、owned slot 1 の runtime が
`stopped` になったことを確認した。

### 再現

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source /absolute/checkout-of-3294bac \
  --script /absolute/checkout-of-3294bac/benchmarks/cuda/linear/validate_dispatch_registry.py \
  --timeout 900 --label cuda-backend-migration-installed-wheel -- 3294bac --full-suite
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py serve --workers 1 --idle-seconds 0
```

この検証で実装の配置・共通宣言の接続を確認した。既定 dispatch を変更せず、
未採用の実験方式を production に加えていない。
