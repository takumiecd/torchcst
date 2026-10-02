# CUDA Linear benchmarks

現在の backend の正しさ・速度・メモリを確認する入口。
試作実装をここへ置かず、測定対象は `src/torchcst/_backends/` にある実装とする。
公開 API の統合確認と、registry の plan を強制する比較を分ける。

| ファイル | 役割 | 測定範囲 |
| --- | --- | --- |
| `run.py` | normalized full/window plan を強制して比較 | 独立 oracle、forward・backward・AdamW、eager / Graph、capture を含む peak |
| `check_normalized.py` | 公開 CSTLinear の統合確認 | 独立 oracle、全5 atom 勾配、更新後の支持、Graph replay と AdamW 状態。任意で完全 step を測定 |
| `validate_wheel.py` | shared L4 pool の検証 driver | 配布 wheel の全維持テスト、上記の統合確認、通常幅・鋭い支持・dense の完全 step |
| `strip_torus.py` | 既存 Strip/Torus の fused / split と参照経路の診断 | 準備、forward、forward/backward。optimizer は含まない |
| `strip_torus_dense.py` | Strip/Torus と保存済み dense の比較 | forward / backward、FP32 と別条件 BF16。メモリは warmed baseline からの追加割当 |
| `strip_torus_split.py` | Strip/Torus の split reductions の比較 | forward / backward、forward Graph。optimizer は含まない |
| `fixtures.py` | Chart・Kernel・model の共通構築 | 純粋な Spec と設定境界の State 構築。テストからも利用 |
| `reference.py` | 正規化 Triweight の独立 dense oracle | backend の距離・重み生成を呼ばず、サンプル点から直接計算 |

`__init__.py` を含めて9 Python file。benchmark から `tests/` を読み込まない。
旧 `experiments/`、探索用 CLI、移行段階ごとの validation driver は削除した。
旧 module 名や `--tests-file` / `--chart-suite` の互換入口は用意しない。

## 正規化 kernel の確認

```bash
python -m pip install -e '.[dev,cuda]'
python -m benchmarks.cuda.linear.run --algorithm normalized_window --correctness-only --output output/normalized-check.json
python -m benchmarks.cuda.linear.run --algorithm normalized_window --size 1024 --rows 128 --profile broad --dense --source-commit COMMIT --output output/normalized-step.json
python -m benchmarks.cuda.linear.check_normalized --output output/public-check.json
```

`run` は選択 plan と同じ演算の normalized full baseline を新しい process で照合する。
`--dense` は通常の dense Linear を追加する。dense は性能の参照で、CST と同じ Parameter
空間の演算ではない。`--profile sharp` は鋭い支持の別条件。通常の sigma3 と混ぜない。

`check_normalized` と `validate_wheel` は従来の NVIDIA L4 検証条件を維持する。
`run` と Strip/Torus の CLI は他の CUDA GPU でも使える。
全 CLI の設定は `--help` を参照。8192² を既定の測定にしない。

## 配布 wheel の L4 gate

[`colab-l4-pool`](../../../tools/colab-l4-pool/SKILL.md) を使う。
この driver は system package を変更せず、job-local の Torch 2.11.0+cu128 / CUDA 12.8
環境でテストする。初期 system probe の CUDA 13.0 と実際の検証 runtime を区別する。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_wheel.py" --label wheel-gate --timeout 600 -- SOURCE_COMMIT
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py serve --workers 1 --idle-seconds 0
```

N1024²、M128、約5% atoms、FP32、TF32無効、fused capturable AdamW の契約を維持する。
各時間・メモリ測定を別 process にし、peak allocated / reserved は Graph capture と replay
を含める。GPU process usage は別の指標で、この driver では未測定。
大きい fixture の完全 step 測定と、小さい fixture の独立全勾配 oracle を区別する。

## Strip/Torus の診断

```bash
python -m benchmarks.cuda.linear.strip_torus --output output/strip-torus.json
python -m benchmarks.cuda.linear.strip_torus_dense --source-commit COMMIT --output output/strip-torus-dense.json
python -m benchmarks.cuda.linear.strip_torus_split --output output/strip-torus-split.json
```

これらは現在の fused backend の診断であり、完全学習 step の結果とは比較しない。
BF16 は別精度の参照。Parameter bytes や追加 tensor 割当を GPU 全体の peak と呼ばない。

## 記録

作業出力は ignored `output/`、raw log / trace / tensor / snapshot は ignored
[evidence/](evidence/README.md)、小さい検証記録は [results/](results/) に保存する。
GPU・runtime・source commit/hash・演算条件・時間分布・誤差・メモリの範囲を記録する。
benchmark の成功は kernel / dispatch の認証や既定採用を自動で行わない。

過去の数値や不採用理由は [研究履歴](../../../docs/research-history/cuda-linear/README.md)、
削除した実行コードの取得方法は [削除台帳](../../../docs/research-history/cuda-linear/retired-code.ja.md)
に保存している。履歴中の旧コマンドは記録当時の source commit で再現する。
[今回の整理と検証](../../../docs/benchmark-cleanup.ja.md)を参照。
