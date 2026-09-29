# 旧 `prototypes/` の参照

`prototypes/` は 2026-09-29 の開発環境整理で廃止した。整理直前の全 141 ファイルは
Git commit `190a1bb3fa6259fde493a30d901d5cab7c2ebb82` に残っている。
過去の測定報告が旧ファイル名や実行コマンドを示す場合は、その commit の
ソースと測定条件を組にして読む。

```bash
git show 190a1bb3fa6259fde493a30d901d5cab7c2ebb82:prototypes/<file>.py
```

必要なら当時の commit を別 checkout で実行する。現在の package は
`src/torchcst/`、開発中の CUDA 実装は [`experiments/cuda/linear/`](README.md)、
継続測定の入口は [`benchmarks/cuda/linear/`](../../../benchmarks/cuda/linear/README.md)。
旧コードを現行 `auto` の一部として扱わない。
