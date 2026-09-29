# CUDA Linear benchmark

[開発用カーネル](../../../experiments/cuda/linear/README.md)の正確性、速度、
メモリを比較する入口。これらの測定は公開 `backend="auto"` を変更しない。
CUDA と Triton が必要。

```bash
python -m pip install -e '.[dev,cuda]'
python -m benchmarks.cuda.linear.profile_paired_dense_cst --size 1024 --rows 128 --graph --rounds 24 --output output/mapped-1024.json
python -m benchmarks.cuda.linear.profile_anchor_atom_training --size 1024 --rows 128 --basis-mode block --decode-mode torch --rounds 24 --output output/anchor-1024.json
python -m benchmarks.cuda.linear.profile_mapped_training_kernels --size 1024 --rows 128 --graph --output output/mapped-kernels-1024.json
```

`--window-rows`、`--cache-windows`、`--weight-tile-rows`、`--listed-unroll`、
`--listed-builder-ba`、`--listed-builder-warps` などは mapped 比較で変更できる。
アンカー側には `--anchor-rows`、`--anchor-column-segments`、
`--backward-lanes` がある。各 CLI の全項目は `--help` を参照。
8192²は大きな GPU メモリを要するので、まず小さい形状で正しさを確認する。

時間は同一環境の対応比較、ピーク割当は必要に応じて別プロセスで確認する。
作業中の結果は `output/` へ置き、採否に使うものだけ
[evidence](evidence/README.md) と [判断台帳](dispatch-evidence.ja.md)に残す。
[測定手順と将来の共通 runner](benchmark-workflow.ja.md)には、実装済みの個別
CLI と構想中の CLI を区別して記している。
