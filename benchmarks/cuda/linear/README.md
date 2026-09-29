# CUDA Linear benchmark

現在動く比較入口をここに集めた。開発中の mapped / anchor 実装を測るもので、
`CSTLinear(backend="auto")` の挙動は変えない。CUDA と Triton が必要。

```bash
python -m pip install -e '.[dev,cuda]'
```

```bash
python -m benchmarks.cuda.linear.profile_paired_dense_cst --size 1024 --rows 128 --graph --rounds 24 --output output/mapped-1024.json
python -m benchmarks.cuda.linear.profile_anchor_atom_training --size 1024 --rows 128 --basis-mode block --decode-mode torch --rounds 24 --output output/anchor-1024.json
python -m benchmarks.cuda.linear.profile_mapped_training_kernels --size 1024 --rows 128 --graph --output output/mapped-kernels-1024.json
```

`--window-rows`、`--cache-windows`、`--weight-tile-rows`、`--listed-unroll`、
`--listed-builder-ba`、`--listed-builder-warps` などの値は mapped 比較で変更できる。
アンカー側には `--anchor-rows`、`--anchor-column-segments`、
`--backward-lanes` などがある。個別 CLI の全項目は `--help` を参照。
8192²は大きな GPU メモリを要する。最初は小さい形状で正しさを確認する。

時間は同一環境の対応比較、ピーク割当は必要に応じて別プロセスで確認する。
結果 JSON は作業中 `output/` へ保存し、採否に使うものだけ `docs/data/` と
[evidence 台帳](../../../docs/dispatch-evidence.ja.md)へ移す。将来の共通 case
runner は[benchmark 運用](../../../docs/benchmark-workflow.ja.md)の設計に従う。
