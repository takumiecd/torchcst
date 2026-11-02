# 固定グリッド・幾何の非 Tensor 化（Step 4）

## 境界と変更

Step 3 の計算ソースは `cfcd9524`。共有 L4 で **433 passed / 1 skipped** を確認した後、
ユーザーの指示で Step 4 を続行した。Step 3 の証拠は
[API 統一記録](api-unification-step3-20261102.ja.md)に保全する。

- RegularGrid の origin / spacing と Geometry の radius / chart_margin / periods を
  immutable な Python metadata に移した。これらの State は固定 Tensor buffer を持たない。
- `.to()` は Tensor を保持しない placement hints を更新する。空の一時 Tensor に変換関数を
  適用して device/dtype を確認するが、固定値や Tensor storage を保存・量子化しない。
- 実行時の Grid 距離は中心 Tensor の device / dtype で座標を生成する。
  初期化後の CUDA 更新の placement は実際の `atoms.p` が正本。
- CUDA Polar の周期 Tensor は AlgorithmState.buffers に置き、周期・atom placement・定数の
  version が変わると再構築する。compiled launch も定数 version を確認する。
- 固定状態の signature は Spec の値を含め、native guard は Spec の辞書 version を確認する。
  property getter の code 変更も prepared callback の認証から外れることを確認する。
- checkpoint は全固定 metadata を extra state に保存し、復元先の一致を要求する。
  旧 buffer 形式や異なる固定値を読み替えない。

数学的な距離・積／radial・正規化・Polar 更新の式は変更しない。固定値は宣言の精度で保持し、
従来の FP32 buffer を後から FP64 化した値としては扱わない。
小さい spacing は metadata のまま保持し、出力座標の低精度丸めと区別する。
非有限の出力座標になる dtype / 配置は生成時に拒否する。

## 検証と再現

- CPU: `python -m tools.kernel_dev test --suite cpu` — **1,699 passed / 1,853 skipped**。
  getter 認証を含む対象検証は **9 passed**。
- runtime ruff、wheel / sdist build、README と Geometry の独立した実行例を確認。
- 固定 Tensor ゼロ、dtype 往復の metadata 精度、異なる constructor dtype のまま FP64 atom
  中心を渡す距離と勾配、weights-only reload、宣言の不正変更後の cache 検査を確認。
- 2 種の非対称周期を使う CUDA 更新と、geometry placement hints を変えた後も atom placement
  に従う検査を追加した。実機は別 snapshot で検証し、完了後に結果と hash を追記する。

ローカルログ: ignored `output/handoff-step4-*.log`。新しい速度や完全 step のピークメモリの
測定は行わない。固定状態の Tensor 数と学習全体の memory は区別する。


## 最終結果

計算ソース `9bb0557d`、job `l4job-af704f1b6a084250a1cffcb0faa52057` は
**716 passed / 3 skipped / 0 failed**。この件数には GPU 数値検査と CPU metadata 検査の両方を含む。
3 skip は L4 環境の host C++ compiler 検査（wrap / local-tile syntax / host local-mode compile）。
主要 CUDA 数値検査・非対称周期更新・native / prepared Linear・optimizer 検査は通過した。

NVIDIA L4 / driver 580.82.07 / Python 3.13.16 / Torch 2.11.0+cu130 / CUDA 13.0 /
Triton 3.6.0。ローカル全 CPU **1,699 passed / 1,853 skipped**、ruff、build、実行例、
radius catalog の宣言検査も通過。`9bb0557d` の GitHub CPU validation も通過した。

- source archive SHA256: `e3c976673a84ba86ee8b299a7f0678c4609645d17852b5bb979eaf65c9d5b519`
- result archive SHA256: `07fd4c1f60bbc423f2e8c89239eede719c07e88450672c54fc584925d6b02965`
- raw artifacts: `~/.local/state/colab-l4-pool/jobs/l4job-af704f1b6a084250a1cffcb0faa52057/results/`
- driver: ignored `output/handoff_step4_gpu_driver.py`（pool source archive に凍結コピーを含む）
- source / result archive の hash は pool の受信検証後、手元でも再照合した。
- pool supervisor の終了と slot の `stopped` を確認済み。他セッションは停止していない。

GPU 側の再現テスト一覧（同じ source と対応 runtime 上で実行）:

```bash
python -m pytest -q \
  tests/test_radius_cell_mass_cuda.py \
  tests/test_radius_cell_mass_fused_backward.py \
  tests/test_radius_global_native.py \
  tests/test_flat_torus_polar_update.py \
  tests/test_radius_center_canonicalization.py \
  tests/test_kernel_composition.py \
  tests/test_radius_prepared_callbacks.py \
  tests/test_radius_prepared_entry.py \
  tests/test_prepared_optimizer_update.py \
  tests/test_static_coordinate_metadata.py \
  tests/test_radius_polar_cuda_update.py \
  tests/test_regular_grid.py \
  tests/test_geometry.py \
  tests/test_coordinate_state.py \
  tests/test_linear_parameters.py \
  tests/test_normalized_strip_public.py \
  tests/test_atom_update_context_cache.py \
  tests/test_atom_update_metadata_cache.py \
  tests/test_linear_context_cache.py \
  tests/test_radius_local_tile_moments.py
```

この後の証拠追記 commit は文書のみであり、上記の計算ソースを変更しない。
完全 step 時間・allocated / reserved ピーク・新規 CUDA Graph 資格付け・DB / dispatch 採用は
この回帰検証の結果から主張しない。
