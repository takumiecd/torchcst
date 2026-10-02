# 削除した研究コード（2026-10-02）

未採用の試作を現在の benchmark や tests の依存先として維持する方式を廃止した。
数値・採否・失敗記録はこのディレクトリへ移し、実行コードは Git 履歴に保存する。
削除直前の checkpoint は `5341ace`（branch `codex/cuda-dispatch-registry-v1`）。
本体の backend 実装は今回変更していない。

```bash
git show 5341ace:experiments/cuda/linear/block_strip_linear.py
```

実験を再開するときは当時の commit から独立した研究 branch / worktree を作る。
古いコマンドや package 名を現在のツリーへ読み替える互換 loader はない。
現在の検証は [benchmark 入口](../../../benchmarks/cuda/linear/README.md) を使う。

削除した実行コード・専用テストは以下の54ファイル。

- `benchmarks/cuda/linear/_profile_trace.py`
- `benchmarks/cuda/linear/benchmark_large_forward.py`
- `benchmarks/cuda/linear/benchmark_mapped_training_memory.py`
- `benchmarks/cuda/linear/benchmark_pod_anchor_basis.py`
- `benchmarks/cuda/linear/benchmark_tile_study.py`
- `benchmarks/cuda/linear/check_anchor_atom_gradients.py`
- `benchmarks/cuda/linear/check_fp16_full_cache.py`
- `benchmarks/cuda/linear/diagnose_cst_sampled_blocks.py`
- `benchmarks/cuda/linear/probe_csr_graph_step.py`
- `benchmarks/cuda/linear/profile_anchor_atom_training.py`
- `benchmarks/cuda/linear/profile_anchor_large_memory.py`
- `benchmarks/cuda/linear/profile_bounded_list_builder.py`
- `benchmarks/cuda/linear/profile_candidate_csr.py`
- `benchmarks/cuda/linear/profile_current_paths.py`
- `benchmarks/cuda/linear/profile_mapped_training_kernels.py`
- `benchmarks/cuda/linear/profile_paired_dense_cst.py`
- `benchmarks/cuda/linear/profile_small_tile_shapes.py`
- `benchmarks/cuda/linear/sweep_pod_anchor_basis.py`
- `benchmarks/cuda/linear/validate_dispatch_registry.py`
- `benchmarks/cuda/linear/validate_normalized_strip_wheel.py`
- `experiments/__init__.py`
- `experiments/cuda/__init__.py`
- `experiments/cuda/linear/__init__.py`
- `experiments/cuda/linear/anchor_atom_training.py`
- `experiments/cuda/linear/anchor_basis.py`
- `experiments/cuda/linear/atom_prefix_linear.py`
- `experiments/cuda/linear/block_atom_major_backward.py`
- `experiments/cuda/linear/block_fused_config.py`
- `experiments/cuda/linear/block_interval_backward.py`
- `experiments/cuda/linear/block_materialize_kernel.py`
- `experiments/cuda/linear/block_materialize_listed.py`
- `experiments/cuda/linear/block_materialize_parallel.py`
- `experiments/cuda/linear/block_shared_kernel.py`
- `experiments/cuda/linear/block_streamed_backward.py`
- `experiments/cuda/linear/block_streamed_forward.py`
- `experiments/cuda/linear/block_strip_kernels.py`
- `experiments/cuda/linear/block_strip_linear.py`
- `experiments/cuda/linear/block_support.py`
- `experiments/cuda/linear/block_support_diagnostics.py`
- `experiments/cuda/linear/block_tile_atom_lists.py`
- `experiments/cuda/linear/bounded_gemm.py`
- `experiments/cuda/linear/bounded_gemm_fp16x3.py`
- `experiments/cuda/linear/direct_diagnostics.py`
- `experiments/cuda/linear/local_atom_kernels.py`
- `experiments/cuda/linear/local_atom_linear.py`
- `experiments/cuda/linear/support_box_routing.py`
- `experiments/cuda/linear/torus_decode_trainable.py`
- `tests/test_atom_prefix_linear.py`
- `tests/test_block_streamed_backward.py`
- `tests/test_block_strip_linear.py`
- `tests/test_fused_compute_configs.py`
- `tests/test_local_atom_linear.py`
- `tests/test_streamed_materialization.py`
- `tests/test_support_box_routing.py`
