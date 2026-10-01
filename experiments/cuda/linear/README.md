# CUDA Linear の開発用カーネル

ここは `CSTLinear` の新しい計算方式を試す場所。配布 wheel と公開
`backend="auto"` からは使わない。現行の `auto` は正確な CST 演算を選ぶ。
採否と制約は実装の隣に記録し、測定は [benchmark](../../../benchmarks/cuda/linear/README.md)
から再現する。

| 方式 | 主な実装 | 意味と状態 | 測定・判断 |
| --- | --- | --- | --- |
| Mapped streamed | [学習経路](block_streamed_backward.py)、[局所 W](block_materialize_listed.py)、[候補リストと atom 勾配](block_tile_atom_lists.py) | Exact CST の候補。bounded list の overflow、勾配、メモリを検証してから本体へ昇格する。 | [選別](cuda-kernel-shortlist.ja.md)、[Ada 実測](notes/five-percent-bounded-lists-ada.ja.md) |
| Fixed anchor | [学習経路](anchor_atom_training.py)、[基底](anchor_basis.py)、[Torus decode](torus_decode_trainable.py) | atom×site 評価を減らす近似演算。品質と長期学習が未確定なので開発用。融合 decode は 8192² で精度上の問題があり、通常の測定では PyTorch decode を使う。 | [L4 8192²](notes/l4-anchor-8192-dispatch-handoff-20260929.ja.md) |
| 比較用の旧方式 | 同じディレクトリのその他の実装 | 既存テストや比較から参照するため保持。採用予定を意味しない。 | [過去の記録](notes/README.md) |

新しい方式を追加するときは、演算の意味、対応形状・dtype・勾配、workspace、
既知の失敗例をこの近くの README に記す。kernel 単体と完全学習ステップを
区別して測る。[配置と昇格の方針](repository-layout.ja.md)と
[dispatch 提案](../../../docs/backend-history/cuda-dispatch-design.ja.md)は
未実装の部分を明示している。削除済みの探索コードは
[Git 履歴から参照](legacy-prototypes.ja.md)できる。

このディレクトリの GPU コードは CUDA/Triton を要する。GPU のない環境では
関連テストが skip されるため、GPU 上の正確性・速度を別途確認する。
