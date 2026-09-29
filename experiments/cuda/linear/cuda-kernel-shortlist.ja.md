# CUDA kernel 移植候補の暫定選別

2026-09-29。`experiments/cuda/linear/` から本体へ昇格させる**候補**。既存の測定・テスト・
呼び出し経路を読み直した結果であり、この文書は新しい GPU 実測や公開 `auto`
への採用を意味しない。評価対象はまず Strip + Torus + Triweight、64×64 block、
FP32、5% atom の `CSTLinear` 相当。演算の意味を保つ経路とアンカー近似は分ける。

## 先に残す核

| 役割 | 移植元・核となる関数 | 判断 |
| --- | --- | --- |
| 既存の公開経路 | `src/torchcst/nn/_backends/` の Torch 実装、`_triton_kernels.py`、`_triton_preparation_kernels.py` | 現行 backend と oracle の基礎として維持。既存 `auto` の挙動は変えない。 |
| 準備・routing | `experiments/cuda/linear/support_box_routing.py` の `station_site_boxes`、`balanced_home_columns`、`_bucket_histogram`、`_bucket_scatter`、`atomic_bucket_sort` と `experiments/cuda/linear/block_streamed_backward.py` の `trainable_boxed_prepare` | mapped と anchor が共用。`support_buckets_batched_box` は既に本体へ移した。移植時に `torch.sort` の一時的なグローバル置換を明示的な sorting 接続へ変える。 |
| 有界候補リスト | `experiments/cuda/linear/block_tile_atom_lists.py::build_tile_atom_lists` の `BOUNDED=True` と `experiments/cuda/linear/block_streamed_backward.py::build_listed_forward_candidates_bounded` | 8 bit 順位、192件、overflow 時の全 bucket 走査を一組で残す。候補数の初期分布だけで安全性を決めない。 |
| 局所 W | `experiments/cuda/linear/block_materialize_listed.py::materialize_listed` の bounded 経路 | window 内の W を生成。`BR=16,BC=64` を基準とし、L4 の小形状では W だけ `BR=8` を候補にする。 |
| atom 勾配 | `experiments/cuda/linear/block_tile_atom_lists.py::mapped_backward_atoms_listed` の bounded 経路 | `BR=16,BC=64`、別の局所 dW GEMM と組み合わせる。deterministic algorithms 有効時は atomic backward を拒否する。 |
| 大きい M の GEMM | `experiments/cuda/linear/bounded_gemm_fp16x3.py` の 3 項版、`experiments/cuda/linear/bounded_gemm.py` の TF32x3 版 | IEEE FP32 の Torch GEMM を基準に保ち、検証済み形状だけ高速数値レシピとして使う。GEMM は kernel ID と誤差条件を明示する。 |

この核から、`experiments/cuda/linear/block_streamed_backward.py::_MappedStreamed` と
`experiments/cuda/linear/block_streamed_forward.py::streamed_forward` の**採用経路だけ**を
取り出して接続する。現状の `_MappedStreamed` は複数の旧方式を切り替える
約500行の分岐を持つため、ファイル全体を移すのではなく、bounded 経路を
小さく組み直す。局所 dW は現在の `torch.mm` をまず使い、独立した Triton
dW kernel を増やさない。

速度・メモリの根拠は [Ada の bounded 測定](notes/five-percent-bounded-lists-ada.ja.md)、
[Blackwell 再測定](notes/blackwell-bounded-cst-20260929.ja.md)、
[Ada/Ampere/L4 の小形状比較](notes/small-shape-ada-ampere-20260929.ja.md)、
[L4 tile 比較](notes/l4-small-tiles-20260929.ja.md)。大形状 Ada の推奨試作は
`listed_bounded`、`window_rows=1024`、`cache_windows=4`、
`listed_unroll=4`、builder `BA=32/warps=1`。小形状 L4 では
`window_rows=512`、W は8×64・勾配は16×64、FP16全行キャッシュが速い。
FP16キャッシュは入力勾配の誤差条件が広がるため別レシピにする。
これらは対象 GPU・形状の**bench case**であり、機種全体への既定値ではない。

## 近似アンカーとして別に保持

`experiments/cuda/linear/anchor_atom_training.py` の `AnchorLayout` と局所基底、
`build_anchor_candidate_lists`、`materialize_anchor_listed`、
`backward_anchor_listed`、`anchor_trainable` を一つの近似アルゴリズムとして
移植候補にする。共通 bounded list を使う `full_tile` と、実アンカー site の
AABB を使う `anchors` は bench で比較可能にする。L4 の1024²では局所基底・
アンカー候補・atom 勾配の4/8レーンに実測の利点がある。8192²では
16×32と24×48の速度・品質差が測られたが、27更新後の真のCSTとの差も
残る。Exact の `auto` には接続しない。

`experiments/cuda/linear/torus_decode_trainable.py` の融合 decode は1024²のL4で有利だったが、
8192²ではアンカー値が参照から約0.44%ずれて失格となった。移植するなら
**小形状専用の明示的候補**に限定し、標準経路は PyTorch decode にする。
[L4 anchor 最適化](notes/l4-anchor-optimization-20260929.ja.md)と
[8192²の引き継ぎ](notes/l4-anchor-8192-dispatch-handoff-20260929.ja.md)を参照。

### デフォルトに関する暫定判断

既存の `CSTLinear(backend="auto")` は正準CST演算を選ぶ契約として維持する。
atom×site 評価点を大幅に減らした**実測済み**の方式はアンカー補間で、
演算そのものが近似に変わる。1024²・L4で最速だった12×24 PODは校正済み
基底が必要で、異なる seed の結果を16×32固定アンカーと単純比較できない。
8192²・L4の
16×32固定アンカーは現行CSTより速い一方、27更新後の出力差は4.60%。
したがって、最初は `approximate` の明示選択または bench preset とし、
その中の基準候補を16×32固定アンカー＋局所基底＋PyTorch decodeにする。
実タスクの品質閾値と長期学習を測ってから、近似を許可した利用者向けの
`auto` 方針を検討する。局所10項基底による正確な評価削減は未検証の研究案で、
現時点のデフォルト候補に数えない。

## 本体へ移さない候補

| 候補 | 扱いと根拠 |
| --- | --- |
| `prototypes/cuda/materialize_bounded.cu` | 同じ局所 W で Triton より遅く、完全ステップへ接続していない。実装は Git 履歴へ残す。 |
| `block_materialize_parallel`、直接 atom 走査、`flash_listed_forward`、融合局所 dW | 対象の完全ステップで勝ちが確認されていないか、別条件で悪化。報告を残し、runtime kernel には入れない。 |
| `block_atom_major_backward`、`block_interval_backward`、`block_streamed_backward` 内の baseline / factored / staged / partial / fused atom backward | 正確性比較に使ったが、現時点の主レシピは listed atom backward。必要な oracle は Torch 参照と現行テストへ移す。 |
| `listed_csr` と非 bounded `listed` | 比較用 bench としては有用。新しい機種や高密度で bounded の overflow が多い場合に再測定してから runtime 候補へ昇格する。 |
| `bounded_gemm_fp16x3` の追加残差項 `fp16x4_dw` | atom 勾配の要素別許容を超えたため採用しない。 |

kernel 単体の優劣と完全ステップの優劣は別に記録する。削除する prototype に
言及する古い文書には元 commit を残し、現行 bench の入口を別途示す。
移植した核は forward、`dX`、atom 勾配、atom 移動、支持域境界、
候補 overflow、eager/Graph、ピーク割当を再確認してから runtime に接続する。
