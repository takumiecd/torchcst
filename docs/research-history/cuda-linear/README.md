> 研究履歴。ここにある旧コマンド・配置案・実装の状態は記録当時のもの。
> 現在の入口は [benchmarks](../../../benchmarks/cuda/linear/README.md)、旧コードは [削除台帳](retired-code.ja.md) を参照。

# CUDA Linear の開発用カーネル

ここは `CSTLinear` の新しい計算方式を試す場所。配布 wheel と公開
`backend="auto"` からは使わない。現行の `auto` は正確な CST 演算を選ぶ。
採否と制約は実装の隣に記録し、測定は [benchmark](../../../benchmarks/cuda/linear/README.md)
から再現する。

| 方式 | 主な実装 | 意味と状態 | 測定・判断 |
| --- | --- | --- | --- |
| Mapped streamed | [学習経路](retired-code.ja.md)、[局所 W](retired-code.ja.md)、[候補リストと atom 勾配](retired-code.ja.md) | Exact CST の候補。bounded list の overflow、勾配、メモリを検証してから本体へ昇格する。 | [選別](cuda-kernel-shortlist.ja.md)、[Ada 実測](notes/five-percent-bounded-lists-ada.ja.md) |
| Fixed anchor | [学習経路](retired-code.ja.md)、[基底](retired-code.ja.md)、[Torus decode](retired-code.ja.md) | atom×site 評価を減らす近似演算。品質と長期学習が未確定なので開発用。融合 decode は 8192² で精度上の問題があり、通常の測定では PyTorch decode を使う。 | [L4 8192²](notes/l4-anchor-8192-dispatch-handoff-20260929.ja.md) |
| 比較用の旧方式 | 同じディレクトリのその他の実装 | 既存テストや比較から参照するため保持。採用予定を意味しない。 | [過去の記録](notes/README.md) |

新しい方式を追加するときは、演算の意味、対応形状・dtype・勾配、workspace、
既知の失敗例をこの近くの README に記す。kernel 単体と完全学習ステップを
区別して測る。[配置と昇格の方針](repository-layout.ja.md)と
[dispatch 提案](../../backend-history/cuda-dispatch-design.ja.md)は
未実装の部分を明示している。削除済みの探索コードは
[Git 履歴から参照](legacy-prototypes.ja.md)できる。

このディレクトリの GPU コードは CUDA/Triton を要する。GPU のない環境では
関連テストが skip されるため、GPU 上の正確性・速度を別途確認する。

階層Fourierタイルの探索は2026-10-21に一旦終了。
[再開用引継ぎ](hierarchical-fourier-handoff-20261021.ja.md)と
[回収済み比較・数式](hierarchical-fourier-dispatch-20261021.ja.md)を参照。

2026-10-23の設計相談は
[128サイト版：atom群の並列処理と集団単位のjob割り当て](hierarchical-fourier-block-design-20261023.ja.md)
に整理した。処理名、部分同期、Z/H/Cと出力の保持、実装順をまとめた未実装のCUDA融合案であり、
既存Triton経路の測定結果とは区別する。
担当人数をthread単位で調整する[関数別擬似コード](hierarchical-fourier-thread-functions-20261023.ja.md)では、
`init_block()`からZ/H/C・最終出力までの処理と、固定集団の同期・メモリ配置を具体化している。
atom群の全Hを各threadで別々に生成し、空いた集団がready jobを取得する方針へ更新した。
一つのatomの協調H生成は撤回済み。動的な配分で傾向を調べ、後に仕事量からの配分規則へ進む。
[backward128の設計](hierarchical-fourier-backward-128-20261023.ja.md)には、G生成からdXへの展開、
Hの再計算・保存・一部保存、正規化を含む全atom勾配、batch間合算と保存・同期の範囲を記した。
保持するatom群とvalue／中心・幅の微分縮約値を選び、不足項目だけを再計算する未実装の候補である。
