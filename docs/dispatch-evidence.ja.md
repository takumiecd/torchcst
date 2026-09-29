# Dispatch の実験台帳と知見

2026-09-29。CUDA dispatch の[設計](cuda-dispatch-design.ja.md)が参照する、
実験と採否の入口。性能の数値は測定条件付きの事実として扱い、異なる GPU、
PyTorch / Triton 版、shape、M に無条件で外挿しない。この台帳は実行時に
読み込まない。配布する選択木の各 plan は、ここから辿れる安定した evidence ID
を持つ。

## 公開プロジェクトでの置き場所

- **GitHub Issue**: 新しい仮説、再現したい性能問題、実験の分担、未決の設計。
  途中の質問や結果への反論もここで議論する。Issue は進行中の会話であり、
  最終的な性能根拠の唯一の保存先にはしない。
- **リポジトリの文書とデータ**: 再現できる実験結果、失敗例、採否、現在の知見。
  選択木の plan は、この台帳の evidence ID と source commit を参照する。
  既存の companion `cst-experiments` に測定を置く場合も、採用判断とリンクは
  このリポジトリから辿れるようにする。
- **PR**: 実装と、知見を選択木に反映する変更をレビューする場所。
  Issue・evidence ID・比較した候補へのリンクを付ける。
- **ARCTX（任意）**: 長い試行錯誤や比較 sweep の作業記録に使える。
  貢献者にインストールを要求しない。ARCTX を使った場合も、採用判断は
  通常の Markdown と再現データへ要約し、ARCTX なしで読めるようにする。

ARCTX の導入を必須条件にはしない。将来、複数人が ARCTX の trial / topic
機能を使いたい場合は、まず小さな実験で export と新規貢献者の導線を確認する。

## 記録するもの

1. **演算の契約**: Exact CST か Approximate CST か、forward と勾配の意味。
2. **実験**: 仮説、比較した全候補、source commit、再現コマンド、GPU と
   ソフトウェア環境、測定手順、結果、誤差、ピーク割当、制限。
3. **判断**: 採用・見送り・保留の理由と、適用できる workload 範囲。
4. **選択木への反映**: 採用した plan ID と節点、evidence ID、未測定領域での
   fallback。木の revision が変わったら変更理由を追加する。

失敗した試行も残す。次の貢献者が同じ候補を試す場合、条件が違うのか、
既知の反例を覆したのかを判断できるため。kernel 単体の勝利と完全学習ステップの
勝利、eager と CUDA Graph、速度と正しさ・ピークメモリは別々に記録する。

## 現在の入口

| Evidence ID | 対象 | 現在の判断 | 詳細 |
| --- | --- | --- | --- |
| `E-20260929-ADA-BOUNDED` | Ada・5%・局所 W の bounded 候補リスト、窓、GEMM | 明示試作。複数設定と反証があり、公開 `auto` には未登録。 | [測定と採否](five-percent-bounded-lists-ada.ja.md) |
| `E-20260929-BLACKWELL-BOUNDED` | Blackwell・同系列の完全学習ステップ | 別 GPU の測定。Ada の選択を世代へ外挿する根拠にはまだ不足。 | [再測定](blackwell-bounded-cst-20260929.ja.md) |
| `E-20260929-SMALL-ADA-AMPERE` | 1024²・M=16/128/2048 | 小 M と GPU 差を確認。限定した設定候補で、公開 `auto` には未登録。 | [比較](small-shape-ada-ampere-20260929.ja.md) |
| `E-20260929-L4-SMALL-TILES` | L4・1024²・小 tile | W 生成と atom 勾配で有利な tile が異なる。機種と形状を限定した知見。 | [測定](l4-small-tiles-20260929.ja.md) |
| `E-20260929-L4-ANCHOR-8192` | L4・8192²・アンカー補間 | Approximate CST。速度の利点と学習後の出力差を確認。Exact の `auto` へは入れない。 | [引き継ぎ](l4-anchor-8192-dispatch-handoff-20260929.ja.md) |

この索引は既存報告を消したり重複させたりしない。詳細報告と小さな生データは
`docs/` と `docs/data/` に残す。巨大な profiler trace や一時 sweep は、
source commit・ハッシュ・入手先を報告に記し、リポジトリを肥大化させない。
`output/` の一時ファイルだけを根拠にした結果は、選択木へ採用する前に
再現に必要な要約とデータを `docs/` 側へ移す。

## 新しい実験の最小テンプレート

```text
Evidence ID / 状態: 仮説・検証済み・見送り・選択木へ採用・更新で失効
演算: Exact または Approximate、意味の版、対象の勾配
仮説: どの条件で何が改善する見込みか
コード: source commit、実装・benchmark・oracle の場所
環境: GPU 名、compute capability、SM 数、CUDA・PyTorch・Triton 版
workload: N/K/M、atom 数と密度、chart、profile、dtype、学習設定
比較: 全候補の recipe ID と基準、warmup、測定順、反復数
結果: 対応ペアの時間分布、PyTorch 割当ピーク、forward/dX/dp 誤差
判断: 採否、対象範囲、反例、未測定条件、選択木の節点 ID
再現: コマンド、保存した JSON/CSV、外部データのハッシュ
```

選択木に上げる前に、独立 oracle、境界・overflow・atom 移動、必要な勾配、
完全ステップ、メモリ、対象 GPU の再測定を確認する。速度差が揺れと同程度なら
勝者を断定しない。既定の選択を変える PR では、対象の evidence ID と
fallback をレビューできるようにする。

## 現時点で再利用できる知見

- `M` は線形層へ渡る総入力行数で、microbatch size と同義ではない。
- 5% atom 密度は生成される W の非ゼロ率を意味しない。
- GPU 上で候補を管理すると、訓練中の atom 移動や Graph replay に対応できる。
  候補容量の overflow と追加 workspace は方式ごとに検証する。
- 同じ GPU でも小 M と大 M の最良設定は変わる。kernel 単体の短縮が
  完全ステップの短縮になるとは限らない。
- 近似アンカー経路は速度だけで Exact CST の代替にしない。実タスクでの
  品質と長期学習の評価が別途必要。
