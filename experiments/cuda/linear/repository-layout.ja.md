# CUDA アルゴリズムと実験の配置

2026-10-01 追記。宣言と実行の境界、Kernel / Geometry / Chart の現在の移行状態は
[宣言と実行の配置規約](../../../src/torchcst/_backends/README.md)を基準にする。
以下の CUDA 配置は先行する段階的移行案であり、`nn/_backends/cuda/` は将来
`torchcst/_backends/cuda/` へ集約する。演算の契約は backend の外へ置く。

2026-09-29。これは [CUDA dispatch 設計](../../../src/torchcst/nn/_backends/notes/cuda-dispatch-design.ja.md)を実装する際の
ディレクトリ規約。公開 API は `torchcst.nn` のままにし、内部の CUDA 最適化を
アルゴリズム単位で読めるようにする。既存ファイルの一括移動は行わず、
実装を昇格させる単位で移す。

## 目標の形

```text
src/torchcst/nn/
  linear.py                         # 利用者向け CSTLinear
  conv.py                           # 利用者向け CSTConv2d
  _backends/
    __init__.py                     # 当面は既存 backend API との接続
    cuda/
      dispatch/
        schema.py                   # context、recipe、版、検証
        select.py                   # 最深の有効 plan と祖先 fallback
        load.py                     # 同梱設定を初回に読み、検証して固定
        registry.py                 # 安定 ID と Python 実装の明示的な対応
        tree.v1.yaml               # 選択木。採用済み plan ID だけを参照
      algorithms/
        mapped_streamed/
          README.md                 # 数式、対応範囲、制約、採否と evidence
          impl.py                   # 準備、forward、backward の接続
          kernels.py                # Triton kernel
          recipes.yaml             # 検証済みの名前付き設定
        fused_strip_torus/
          README.md
          impl.py
          kernels.py
          recipes.yaml
      approximate/
        anchor/
          README.md                 # Exact CST とは別の演算契約
          impl.py
          kernels.py
          recipes.yaml
tests/
  nn/cuda/dispatch/                # GPU 不要の木・schema テスト
  nn/cuda/algorithms/              # oracle、勾配、境界、GPU テスト
benchmarks/
  cuda/linear/
    run.py                          # 共通の正確性・速度・メモリ測定入口
    cases/                          # 利用者が数値を変えられる実験設定
    mapped_streamed.py              # アルゴリズム別の構築と oracle
    anchor.py                       # 近似として別の品質指標を測る
    dispatch-evidence.ja.md         # evidence ID の索引
    evidence/                       # 小さな測定結果。大きな trace は置かない
experiments/
  cuda/linear/                      # 採否未決の一時的な探索。昇格後は整理
  cuda/linear/notes/                # 実装に対応する過去の測定報告
```

これは**最終的な配置先**であり、空の雛形を先に大量生成する指示ではない。
`materialized`、`factored`、`tiled` の既存実装は、責務とテストを保ったまま
順次同じ形に寄せる。Conv は Linear の選択木に混ぜず、演算 ID と対応する
アルゴリズムを追加するときにディレクトリを設ける。複数方式で共用する
kernel だけを `cuda/common/` に置き、早まって共通化しない。

各アルゴリズムの `README.md` は、演算の意味、対応する chart・dtype・勾配、
workspace の上界、正確性の oracle、レシピの意味、既知の失敗例、代表的な
evidence ID、再現コマンドを記す。詳細な時系列の測定報告は実装の近くの
`notes/`、小さな測定結果は `benchmarks/cuda/linear/evidence/` に残し、README から相互リンクする。再現可能な
bench は `benchmarks/`、未整理の探索は `experiments/` に置き、どちらも配布
wheel に含めない。実装を開けば判断理由と対応する bench に到達できる。

bench の入力、出力、測定手順は[benchmark 運用](../../../benchmarks/cuda/linear/benchmark-workflow.ja.md)に記す。
最初に移す kernel は[暫定選別](cuda-kernel-shortlist.ja.md)で絞る。
`recipes.yaml` はレビュー済みの実行設定、bench の case は比較する仮説と
数値の上書き。bench の数値を変えても配布する `auto` は変わらない。

## 設定を読む流れ

1. `tree.v1.yaml` は、分岐 key、互いに重ならない条件、各節点の plan ID、
   evidence ID、木の revision を持つ。選択の優先順位は木の深さで決まり、
   ファイルの記述順では決めない。
2. 各 `recipes.yaml` は、そのアルゴリズムで検証済みの完全な設定を ID 付きで
   定義する。子節点で差分記法を許す場合も、ロード時に完全な plan へ展開する。
3. `registry.py` が実装 ID を Python callable に対応づける。YAML には import
   文字列、式、任意コードを書かない。未知 ID・schema 版、重複 ID、重なる兄弟、
   必要項目の欠落はロード時に拒否する。evidence ID が
   [台帳](../../../benchmarks/cuda/linear/dispatch-evidence.ja.md)から辿れるかは CI で確認する。
4. 同梱 YAML は `importlib.resources` で初回に読み、`safe_load` で解析し、
   型検証済みの不変オブジェクトにする。forward ごと、CUDA Graph replay ごとに
   ファイルを読まない。Tensor 値による CPU 同期も行わない。
5. selector は context のメタデータで最深の対応 plan を選ぶ。plan が非対応なら
   有効な祖先に戻し、選択 path と理由を診断で示す。実験結果の CSV/JSON を
   実行時に順位付けしない。

YAML を採用する場合、ローダ導入時に依存と wheel への設定同梱を明示し、
ビルドした wheel からの読み込みを確認する。設定ファイルはレビュー対象の
ソースであり、利用者が実行中に書き換える設定ではない。

## 今のリポジトリからの移行順

| 順 | 対象 | 完了条件 |
| --- | --- | --- |
| 1 | `src/torchcst/nn/_backends/_triton_preparation.py` から旧試作への import | 完了。boxed support kernel を本体へ移し、`src/` から旧試作への依存をゼロにした。 |
| 2 | 現行 `_backends` の選択 | `dispatch/` を追加し、既存 `auto` の結果を維持したまま木、検証、診断を導入する。明示 backend も維持する。 |
| 3 | `experiments/cuda/linear/block_streamed_backward.py` と関連 kernel | 開発候補として保管。実装者が方式を確定した後、一つのアルゴリズムとして本体へ移す。oracle、forward・`dX`・atom 勾配、CUDA Graph、メモリの証拠が揃うまで `auto` に加えない。 |
| 4 | `experiments/cuda/linear/anchor_atom_training.py` | 開発途中の近似演算として保管。品質と長期学習を評価するまで既定にしない。Exact の木から到達させない。 |
| 5 | 旧 `prototypes/` と報告 | 完了。開発用 44 モジュールを移し、97 ファイルを削除。歴史的な報告には元コードの commit と[参照方法](legacy-prototypes.ja.md)を記録した。 |

一つの方式を移す PR には、実装、短い README、名前付きレシピ、oracle、
benchmark case、evidence ID、選択木の変更を必要な範囲でまとめる。
探索段階の方式には木の変更を要求しない。`output/` は再生成可能な作業出力の
ままにし、選択の根拠に使う要約と小さなデータだけを保存する。
