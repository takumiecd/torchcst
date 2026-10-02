# Benchmark と実験コードの整理（2026-10-02）

実験用の本体未採用実装を現在の検証ツリーへ維持する方式を廃止した。
`experiments/` の27 Python file、そこに依存する探索用 benchmark 20 file と
試作専用テスト7 file を削除した。現在の本体 backend 実装・登録・選択規則は変更しない。
research commit `5341ace` から旧実行コードを取り出せる。

benchmark は9 Python file（28 file から整理）とし、normalized の強制 plan / 公開 API /
wheel gate、既存 Strip/Torus の3つの診断、共通 fixture / oracle に絞った。
移行時点ごとの validation driver を `validate_wheel.py` に統合し、維持しているテストを
全て実行する。旧名の互換 module・CLI 引数は残さない。

独立 sampled-site oracle と mixed fixture を `reference.py` へそのまま移した。
backend の距離・正規化・重み生成を呼ばない独立の式を維持する。benchmark が test file を
runpy で読み込む仕組みを廃止し、test も fixture / oracle を直接 import する。
Strip/Torus model の構築を測定 CLI から `fixtures.py` に移した。

研究文書59 file を `docs/research-history/cuda-linear/` へ移し、数値・採否・負の結果を
保存した。raw evidence と既存 results は削除せず、履歴中のコマンドは記録当時の
source commit で再現する。今後の入口と測定の範囲は
[benchmark README](../benchmarks/cuda/linear/README.md)、削除対象は
[削除台帳](research-history/cuda-linear/retired-code.ja.md) を参照。

## 検証

CPU source と配布 wheel はともに481 passed / 117 skipped。削除した試作専用テストは
43 CPU pass / 84 CUDA skip に相当する。現在の本体のテストを減らすための変更ではない。
lint / format、6 CLI の CPU 上の --help、文書のローカルリンクも確認した。
旧 `5341ace` の oracle / mixed fixture / 正規化 Chart を4条件で照合し、
独立 oracle の W・勾配は bitwise 一致。Strip/Torus の初期 atom・y・dX・dP も一致した。

L4 の新しい wheel gate の結果は完了後に追記する。
