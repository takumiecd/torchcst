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
43 CPU pass / 84 CUDA skip に相当する。削除したテストは未採用の試作だけを対象とし、本体のテストは維持した。
lint / format、6 CLI の CPU 上の --help、文書のローカルリンクも確認した。
旧 `5341ace` の oracle / mixed fixture / 正規化 Chart を4条件で照合し、
独立 oracle の W・勾配は bitwise 一致。Strip/Torus の初期 atom・y・dX・dP も一致した。

L4 の新しい wheel gate は **597 passed / 1 skipped / 0 failed**。
Linux では使えない macOS socket binding のテスト1件を skip した。
source は `25f9c3a`、job は `l4job-0eebb834bd0f404e80f78354e0fdd4db`。
Torch 2.11.0+cu128 / CUDA 12.8 / Triton 3.6.0 / NVIDIA L4 / driver 580.82.07。
job-local 環境を使用し、system probe の Torch 2.11.0+cu130 / CUDA 13.0 と区別する。
未使用の共有 RAPIDS/CUDA 13 package との pip 依存警告はあるが、system package は
変更していない。対象の Torch/Triton テストと測定の範囲を記録する。

独立 FP64 oracle の W / y / dX / 全5 atom 勾配、鋭い支持・floor / clip、
中心・幅の変更後の Graph replay と AdamW の Parameter / 状態を確認した。
result archive の SHA256 を receipt と照合し、152 Python file を source / wheel /
installed の間で byte 比較した。最初の CPU-tested local wheel は最後の README 修正だけ
異なるため、README 修正後に再構築し、GPU-tested wheel と archive SHA256 まで一致した。
Python file に差はなく、docs だけの変更による CPU テストの再実行は行っていない。

N1024² / M128 / 52,429 atoms（約5%）/ FP32 / TF32無効 /
fused capturable AdamW lr1e-4・weight_decay .01。各条件は別 process。
時間は forward・backward・optimizer を含む Graph replay の同期 wall time 中央値。
peak は Graph capture から測定する。

| 条件 | 経路 | Graph ms | peak allocated MiB | peak reserved MiB |
| --- | --- | ---: | ---: | ---: |
| 通常 sigma3 | full | 0.648171 | 51.8511 | 106 |
| 通常 sigma3 | window | 1.543463 | 46.1655 | 86 |
| 通常条件の比較 | dense | 0.129759 | 50.5024 | 106 |
| 鋭い支持 | full | 0.557544 | 51.8511 | 106 |
| 鋭い支持 | window | 0.306782 | 46.1655 | 86 |
| 鋭い条件の比較 | dense | 0.128805 | 50.5024 | 106 |

GPU process usage は未測定。大きい fixture は完全 step の測定であり、全 atom の
独立勾配 oracle ではない。同じ profile の full/window 初期 atom SHA は一致する。
既存結果との速度差を algorithm 改善・採用の根拠にはしない。owned GPU は停止済み。
[機械記録と hash](../benchmarks/cuda/linear/results/benchmark-cleanup-20261002.json)を参照。
