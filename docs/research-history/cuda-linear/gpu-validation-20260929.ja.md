# CUDA 実機テスト: `0faed55`

2026-09-29。`0faed556bc7072d70166b0974fbbc72bb64a6698` の Git archive を
隔離ディレクトリへ展開して実行した。転送した gzip archive の SHA256 は
`dd67a706260fd8bc9abfbdfeecf06764c939e70d959087c1819b50a06dadb671`。
以下は正確性・互換性テストの結果であり、性能比較ではない。

| 実機 | 環境 | 実行範囲 | 結果 |
| --- | --- | --- | --- |
| GeForce RTX 3070 8 GiB | Python 3.10.20、PyTorch 2.6.0+cu126、Triton 3.6.0、pytest 9.0.3 | `python -m pytest -q --disable-warnings --maxfail=1` | 671 passed、1 skipped、270.44 秒 |
| A100 80GB PCIe MIG 3g.40gb | Python 3.12.9、PyTorch 2.6.0+cu126、Triton 3.2.0、pytest 8.3.4 | `PYTHONPATH=src:. python -m pytest -q -rs tests/test_fused_compute_configs.py tests/test_support_box_routing.py tests/test_batched_support_preparation.py` | 35 passed、63.28 秒 |

RTX 3070 での唯一の skip は
`test_default_fp_fusion_matches_explicit_dense_a100_tile`。A100 専用の
FP fusion 設定を検査するためで、A100 の対象テストでは実行して通過した。
本体へ移した boxed support routing と開発用経路の関連テストも両機で通過した。
GPU ごとの速度優劣や、未測定の GPU 世代への適用はこの結果から判断しない。
