# 測定 JSON

任意の Colab 測定補助Actionsの `request_file` に、以下の既存ファイルのパスを指定する。
GPU・Case・Plan・回数は JSON に固定され、Actions 側で個別に上書きしない。

| ファイル | GPU | 独立測定 |
| --- | --- | --- |
| `benchmarks/requests/colab-linear.json` | L4 | 1回 |
| `benchmarks/requests/colab-linear-t4.json` | T4 | 1回 |
| `benchmarks/requests/colab-linear-a100.json` | A100 | 1回 |
| `benchmarks/requests/colab-linear-h100.json` | H100 | 1回 |
| `benchmarks/requests/colab-linear-g4.json` | G4 / Blackwell | 1回 |
| `benchmarks/requests/colab-linear-l4-g4.json` | L4、G4を順番に | 各1回 |

全て同じ1024²・通常 sigma-three Case の full / window512 と dense 参照を比較する。
正しさ・全勾配・完全な training step の時間・capture/replay の peak memory を記録する。
機種によってモデルや精度条件を変更しない。

選んだ GPU を Colab が提供できるかは、利用者のプラン・残量・空き状況による。
別 GPU へ自動変更しない。これらの追加 JSON は設定として用意したもので、
全ての GPU 上で実機検証済みという意味ではない。

新しい条件は JSON / Case / Plan を PR で main に追加してから選ぶ。
[参加・設定手順](../../docs/benchmark-contributions.ja.md)を参照。

測定補助Actionsはartifact保存まで。各完成 `artifacts/benchmark.json` を取り出し、
同じrepositoryの提出Issueに添付する。ローカルGPUでの測定にも同じ提出フォームを使う。
