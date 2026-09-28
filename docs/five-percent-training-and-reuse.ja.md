# 5% atom：学習ステップの基準値と重み再利用

2026-09-28。A100 80GB PCIe MIG 3g.40gb（42 SM）、FP32、TF32無効、seed 21。新しい二つの測定は**異なるCST幾何**を使うため、速度を一つの学習経路として合算しない。

## mapped層：同じatom値で複数入力を処理する

`BlockStripLinear` のatom数を `round(0.05 × N × K)` とする5%を主条件にした。これは重みの非ゼロ率ではない。64×64 tile、入力128行。各入力は別の乱数tensor。同じatom値に対し、正しい現在の配置準備を毎回行う `stream_each`、配置だけ1回行う `stream_prepare_once`、現在の全論理重みを1回生成して使う `refresh_once`、あらかじめ生成済みのdense重みを読む `stored_dense` を比較した。`refresh_once` は配置準備と全重み生成を計時に含み、固定幾何・列hintの構築は含まない。atomを更新したら再生成が必要で、保持重みを更新後にそのまま使う計測ではない。CUDA Graph、各経路3 round、rep=20 msの中央値。canonicalな全atom評価を1,152点で確認し、全経路の全出力が `atol=rtol=3e-5` に合格した。

| 形状 | 入力数 | 毎回streamed | 配置1回・毎回streamed | 重み1回生成 | 生成済みdense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096² | 1 | 11.734 ms | 11.734 ms | 11.600 ms | 0.658 ms |
| 4096² | 2 | 23.446 ms | 21.404 ms | 12.259 ms | 1.315 ms |
| 4096² | 4 | 46.871 ms | 40.742 ms | 13.575 ms | 2.629 ms |
| 4096² | 8 | 93.740 ms | 79.416 ms | 16.203 ms | 5.267 ms |
| 4096² | 16 | 187.540 ms | 156.900 ms | 21.542 ms | 10.579 ms |
| 8192² | 1 | 46.781 ms | 46.761 ms | 46.700 ms | 3.023 ms |
| 8192² | 2 | 93.451 ms | 85.136 ms | 49.756 ms | 6.046 ms |
| 8192² | 4 | 186.859 ms | 161.898 ms | 55.758 ms | 12.129 ms |
| 8192² | 8 | 374.265 ms | 315.591 ms | 67.959 ms | 24.202 ms |
| 8192² | 16 | 747.421 ms | 622.444 ms | 92.121 ms | 48.398 ms |

8入力では重み再利用が毎回streamedより5.79倍／5.51倍速い。ただし生成済みdenseより3.08倍／2.81倍遅い。16入力ではdenseとの差が2.04倍／1.90倍まで縮む。重み生成の固定費を多くの入力で償却できるが、毎入力でatomを更新する場合には使えない。全重みを保持すると64／256 MiBを追加で占め、streamedの一時重み窓16／32 MiBより大きい。この測定はforwardのみで、backward中の再利用や学習時ピークを示さない。

## native Strip層：denseとAdamWの学習ステップ

勾配実装がある標準 `CSTLinear` を使い、同じ出力形状・入力・初期重み値のdense層と比較した。atom数は論理重み要素数の5%、batchは128。dense重みの初回生成は計測外。各ステップはforward、入力勾配、パラメータ勾配、`torch.optim.AdamW(foreach=True, lr=1e-3)`、新しいatom配置準備を含む。最初のoptimizer状態生成を別扱いとし、以後5回の同期wall timeの中央値。各モードを独立プロセスで実行した。初期重みのcanonical 1,152点と初期全出力の照合は通過した。optimizer後のCSTとdenseのパラメータ更新は表現が違うため一致を求めない。

| 形状 | dense AdamW step | native CST AdamW step | dense parameter | CST atom parameter | dense warmed step peak | CST warmed step peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024² | 0.531 ms | 110.584 ms | 4 MiB | 1 MiB | 40.3 MB | 18.7 MB |
| 4096² | 3.785 ms | 6468.363 ms | 64 MiB | 16 MiB | 358.9 MB | 144.0 MB |

メモリ列は独立プロセス内の `torch.cuda.max_memory_allocated()` で、モデル・初期化済みoptimizer状態・入力・勾配・一時領域を含む。allocator reserved、CUDA context、他のモデル層は含まない。5回のサンプルとbaselineからのピーク増分はJSONに残した。

このnative層は列方向を64列のstationに分けるmapped試作とは異なる幾何である。学習速度の数値を `triton_streamed` に外挿しない。mapped試作はforward専用で、同じ5%・4096²の正しい学習ステップ比較には専用backwardが必要である。native基準値は現在の標準学習経路の速度とメモリを示す。

## 判断

- 同じatom値に複数入力を通す場合、重みを一度生成する経路には大きな効果がある。更新ごとに全重みを生成する費用と64／256 MiBの保持を許容できる用途で検討する。
- dense並みの単発forwardを狙う場合、配置準備の改良だけでは足りない。現在の1入力では全重み生成込みでdenseの約18倍／15倍を要する。
- 学習での採否には、mapped層の `dX` とatom `dP` を実装し、forward・backward・optimizer・メモリを同じ幾何で再測定する必要がある。現行のforward試作の速度改善を学習速度と呼ばない。

測定ソース `b9619f2`、git archive SHA256 `a6965da0a55b668c71e5345bbfd8d1f21a741e7c76a01b60c139a97e078b8e38`。A100側の展開前archiveと一致。結果は `output/triton-a100-20260928/` に保存した。

| 結果JSON | SHA256 |
| --- | --- |
| `reuse-4096.json` | `e008a889c47365c71fd9875ffe10cc231b88032de16aaa41e1c8ad6486b2f679` |
| `reuse-8192.json` | `ac1af7fe09029066b989e3e7084914b222d92cb849f5e0a8bd14a58ecf3a4ce7` |
| `reuse-4096-16.json` | `7c9b796b14dd6c5a5ccde0310b65fcece71de1ba0238d9c302602d1c0859fc98` |
| `reuse-8192-16.json` | `271d9686a15ecce18cad099a7474a7880b64d39cf1074ef03a64ea2d1ae4394c` |
| `train-1024-dense.json` | `db7cddd39fd0cf29e65812a66a59a29024707d41f2f6c8f13ccc482847887993` |
| `train-1024-cst.json` | `27b8ad961fb25af5ee76760aaed294e4bd9a41cb8b6839a0e4435665a9cb9c96` |
| `train-4096-dense.json` | `d1a6bd1bc0749ed9cc67ba5d5da60335e691d66179927f5fed3694be4933bd43` |
| `train-4096-cst.json` | `740da635b4555dbc5dff902aad733ea02463832abb47536aae9aba48ce7de3e3` |

```bash
PYTHONPATH=src:. python -m prototypes.benchmark_mapped_reuse --size 4096 --source-commit b9619f2 --output reuse-4096.json
PYTHONPATH=src:. python -m prototypes.benchmark_mapped_reuse --size 8192 --source-commit b9619f2 --output reuse-8192.json
PYTHONPATH=src:. python -m prototypes.benchmark_mapped_reuse --size 4096 --microbatches 16 --source-commit b9619f2 --output reuse-4096-16.json
PYTHONPATH=src:. python -m prototypes.benchmark_mapped_reuse --size 8192 --microbatches 16 --source-commit b9619f2 --output reuse-8192-16.json
PYTHONPATH=src:. python -m prototypes.benchmark_five_percent_training --size 1024 --mode dense --source-commit b9619f2 --output train-1024-dense.json
PYTHONPATH=src:. python -m prototypes.benchmark_five_percent_training --size 1024 --mode cst --source-commit b9619f2 --output train-1024-cst.json
PYTHONPATH=src:. python -m prototypes.benchmark_five_percent_training --size 4096 --mode dense --source-commit b9619f2 --output train-4096-dense.json
PYTHONPATH=src:. python -m prototypes.benchmark_five_percent_training --size 4096 --mode cst --source-commit b9619f2 --output train-4096-cst.json
```
