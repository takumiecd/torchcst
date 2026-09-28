# 5% atom密度：メモリ優先の判定

入力行をまとめて局所重み窓を使い切る[続きの測定](five-percent-weight-first-streaming.ja.md)では、大きい入力バッチに限って全Wなし経路がdenseより低いピークに達した。

2026-09-28。`atom数 = round(0.05 × N × K)` を主条件とする。重みの非ゼロ率5%ではない。全重みを保持しないこと、および測定中のピークGPU tensor割当量が生成済みdenseより小さいことを実装候補の一次条件にした。速度はこの条件を満たす経路について改善する。

## 「生成済みdense」の意味

CSTのatomから論理重み `W[N,K]` を全要素生成して保持し、PyTorch/cuBLASの `F.linear(X,W)` を呼ぶ比較用経路である。測定時間にはW生成を含まない。4096²と8192²のFP32 Wはそれぞれ64 MiBと256 MiB。重み更新ごとに再生成する案でも、保持中の全Wは同じ容量を使う。どちらもメモリ優先の実装候補から外す。

`triton_streamed` は最大1024行のW窓だけを作って掛け、窓を再利用する。窓は16／32 MiB。ただし配置準備、atomのpacked表、並べ替え用領域など、窓以外のメモリも測る必要がある。

## 独立プロセスでのA100実測

A100 80GB PCIe MIG 3g.40gb（42 SM）、batch128、FP32、TF32無効、seed21。各モードで同じatom・入力・論理Wを用いた。比較用Wで正しさを確認した後、CSTモードではそのWを解放し、`torch.cuda.memory_allocated()` と `max_memory_allocated()` を測定した。各モードは独立プロセス。固定箱・列hintはwitness版の常駐量に含めた。全経路がcanonical Wの1,152点と全出力 `atol=rtol=3e-5` の検査に合格した。時間はCUDA Graphで測った参考値で、メモリ測定自体は通常forward呼び出し。

| 形状 | 経路 | 常駐割当 | forward中の追加ピーク | 合計ピーク | forward時間 |
| --- | --- | ---: | ---: | ---: | ---: |
| 4096² | 生成済みdense（比較用） | 74.12 MiB | 2.00 MiB | **76.12 MiB** | 0.663 ms |
| 4096² | fused（全Wなし） | 28.20 MiB | 56.00 MiB | 84.20 MiB | 14.138 ms |
| 4096² | streamed（全Wなし） | 28.20 MiB | 56.00 MiB | 84.20 MiB | 12.449 ms |
| 4096² | streamed＋witness（全Wなし） | 31.53 MiB | 55.20 MiB | **86.73 MiB** | 11.738 ms |
| 8192² | 生成済みdense（比較用） | 268.12 MiB | 4.00 MiB | **272.12 MiB** | 3.023 ms |
| 8192² | fused（全Wなし） | 84.39 MiB | 219.20 MiB | 303.59 MiB | 55.854 ms |
| 8192² | streamed（全Wなし） | 84.39 MiB | 219.20 MiB | 303.59 MiB | 49.848 ms |
| 8192² | streamed＋witness（全Wなし） | 97.69 MiB | 219.60 MiB | **317.29 MiB** | 47.088 ms |

常駐割当にはそのモードの重みまたはatom、入力、固定幾何と測定時点の生存tensorが含まれる。allocator reserved、CUDA context、勾配、optimizer状態、他層は含まない。生成済みdenseのW作成時間も含まない。witness版の固定箱とhintは3.33／13.30 MiBである。

**現在の全Wなし経路もピークメモリ条件を満たさない。** 速度最速のwitness版は、dense参考値より10.60／45.17 MiB多い。atom自体のパラメータは16／64 MiB（dense Wの25%）だが、forward時の配置準備の割当ピークが55／220 MiB程度ある。重み窓を小さくするだけではこのピークは解消しない。

次の優先順位は配置準備の一時領域を削ること。`route_and_layout` は全atomのキーに `torch.sort(stable=True)` を使い、前後にdecode、order、packed配列を持つ。実際のピークへの各配列の寄与は未分離なので、まずsort前後のallocationを計測し、同じ正しさと更新規則でキー分類・並べ替え・packingを少ないscratchに置き換える案を測る。**全Wを保持せず、合計ピークがdenseより小さく、かつforwardが速い**ことを採用基準とする。学習時はbackwardとoptimizerを含む別のピーク検証が必要。

測定ソース `5f1d7cd`、git archive SHA256 `d07ab6ad49dd76bc52fe52ccdc53c4a9d07a1e7187455f7577205cbe51a19f30`。A100側のarchiveと一致。結果は `output/triton-a100-20260928/mem-*.json` に保存した。

| JSON | SHA256 |
| --- | --- |
| `mem-4096-dense.json` | `a6a2c453bb4cd84b8c76fb5c7fa1a6378863af5b3d68705993fbc31e7e7d7023` |
| `mem-4096-fused.json` | `6985dac2d0ea14bc697b55421531f3d0b22cb36c22e2e1f9dc0e99ee907e068f` |
| `mem-4096-stream.json` | `d473994a1f206b9b4e0f594afd5bc2e4bcde668185e1b1198c4cfa8bf770aa19` |
| `mem-4096-stream-fast.json` | `ade3918d4d77f944bf63234606cc32291ca2e0e8b2d34696f712719a1a3fd961` |
| `mem-8192-dense.json` | `278d5640f313cc413c4b199d467c50e51a78a3c7f0f2bd0b21b1a01e1056e320` |
| `mem-8192-fused.json` | `72d850809c1f726eb08da1c92f6804e954c498f8eac93fc8ae4d31216f1e9bce` |
| `mem-8192-stream.json` | `9fb1683a3d0fbdcc45089ce38665f5d1d475566d78b5018007474734b8cfb156` |
| `mem-8192-stream-fast.json` | `5058e515b46093ab0bdccd2ee7e67f321383939530d58d565cfe1018988ba00c` |

```bash
PYTHONPATH=src:. python -m prototypes.benchmark_mapped_memory --size 4096 --mode dense --source-commit 5f1d7cd --output mem-4096-dense.json
PYTHONPATH=src:. python -m prototypes.benchmark_mapped_memory --size 4096 --mode fused --source-commit 5f1d7cd --output mem-4096-fused.json
PYTHONPATH=src:. python -m prototypes.benchmark_mapped_memory --size 4096 --mode stream --source-commit 5f1d7cd --output mem-4096-stream.json
PYTHONPATH=src:. python -m prototypes.benchmark_mapped_memory --size 4096 --mode stream_fast --source-commit 5f1d7cd --output mem-4096-stream-fast.json
PYTHONPATH=src:. python -m prototypes.benchmark_mapped_memory --size 8192 --mode dense --source-commit 5f1d7cd --output mem-8192-dense.json
PYTHONPATH=src:. python -m prototypes.benchmark_mapped_memory --size 8192 --mode fused --source-commit 5f1d7cd --output mem-8192-fused.json
PYTHONPATH=src:. python -m prototypes.benchmark_mapped_memory --size 8192 --mode stream --source-commit 5f1d7cd --output mem-8192-stream.json
PYTHONPATH=src:. python -m prototypes.benchmark_mapped_memory --size 8192 --mode stream_fast --source-commit 5f1d7cd --output mem-8192-stream-fast.json
```
