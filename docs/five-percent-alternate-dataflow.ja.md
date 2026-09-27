# atom 密度 5%：小分け重み生成と GEMM の分離

2026-09-27。前回の [X 再利用と atom 群](five-percent-atom-group-and-x-reuse.ja.md) の後、5% を主対象に計算経路を変えた。結論は、列方向の距離項を共有して論理重みを **1024 出力行ずつ**作り、その都度 cuBLAS GEMM に渡す明示的な `triton_streamed` 試作 backend が最も有望だった、というもの。dispatch は変更していない。この backend は forward 専用であり、学習全体の高速化を示す結果ではない。

## 条件と主結果

A100 80GB PCIe MIG 3g.40gb（42 SM）、FP32、TF32 無効、seed 21、入力 128 行、64×64 tile、Triweight、atom 数 `round(N² × 0.05)`。各形状で同じ atom・入力・dense oracle を用い、CUDA Graph 3 round の中央値を比較した。`full` は毎回 `prepare` を含む。出力全体を atol=rtol=3e-5 で照合し、全候補が合格した。生成済み dense W の速度は重み生成を含まないため、主比較には使わない。

| 形状 | 現行 `triton_fused` full | `triton_streamed` full、1024行、重みタイル64×32 | 短縮率 | 一時重み |
| --- | ---: | ---: | ---: | ---: |
| 128×4096×4096 | 14.189 ms | 12.461 ms | 12.2% | 16 MiB |
| 128×8192×8192 | 55.962 ms | 49.883 ms | 10.9% | 32 MiB |

CUDA Graph を使わない通常呼び出しの同期 wall time 中央値も 4096² は 14.308→12.650 ms（11.6%短縮）、8192² は 56.180→50.095 ms（10.8%短縮）。通常呼び出しのサンプルは各経路 10 回連続を 3 round、順序を交互にした。入力/モデル/出力以外で単発 forward 時に増えた PyTorch allocated bytes のピークは、4096² で双方 58,721,280 byte、8192² で双方 231,106,560 byte。この値は測定 run の既存 tensor を差し引いた増分であり、学習全体のピークや allocator reserved bytes ではない。

`triton_streamed` は `prototypes/block_strip_linear.py` から明示的に選択できる。既定の重み生成 tile は64×32、重み窓は1024行。4D の chart、64×64 tile、行数が64の倍数という条件に限定した forward-only 試作で、勾配が必要な呼び出しは拒否する。重み窓を生成して同じ buffer を使い回し、`torch.mm` で出力の該当列へ直接書く。atom 更新後の CUDA Graph 再実行を含む A100 テスト `tests/test_streamed_materialization.py` は 1 件合格した。これを既定 dispatch に組み込んでいない。

## 経路の内訳とメモリの調整

5% で現在の fused kernel は 4096² の prepared forward のうち約11.4 ms を占め、`support_buckets_batched` は約2.0 ms。fused kernel は167 registers/thread、spill なし。論理重み生成を別 kernel に分けると、列座標だけで決まる距離の z/w 項を atom×列について一度だけ評価でき、X の再利用は cuBLAS に任せられる。

最初の全重み materialization（32×32タイル）は生成のみ 9.276/36.955 ms（4096²/8192²）、cuBLAS を含む prepared forward は 9.969/39.972 ms、full は 12.743/51.336 ms。全重みの一時領域は64/256 MiB。小分け方式なら 1024 行で16/32 MiB、512 行では8/16 MiBとなる。512 行の full は13.066/50.567 msで、メモリを半分にすると速度が少し落ちる。64行まで細かくすると launch/GEMM の繰り返しが増え、4096² の prepared で14.481 msとなった。

重み生成の tile を32×32から64×32へ変えると、全重み prepared の生成＋cuBLAS は 4096² で9.942→9.560 ms、8192² で39.883→38.463 ms。小分け full でも、4096² の1024行では32×32が12.850 ms、64×32が12.461 ms、64×64が12.790 ms。8192² では51.284/49.883/49.872 ms。64×64の単独生成は速いが register 使用が168/thread まで上がり、4096² の小分け full では64×32より遅いため64×32を試作の既定値にした。64×32は96 registers/thread、spillなし。

## 別案の測定

- 8 atom を近接順に並べた理想群で site の箱との距離下界を判定すると、4096² prepared kernel は現行32行11.447 ms、群内分岐11.195 ms、事前CSR候補列挙10.271 ms。全出力は合格。ただし群作成がCPUで10.77秒、CSR作成が0.156秒かかり、timed forward から除外した。CSRは候補群の77.45%を保持した。このままでは full forward に採れない。
- BF16 を high/low に分けた3項・4項の Tensor Core dot は正しさに合格したが、4096² full は14.971/14.983 msで現行14.166 msより遅い。実験 kernel は取り除いた。
- fused kernel の batch tile BM を128から64/32へ減らすと、4096² full は14.215→23.205/41.465 msに悪化した。8 warps も16.832 msで遅い。これは別資料の出力行 tile BN=32 の選択とは別の sweep。
- Torus 上の chord から support row を予測し近傍だけ検査する準備案は、出力と準備済み tensor が一致したが、4096² の準備時間が2.779→6.111 ms、full が14.220→17.554 msに悪化した。実験コードは履歴に残し、現行実装から除いた。

## 再現

試作 backend とテストは commit `b589248`、最終タイル比較は `b93bef4`。隔離した git archive の SHA256 はそれぞれ `591a63ad31810e570677a219b80258f0c617992cb489b88354b3506ca6adc149` と `6d13e18e018aa2a7967e071faaf89ed2801bf25bbfa03dd8bcf12199ca7df000`。いずれもA100側で一致を確認した。結果 JSON は gitignored の `output/triton-a100-20260927/` にある。

| JSON | SHA256 |
| --- | --- |
| `profile-4096.json` | `e0153e4962e7cc1d89b9e58fecae044987b95a43c6ff83af5daf1c1bca599722` |
| `bf16-4096.json` | `db8b3afa2adb6a71f5280a8043340d9d81b1ee8ad510a297685ea0db499ef70d` |
| `warps-4096.json` | `caf56a29b737a2afa5998dfae9295ebe6a7eb36a4260ae09215e093ba4e00ea8` |
| `chord-4096.json` | `2e2a8ff59ac4746118c0f4eb44a30948c685690a98aa1e91dbc8d984f9de2291` |
| `grouped-csr-4096.json` | `3b8fbeb230c8297087fd3f7febed3b66ec1f4c2116e5ec8033a333fd1e112977` |
| `materialize-4096.json` / `materialize-8192.json` | `df0f26d141104343702887766b923973d4f9b8c9c38d6028b8a5b1a7588c22dc` / `c87d8d06bf4d2128caac09ae3fcd919c64ea14c001695b00c922b985ab230407` |
| `factored-sweep-4096.json` / `factored-sweep-8192.json` | `f7f8e967a10aa16ee7198fe3addd9bce1e7a8e2d98d06305d2f86cffb4efdb15` / `a3e99e09f689c919dc58ffbad1a00a9edd34649221e3417435176c1aa5390173` |
| `streamed-tile-4096.json` / `streamed-tile-8192.json` | `322671c5619be7f44bb66e52c8c353765d704e52e5fbe36264b88685f40c6f48` / `2b90da578279adea6fcc3d6e8d8e585bc47b99d7114a76b8b2e71405d5397c63` |

隔離 checkout で次を実行する。最後の pytest は CUDA/A100 が必要。

```bash
PYTHONPATH=src:. python -m prototypes.benchmark_streamed_materialize \
  --size 4096 --eager --source-commit b93bef4 --output streamed-tile-4096.json
PYTHONPATH=src:. python -m prototypes.benchmark_streamed_materialize \
  --size 8192 --eager --source-commit b93bef4 --output streamed-tile-8192.json
PYTHONPATH=src:.:tests python -m pytest -q tests/test_streamed_materialization.py
```

次の技術課題は backward と学習時ピーク、学習後に移動した atom の分布、長い系列で重み窓を使い回せる条件の確認。CSR案を採るなら群と候補を更新するGPU処理の総時間を full forward に含める必要がある。
