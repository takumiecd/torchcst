# 5% atom密度：全Wを作らずdense未満のforwardピークに到達

小さい入力行数と学習時のbatch設計は[追加検討](five-percent-batch-memory-strategy.ja.md)を参照。
[mapped学習ステップの測定](five-percent-mapped-training-memory.ja.md)ではbackwardとAdamWまで比較した。

2026-09-28。ここでの5%は `A = round(0.05 × N × K)` 個のCST atomを意味し、論理重みの非ゼロ率ではない。A100 80GB PCIe MIG 3g.40gb（42 SM）、FP32、TF32無効、64×64論理tile、seed21で測定した。CST経路は全重み `W[N,K]` を作らず、1024行以下の局所窓を作り、入力行全体へ適用して再利用する。

## 到達点

比較のdense側は、生成済みの全Wを保持して `F.linear` を呼ぶ。W生成時間は含まない。CSTとdenseは独立プロセスで測り、正しさの確認に使ったWはCSTのメモリ測定前に解放した。ピークは `torch.cuda.max_memory_allocated()` によるGPU tensor割当量で、CUDA context、allocator reserved、backward、optimizerは含まない。時間はCUDA Graph中央値。

| 論理重み | 入力行 | denseピーク | CSTピーク | dense時間 | CST時間 | CSTの全出力照合 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4096² | 128 | 76.12 MiB | **73.99 MiB** | 0.663 ms | 11.396 ms | 合格 |
| 8192² | 128 | 272.12 MiB | **265.54 MiB** | 3.023 ms | 45.082 ms | 合格 |
| 4096² | 2048 | 136.12 MiB | **130.39 MiB** | 9.582 ms | 21.532 ms | 合格 |
| 8192² | 2048 | 392.12 MiB | **330.74 MiB** | 37.197 ms | 83.261 ms | 合格 |

全出力はCSTから作ったdense oracleと `atol=rtol=3e-5` で比較し、違反0。canonical Wの1,152点も合格。128行でのピーク削減幅は4096²で2.13 MiB、8192²で6.58 MiBとまだ小さい。時間はdenseの約17.2倍／14.9倍で、速度目標には届いていない。2048行ではどちらも約2.24倍。

## ピークを下げた二つの変更

1. `decode_intrinsic_torus_compact` はTorusの中心座標を1つのTriton kernelで復号し、PyTorchの大きな中間tensorを省く。通常のTriton除算は大きな主半径を掛けた後に中心座標の誤差を生み、4096²の全出力照合に失敗した。`tl.div_rn` を使うと円周方向の2座標がPyTorchとbitwise一致し、4096²・8192²の全出力照合を通過した。
2. `atomic_bucket_sort` はbucketを数えてprefix sumを取り、atom IDを原子加算で配置する。現在の実装は試作で、`torch.sort(stable=True)` を測定時だけ差し替える。bucket内の順序は非決定的だが、各atomの配置は一意で、offsetsは既存経路と一致する。atomを移動させた小規模テスト、permutation検査、5%の全出力検査を通過した。

準備処理のみのピークは、4096²／8192²で従来の復号を使うと79.4／310.0 MiB、復号融合後に安定sortを使うと75.7／298.6 MiB、復号融合と原子bucket配置を使うと**65.9／257.0 MiB**。復号融合後にピークがsortへ移ることを段階計測で確認した。`argsort(stable=True)` と `sort(stable=False)` は、この条件では元の安定sortとピークが同じだった。

## 速度の残り

段階別のCUDA Graph計測では、8192²・128行の45.1 msのうち、atomの配置準備が約6.7 ms、局所重み窓生成が8窓で約35.8 ms、GEMMが合計約2.6 ms。2048行でも窓生成はほぼ同じ約35.8 msで、GEMMが約40.9 msへ増える。次は**局所重み生成で各atomとsiteを評価する回数**を減らす必要がある。atomレーンを2〜8本に増やす方法は4096²・8192²とも既定より遅く、factored kernelの16×32、32×32、32×64、64×64 tileも8192²では64×32の約4.47 ms／窓を改善しなかった。

窓を1024行から256行にすると、128行入力で4096²・8192²ともピークは変わらず、時間だけ11.396→12.594 ms／45.082→47.266 msになった。2048行・8192²ではピーク330.74→325.54 MiB、時間83.261→85.558 ms。窓サイズより配置準備が小バッチのピークを決めている。

## 適用範囲と次の設計

これは**forward専用の試作**である。融合decodeと原子配置はautograd経路を持たず、学習のbackwardとoptimizerを含むピークは未測定。原子配置のbucket内順序は実行ごとに変わりうるため、厳密な再現性が必要な学習用途へ直接採用しない。現在のPython側 `torch.sort` 差し替えも正式APIではない。以前の[支持域分析](five-percent-mapped-gemm.ja.md)では32列boxで除外できたatom×列小片は約0.3%だけだった。次は単純なbox除外より、atom側からsiteへ寄与を集める方式や距離計算の再利用を優先して検証する。全Wを保持しない条件は維持する。

モデル相談ではAstra、Gemini、Claudeから配置・窓再利用・融合処理の案を得た。ただし実際のピーク位置はA100計測で判断した。Grok 4.7 highへの `cursor-agent` 質問は時間切れで返答を得られなかった。外部提案だけで実装を選ばず、候補をA100で測った。

測定ソースは `de022df`（forward）、`efd1bcc`（配置準備）、`ec7f205`（段階別）、`9ac4e95`（tile比較）、`4441a6e`（seed追加）。forward測定のarchive SHA256は `85a08b9edce506a4ad40b5364a98fd215929d97bc4f7f4ef3210fd440686a21b`。A100側と一致を確認した。seed22・23の8192²も全出力照合に合格し、ピーク265.54 MiB。A100の `tests/test_support_box_routing.py` と `tests/test_streamed_materialization.py` は3件合格。詳細JSONは `output/triton-a100-20260928/atomic-forward-*.json`、`atomic-*.json`、`stages-*.json`、`scan-*.json`、`seed-*.json` にある。
