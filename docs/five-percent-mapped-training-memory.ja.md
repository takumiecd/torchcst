# 5% mapped CST：学習ステップのピークメモリ

backwardの後続速度改善は[こちら](five-percent-backward-speed.ja.md)に記録した。
Adaで1024²を境界ケースとして測り直した結果と対象形状の優先順位は[こちら](five-percent-target-shapes.ja.md)。

2026-09-28。`A = round(0.05 × N × K)` 個のatomを持つ `BlockStripLinear` で、forward、入力勾配、atom勾配、AdamW更新まで実行した。CST経路は全重み `W[N,K]` と全重み勾配 `dW[N,K]` を保持しない。forwardと入力勾配では最大1024行、かつWの行数の半分以下の局所重み窓を使い回す。atom勾配では各16×16 site tileの `dW` を局所的に計算し、現在のatomへ加算する。実装は[backward試作](../prototypes/block_streamed_backward.py)と[測定コード](../prototypes/benchmark_mapped_training_memory.py)。

条件はA100 80GB PCIe MIG 3g.40gb（42 SM）、FP32、TF32無効、seed21、AdamW `foreach=True`。dense側はCSTから生成した初期Wをパラメータとして保持し、`F.linear` を使う。CSTの正しさ確認に使った全Wはメモリ測定前に解放した。各経路を独立プロセスで測り、optimizer状態生成後のステップで `torch.cuda.max_memory_allocated()` を取った。数値はGPU tensor割当量で、CUDA allocator reservedとcontextは含まない。時間はウォームアップ後1ステップの同期wall timeで、JITと初期W生成を含まない。

## 単一microbatchの結果

ピークと時間。メモリ単位は10進MB。Mは線形層に入力される総行数で、系列モデルのmicrobatch sizeと同義ではない。

| W形状 | M | denseピーク | CSTピーク | 削減 | dense時間 | CST時間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024² | 1 | 38.73 | 27.79 | 28.3% | 1.02 ms | 6.74 ms |
| 1024² | 128 | 40.29 | 29.87 | 25.9% | 1.27 ms | 6.66 ms |
| 1024² | 2048 | 68.21 | 61.45 | 9.9% | 2.32 ms | 9.93 ms |
| 4096² | 1 | 352.63 | 163.77 | 53.6% | 2.74 ms | 70.56 ms |
| 4096² | 16 | 353.37 | 164.75 | 53.4% | 2.51 ms | 71.10 ms |
| 4096² | 128 | 358.88 | 171.41 | 52.2% | 3.94 ms | 73.95 ms |
| 4096² | 2048 | 453.25 | 297.92 | 34.3% | 30.29 ms | 119.91 ms |
| 8192² | 1 | 1359.31 | 557.20 | 59.0% | 7.92 ms | 279.34 ms |
| 8192² | 128 | 1371.80 | 574.69 | 58.1% | 14.93 ms | 290.83 ms |
| 8192² | 2048 | 1560.54 | 825.93 | 47.1% | 119.06 ms | 479.75 ms |

小さい重み1024²ではM=2048の入力・出力・その勾配が重み差に比べて大きくなり、CSTの節約率が28.3%から9.9%へ下がる。一方、4096²・8192²ではM=1でもそれぞれ約189 MB・802 MB少なく、128行でも約187 MB・797 MB少ない。2048行でも約155 MB・735 MB少ないが、削減率は34.3%・47.1%へ下がる。**activationが支配的になると相対的な利点は薄れるが、形状次第で常駐メモリの削減は大きく残る。** この表は単層なので、深いモデル全体のactivationピークは示さない。

1024²の旧測定は1024行の一時窓が論理W全体と同じ形状だったため、全Wを作らないという条件に合わなかった。上の1024²のCST値は窓を512行に制限して再測定した値である。4096²・8192²の窓は元から1024行で、表の値は変わらない。

## 勾配蓄積はピークと速度の両方で判断する

同じ実効行数について、一度の大きいMと小さいmicrobatchの蓄積を比較した。蓄積実験は同じ入力bufferと上流勾配を繰り返し使い、上流勾配を回数で割った。異なるデータによる統計的な学習比較ではなく、メモリ寿命と処理費用の測定である。

| W形状・実効行数 | 実行 | denseピーク | CSTピーク | dense時間 | CST時間 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1024²・2048 | M=2048を1回 | 68.21 MB | 61.45 MB | 2.32 ms | 9.93 ms |
| 1024²・2048 | M=128を16回蓄積 | **41.34 MB** | **31.44 MB** | 3.72 ms | 82.63 ms |
| 4096²・128 | M=128を1回 | 358.88 MB | **171.41 MB** | 3.94 ms | **73.95 ms** |
| 4096²・128 | M=16を8回蓄積 | **353.89 MB** | 182.63 MB | 8.01 ms | 557.62 ms |

1024²では蓄積によってactivationが大きく減り、両経路のピークが下がる。ただしCSTは局所重みと配置を16回再生成し、時間が約8.3倍になる。4096²では、蓄積の2回目以降のforwardが保持済みatom勾配と重なるため、CSTのピークは単発128行より**11.22 MB増える**。microbatchを小さくすれば必ずCSTピークが下がる、という判断はできない。

## 速度の内訳と残る課題

128行の別runで同期フェーズを分けた。4096²はforward 12.64 ms、backward 61.32 ms、AdamW 1.01 ms。8192²は47.16／240.53／2.42 ms。ピークはいずれもbackwardで生じる。forwardは以前の全Wなし試作と同じ局所窓方式だが、trainable経路ではTorus復号をPyTorch autogradに通すため、融合decode版と同じメモリ・速度ではない。次はatom勾配の局所 `dW` と支持域評価の費用、およびbackward scratchを減らすことが主対象。

## 検証と適用範囲

初期Wのcanonical 1,152点と大形状の全出力をdense oracleと照合し、全件合格。入力勾配とatom勾配は128²、192²、およびatomを境界・idleへ移動させた128²で独立したPyTorch referenceと照合した。A100関連テスト6件合格。4096²・8192²の128行ではoptimizer更新後の入力勾配とatom勾配が有限かつ非ゼロであることも確認した。atomパラメータ／勾配／二つのAdamW momentは4096²で16／16／32 MiB、8192²で64／64／128 MiBだった。大形状の全atom勾配を独立oracleで照合したわけではない。幅パラメータの勾配は既存 `DirectAmpWidth` と同様にdetachする。atom勾配の加算順はatomicで非決定的であり、厳密な再現性が必要な用途にはまだ適用しない。準備処理でPythonの `torch.sort` を一時的に差し替える実装も試作用である。

性能値は単層・単一GPU・1サンプルの目安。AMP、別optimizer、長い系列、他層のactivationを含む全モデルのピークは未測定。source `058363e` のarchive SHA256は `3792e2b72bdaec8ca483d7e1d4d548feb23b67d1cad6c183b9e13ee30732ce98`。A100側で一致を確認した。phase計測source `528bee6` のarchive SHA256は `f85fe9e305b1c8387b06cb2ed71a0d5426de1635dd2c5d22dda2ae8b8596ea1c`。大形状の勾配有限性確認source `fd61809` のarchive SHA256は `04fa0f64f1172335ea92e491407394e99ae8467751050fd63946cb61a05c65e3`。結果JSONは `output/triton-a100-20260928/mapped-train-*.json`、`phases-mapped-*.json`、`gradcheck-mapped-*.json` に保存した。

1024²の窓制限後のsource `75dd106` のarchive SHA256は `eeabdbd9a972a9ca1bc149d20d780682bbf0630a837c6ec3495a880b2bc484ee`。A100側で一致し、関連テスト9件合格。結果JSONは `output/triton-a100-20260928/bounded-train-1024-*.json`。
