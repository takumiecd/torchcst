# 5% CST：学習時の入力行数とピークメモリ

その後、同じmapped幾何でbackwardとAdamWまで実行した[学習ピーク測定](five-percent-mapped-training-memory.ja.md)を追加した。以下のforward表と当時の設計判断はその前段階の記録。

2026-09-28。ここでの5%は `A = round(0.05 × N × K)` 個のatom数を指す。この時点のmapped `BlockStripLinear` のCST経路はforward専用で、以下の表は**学習ピークではない**。生成済みdenseとの比較では、dense側だけが全重み `W[N,K]` を保持する。CST側は全Wを作らず局所重み窓を再利用する。A100 80GB PCIe MIG 3g.40gb、FP32、TF32無効、seed21。各経路を独立プロセスで測り、全出力はdense oracleと照合した。GPU tensor割当ピークであり、CUDA contextとallocator reservedは含まない。

## 「batch size」ではなく行列に入る総行数

ベンチマークの `M=128, 2048` は線形層へ入力する行列 `X[M,K]` の**行数**。系列モデルで時系列をflattenすれば通常 `M = microbatch size × sequence length` となる。同じMでも、深いモデルのactivation保持量は層数、系列長、attention方式、checkpointingに左右される。実効的な学習batchは、勾配蓄積を使う場合 `microbatch size × 蓄積回数 × データ並列数` であり、1回のforwardのMとは別に設計できる。

## 小さいMも含めたforwardピーク

単位MiB。差は `dense − CST` で、正ならCSTが少ない。時間はCUDA Graph中央値。128行と2048行は[前回の測定](five-percent-compact-native.ja.md)、1・16・64行は同じforwardソース `de022df` で追加測定した。

| W形状 | M | denseピーク | CSTピーク | 差 | dense時間 | CST時間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096² | 1 | 72.156 | 73.480 | −1.324 | 0.089 ms | 10.729 ms |
| 4096² | 16 | 72.625 | 73.714 | −1.089 | 0.195 ms | 10.882 ms |
| 4096² | 64 | 74.125 | 74.464 | −0.339 | 0.449 ms | 11.144 ms |
| 4096² | 128 | 76.125 | 73.990 | +2.135 | 0.663 ms | 11.396 ms |
| 4096² | 2048 | 136.125 | 130.389 | +5.736 | 9.582 ms | 21.532 ms |
| 8192² | 1 | 264.188 | 261.571 | +2.617 | 0.327 ms | 42.730 ms |
| 8192² | 16 | 265.125 | 262.040 | +3.085 | 0.701 ms | 43.272 ms |
| 8192² | 64 | 268.125 | 263.540 | +4.585 | 1.304 ms | 44.148 ms |
| 8192² | 128 | 272.125 | 265.540 | +6.585 | 3.023 ms | 45.082 ms |
| 8192² | 2048 | 392.125 | 330.739 | +61.386 | 37.197 ms | 83.261 ms |

小さいMなら必ずCSTが少ない、とは言えない。4096²は64行まで現行の配置準備メモリが重み削減を上回り、128行で逆転する。8192²は1行でもCSTが少ない。大きいMでactivationが増えても、この単層forwardではCSTの**絶対的な削減量**は消えず、2048行ではむしろ大きい。ただし深いネットワークのactivationが圧倒的な場合、削減量の総ピークに対する比率は小さくなる。どの演算フェーズでピークが起きるかも重要である。

## 学習で期待できる部分と未確定な部分

このmapped試作のatomパラメータは1 atomあたりFP32で5値なので、5%ではパラメータ要素数がdenseの約25%。FP32のパラメータ、勾配、AdamWの二つのmomentを同じ比率で保持できれば、4096²の1層でdense約256 MiB対CST約64 MiB、8192²で約1024 MiB対約256 MiBとなる。これは**常駐する4配列だけの容量見積もり**で、幾何、activation、routing scratch、backward中間値、optimizer一時領域を含まない。多層ではこの常駐差が各層に積み上がる可能性がある。

既存の勾配対応native `CSTLinear` では4096²・batch128・AdamWの学習ステップピークがdense 358.9 MB対CST 144.0 MBだった。ただし[既存測定](five-percent-training-and-reuse.ja.md)のnative Stripはmapped `BlockStripLinear` と異なる幾何で、速度もCST 6468 ms対dense 3.785 msだった。この値の外挿は避け、その後にmapped経路自体の[学習測定](five-percent-mapped-training-memory.ja.md)を実施した。

native Stripでもmicrobatchを1・16・128行に変えて学習を測り直した。4096²のdense／CSTピークは1行で352.63／138.97 MB、16行で353.37／139.46 MB、128行で358.88／143.97 MB。1024²では1行の38.73／16.62 MBから2048行の68.21／50.29 MBへ増え、CSTの削減率は57.1%から26.3%へ低下した。4096²の16行×8回の勾配蓄積ではCSTピークが158.18 MBとなり、128行一括の143.97 MBより高い。2回目以降のforwardで既存のatom勾配と準備領域が重なるためである。source `5022253`、archive SHA256 `afa6b1dec5952e881a65c1b19c7c323effad923e81836b5a3d00f972970fa68d`。A100側で一致を確認し、結果は `output/triton-a100-20260928/train-{1024,4096}-m*.json` に保存した。

## 開発上の判断

1. **主評価は小さいmicrobatchの学習ステップ**に置く。Mは1、16、64、128を含め、forward、backward、optimizer、activation保存を同じモデル・同じ更新規則で測る。目標は「denseを数MiBだけ下回る」ではなく、常駐パラメータ・勾配・optimizer状態の削減が総ピークに実際に現れること。
2. **大きいMも維持する。** 512〜2048行では局所重み窓を入力行に再利用でき、速度差が縮む。大きい実効batchが欲しい場合はmicrobatchと勾配蓄積を分けて設計し、activationのピークと窓の再生成費用を両方測る。
3. 単層だけでなく、実際に想定する層数・系列長を固定したend-to-endモデルで、ピークの瞬間に何が生存しているかを記録する。activationが支配的ならcheckpointingなどの影響も同じ条件で比較する。

追加測定JSONは `output/triton-a100-20260928/small-{4096,8192}-{1,16,64}-{dense,cst}.json`。全12件でsource commit、A100、5% atom数、完了flag、canonical照合、全出力照合を確認した。ソースarchive SHA256は `85a08b9edce506a4ad40b5364a98fd215929d97bc4f7f4ef3210fd440686a21b` で、A100側と一致した。
