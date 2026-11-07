# 大きな幅での応答再計算

[English](large-recompute-20261108.md) | **日本語**

同じA100 MIG 3g.40gbでR=256、正方N=8192/16384、A=8Nの応答保存と
再計算を比較する。振幅・半径の法則、初期化、FP32演算、TF32無効、
optimizerは変更しない。追加Algorithmは既存の再計算実装を検証済みの
大規模正方スコープへ宣言するだけで、benchmark限定の明示選択とする。
この宣言に新しいCUDA数値計算コードはない。

幅比較で検証した凍結P/x/target/dYを使う。N=2048の検証用payloadも使う。
prefix adapterはpayloadハッシュと再生成したP・初期化校正を確認して、
既存Linear workerを呼ぶ。コミット済みソースの新しいスナップショットと
独立した拡張ビルドで識別する。CUDA回帰後、N2048再計算の正しさ・optimizer
検証を実施する。大きな幅でも初期・更新後の全要素FP64 VJP検証を維持する。

各arm・幅で独立プロセス2回。Aは幅8192→16384、保存→再計算→Dense、
Bは幅16384→8192、Dense→再計算→保存。各10 warmupと51計測の同期した
完全stepを実行する。zero_grad、dX消去、forward/MSE、backward、optimizerを
含め、準備・CPU検証・結果保存を除く。allocatorピークは全61stepを含む。
allocatedとreservedを別記し、プロセス全体のメモリとは呼ばない。
正しさ失敗・非有限値・OOM・timeoutで停止し、証拠を残し自動再試行しない。
状態ハッシュ、保持方式の保存領域、全観測を比較する。長時間学習・LM全体の
精度・CUDA Graph・別デバイスについては主張しない。

再計算ではforwardのZ/H保持を解除するが、backwardでは逆向き中間値が
生きたままZ/Hを再生成する。単独の大きなLinearでは一時領域のピークが
変わらない可能性がある。節約を仮定せず交換条件を測る。多層LMでの以前の
結果だけでは、このピークは決まらない。

これは計測前のプロトコルであり、監査後に結果を追記する。

## 作業領域の内訳

正方chart・M=8・direct levels2では、FP32 moment一つが
`Z_bytes = 4 R (N-2) (2M+1)`、応答一つが`H_bytes = 4 R A`。
勾配partialは`12 A ceil(R/8)` byte、geometryは`100 A` byteである。
N16384/A131072/R256ではZ一つ271.967MiB、H/G一つ128MiB、partial48MiB、
geometry12.5MiB。両向き分を合わせて約860.434MiBになり、さらに入力・出力・勾配・
optimizer・owner metadataが加わる。通常の再計算もbackwardピークでは同じ作業領域を
再生成する。これはshapeから求めたTensor payloadで、allocatorピークの実測や
GPUプロセス全使用量ではない。
