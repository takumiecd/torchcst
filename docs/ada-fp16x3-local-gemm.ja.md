# Adaの局所Wに対するFP16分解GEMM

2026-09-28。RTX 6000 Ada、CST 8192×8192、atom密度5%、64×64幾何タイル、1024行の局所W窓。学習経路は全W・全dWを保持しない。試作コードは`prototypes/bounded_gemm_fp16x3.py`。既定のdispatchは変更していない。

## 計算方法

入力のFP32値`a`と局所WのFP32値`b`をレジスタ内で`ah=fp16(a)`、`bh=fp16(b)`、`al=fp16(4096*(a-ah))`、`bl=fp16(4096*(b-bh))`へ分ける。Tensor Coreで`ah@bh`、`ah@bl`、`al@bh`を計算し、FP32累積して`ah@bh+(ah@bl+al@bh)/4096`を出す。`al@bl/4096²`は省略する。分解した成分の全行列はメモリに作らず、各GEMMタイルのレジスタ内だけで使う。

補正成分を拡大しない初版は人工Wでforwardの3e-5許容を約13.3万/209.7万要素で外した。4096倍すると同条件で0件になった。ただしFP16の表現範囲と省略項に依存する近似であり、任意の入力値に対するIEEE同等性は保証しない。

## 5% CST学習での選択

`fp16x3_dx`はforwardと入力勾配`dX`に新GEMMを使い、atom勾配用の局所`dW`にはIEEE `torch.mm`を使う。局所dWまで新GEMMにした`fp16x3`は、seed 21でatom勾配の3e-4許容を10,421/16,777,215要素で外したため採用しない。`dW`は1024×8192の窓で逐次計算し、atom勾配へ畳み込んでバッファを再利用する。

seed 21/22/23、M=2048のIEEE学習経路との比較では、出力・`dX`の許容`3e-5+3e-5|reference|`違反は0。atom勾配は`3e-4+3e-4|reference|`違反が0。seed 22/23のatom勾配では、さらに厳しい3e-5基準に25/8要素が外れたが、同じseed 22でIEEE経路を二度実行しても13要素外れた。atom勾配kernelのatomic加算順序に伴う揺れと整合する。seed 21の`dX`相対L2誤差は1.77e-6、atom勾配は2.85e-8。

## 速度とメモリ

完全なAdamWステップを同一プロセス・同一モデルで交互に10回ずつ実行した。配置準備、forward、backward、optimizerを含む。両経路とも局所dWはIEEE。

| 8192²・5% | 対照 | 新方式 | 中央値の差 | 対応ペアの速度比中央値 |
| --- | ---: | ---: | ---: | ---: |
| M=2048、対照TF32x3+IEEE dW | 46.25 ms | 38.81 ms | 16.1%短縮 | 0.827 |
| M=128、対照IEEE | 17.72 ms | 18.86 ms | 6.5%遅い | 1.036 |

交互実行でも実行時間は揺れる。独立プロセスの5回中央値は`fp16x3_dx`が43.62 msで、反復中に51.85→38.47 msへ変動した。したがって性能判断には交互実行を優先する。M=128はIEEEを維持し、M=2048をAda向け新方式の候補とする。M=2048の新方式を通した単独ステップの`torch.cuda.max_memory_allocated()`は839,612,416バイト（約801 MiB）。従来のAda dense対照の約1561 MBより低い。学習実装は全W・全dWを作らない。メモリ測定用スクリプトは開始前に正しさ確認用のdense oracleを一度作り、削除後にピーク統計をリセットする。

## 再現

単一窓比較: `python -m prototypes.profile_bounded_gemm_fp16x3 --size 8192 --rows 2048 --output ...`。

実CSTでの勾配比較: `python -m prototypes.compare_bounded_gemm_gradients --size 8192 --rows 2048 --seed 22 --compare-mode fp16x3_dx --forward-gemm-mode fp16x3 --materialize-mode listed --output ...`。

交互学習ステップ: `python -m prototypes.profile_paired_step_gemm --size 8192 --rows 2048 --rounds 10 --output ...`。M=128比較では`--rows 128 --control-mode ieee`。

測定JSONは`output/ada-20260928/cache-hypotheses/`に保存した。`full-fp16x3-dx-gradients-seed22.json`、`full-fp16x3-dx-gradients-seed23.json`、`ieee-repeat-gradients-seed22.json`、`paired-full-step-fp16x3-dx-m2048.json`、`paired-full-step-fp16x3-dx-m128.json`、`full-step-fp16x3-dx-m2048.json`を参照。
