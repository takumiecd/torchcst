# 5% CSTのforward：局所Tensor Coreと重み生成の削減

2026-09-28。A100 80GB PCIe MIG 3g.40gb（42 SM）、FP32、5% atom密度、64×64 CST tile、1024行の局所W窓で測った。全Wと全dWはCST学習経路で生成・保持しない。単層AdamWのforward、backward、更新を含む測定で、dense側は同じCSTから生成したWを保持する対照である。

## 重み生成とGEMMの内訳

8192²のforwardをGPU eventで分けると、局所Wの生成はバッチM=128とM=2048の両方で約36.1 msだった。IEEE FP32 GEMMはM=128で2.68 ms、M=2048で40.92 ms。したがってバッチを増やした時の時間増は重み生成ではなくGEMMに由来し、小バッチでは重み生成が支配的だった。Tritonの `tl.dot(input_precision="tf32x3")` でM=2048のGEMMを23.37 msへ短縮できたが、M=128は2.69 msで効果がなかった。forwardはバッチが大きい場合だけ明示的に `--forward-gemm-mode tf32x3` を使う。

次に、16×64サイトタイルと交差する可能性のあるatomの一覧をforwardで一度作り、重み生成とbackwardのatom勾配で共有した。8192²では一覧作成3.14 ms、一覧を使う重み生成27.0 ms、従来の生成36.1 ms。重み窓の値は全要素で一致した。さらにbackwardの入力勾配用W窓再生成にも同じ一覧を使用する。一覧はatomが支持域を持ち得ないサイトタイルだけを除外し、近傍の境界でも保守的な判定を使う。

## 学習ステップ

単位は同期wall msと10進MB。各3回の中央値。M=128はIEEE FP32 GEMM、M=2048はforwardと入力勾配にTF32x3、局所dWにIEEE FP32を使った。`新経路`は候補一覧をforward・backwardで共有し、W窓を必要な時だけ最大1024行分作る。`従来CST`は候補一覧をbackwardでのみ使う窓2枚キャッシュ経路で、全GEMMがIEEE FP32である。

| W形状 | M | dense | 従来CST | 新経路 | 新経路ピーク | denseピーク |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096² | 128 | 3.94 | 31.54 | **28.59** | 187.96 | 358.88 |
| 4096² | 2048 | 30.29 | 59.47 | **47.56** | 313.79 | 453.25 |
| 8192² | 128 | 14.93 | 127.47 | **110.23** | 588.79 | 1371.80 |
| 8192² | 2048 | 119.06 | 237.37 | **187.96** | 840.03 | 1560.54 |

8192²・M=2048でforward W窓を4枚まで保持すると180.07 ms、ピーク893.72 MB。forward約57 ms、backward約103 msで、backwardはなお明確なボトルネックである。従来CST・denseの値には別コミットで測った条件もあり、GPU実行環境による数%の揺れを考慮する。小バッチでTensor Coreを使わず、候補一覧だけを使う判断は、M=128のGEMMが約2.7 msでTF32x3との差がなかったことに基づく。

8192²・M=2048で新経路と従来IEEE経路の全出力16,777,216要素、入力勾配16,777,216要素、atom勾配16,777,215要素を `atol=rtol=3e-4` で照合し、違反は0だった。学習ステップの出力はdense対照とのより厳しい `atol=rtol=3e-5` も通過した。支持域境界とatom移動を含め、A100の勾配テスト41件が通った。この精度は指定した初期値と入力での結果であり、全学習状態に対する保証ではない。

## 次の設計判断

追加のA100測定では、forwardのW生成とTensor Core GEMMを二重バッファで重ねても50.20→48.30 msで、32 MiBの窓を追加する価値は小さかった。backwardの局所dW GEMMとatom勾配を重ねた場合は78.83→78.31 msで、同じく32 MiB追加した。16×64サイトタイルを8×64にするとW生成は26.92→24.66 msへ短縮したが、候補一覧の構築は3.14→5.89 ms、一覧容量は13.43→26.87 MBとなる。backwardのatom勾配は5.28 msでほぼ変わらず、学習ステップへの採用は見送った。入力勾配GEMMのタイル変更は単体で3.00→2.68 ms/窓だったが、8192²・M=2048の学習ステップを3回測ると、変更前180.72 ms、変更後178.78 msと180.78 msで、安定した改善を確認できなかった。これらの値は `profile_overlap_forward.py`、`profile_overlap_dw_atoms.py`、`profile_listed_materialize.py`、`profile_listed_tile_shapes.py`、`profile_bounded_gemm_tiles.py` の試作測定で、結果JSONは同じ出力ディレクトリにある。

局所Wを完全に消してatom計算とGEMMを単一カーネルにすると、複数の入力バッチタイルが同じWタイルを必要とする。現行の生成一回・複数入力行で利用する順序に比べ、atom評価の重複が増える可能性がある。M=2048ではTensor Core化後のGEMMが23.4 ms、候補一覧を使うW生成が27.0 msなので、単純な融合では生成費用を繰り返す危険がある。融合案は同じピーク・出力精度・学習ステップ時間で比較してから採用する。

今回の実装は[局所forward](../prototypes/block_streamed_forward.py)、[候補一覧W生成](../prototypes/block_materialize_listed.py)、[局所Tensor Core GEMM](../prototypes/bounded_gemm.py)、[共有候補一覧を使うbackward](../prototypes/block_streamed_backward.py)。結果JSONは `output/triton-a100-20260928/` に保存した。
