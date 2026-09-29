# 1024²・小入力行数のCST学習をAdaとAmpereで比較

2026-09-29。1024²を小形状の必須評価対象とし、全W・全dWを作らない5% Triweightの単層AdamW完全ステップを測った。`M`は線形層への入力総行数で、microbatch sizeそのものではない。主比較は`M=128/2048`、さらに小さい入力として`M=16`をRTX 3070で測った。

同じコミット`bfac322`のGit archive（SHA256 `6d7913f75386098a40654ab7d68e7f531d300e5b835b9284a52cf71f1568426e`）を隔離展開した。比較用CLIへの`M=16`追加とoptimizer診断スクリプトだけを後から転送し、Tritonの学習kernelは変えていない。RTX 6000 AdaはPyTorch 2.9.1+cu128／Triton 3.5.1、GeForce RTX 3070はPyTorch 2.6.0+cu126／Triton 3.6.0。各速度測定前にGPU空きを確認し、別ジョブが約9～16 GBを使い始めてからはAdaの速度測定を行っていない。A100 MIGは約42 GB中約21 GBしか空いておらず、性能測定を重ねなかった。

`listed_bounded` 8 bit候補、局所W生成4回展開、builder BA32／1 warp、512行W窓、FP32。`M=2048`のforwardと入力勾配はFP16x3、局所dWはIEEE FP32。`M=16/128`はIEEE FP32。forwardで作った512行W窓を0枚／1枚保持する設定を、denseとともに**同一プロセスで32回ずつ交互にCUDA Graph再実行**した。時間比は同じ回の対応ペア比中央値。初期出力は全条件でdense oracleの`atol=rtol=3e-5`に合格した。

| GPU | M | dense中央値 | CST保持0枚 | CST保持1枚 | 保持1／0対応ペア比 | 保持1／dense対応ペア比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 6000 Ada | 128 | 0.113 ms | 0.673 ms | 0.637 ms | **0.9429** | 5.59 |
| RTX 6000 Ada | 2048 | 0.489 ms | 0.875 ms | 0.844 ms | **0.9658** | 1.72 |
| RTX 3070 Ampere | 16 | 0.405 ms | 1.559 ms | 1.444 ms | **0.9265** | 3.52 |
| RTX 3070 Ampere | 128 | 0.440 ms | 1.560 ms | 1.465 ms | **0.9351** | 3.23 |
| RTX 3070 Ampere | 2048 | 1.425 ms | 2.559 ms | 2.475 ms | **0.9589** | 1.73 |

512行W窓1枚の保持は、同じ形状の既定0枚に対してAdaで約3.4～5.7%、Ampereで約4.1～7.3%短縮した。両GPUで方向が揃ったため、小形状の**明示的な速度優先試作設定**として有望。ただし1024²・M=128は依然denseの3～6倍程度で、近接した速度とは言えない。GPU間でPyTorch/Triton版や性能特性が違うため、比率差をGPU世代だけへ帰属しない。

## ピーク割当と正しさ

Graph capture時のPyTorch割当ピークは、CSTとdenseを**別プロセス**で測った。比較用の全WはCST学習ステップに持ち込んでいない。

| GPU | M | CST保持1枚 | dense Graph | 削減率 |
| --- | ---: | ---: | ---: | ---: |
| Ada | 128 | 47.24 MB | 57.33 MB | 17.6% |
| Ada | 2048 | 64.59 MB | 85.25 MB | 24.2% |
| RTX 3070 | 16 | 45.40 MB | 55.96 MB | 18.9% |
| RTX 3070 | 128 | 47.24 MB | 57.33 MB | 17.6% |
| RTX 3070 | 2048 | 63.54 MB | 85.25 MB | 25.5% |

全CST条件でGraph captureとパラメータ更新に成功。Graph／eagerの3更新照合も通過し、最大パラメータ差はAda・M=2048の1.91e-6が最大だった。RTX 3070で強制overflowとGraph再実行を含むGPUテスト8件が通過した。Adaで以前測った**eager dense**のM=128ピーク40.29 MBに対しては、今回のCST Graph 47.24 MBはまだ大きい。速度とピークをともに最良のdense運用へ対抗させる課題は残る。

AdaのM=128・保持1枚のGPU profilerではkernel合計0.667 ms。atom勾配0.210 ms、局所W生成0.131 msが大きく、GEMM数個の変更だけではdenseとの差を埋められない。小kernelの多いAdamWを`foreach`からPyTorchの`fused=True`へ変える案も完全ステップで交互測定したが、fused／foreach対応ペア比はAda **1.3265**、RTX 3070 **1.1870**で遅く、不採用。

生JSONは[`data/small-shape-20260929/`](data/small-shape-20260929/)に保存した。速度用`cache-*.json`、ピーク用`peak-*.json`、optimizer用`optimizer-*.json`、Ada内訳`profile-1024-m128-c1.json`。計測コードは`prototypes.profile_paired_dense_cst`、`prototypes.probe_csr_graph_step`、`prototypes.probe_dense_graph_step`、`prototypes.profile_small_optimizer_modes`。

次の速度改善には局所W生成・atom勾配の計算方式を変える必要がある。512行窓の保持は再生成1回を省くが、atom勾配2回と残りのW生成3回は残る。公開backendの既定dispatchは変更していない。
