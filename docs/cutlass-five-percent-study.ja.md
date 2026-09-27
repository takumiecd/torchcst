# CUTLASSからの5% mapped GEMM検討

2026-09-27。CUTLASS checkout `0b55a2f691d69981583568fd9eb69687b1f0de8a`を参照し、A100上のatom密度5%（N=K=4096/8192、M=128、CST tile64×64）に移せる手法を調べた。CUTLASS自体は変更していない。

## 設計との対応

- `media/docs/cpp/efficient_gemm.md`の階層タイルとwarp内再利用：現行`block_fused`は16×32のW小片をatomから生成し、128入力行のdotに一回使う。全Wは保持しない。
- 同資料のparallel split-K：現行8分割と部分出力のreduceで実装済み。5%では旧BA8比で4096²が23.637→16.281ms、8192²が85.239→65.131ms（既存の同一run測定、詳細は`five-percent-mapped-gemm.ja.md`）。
- 同資料のthreadblock rasterization：今回M方向のCTA数は128/128=1。M方向の順序を入れ替える余地がなく、N方向の順序変更だけではatom×サイト評価を減らせない。
- `include/cutlass/gemm/threadblock/mma_pipelined.h`の二重バッファ：入力とWの先読みを計算と重ねる発想は適用候補。ただし現行のWはメモリからのタイル読み出しでなく、atomを走査してその場で生成する。先行診断では4096²でW生成13.246msに対し融合全体16.737ms、8192²で52.725msに対し66.944ms。単純な入力先読みの上限は小さい。
- AmpereのTensor Core：CUTLASS例`examples/23_ampere_gemm_operand_reduction_fusion/ampere_gemm_operand_reduction_fusion.cu`はFP16/BF16を使う。現行はFP32、TF32無効、出力照合`atol=rtol=3e-5`なので、そのまま精度を置き換えられない。

## A100での実験

同じ層・atom・入力・dense Wを共有し、各経路を交互にCUDA Graph 3ラウンド、rep=20msで測定した。準備込み、dense Wは事前生成。A100 80GB PCIe MIG 3g.40gb、42 SM、seed21、FP32、TF32無効。全測定経路はcanonical Wの1,152点照合とdense出力に対する`atol=rtol=3e-5`を通過した。

| 実験 | 4096²：既定→候補 | 8192²：既定→候補 | 判定 |
| --- | ---: | ---: | --- |
| Triton `num_stages=1` | 16.290→16.287ms | 65.158→65.049ms | 有意な改善なし |
| `num_stages=2～5` | 16.289～16.293ms | 65.035～65.094ms | 段数に比例した改善なし |
| atomの振幅・精度を距離計算より先に読む | 16.355→16.379ms | 65.147→65.126ms | 測定変動内 |

パイプライン段数の8192²既定3サンプルは65.0367～65.1643ms。候補との差はこの範囲以下である。先読み候補も各3サンプルで出力照合済み。先読み候補のkernel変更はrevertし、既定は維持した。

現在の16×32タイルで`tl.dot(input_precision="tf32x3")`も試したが、A100で`LLVM ERROR: Failed to emit transfer from register to shared`、終了コード134でコンパイルが停止した。正しさと速度は測れていない。以前の32×16タイルの同種試作もコンパイル中に停止しており、今回の実験変更はrevertした。

## 再現

- 段数比較commit `bdc8205220f69487df2323ac16093517c3d6ffa8`、source archive SHA256 `cd1a757df2080a10a95ffdb84fcf88322258ce79e039957a4c35559e2755ee0a`。結果JSONは`output/triton-a100-20260927/cutlass-stages-{4096,8192}.json`、SHA256は順に`9da46ea1424f17cd4a2b151982f17aca74e67977289930e86810f43f698bbff7`、`dccdae53159b071ab0697dffccba84ab9c7d369569c631f863b2d678016f0299`。
- 先読み比較commit `369e4597fd95b438c6428304e55ef7044bd48402`、source archive SHA256 `4683b8c4355aa7b033af64d36d705c9acb9aa48181d71c2d792305701c713a19`。結果JSONは`output/triton-a100-20260927/cutlass-early-atom-{4096,8192}.json`、SHA256は順に`0816a8f4cefa509ddc7c8ceb04d3d8f23e5cd745ee6ef9a249b1713d3e666508`、`c1c0522518ee2a7eaa3280666ee01e8207f12c3ca1088cb37dbd829d71d0b8f2`。
- Tensor Core試作commit `27f3e582c4a407a7c6d8b24f9407608c0dca107c`、source archive SHA256 `4f3358c26a0aa5326cf5e0559780f62d21a62e238eaa9fc5e0e9f676305ae97a`。4096²でコンパイル停止。8192²は実行していない。
- remote snapshotは順に`srv11/cst-lab/torchcst-cutlass-bdc8205`、`srv11/cst-lab/torchcst-cutlass-369e459`、`srv11/cst-lab/torchcst-cutlass-27f3e58`。アップロードした各archiveは展開前にremote SHA256と照合した。結果JSONもremote/local hash一致。

```bash
PYTHONPATH=src:. python -u -m prototypes.benchmark_split_k_five_percent --output cutlass-stages-4096.json --source-commit bdc8205220f69487df2323ac16093517c3d6ffa8 --size 4096 --batch 128 --full --splits 8 --stages 1 2 3 4 5
PYTHONPATH=src:. python -u -m prototypes.benchmark_split_k_five_percent --output cutlass-stages-8192.json --source-commit bdc8205220f69487df2323ac16093517c3d6ffa8 --size 8192 --batch 128 --full --splits 8 --stages 1 2 3 4 5
```

次の性能課題は、CUTLASSのスケジューリング移植より、atom×サイトの寄与評価を減らす方法である。atom密度5%の初期配置では32列小片の保守的なsupport cullは約0.3%しか省けなかった。学習後にatom配置が偏った場合は、ここを再測定する価値がある。Tensor Core利用はTritonコンパイル失敗の切り分けと許容誤差の確認が先になる。
