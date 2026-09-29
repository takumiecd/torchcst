# RTX 3070とA100 MIGの5% CST学習比較

2026-09-28。コミット `1f0d7b8` の同じGit archive（SHA256 `c2804232c63a7b505ff710c98fd685783024115247e0f5707ba757bca9411ac0`）を両GPUへ転送し、`benchmarks.cuda.linear.benchmark_mapped_training_memory` で測った。Wは8192×8192、atom数3,355,443（5%）、単層AdamW、FP32、seed 21。CSTは全Wと全dWを保持せず、1024行の局所W窓、候補atom一覧共有、atom勾配のlisted経路を使用する。M=128はIEEE FP32 GEMM・窓2枚、M=2048はforwardと入力勾配にTF32x3・局所dWにIEEE FP32・窓4枚。denseは同じCSTから生成したWを常駐させ、W生成時間は測定に含めない。各モードは別プロセスで、最初のステップを含む10ステップをウォームアップした後、同期wall時間7回の中央値を取った。

| GPU | M | dense ms | CST ms | CST / dense | denseピーク MB | CSTピーク MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 3070 | 128 | 17.34 | 54.40 | 3.14 | 1371.80 | 588.79 |
| A100 80GB PCIe MIG 3g.40gb | 128 | 14.94 | 78.03 | 5.22 | 1371.80 | 588.79 |
| RTX 3070 | 2048 | 72.62 | 151.85 | 2.09 | 1560.54 | 893.72 |
| A100 80GB PCIe MIG 3g.40gb | 2048 | 119.48 | 152.51 | 1.28 | 1560.54 | 893.72 |

RTX 3070はWSL上のPyTorch 2.6.0+cu126・Triton 3.6.0、A100 MIGはPyTorch 2.6.0+cu126・Triton 3.2.0。3070のGPUは測定開始時に他の負荷0%。両GPUでcanonical Wおよび全forward出力の照合に合格し、勾配は有限・非ゼロ。3070では `tests/test_block_streamed_backward.py` の41件が通過した。大きな形状の全勾配をdenseと要素ごとに比較した試験ではない。

RTX 3070のL2は4 MiB、A100全体は40 MBで、3g.40gb区画にはL2の4/8が割り当てられる。3070の帯域公称値は448 GB/s、A100 80GB PCIe全体は1935 GB/sで、MIGでは帯域資源も分割される（[RTX 3070仕様](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)、[A100仕様](https://www.nvidia.com/en-us/data-center/a100/)、[MIG配分](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/concepts.html)）。局所W窓32 MiBは両GPUのL2容量より大きい。キャッシュ容量だけで速度を予測できない。M=128では3070のCSTがA100 MIGより約30%短い一方、M=2048ではほぼ同速だった。denseはM=2048で3070が約39%短く、CST/denseの比は3070で悪化した。FP32演算器、Tensor Core、帯域、キャッシュ、ライブラリ実装の寄与はこの学習ステップ測定だけでは分離できない。

CSTの各サンプルは学習の更新が進むにつれて変化し、M=128は3070で49.97–56.57 ms、A100 MIGで67.39–82.01 ms、M=2048は3070で146.89–153.69 ms、A100 MIGで141.34–156.94 msだった。denseは比較的安定していた。中央値はこのウォームアップ数と7ステップに対する値であり、長期学習全体の代表値とは限らない。

結果JSONは `output/triton-a100-20260928/{rtx3070,a100mig}-{cst,dense}-8192-*-w10.json` に保存した。RTX 3070へは `~/.ssh/config` の `win` で接続し、同GPUの再利用手順は `~/.codex/skills/win-rtx3070/SKILL.md` に記録した。
