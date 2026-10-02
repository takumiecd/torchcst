# Colab Pro L4での5% CST検証とL2計測

2026-09-28。共有のRTX 6000 Adaで別プロジェクトが再開したため、CLIでColab ProのL4を使用した。`colab whoami`で契約アカウントを確認し、両実験前に既存セッション0件を確認した。`--gpu L4`の割当実機は **NVIDIA L4**、compute capability 8.9、58 SM、GPUメモリ23,659,151,360 bytes。PyTorch 2.11.0+cu128、Triton 3.6.0。結果を回収して2つの名前付きセッションを停止し、最終的に割当0件を確認した。Blackwellは使っていない。

L4はRTX 6000 Adaと同じAda世代だが、公式仕様ではL2が48対96 MiB、FP32公称値が30.3対91.1 TFLOPS、メモリ帯域が300対960 GB/sである。L4はアルゴリズムの探索と正確性確認に使い、キャッシュ容量に依存する採否はRTX 6000 Adaで再確認する（[L4仕様](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)、[RTX 6000 Ada仕様](https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf)）。

## 同一条件の学習ステップ

コミット`7fb6a9c`のGit archive SHA256は`ab5d0793e89cc8f1625684fc419c9e329252b4852242ae9332ada4f7969d326f`。8192×8192、atom数3,355,443（5%）、FP32、seed 21、単層AdamWをCST/dense別プロセスで測定した。CSTは1024行局所W、先頭2窓保持、listed候補、`staged_listed` backward。M=128はIEEE FP32、M=2048はforwardと入力勾配がTF32x3、局所dWがIEEE FP32。初回を含めて10ステップをウォームアップし、同期wall時間7ステップの中央値を取った。全Wは照合にだけ一時生成し、CST学習ステップ前に解放した。CST学習は全Wも全dWも保持しない。

| M | CST ms | dense ms | CST / dense | CST割当ピーク MB | dense割当ピーク MB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 65.88 | 27.25 | 2.42 | 588.37 | 1371.80 |
| 2048 | 142.41 | 93.89 | 1.52 | 839.61 | 1560.54 |

全出力照合に合格し、入力・atom勾配は有限・非ゼロ。これは大形状の全勾配をdense勾配と要素ごとに照合した結果ではない。別ステップの工程時間はM=128でCST forward/backwardが26.36/35.71 ms、denseが2.05/3.31 ms。M=2048ではCSTが50.69/84.50 ms、denseが24.24/46.47 ms。工程時間は上表の反復中央値と同一ステップではない。生データは`output/colab-20260928/l4-8192-*.json`と`manifest.json`。

## アルゴリズム候補

M=128の準備済みforwardで、局所W＋GEMMは18.84 ms。Ampereで調整したatom直接融合構成は28.33 ms、単純な融合構成は41.64 msだった。両方とも局所W方式の出力と照合に合格したが、L4上では採用しない。この測定はforward本体だけで、候補一覧・準備やbackwardを含む学習ステップではない。

backwardでatom勾配と局所Wを同時に計算してdW窓をWに上書きする試作は、最初の1024行窓で別々のW生成1.601 ms＋atom勾配2.422 msから同時生成3.582 msへ短縮した。Wは参照と全要素一致し、このL4実行でのatom勾配の相対L2差は`1.49e-7`。ただしレジスタは127から254/threadへ増え、RTX 6000 Adaの別実測では勾配差が`3.81e-4`だった。学習ステップ全体での速度・精度未検証のため、既定経路へは入れない。結果は`output/colab-20260928/l4-forward-alternatives-8192.json`と`l4-listed-weight-reuse-8192.json`。

## L2カウンタ

Nsight ComputeでM=2048の`bounded_gemm_kernel`を1窓ずつ測った。各局所Wを生成した直後にGEMMを実行し、`--cache-control none`を指定した。W窓は512/1024/2048行で16/32/64 MiB。入力Xは64 MiB。各窓につき1回のカウンタ値である。

| W窓行数 | L2セクタヒット率 | DRAM読み出し MB |
| ---: | ---: | ---: |
| 512 | 85.23% | 248.72 |
| 1024 | 80.96% | 641.04 |
| 2048 | 79.20% | 1400.53 |

局所性とL2再利用がGEMMに寄与していることは支持される。しかしヒット率は**カーネル全体**であり、WとXを分けた値ではない。窓サイズの変更はGEMM形状・CTA数も変え、Nsight Computeもcache状態が制御されない旨を警告した。この単発結果からL2容量が主要ボトルネック、あるいは最適窓が512行と断定しない。生出力は`output/colab-20260928/l4-ncu-cache.txt`、要約は`l4-ncu-cache-summary.json`。

Colab CLI 0.6.0には計算単位残高やL4/G4の単価を表示するコマンドがなく、消費単位は算定できていない。ColabのGPU提供状況・利用枠は変動する（[Colab FAQ](https://research.google.com/colaboratory/faq.html)）。今後も短いセッションで実機確認、結果回収、停止を行う。
