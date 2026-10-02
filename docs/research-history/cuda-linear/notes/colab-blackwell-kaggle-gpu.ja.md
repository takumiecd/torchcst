# Colab Pro BlackwellとKaggle CLIのGPU確認

2026-09-28。研究用GPUをCLIから追加利用するため、両サービスを実測した。Colab Proは正しい契約アカウントを確認して利用した。授業用RTX A6000は使用していない。

## Colab Pro: RTX PRO 6000 Blackwell

ローカルの `google-colab-cli` 0.6.0で `colab whoami` と `colab sessions` を確認し、`G4`を要求した。割当実機は **NVIDIA RTX PRO 6000 Blackwell Server Edition**、compute capability 12.0、GPUメモリ101,974,081,536 bytes（`nvidia-smi` 97,887 MiB）、188 SM。PyTorch 2.11.0+cu128、Triton 3.6.0。短い診断は `colab run` で自動解放し、学習ベンチマーク用の名前付きVMも結果回収後に `colab stop` して、割当0件を確認した。

コミット `d22a2f0` をGit archive化し、SHA256 `dabe6a46f5e35c2eac7b1470a47f779653539f2bcc07d566b94d20af62ee2afb` をColab側でも照合した。8192×8192、atom数3,355,443（5%）、単層AdamW、FP32、seed 21、各方式を別プロセスで実行。最初のステップを含む10ステップをウォームアップし、同期wall時間7ステップの中央値を測定した。CSTは1024行の局所W窓、listed候補、`staged_listed` atom勾配。M=128はIEEE FP32、M=2048はforwardと入力勾配にTF32x3、局所dWにIEEE FP32。denseは同じCSTから作った全Wを常駐させ、W生成時間は含めない。

| M | 方式 | 窓キャッシュ | 学習ステップ ms | PyTorch割当ピーク MB | CST / dense |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | CST | 2 | 13.53 | 588.37 | 2.89 |
| 128 | dense | - | 4.69 | 1371.80 | - |
| 2048 | CST | 2 | 24.30 | 839.61 | 1.62 |
| 2048 | CST | 4 | 24.03 | 893.72 | 1.60 |
| 2048 | dense | - | 14.98 | 1560.54 | - |

全CST設定でcanonical Wと全forward出力の照合に合格し、入力とatomの勾配は有限・非ゼロ。大形状の全勾配をdense勾配と要素ごとに比較した結果ではない。M=128の7回はCST 12.67–13.89 ms、dense 4.67–4.70 ms。M=2048、窓4枚はCST 23.21–24.35 ms、dense 14.97–14.99 ms。工程別の別ステップでは、M=2048のCST forward 10.25 ms、backward 13.67 ms、optimizer 0.85 ms。denseはforward 4.65 ms、backward 6.85 ms、optimizer 3.89 ms。CSTのbackwardが最大の工程だが、denseとの差にはforwardも寄与する。

計測プログラムは照合用の全dense Wを学習開始前に一時生成する。CST**学習ステップ**は全Wも全dWも保持せず、照合Wを解放してからピーク統計をリセットしている。表はPyTorchの割当ピークであり、CUDAコンテキストと予約メモリを含まない。

結果は `output/colab-20260928/blackwell-*.json` に保存。ColabのGPU機種と利用枠は変動するため、再実行時はCLIの指定だけで判断せず実機を確認する（[Colab FAQ](https://research.google.com/colaboratory/faq.html)、[Colab CLI](https://github.com/googlecolab/google-colab-cli)）。

## Kaggle CLI: 現アカウントの割当実機

Kaggle CLI 2.2.4は認証済み。`kaggle quota` は実験前に週30時間残を示した。非公開の短い診断kernelで `NvidiaL4`、`NvidiaRtxPro6000`、`NvidiaTeslaA100` をそれぞれ指定したが、3回とも実機は **Tesla T4 ×2**、各15,636,037,632 bytes（`nvidia-smi` 15,360 MiB）、compute capability 7.5、PyTorch 2.10.0+cu128、Triton 3.6.0だった。原因や他機種へのアカウント権限はこの結果だけでは断定できない。指定機種を使えたことにはしない。

診断出力は `output/kaggle-20260928/` に保存。Kaggle CLIのGPU識別子一覧にはアカウント/大会限定機種も含まれる（[Kaggle kernel CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)）。今後も割当実機をジョブ内で検査する。
