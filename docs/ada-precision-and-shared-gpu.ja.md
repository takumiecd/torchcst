# RTX 6000 Adaでの5% CST学習: 重み精度と共有時の暫定測定

2026-09-28。8192×8192、atom数3,355,443（5%）、単層AdamW、FP32、seed 21。`39c0d66` の隔離snapshotに本記録のコード差分を転送して測った。RTX 6000 Ada 48GBは `parc_final` と同時使用中だったため、以下の時間は探索用であり、空き状態での性能比較として扱わない。授業用RTX A6000は使用していない。

## 正確性

Adaでは、局所W生成に `enable_fp_fusion=True` を指定すると、canonical dense Wとの差が最大 `1.999e-6`、窓ごとの相対L2誤差が最大 `7.59e-4` だった。listed経路とunlisted経路のWは一致した。差は候補列挙に由来しない。GEMM後には出力誤差が最大約 `1.0e-4` となり、`atol=rtol=3e-5` の全出力照合に失敗した。forwardのGEMMをTF32x3からIEEEに変更しても解消しなかった。

局所W生成だけ `enable_fp_fusion=False` にすると、canonical Wとの差は最大 `6.985e-10`、窓ごとの相対L2誤差は最大 `2.03e-7` に縮小した。8192²、M=2048の全出力照合に合格した。forwardおよび入力勾配向けに再生成するWへ同じ設定を適用した。Ampere（A100、RTX 3070）では既存の速い融合設定を維持し、Ada以降の計算能力では非融合設定を選ぶ。Blackwellは未実測なので、同じ保守的な設定から検証を始める。

`prototypes.diagnose_crossgpu_weight` は誤差原因を調べるため一時的に全dense Wを生成する診断プログラム。**CST学習経路は全Wも全dWも常駐させず**、1024行の局所W窓を使う。学習ステップのメモリ値は診断Wを解放してから測ったPyTorch割当ピークである。

## 共有GPUでの学習ステップ

`prototypes.benchmark_mapped_training_memory` を使用。10ステップのウォームアップ後、同期wall時間7回の中央値。M=128はIEEE FP32、M=2048はforwardと入力勾配にTF32x3。CSTはlisted生成とatom勾配のlisted経路、1024行窓、AdamW。denseは同じCSTから生成したWを常駐させ、生成時間を含めない。各モードは別プロセス。

| M | 方式 | 窓キャッシュ | 中央値 ms | 学習ステップ割当ピーク MB |
| ---: | --- | ---: | ---: | ---: |
| 128 | CST | 2 | 23.58 | 588.37 |
| 128 | dense | - | 15.63 | 1371.80 |
| 2048 | CST | 2 | 69.75 | 839.61 |
| 2048 | CST | 4 | 67.89 | 893.72 |
| 2048 | dense | - | 50.12 | 1560.54 |

同時実行のためM=128の各試行はCST約19.7–41.8 ms、dense約8.4–17.4 msと大きく揺れた。M=2048でもCST窓4枚は約65.0–70.9 ms、dense約45.1–62.0 ms。これらの比率から単独使用時の速度差を断定しない。M=2048、CST窓4枚の別途同期した工程時間はforward 38.77 ms、backward 30.40 ms、optimizer 3.45 msで、Adaではforwardが大きい。工程時間は上表の繰り返し中央値と同一ステップの分解ではない。

Adaでの局所W生成単体は、全行分のlisted生成で融合時約3.69 ms、非融合時約4.30 ms、候補一覧生成約0.69 ms。共有中の概算だが、forward全体の約39 msをW生成だけで説明できない。TF32x3 GEMMタイル探索では現行 `(BM, BN, BK, warps)=(32,128,32,4)` の約1.01 ms/窓を明確に上回る候補はなかった。共有メモリ上限を超えるタイルは計測から除いた。Ada上のstreamed backward・forward既存テストは42件通過した。

実測JSONは `output/triton-a100-20260928/ada-*.json` に保存。AdaのPyTorchは2.9.1+cu128、Tritonは3.5.1。次は`parc_final`の処理終了とGPUの空きを確認し、同じコミットの隔離snapshotでM=128/2048のCSTとdenseを再測定する。

Colab Proで単独利用できたRTX PRO 6000 Blackwellの同じ5%学習ステップは[別記録](colab-blackwell-kaggle-gpu.ja.md)を参照。Colabの実測値と本稿のAda共有時の値を直接の世代間性能比として扱わない。
