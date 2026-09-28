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

共有時の実測JSONは `output/triton-a100-20260928/ada-*.json` に保存。AdaのPyTorchは2.9.1+cu128、Tritonは3.5.1。GPUが空いた時の再測定を後段に記す。

Colab Proで単独利用できたRTX PRO 6000 Blackwellの同じ5%学習ステップは[別記録](colab-blackwell-kaggle-gpu.ja.md)を参照。Colabの実測値と本稿のAda共有時の値を直接の世代間性能比として扱わない。

## Adaが空いていた時の再測定と設計判断

同日19:52–20:03頃（日本時間）、RTX 6000 Adaの使用率0%、割当39 MiB、`parc_final`のPython処理が見えない状態を確認した。コミット`ce65f35`のGit archive（SHA256 `4fdcd64f72d90492a2444c365768b7a3d1819ea5366b80c6b98499f652afef91`）を隔離展開し、上と同じ8192²・5%・seed 21・AdamW・10ウォームアップ・7反復で再測定した。CSTは1024行窓、先頭2窓をforwardから保持、listed生成・`staged_listed`勾配。M=128はIEEE FP32、M=2048はforwardと入力勾配にTF32x3、局所dWにIEEE FP32。各方式は別プロセスで、dense WはCSTの測定前に照合して解放した。

| M | CST ms | dense ms | CST / dense | CSTピーク MB | denseピーク MB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 18.28 | 8.05 | 2.27 | 588.37 | 1371.80 |
| 2048 | 45.85 | 35.11 | 1.31 | 839.61 | 1560.54 |

全条件で出力照合に合格し、勾配は有限・非ゼロ。M=128のCSTは17.07–19.24 ms、denseは8.02–8.07 ms。M=2048のCSTは44.56–48.00 ms、denseは34.07–36.26 ms。別の工程計測ステップでは、CSTのforward/backward/optimizerがM=128で8.35/9.71/1.40 ms、M=2048で20.08/26.49/1.40 ms。denseは順に0.77/1.15/6.52 msと11.91/18.36/6.59 ms。工程値は上表の反復中央値と同一ステップの分解ではない。結果は`output/ada-20260928/ada-clean-*.json`に保存した。

AdaのL2容量を想定して窓を増減するだけでは、明確な速度改善は得られていない。M=2048で同じ先頭2048行分を保持した探索値は、512行×4窓が46.02 ms・822.84 MB、1024行×2窓が45.85 ms・839.61 MB、2048行×1窓が45.71 ms・873.59 MB。1024行×4窓は47.08 ms・893.72 MBだった。各7反復は実行中に数ms上昇する傾向があり、後半には別プロジェクトの評価処理が再開したため、約1%の差を優劣と解釈しない。追加の保持はメモリを増やすので、現時点の基準は1024行×2窓とする。

backwardの候補一覧カーネルがatom勾配と局所Wを同時に作り、dW窓をその場でWへ上書きする試作を追加した。Adaの最初の1024行窓では、別々のW生成0.543 ms＋atom勾配0.822 msに対して同時生成1.202 msで、Wは参照と全要素一致した。しかしレジスタは128から255/threadへ増え、基準勾配との相対L2差は`3.81e-4`。W精度のため融合積和を無効化した影響を含み、学習ステップの速度と勾配許容差は未検証である。既定の学習経路には選ばない。診断は`prototypes.profile_listed_weight_reuse`と`output/ada-20260928/ada-listed-weight-reuse-8192.json`。

小バッチ向けのatom直接融合forwardも試したが、その時点で別プロジェクトの`policy_server.py`がGPUを使用しており、各方式の時間が大きく揺れた。この値で採否は判断しない。`prototypes.profile_ada_forward_alternatives`と`output/ada-20260928/ada-forward-alternatives-8192-m128.json`を再実行可能な診断として残す。

今後はAdaを主な開発環境にし、アルゴリズムはGPU世代で固定しない。次に検証するのは、M=128のW生成・小バッチGEMMと、M=2048のforward/backward GEMMおよびatom勾配を分けた改善である。採否はステップ全体の時間、denseより低いピーク、出力・入力勾配・atom勾配の精度で判断する。Blackwellは候補を絞った後の確認に使い、Ampereは回帰確認に使う。L2 hit率は未測定なので、窓サイズと速度の関係をcache効果と断定しない。
