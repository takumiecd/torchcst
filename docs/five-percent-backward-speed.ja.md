# 5% mapped CST：backwardの計算順序と速度

2026-09-28。前段の[学習メモリ測定](five-percent-mapped-training-memory.ja.md)と同じ5% atom密度、FP32、TF32無効、A100 80GB PCIe MIG 3g.40gb（42 SM）、64×64 CST tileを使った。Mは線形層へ入る総行数。単層のAdamWステップでforward、入力勾配、atom勾配、更新を含む。全重みWと全重み勾配dWはCST経路で保持しない。

## 時間を使う場所

4096²・M=128の旧backwardを個別に計時すると、入力勾配は9.74 ms、atom勾配は49.64 msだった。旧atom勾配は16×16の論理サイトタイルで局所dWを計算し、そのタイルの候補atomを評価してglobal gradientへatomic加算する。forwardで使っていた列方向の幾何情報共有は、この経路では使っていなかった。

二つの試作を追加した。

1. `factored`：forwardの列方向距離計算の共有をatom勾配へ移し、サイト座標をatomループの外で読み、atomとサイトの差を振幅・中心勾配に使い回す。16×32のdWタイル、atomレーン8を使う。
2. `staged`：入力勾配の局所重み窓が終わったら、**同じ最大1024×Kのバッファ**を局所dW窓に転用する。各行窓のdWを `torch.mm(dY_window.T, X, out=window)` で計算し、factored kernelでatom勾配へ縮約してから上書きする。4096²では窓16 MiB、8192²では32 MiB。全dWを保持しない。

`staged` の行窓はforwardから保存したpacked atom配置と固定サイト幾何を再利用する。重み窓自体をforwardから保持するとactivationと共存してピークを上げるため、現試作ではbackwardの入力勾配時に再生成する。

## 学習ステップの結果

単位は10進MBと同期wall ms。旧・factored・stagedは同じatom数、初期値、入力を使う独立プロセス。denseは従来の対照。時間はウォームアップ後のステップ。ピークはoptimizer状態初期化後の `torch.cuda.max_memory_allocated()`。各経路で初期重みのcanonical点と全出力を照合した。

| W形状 | M | dense | 旧CST | factored | staged | CSTピーク |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096² | 128 | 3.94 ms | 73.95 ms | 63.37 ms | **63.29 ms** | 171.41 MB |
| 4096² | 2048 | 30.29 ms | 119.93 ms | 100.50 ms | **90.46 ms** | 297.92 MB |
| 8192² | 128 | 14.93 ms | 290.83 ms | 249.08 ms | **247.93 ms** | 574.69 MB |
| 8192² | 2048 | 119.06 ms | 475.44 ms | 未計測 | **354.04 ms** | 825.93 MB |

2048行の旧CSTとstagedは同一ソース `98ce184` で各3回計測した中央値。4096²は旧119.93/119.91/120.12 ms、staged90.69/90.46/90.40 ms。8192²は旧479.22/475.44/458.77 ms、staged356.16/354.04/337.48 ms。128行は各1回の確認値。最大の速度改善は4096²・2048行で約24.6%、8192²・2048行で約25.5%。ピークは計測した各経路で同じだった。4096²・M=512の予備測定ではfactored70.91 ms、staged68.66 ms、ピークはいずれも198.93 MB。

小形状128²・192²の入力勾配とatom勾配を独立したPyTorch参照と照合し、atomを支持域境界・idleへ動かした128²も含めて9テストがA100で通った。大形状の全atom勾配を独立参照と照合したわけではないが、測定ステップではパラメータ勾配と入力勾配が有限かつ非ゼロであることを確認した。atomic加算の順序は非決定的。準備処理の `torch.sort` 一時差し替えも試作のまま。

## 次に減らす計算

`staged` でも4096²・M=128はdenseの約16倍、M=2048は約3倍かかる。4096²・M=128では局所dWのGEMMより、**候補atomを各サイトタイルで繰り返し評価し、各タイルから勾配をatomic加算する処理**が支配的だった。16×32タイルでは64×64 station当たり8個のプログラムが同じ候補atomを扱う。局所dW窓を用意できたので、次はatom側を作業単位とし、station内のサイト寄与をまとめてから少数の勾配を書き込む方式を検証する。これならdW窓とatom配置を再利用しつつ、候補atomのロードとatomic加算の反復を減らせる可能性がある。正しさ、ピーク、速度はまだ未検証。

単純な支持域box除外は、以前の5%初期配置で32列小片の約0.3%しか省けなかったため、主な改善策には据えない。dense並みの速度を保証する根拠は現時点ではない。

実装は[backward試作](../prototypes/block_streamed_backward.py)、[学習ステップ測定](../prototypes/benchmark_mapped_training_memory.py)、[内訳計測](../prototypes/profile_mapped_backward_parts.py)、[勾配テスト](../tests/test_block_streamed_backward.py)。A100で検証した `98ce184` のgit archive SHA256は `bc17e0c3051af286dbe21070bc31cdc7a1ae7f6c0f956238217755f81f35d181`。結果JSONは `output/triton-a100-20260928/{profile-,factored-train-,staged-train-,paired-}*.json` に保存した。
