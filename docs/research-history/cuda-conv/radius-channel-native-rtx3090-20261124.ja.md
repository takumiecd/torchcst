# RTX 3090 での半径セル質量チャネル重みのネイティブ実行

[English](radius-channel-native-rtx3090-20261124.md) | **日本語**

[チャネル Chart 設計](radius-cell-mass-channel-charts-20261124.ja.md)の実装と計測を行った。
空間タップごとに Atom 行を持つ。内部の `channel_weights` は連続な FP32 物理座標
`[tap, atoms, 4]` を受け取り、`[tap, 128, input_sites]` を返す。両軸で M8、
spacing 1、base tile 1、merge factor 2、半径上限32サイトを使用する。既存の物理座標への
復号で rational の振幅と半径割合から物理半径を求め、その autograd で rational 側の
勾配を得る。通常の Conv2d が合成重みを受け取り、重み余接を返す。native の随伴計算は
この余接と forward で保存した幾何情報を使う。

上限付き parser と重み専用の入口はこの研究経路向けである。直接加算は全タップを一括で
launch する。Fourier 経路は Linear の幾何生成、owner ペアの加算、fold、余接の縮約を
再利用し、縮約はタップ単位で実行する。Fourier を始める level 3、4、5は、それぞれ
半径4、8、16を超えたところから Fourier 加算する。level 6は到達する全階層を直接加算する。

## 学習ステップの対応比較

RTX 3090 の独立した2プロセス（PID 4400、4415）で、seed 内で入力と初期重みを揃え、
測定順を反転した。合成 FP32 channels-last 入力 `[256, 96, 8, 8]`、stride 2、groups 1、
出力幅128とし、padding は1×1で0、3×3で1。Atom は1,773個で、3×3では1タップ197個。
半径割合は [0, 0.25] または [0.65, 0.98] の一様分布で初期化した。初期の物理半径は、
それぞれおよそ1.44–6.71と11.32–30.68だった。各ステップに rational の復号、重み構築、
レイアウト変換、forward、固定 dY と dX を含む backward、fused AdamW（学習率0.001、
weight decay 0）と Atom 群の振幅・半径割合の射影を含めた。Dense は対応する実効重みを
初期値とし、通常の AdamW を使った。Torch 参照は同じ Chart の因子をバッチ行列積で合成した。
各経路で warmup 8回、計測24回。CUDA 12.6、driver 580.95.05、cuDNN 9.5.1、
PyTorch 2.6.0+cu126を使い、TF32を無効にした。

| 形状・半径範囲 | Dense | Torch | CUDA 直接 | Fourier 3 | Fourier 4 | Fourier 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1×1 狭い半径 | 0.278–0.289 | 15.405–15.437 | 1.562–1.704 | 1.860–1.891 | 1.793–1.815 | 1.734–1.765 |
| 1×1 広い半径 | 0.285–0.393 | 11.415–11.935 | 2.419–2.561 | 1.886–2.580 | 1.735–1.816 | 1.748–1.767 |
| 3×3 狭い半径 | 1.011–1.012 | 14.785–14.818 | 1.600–1.612 | 3.650–3.665 | 3.527–3.551 | 3.280–3.327 |
| 3×3 広い半径 | 0.998–1.002 | 10.802–10.976 | 2.811–2.841 | 3.634–4.363 | 3.141–3.156 | 5.257–5.273 |

時間は CUDA event 中央値の範囲（ms）。同期した実時間と全サンプルは結果 JSON に保存した。
3×3の直接加算は allocated 42.53–42.57 MiB、reserved 62 MiB だった。Torch 参照の
allocated は狭い半径で63.33–63.35 MiB、広い半径で110.72–110.74 MiB、reserved は
それぞれ76と134 MiB。Dense は allocated 43.01–43.05 MiB、reserved 64 MiB。
生存テンソルを含む allocator 全体のピークであり、GPU プロセス全体は未計測である。

今回の候補では、1×1の狭い半径と3×3の両条件で直接加算が速かった。1×1の広い半径では
level 4と5の Fourier が両プロセスで直接加算を上回った。4と5の順位は入れ替わった。
3×3の Fourier はタップ単位 launch の結果で、複数タップをバッチ化した場合の性能は
別途検証する。公開 dispatcher での選択には学習済み checkpoint、モデル統合、各演算精度での
検証が必要になる。

## 数値検証と再現

計測条件の W、Y、dX、rational Atom 全4フィールドを CPU FP64 参照と照合した。
native 経路の最大相対 L2 誤差は1.84e-6だった。個別テストでは、入力幅96と128の両空間形状で
独立した Chebyshev 数値積分、物理半径の階層境界、周期 owner、全一次勾配を確認した。
準備した物理座標で入力・中心・半径の bank を変えた CUDA Graph replay も検証した。
拡張したスイートは19件成功した。計測の rational 復号と optimizer step は eager で実行した。
同じ extension で既存の幅2048の Linear 勾配テスト2件も成功した。CPU スイートは2,219件成功、
2,605件スキップ。wheel と source distribution のビルドも通った。

計測した実行コードは `ded7540a086a0500a7288931080e0a6daaf3c539`。ソース、ログ、
driver、結果は worktree の Git 管理対象外の `output/conv-radius-native-20261124/` と
Kyutech ホーム相対の `cst-conv-channel-20261124/ded7540a/` に保存した。
隔離したビルドに Ninja 1.13.2、CUDA architecture 8.6、インストール済み NVIDIA Python
パッケージの CUDA ヘッダーを使った。ソースと driver を展開し、これらのヘッダーと Ninja を
設定してから driver のあるディレクトリで実行する。

```sh
export PYTHONPATH=source/src:source:source/tests TORCH_CUDA_ARCH_LIST=8.6
python -m pytest -q test_radius_channel_weight_cuda.py
python measure.py --seed 17 --source-commit ded7540a086a0500a7288931080e0a6daaf3c539 --output run-17.json
python measure.py --seed 23 --source-commit ded7540a086a0500a7288931080e0a6daaf3c539 --output run-23.json
```

拡張したスイートには、同じ pytest コマンドで記録ディレクトリの
`test_radius_channel_weight_v2.py` を指定する。

| ファイル | SHA-256 |
| --- | --- |
| `source.tar.gz` | `861c17fabac10a812a637b6898d7f0ad30487259e35d3ca16414525b4d47b1f6` |
| `driver.tar.gz` | `8079e79d12063243e8524fd1738f83529d5a9a137662c46682fe443a040143ba` |
| `measure.py` | `9f60b544e9534e3c897d5d0c687bed4564fd3e175a6939a30fdd3ea11c0f3873` |
| `test_radius_channel_weight_v2.py` | `dcacc91d42b03b8fb64d472557531fb5dcfe88d07d3ede3097178c915ff756db` |
| `run-17.json` | `fb24f893d000a00d84bf791de95eb9dbc7a7d34c0dcc111c7cbf330a32b3263e` |
| `run-23.json` | `b9a9c2e73a5541940aed53767dc0eef0aa5cf3785c45283e92d0ca5e17a693f0` |
