# RTX 3090 での Conv2d の重み合成

[English](composed-weight-rtx3090-20261124.md) | **日本語**

`CSTConv2d(backend="composed")` は `W = F_out @ F_in.T` を計算し、PyTorch Conv2d を適用する。
両方の演算を PyTorch で微分する。合成した重みはバッチ内の全空間位置で再利用する。
既存の2段の畳み込みは中間の畳み込みで Atom ごとに1出力チャネルを作る。
既存の materialized 経路は Atom ごとの重みテンソルを作ってから和を取る。

RTX 3090 の2つのプロセス（PID 3763、3787）を使い、channels-last FP32 の合成入力
`[256, 96, 8, 8]`、出力 `[256, 128, 4, 4]`、stride 2、groups 1、Atom 数1,774で測定した。
1×1は padding 0、3×3は padding 1で、後者の平坦化した入力 Chart は864サイトとした。
両形状で L2 正規化した固定 σ=0.75 の周期 Gaussian 因子を用い、振幅と中心を更新した。
同じプロセス内の各 CST 経路で入力、余接、初期 Atom Parameter を共有した。
経路ごとに独立した optimizer を用い、数値誤差の範囲内で等しい実効重みから始めた。
seed 17 と23で測定順を反転した。GPU の compute capability は8.6で、driver 580.95.05、
CUDA 12.6、cuDNN 9.5.1、PyTorch 2.6.0+cu126を使用し、TF32は無効にした。

eager の層ステップには forward、固定 dY による backward、dX、fused CSTAdamW または
AdamW を含む。学習率は0.001、weight decay は0とした。各経路で warmup 8回、同期計測24回。
表は2回の CUDA event 中央値と、測定開始時の生存テンソルを含む allocator のピークを示す。
同期した実時間の中央値は結果 JSON に記録した。GPU プロセスのメモリ使用量は未計測である。

| 空間カーネル | 経路 | 層ステップ (ms) | ピーク allocated (MiB) | ピーク reserved (MiB) |
| --- | --- | ---: | ---: | ---: |
| 1×1 | Dense | 0.264–0.310 | 40.51 | 62 |
| 1×1 | Materialized | 3.269–3.290 | 296.86 | 320 |
| 1×1 | Factored | 3.539–3.780 | 117.87 | 190 |
| 1×1 | Composed | 2.829–2.830 | 45.92 | 70 |
| 3×3 | Dense | 0.967–1.000 | 42.43 | 64 |
| 3×3 | Materialized | 44.347–44.361 | 3046.63 | 3088 |
| 3×3 | Factored | 18.339–18.357 | 2813.72 | 2850 |
| 3×3 | Composed | 3.536–3.539 | 65.57 | 88 |

計測前に独立した FP64 の周期 Gaussian 計算で、初期 Atom 状態の Y、dX、振幅と両中心の
勾配を検証した。Composed の最大相対 L2 誤差は両形状・両 seed で4.57e-7だった。
全経路で optimizer 32ステップの完了を確認した。計測範囲は合成入力での単一 FP32 層である。
学習済み checkpoint、BF16 autocast での学習、CUDA Graph、dispatcher 採用の検証は今後の対象とする。

## 再現

計測したソースは `d8aab933d315d2707b06e7a250756a88049c90bc`。driver、ソースアーカイブ、
照合済み結果は worktree の Git 管理対象外の `output/conv-composed-20261124/` と、
Kyutech のホームからの相対パス `cst-conv-composed-20261124-d8aab933/` に保存した。
ソースを `source/` に展開し、driver のあるディレクトリから実行する。

```sh
PYTHONPATH=source/src python measure-v3.py --seed 17 --order dense,materialized,factored,composed --source-commit d8aab933d315d2707b06e7a250756a88049c90bc --output run-17.json
PYTHONPATH=source/src python measure-v3.py --seed 23 --order composed,factored,materialized,dense --source-commit d8aab933d315d2707b06e7a250756a88049c90bc --output run-23.json
```

| ファイル | SHA-256 |
| --- | --- |
| `source.tar.gz` | `d1f76f5f68d33287c4b582b108755c933a4ce2ee2e773fb3e6907f464f7e64b8` |
| `measure-v3.py` | `f3169d7884149fc161ed2c633e6301a3feb03ac49491ab89a9cde7be69427947` |
| `run-17.json` | `6576d2ae8f5816731ddde1a3249d095057fe1e4a5c69c34633247402b366746e` |
| `run-23.json` | `4032790de620e92d155e75c601d0cb9913a09d9eceec01e9a02ff6fb6251e5ea` |
