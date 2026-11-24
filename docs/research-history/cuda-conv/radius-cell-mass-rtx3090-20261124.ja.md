# RTX 3090 での半径セル質量 Conv2d

[English](radius-cell-mass-rtx3090-20261124.md) | **日本語**

RigL 比較では半径セル質量を使う。この計測は Chebyshev M8 と rational radius 座標で行った。
Torch 参照経路は各軸でセル積分を計算し、因子を `W = F_out @ F_in.T` で縮約して
PyTorch Conv2d を適用する。backward では同じセル積分と rational radius 則を微分する。
層を CPU で初期化してから CUDA に移して eager 実行する。

出力 Chart は128サイト、base tile 1とした。入力 Chart は1×1で96サイト・base tile 3、
3×3で864サイト・base tile 27とした。両軸で spacing 1、merge factor 2を用いる。
サイト数を base tile で割るとそれぞれ2のべきになる。Chart は周期境界を持ち、入力は
channel、高さ、幅の順にパッチを平坦化する。rational radius の上限は32、学習可能な
割合の初期値は0.15とした。半径、振幅、中心は計測中も optimizer で更新する。

RTX 3090 の独立した2プロセス（PID 3879、3891）で、channels-last FP32 の合成入力
`[256, 96, 8, 8]`、stride 2、groups 1、Atom 数1,774、固定 dY を使った。
padding は1×1で0、3×3で1。Torch 2.6.0+cu126、CUDA 12.6、cuDNN 9.5.1、
driver 580.95.05を使用し、TF32は無効にした。各 seed 内の CST 経路で初期 Atom、
入力と dY を共有した。各経路で forward、dX を含む backward、個別の fused CSTAdamW
または AdamW を計測し、学習率は0.001、weight decay は0とした。表は測定順を反転した
独立2プロセスの CUDA event 中央値の範囲を示す。各条件で warmup 8回、計測24回。
allocated と reserved は生存テンソルを含む PyTorch allocator のピークである。

| 空間カーネル | 経路 | 層ステップ (ms) | ピーク allocated (MiB) | ピーク reserved (MiB) |
| --- | --- | ---: | ---: | ---: |
| 1×1 | Dense | 0.275–0.300 | 40.52 | 108 |
| 1×1 | Materialized | 12.940–14.852 | 222.54–222.59 | 352 |
| 1×1 | Factored | 12.928–12.984 | 134.16–134.18 | 176 |
| 1×1 | Composed | 12.945–15.366 | 61.68–61.73 | 118 |
| 3×3 | Dense | 1.002–1.012 | 42.45 | 90 |
| 3×3 | Materialized | 17.174–17.509 | 2323.93–2323.96 | 2456 |
| 3×3 | Factored | 23.415–23.426 | 2838.18–2838.21 | 2860 |
| 3×3 | Composed | 11.408–12.059 | 107.63–107.66 | 130 |

小さい1×1・3×3の CPU と CUDA の条件で、独立した数値積分と有限差分を使い、
Y、dX、全4フィールドの Atom 勾配を確認した。性能計測の条件でも、計測前に Y、dX、
Atom 勾配の各フィールドを CPU FP64 参照と照合した。Composed の相対 L2 誤差の最大値は
両 seed の全照合テンソルで5.09e-7。各経路で optimizer 32ステップを完了した。
計測は単一 FP32 層を対象とする。学習済み checkpoint、BF16 autocast での学習、
CUDA Graph、プロセス全体のメモリ、dispatcher 採用の検証は今後の対象とする。

この3×3条件では composed を性能比較の基準にできる。1×1ではピークメモリが減り、
速度は factored と同程度か遅かった。次の CUDA 候補では、Linear で使った重み合成と
`dW → dAtoms` の pullback を分ける構成を流用し、入力と出力の Fourier routing を
各軸で処理する。既存の正方 wide recipe には別の軸条件がある。今回の Conv2d は Torch
参照経路なので、次は因子生成と pullback を分けてプロファイルしてから最適化する。
[周期点 Gaussian の結果](composed-weight-rtx3090-20261124.ja.md) は別の profile と計測条件に対応する。

## 再現

計測したソースは `1f6389c17d55e2f3baeb64ffc3458cc0f2c84155`。
driver、ソースアーカイブ、SHA照合済み結果は worktree の Git 管理対象外の
`output/conv-radius-20261124/` と Kyutech のホームからの相対パス
`cst-conv-radius-20261124-1f6389c1/` に保存した。ソースを `source/` に展開し、
driver のあるディレクトリから実行する。

```sh
PYTHONPATH=source/src python -m pytest -q source/tests/test_cst_conv_radius_cell_mass.py
OMP_NUM_THREADS=2 PYTHONPATH=source/src python measure.py --seed 17 --order dense,materialized,factored,composed --source-commit 1f6389c17d55e2f3baeb64ffc3458cc0f2c84155 --output run-17.json
OMP_NUM_THREADS=2 PYTHONPATH=source/src python measure.py --seed 23 --order composed,factored,materialized,dense --source-commit 1f6389c17d55e2f3baeb64ffc3458cc0f2c84155 --output run-23.json
```

| ファイル | SHA-256 |
| --- | --- |
| `source.tar.gz` | `bace2d55d9e4a69e15818dc3eeb9be508bf1d5290b970f57c84b62c07fe40652` |
| `measure.py` | `99fd804339e3ef7717beb88fa226c3b533de45cd0bdd4b1a1ca267bb524bcb99` |
| `run-17.json` | `37fc3f2ff818cfcb45b7e7a7cf881e2ef1629dfaabd1ad4c84a8b9ca9ec9a29e` |
| `run-23.json` | `794344df8cabee76ed6bc0141cb1c6bdd649a33975d05923b5fde03e198131b3` |
