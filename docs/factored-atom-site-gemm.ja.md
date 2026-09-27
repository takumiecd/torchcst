# 5% mapped GEMMの列方向距離計算の共有

2026-09-27。A100上のatom密度5%、M=128、CST tile64×64、N=K=4096/8192を対象に、Triweight重み小片を生成する距離計算を変更した。

## 方法

従来は各atomと16×32サイト小片の全512点について、4座標の差と二乗を計算していた。Strip+Torusのサイト座標は、最初の2成分が出力行と入力列に依存し、残りの2成分は入力列だけに依存する。そこで各atomについて、後者の二乗和を32列で一度計算して16行へ共有する。

```text
xy[n,k] = (sx[n,k] - cx)² + (sy[n,k] - cy)²
zw[k]   = (z[k] - cz)² + (w[k] - cw)²
distance²[n,k] = xy[n,k] + zw[k]
```

寄与のプロフィール、支持範囲、atom所属、全Wを保持しない方針は変えない。浮動小数点の加算順序は変わるため、canonicalなWの点検と全出力を同じ許容誤差で照合した。**atom×サイトの組数やprofile評価回数はまだ減っていない。** 16×32小片内で、z/wの差・二乗の計算回数をatomあたり512回から32回へ減らした。4成分合計の差・二乗は2048回から1088回となる。これは論理的な演算数であり、GPU命令数そのものではない。

`block_fused`に実験用`COLUMN_FACTORED`を追加し、A100の測定済み5%形状だけ既定で選択する。`FusedConfig(column_factored=True)`は4次元、BA1、late-reduceを要求する。その他の形状や5%未満の既定は変えていない。

## 同一run比較

同じ層・atom・入力・dense Wで各経路を交互にCUDA Graph 3ラウンド、rep=20ms測定。準備込み。A100 80GB PCIe MIG 3g.40gb、42 SM、FP32、TF32無効、seed21。atom数は4096²で838,861、8192²で3,355,443。dense Wは事前生成。全7経路がcanonical W 1,152点および同じWによるdense出力との`atol=rtol=3e-5`照合に合格。

| N=K | 旧8分割＋座標hoist | 新既定・列共有 | 時間短縮 | dense・事前生成W | 新既定/dense |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16.320ms | **14.592ms** | **10.6%** | 0.660ms | 22.1倍 |
| 8192 | 65.022ms | **58.523ms** | **10.0%** | 3.027ms | 19.3倍 |

4096²の旧3サンプルは16.284～16.402ms、新は14.591～14.653ms。8192²の旧は65.017～65.130ms、新は58.509～58.571ms。速度差は各runの変動より大きい。前回の旧BA8基準23.637/85.239msからの累積時間短縮は38.3%/31.3%だが、これは別runのため参考値とする。

A100で関連69テスト合格。端数形状、I/B境界、Torus seam、CUDA Graph replay後のatom更新を含む。ローカル全体は427件合格、CUDA専用173件skip。実装はforward prototypeであり、GPU backwardと学習後の偏ったatom配置は未検証。

## 再現

- 候補測定source commit `9dd111c756f9d1253848a84bba2fa67212824043`、archive SHA256 `bca2696958163c1bdb886fd5fd14ec1f8f83783fe30964bd0ed51ec592af7c34`。
- 既定接続・最終測定source commit `afae58d4543e2444a75cfcea150883d1ebdb31c4`、archive SHA256 `785cd88a9de6a868f293bd6da6780b4de8b3d3c6c7c1a8e5f935c984336d665b`。remote snapshot `srv11/cst-lab/torchcst-factor-afae58d`。展開前archive hash一致。
- 最終JSONは`output/triton-a100-20260927/factored-final-4096.json`と`factored-final-8192.json`。SHA256は順に`3a613ded6f9139595bd8b0adf2bbfeaefbd9b015a5a903ab1b436a9155e67534`、`2196514bc155c5a2aaa00743bbe6cca5dd5cc97e661265ed92232f66b7d7a991`。A100テストlog SHA256 `334dcd8a25948cb7dd3d28d3681ab793cd6f7ad146c207e5603eff46fd33bf5e`。すべてremote/local hash一致。

```bash
PYTHONPATH=src:. python -m pytest -q --color=no tests/test_fused_compute_configs.py tests/test_block_strip_linear.py
PYTHONPATH=src:. python -u -m prototypes.benchmark_split_k_five_percent --output final-4096.json --source-commit afae58d4543e2444a75cfcea150883d1ebdb31c4 --size 4096 --batch 128 --full --splits 8 --column-factored
PYTHONPATH=src:. python -u -m prototypes.benchmark_split_k_five_percent --output final-8192.json --source-commit afae58d4543e2444a75cfcea150883d1ebdb31c4 --size 8192 --batch 128 --full --splits 8 --column-factored
```

次にprofile評価回数そのものを減らすには、現在の16×32小片より細かい粒度で支持範囲を安全に判定するか、支持範囲が共通するatom群をまとめる必要がある。既存診断では16/32列小片を丸ごと除外できる割合が数%以下で、単純なbox判定は遅かった。判定・詰め直し費用を含むforward全体で評価する。
