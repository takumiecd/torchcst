# 振幅依存kernelの確認と128 step学習率比較

2026-09-09。既存runner `experiments/whitening_learning.py`、数値ソースは
独立白色化λ実装 `985056e` のsnapshotを使用。kernelを固定したまま、
Cholesky λ=1e-4で128 stepの精度をさらに上げられるか比較した。

## kernelの確認

今回のmodelは `AmplitudeBandwidthSeparable`。atomの4座標は振幅a、入力位置2座標、
出力位置1座標である。入力Gaussianのσは0.25固定、出力側の幅だけが振幅に依存する。

$$
s(a)=\operatorname{sigmoid}\!\left(
\frac{\log(a^2+\epsilon_g^2)-2\log\tau}{T}\right),\qquad
\kappa(a)=\sigma_{\rm out}(a)^{-2}
=\sigma_{\rm explore}^{-2}+
(\sigma_{\rm narrow}^{-2}-\sigma_{\rm explore}^{-2})s(a).
$$

設定は `sigma_explore=inf`、`sigma_narrow=0.1`、`tau=0.005`、
`temperature=0.25`、`gate_eps=1e-12`。負の振幅も絶対値が大きいほど狭くなる。
幅自体を線形補間するのではなく、precision κを補間する。

同じkernel実装のFP32診断値:

| 振幅の絶対値 | gate | 出力σ |
| --- | ---: | ---: |
| 0.001 | 約2.56e-6 | 約62.50 |
| 0.005 | 0.5 | 約0.14142 |
| 0.010 | 約0.99611 | 約0.10020 |
| 0.020 | 約0.99998 | 約0.10000 |

小さい振幅で広く、大きい振幅で0.1へ狭まる。数学上は滑らかなsigmoidで、
小さい振幅のatomをifで狭い/広いに切り替える設計ではない。
浮動小数点ではgateが0や1に飽和し得る。

振幅に対する微分には、単なる出力倍率だけでなく幅の変化も入る:

$$
\frac{\partial\{a\phi(\kappa(a))\}}{\partial a}
=\phi+a\frac{\partial\phi}{\partial\kappa}\frac{d\kappa}{da}.
$$

`kernels/_tangent.py` の `amplitude_bandwidth` はこの第2項も計算する。
ゼロ・微小振幅・gate遷移付近・負振幅を含む、specialized/factor_autograd/referenceの
JVP・VJP・Gramのautograd oracleテスト9件が成功した。

## 128 step結果

A100 MIG、MNIST8192 train / 2000 held-out、batch128、64 atoms / 256 parameters。
seeds17/29/43、Cholesky whitening λ=1e-4、一次・更新damping各0.01、
betas(0.9,0.99)、trust regionなし。12試行すべて128 stepで完走。

| lr | seed17 | seed29 | seed43 | 精度中央値 | 精度平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.03 | 75.95% | 75.75% | 77.80% | 75.95% | 76.50% |
| 0.05 | 74.55% | 77.20% | 77.00% | **77.00%** | 76.25% |
| 0.07 | 71.55% | 72.75% | 74.70% | 72.75% | 73.00% |
| 0.10 | 74.35% | 69.85% | 73.40% | 73.40% | 72.53% |

今回の範囲で中央値77%を超える設定は得られなかった。0.03は平均が0.25ポイント高いが、
中央値は下がるので、指標を切り替えて一律の改善とはしない。0.07/0.10は悪化した。

lr=0.05の全seedは、前の512 step実験の128 step評価と精度が完全一致した。
初期parameter・batch順のhashも一致する。今回の試行同士はソースhashも一致する。
学習率はseedごとに0.03→0.05→0.07→0.10の固定順、別processで直列実行。
同じheld-outを設定探索に使用しており、最良設定の独立評価ではない。

全体step時間中央値はそれぞれ18.953 / 19.484 / 18.635 / 18.749 ms。
先頭2更新を除いたwarm集計で、精度の低い設定の方が有用だと解釈しない。
速度測定の反復・順序無作為化はしていない。

## 再現

```bash
PYTHONPATH=src:. python -m experiments.whitening_learning \
  --whitening cholesky --whitening-damping 0.0001 --lr 0.05 \
  --seed 17 --steps 128 --eval-every 16 --stage-timing \
  --data /path/to/MNIST/raw --output lr-results/lr0.05-s17.json
```

`experiments/summarize_whitening_lr.py` で集計する。
[JSON](whitening-lr128-results.json)に全試行、評価曲線、ソースhash、初期値・batch hashを保存。
rawログは `output/whitening-lr128/`、GPU結果は
`/home/jovyan/work/srv11/cst-lab/damping-20260909/lr-results`。
今回はkernel/optimizerの本体を変更していない。

実装の確認先:

- [model構成](../../experiments/mnist_current_api.py)
- [振幅依存kernel](../../src/torchcst/kernels/amplitude_bandwidth.py)
- [振幅依存を含むJacobian](../../src/torchcst/kernels/_tangent.py)
- [oracleテスト](../../tests/test_tangent_ops.py)
- [白色化λ比較](whitening-damping.ja.md)
