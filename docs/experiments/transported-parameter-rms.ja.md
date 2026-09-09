# atom内4×4の移送付き二次履歴

2026-09-09。過去の二次履歴を現在の接空間へ射影し、見えなくなった方向を落としてから
現在の勾配外積を加える実験。公開optimizerのAPI・既定値は変更していない。

## 式と保存状態

atom aのGram `G_a=J_a.T J_a`を小さい固有分解で白色化する。
有効固有値の閾値は既存の`tangent_rtol=1e-6`。相対閾値以下の方向は基底から除外するが、
テンソルの形状は固定して、その列をゼロにする。`Q_a=J_a B_a`として、

\[
T_a=B_{t,a}^T(J_{t,a}^T J_{t-1,a})B_{t-1,a},\quad
h_a=B_{t,a}^T g_{\theta,t,a},
\]
\[
C_{t,a}=\beta_2T_a C_{t-1,a}T_a^T+(1-\beta_2)h_a h_a^T.
\]

Cは中心化しない二次モーメント。現在のatomの接空間と直交する過去の方向はTで落ちる。
atomをまたぐ相関・他atomへの移動は扱わないため、全modelへの厳密な射影ではない。
座標が変わっただけの場合と、表現可能な方向そのものが変わった場合を単体テストで区別した。

更新計量は、`R_a=B_a.T G_a`として、

\[
M_a=R_a^T\left(\sqrt{C_{t,a}/(1-\beta_2^t)}+\varepsilon I\right)R_a.
\]

一次履歴の移送・全結合PCG再圧縮は従来どおり維持する。更新はMのatomブロックを使い、
単一のEuclidean trust半径0.25を課す。非対角を含む4×4計量なので、前回の単純な
parameter二乗EMAからは、履歴移送だけでなくブロック相関・白色化も変更している。
結果の差を履歴移送だけの因果効果とは解釈しない。

64 atoms・4変数の場合の二次状態は、C `(64,4,4)` FP32で4KiB、
前回のB `(64,4,4)` FP64で8KiB、前回point `(64,4)` FP32で1KiB。
合計13KiB（テンソル本体のみ）。一時領域・一次状態・CUDA Graph buffer等は別。
小行列の計算はFP64、C保存はparameterと同じdtype。
既存AtomRMSの移送の考え方を用いるが、入力はatom_square観測ではなく、
集約済みjgの外積。必要なautograd観測はjgだけ。

## 条件と結果

A100 80GB PCIe MIG 3g.40gb、MNIST8,192 train / 2,000 held-out、batch128。
seed17/29/43、512更新、8更新ごと評価。lr0.05、betas0.9/0.99、eps1e-8。
64 atoms・256 parameters、一次再圧縮damping0.01、PCG上限1,024、rtol1e-5。
初期parameter・batch順・srcコードhashを以前の試行と照合する。
比較元は以前の測定であり、この実験と同時に再実行していない。
最初の2更新を除く更新時間中央値、全期間のPyTorch peak allocatedを報告する。

新方式は3/3試行・1,536更新が数値検査を通過した。以下は3seedの中央値。

| 方式 | 最終精度 | 全体 ms/更新 | 更新求解 ms | peak MiB | 再圧縮有効反復 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 従来separable D | 77.05% | 48.64 | 3.75 | 61.73 | 149 |
| 生のparameter二乗EMA | 71.75% | 73.23 | 0.81 | 59.28 | 385 |
| 対角alpha型EMA | 74.50% | 69.69 | 0.81 | 59.28 | 351 |
| 移送付き4×4二次履歴 | 79.30% | 72.94 | 1.33 | 60.53 | 348 |

移送付き4×4の最終精度はseed17/29/43で80.45% / 77.10% / 79.30%。
前回の生のparameter二乗EMAより全3seedで改善し、従来separable Dより2/3seedで高かった。
ただし80%初回観測は2/3、3回連続確認は0/3。従来separable Dはそれぞれ3/3、1/3だった。
85%は今回も全seed未到達。最終精度の中央値の改善を、安定した80%達成と混同しない。

75%連続確認までの学習時間中央値は23.83秒（全3seed達成）、従来12.30秒。
80%初回観測は達成2seedだけの中央値32.96秒であり、全seedの代表値ではない。

今回の条件では、単純な二乗EMAより有望な最終精度が得られた一方、速度・到達の安定性は
改善したとは言えない。一次履歴の再圧縮を残しているので、反復の負担は依然ある。
「見えなくなった二次履歴を落としたこと」だけが改善原因だとは判定できない。
公開既定値は変更せず、比較可能な実験として残す。


学習率は未調整。単一モデル・3seedの探索的実験で、方式全般への結論ではない。
評価subsetを継続監視しているため独立した最終test精度の推定でもない。
今回の実装は精度検証を優先し、小行列eigh等の実行・同期の最適化は未実施。

## 再現

```sh
for seed in 17 29 43; do
  python -m experiments.local_tangent_learning --data DATA/MNIST/raw \
    --output output/transported-rms/transported_block-s$seed.json \
    --approximation full --parameter-rms transported_block \
    --seed "$seed" --steps 512 --eval-every 8
done
python -m experiments.summarize_parameter_rms output/parameter-rms output/dense-comparison \
  --transported output/transported-rms --output docs/experiments/transported-parameter-rms-results.json
```

[全結果・評価checkpoint・ソースhash](transported-parameter-rms-results.json)。
