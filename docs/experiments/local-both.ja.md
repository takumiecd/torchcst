# alphaとCの両方をatom内に限定する実験

2026-09-09。Cの移送付き4×4履歴を維持し、一次履歴alphaの移送と再圧縮も
同じatom内に限定した。公開既定値は変更せず、実験モード`local_both`として保存した。

## 数式と実装

atom aごとに、新旧の重なりRと現在のGram Sだけを計算する。

\[
R_{t,a}=J_{t,a}^TJ_{t-1,a},\qquad S_{t,a}=J_{t,a}^TJ_{t,a},
\]
\[
b_{t,a}=\beta_1R_{t,a}\alpha_{t-1,a}+(1-\beta_1)g_{\theta,t,a},
\qquad (S_{t,a}+\lambda I)\alpha_{t,a}=b_{t,a}.
\]

Sを対称化しFP64のbatched Choleskyで直接解く。alphaをparameter dtypeへ丸めた後、
各atomの真の相対残差を確認し、その最大が1e-5以下であることを要求する。
ここでの残差はブロック近似した方程式に対するもの。元の全結合方程式との一致は要求しない。
失敗時は既存のjoint commit検査に従う。診断iterations=0は直接法を意味する。

C側の基底B、移送T=B_new.T R B_old、勾配外積、二次EMA、計量は
[前回の移送付きC](transported-parameter-rms.ja.md)と同じ。
更新方向はatom内4×4計量と共通trust半径で解く。
大域的な線形反復求解はなくなったが、trust半径の小さいsecular探索は残る。

R、S、Cは(K,p,p)、alphaは(K,p)。各小行列分解の費用はO(Kp^3)、
これらの行列の保存量はO(Kp^2)。kernel factor、データ、solver一時領域は別。
モデル自体のatom間の影響を消すわけではなく、履歴処理・計量側の近似である。

## 条件

前回と同じA100 80GB PCIe MIG 3g.40gb、MNIST8192 train / 2000 held-out、
64 atoms・256 parameters、batch128、lr0.05、betas0.9/0.99、eps1e-8、lambda0.01、
trust半径0.25、FP32 parameters、TF32無効。seed17/29/43、各512更新、8更新ごと評価。
初期parameter、batch順、srcコードhashは比較元と一致。
比較元は以前の測定で、同時再実行ではない。学習率は調整していない。

時間は最初の2更新を除いた中央値を3seedで集約。GPU同期を更新前後に実施し、
評価・診断・JSON保存は更新時間から除外する。到達時間は初回準備込みの学習累積秒。
ピークはデータ・状態・一時領域等を含むPyTorch allocatedであり、状態テンソル単体ではない。

## 結果

全3試行・1,536更新が検査に合格。

| 方式 | 最終精度中央値 | ms/更新 | 更新求解 ms | peak MiB | 80%初回観測 | 80%連続3回 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 従来separable D | 77.05% | 48.64 | 3.75 | 61.73 | 3/3 | 1/3 |
| Cのみatom内、alpha全結合PCG | 79.30% | 72.94 | 1.33 | 60.53 | 2/3 | 0/3 |
| **alphaとCの両方をatom内** | **80.85%** | **19.62** | **1.66** | **72.82** | **3/3** | **1/3** |

最終精度はseed17/29/43で81.15% / 80.85% / 80.45%。
前回のCのみatom内より全3seedで高く、中央値の更新時間は約3.7倍速かった。
75%連続確認まで5.05秒、80%初回観測まで8.25秒（ともに全3seed達成、中央値）。
80%連続確認は1seedだけで10.01秒。85%は全seed未到達。

精度と時間の両方に改善が見えた一方、80%の連続確認はまだ安定していない。
ピークallocatedメモリは増えており、O(Kp^2)の状態量だけで実使用量の削減を主張できない。
増加の内訳は今回未分析。小行列のlibrary呼び出しや同期の最適化も未実施。
この単一モデル・3seedの結果を大規模での性能や方式全般へ一般化しない。

## 検証と再現

ブロック対角dense oracleとの解の一致、重なった別atomの係数が移送に混入しないこと、
ゼロGram/ゼロ右辺を正のdampingで扱えることを単体テストした。

```sh
for seed in 17 29 43; do
  python -m experiments.local_tangent_learning --data DATA/MNIST/raw \
    --output output/local-both/local_both-s$seed.json --approximation full \
    --parameter-rms local_both --seed "$seed" --steps 512 --eval-every 8
done
python -m experiments.summarize_parameter_rms output/parameter-rms output/dense-comparison \
  --transported output/transported-rms --local-both output/local-both \
  --output docs/experiments/local-both-results.json
```

[全試行・評価checkpoint・到達時間・ソースhash](local-both-results.json)。
