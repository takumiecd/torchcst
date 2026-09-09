# parameter勾配の二乗EMAとalpha型のスケール補正

2026-09-09。`g_theta = J.T g_w`の二乗EMAを更新の分母へ使う案を試した。
公開API・既定値は変えず、実験用optimizer subclassで実装した。

## 比較する式

一次履歴は従来どおり。

\[
b_t=\beta_1 J_t^T J_{t-1}\alpha_{t-1}+(1-\beta_1)g_{\theta,t},
\qquad (G_t+\lambda I)\alpha_t=b_t.
\]

更新は`b_hat=b/(1-beta1^t)`を右辺とし、下記の対角計量Mで
`min b_hat.T delta + delta.T M delta/(2 lr)`、`||delta||<=0.25`を解く。
対角のsecular解を使い、全体の線形反復求解やeighは更新側に使わない。
半径内なら要素ごとのAdam型割り算、境界では共通shiftを加える。
再圧縮は全atom間の結合を維持したPCGのまま。

**raw:**

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_{\theta,t}^{\odot2},
\qquad M_t=\operatorname{diag}(\sqrt{\hat v_t}+\varepsilon).
\]

勾配の二乗をparameter座標で蓄積する。`J.T (g_w^2)`でも、従来の
`diag(J.T D J)`でもなく、parameter勾配全体を合計した後の二乗。
分子が移送された一次履歴なので、通常のCST＋Adamとも異なる。

**alpha_diagonal（今回採用したalpha型の一解釈）:**

\[
s_t=\operatorname{diag}(G_t)+\lambda,
\quad a_t=g_{\theta,t}\oslash s_t,
\quad v_t=\beta_2v_{t-1}+(1-\beta_2)a_t^{\odot2},
\quad M_t=\operatorname{diag}\{s_t\odot(\sqrt{\hat v_t}+\varepsilon)\}.
\]

瞬時勾配を対角再圧縮した係数aを二乗EMAへ入れ、現在のsで分母を元の座標へ戻す。
保存済み一次moment alphaを二乗するものではない。EMAのv自体は移送しないため、
幾何学的に厳密な二次momentではない。スケールが時間変化する場合の経験的近似。
このsは二次moment専用であり、一次履歴の再圧縮を対角化していない。

## 条件

A100 80GB PCIe MIG 3g.40gb、FP32、TF32無効、host1 thread。
MNIST8,192 train / 2,000 held-out、batch128、64 atoms・256 parameters。
seed17/29/43、512更新、8更新ごとに評価。lr0.05、betas0.9/0.99、eps1e-8、
再圧縮damping0.01、PCG上限1,024、rtol1e-5。
二乗EMAにはバックプロパゲーション中のjgだけを要求し、不要な行・列二乗観測は要求しない。

baselineは前回[dense比較](dense-comparison.ja.md)のspectral＋pair。
同seedの初期parameter・batch順・srcコードhashの一致を検証する。
新方式の順番はseedごとに入れ替えたが、baselineは以前の測定であり同時に再測定していない。
時間は各更新前後の同期を含み、評価・診断・JSON保存を除く。
準備後時間は最初の2更新を除いた中央値。メモリは常駐データとGraph buffer等を含む
PyTorch peak allocated。失敗試行は未完了として残し、完了runだけの値には完了数を付す。

## 結果

新方式の全6試行・3,072更新が数値検査を通過した。以下は3seedの中央値。

| 方式 | 最終精度 | 更新求解 ms | 全体 ms/更新 | peak MiB | 再圧縮有効反復 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 従来のseparable D | 77.05% | 3.75 | 48.64 | 61.73 | 149 |
| g_theta二乗EMA | 71.75% | 0.81 | 73.23 | 59.28 | 385 |
| 対角alpha型二乗EMA | 74.50% | 0.81 | 69.69 | 59.28 | 351 |

| 方式 | seed17最終 | seed29最終 | seed43最終 | 80%初回観測 | 80%連続3回確認 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 従来のseparable D | 76.15% | 77.35% | 77.05% | 3/3 | 1/3 |
| g_theta二乗EMA | 71.75% | 74.90% | 71.75% | 0/3 | 0/3 |
| 対角alpha型二乗EMA | 74.60% | 74.50% | 71.35% | 1/3 | 0/3 |

75%連続確認までの準備込み学習時間中央値は、従来12.30秒、raw25.64秒、alpha型18.28秒。
85%は全方式で未到達。alpha型の80%初回観測はseed29の1試行のみなので、
その到達時間38.45秒を全seedの代表値とは扱わない。

今回の固定設定では、更新求解を約0.81msへ軽くできた一方、精度と全体時間では従来方式に及ばなかった。
同じPCG予算でも、軌道の変化で再圧縮の有効反復が増えた。全体時間を分解して再計測したわけではないが、
全体が遅くなった結果と整合する。2方式とも一次履歴の再圧縮を厳密経路に残しているため、
今回の変更だけでは主要な再圧縮コストはなくならない。
alpha型はrawより中央値で良かったが、全seedで優位ではない。


学習率の調整は今回行っていない。単一の小さいMNISTモデルでの結果であり、
parameter二乗EMAやalpha型の方式全般の限界を示すものではない。
評価subsetを継続監視しているため、独立した最終test精度の推定でもない。

## 再現

```sh
bash experiments/run_parameter_rms_a100.sh DATA/MNIST/raw output/parameter-rms
python -m experiments.summarize_parameter_rms output/parameter-rms output/dense-comparison \
  --output docs/experiments/parameter-rms-results.json
```

[各seed・全評価checkpoint・到達時間・ソースhash](parameter-rms-results.json)。
