# atom-local optimizerのtrust-regionを外す

2026-09-09。実験用local_both optimizerに、半径制限もクリッピングも行わない更新を追加した。
公開`torchcst.CSTAdam`の既定値には未導入。実装は`experiments/parameter_rms.py`、
`local_first_moment.py`、`transported_parameter_rms.py`、`unconstrained_update.py`。

## 変更

C・alphaのatom内移送・再圧縮を維持し、更新だけを

\[
(M_{t,a}+\mu I)\Delta\theta_{t,a}=-\eta\widehat b_{t,a}
\]

の4×4直接解へ変更した。mu=0.01は学習率で割る前のMに加える。
alpha再圧縮のlambda=0.01とは別。FP64 batched Choleskyで解き、
FP32へ丸めた後の各ブロックの相対残差を検査する。
半径探索・ノルムによるclip・半径を超えた更新の拒否は行わない。
shape/dtype/deviceと有限性、残差、joint commitの検査は維持。

従来のtrustありMには追加muを入れていないため、純粋な半径有無だけのablationではない。
「正則化された直接更新」への変更と学習率の小さいsweepを評価する。

## 条件

A100 80GB PCIe MIG 3g.40gb、MNIST8192 train / 2000 held-out、batch128、
64 atoms/256 parameters、seed17/29/43、512更新、8更新ごと評価。
betas0.9/0.99、eps1e-8、lambda0.01、mu0.01。FP32、TF32無効。
学習率0.005 / 0.01 / 0.05を事前に選んで比較。比較元は以前のlocal_both lr0.05・半径0.25。
初期parameterとbatch順を照合。同時期の比較元再測定はしていない。

時間は準備2更新を除く中央値。評価・診断・ファイルIOは更新時間に含めない。
メモリはdata・model・optimizer・一時領域等を含むPyTorch peak allocated。
観測精度は同じheld-out subsetを繰り返し確認したもので、独立した最終test精度ではない。

## 結果

新規9試行・4,608更新がすべて数値検査合格。3seedの中央値。

| 条件 | 最終精度 | 全体 ms/更新 | 更新求解 ms | peak MiB | 80%初回 / 連続確認 |
| --- | ---: | ---: | ---: | ---: | ---: |
| trustあり lr0.05 | 80.85% | 19.62 | 1.66 | 72.82 | 3/3・1/3 |
| trustなし lr0.005 | 74.75% | 19.18 | 0.93 | 68.91 | 0/3・0/3 |
| trustなし lr0.01 | 78.90% | 18.93 | 0.91 | 68.91 | 1/3・1/3 |
| trustなし lr0.05 | 80.55% | 18.93 | 0.90 | 68.91 | 2/3・1/3 |

lr0.05の最終精度はseed17/29/43で79.45% / 80.75% / 80.55%。
最大変位ノルムは約0.955 / 0.907 / 0.953で、旧半径0.25を超える更新が実際に通っている。
全seedの80%安定達成や85%到達には至っていない。

今回のmu=0.01・lr0.05では、trust-regionなしでほぼ同程度の最終精度を得た。
小行列更新は速くなったが全体時間の改善は小さい。低いlrは512更新の固定予算では精度が低かった。
速度・精度差は3seedの探索結果であり、純粋な半径制約の効果や有意差は主張しない。
公開既定値は変更しない。


## 再現

```sh
for seed in 17 29 43; do
  for rate in 0.005 0.01 0.05; do
    python -m experiments.local_tangent_learning --data DATA/MNIST/raw \
      --output output/no-trust/lr$rate-s$seed.json --approximation full \
      --parameter-rms local_both --no-trust --update-damping 0.01 \
      --lr "$rate" --seed "$seed" --steps 512 --eval-every 8
  done
done
python -m experiments.summarize_no_trust output/no-trust output/local-both \
  --output docs/experiments/no-trust-results.json
```

[全結果・checkpoint・最大変位・raw SHA256](no-trust-results.json)。
