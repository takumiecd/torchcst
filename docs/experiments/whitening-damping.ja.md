# Cholesky白色化のλを一次モーメントから分離する

2026-09-09。数値実装とrunner: `985056e`。設定・数式説明: `2d9f326`。
白色化用の `whitening_damping` を追加し、一次再圧縮・最終更新の正則化0.01は
固定したまま、白色化λだけを0.01 / 1e-4 / 1e-6 / 1e-8に変えた。
固有値版も同じソース・初期値・batch順で再測定した。

## 結果

3 seed（17/29/43）、512 step、全15試行・7680更新がvalidで完走。
時間は各試行で最初の2更新を除いた中央値を取り、さらに3 seed間の中央値。

| 方式 / 白色化λ | 精度中央値 | 精度平均 | 学習step ms | 白色化区間 ms |
| --- | ---: | ---: | ---: | ---: |
| eigen | 80.55% | 80.25% | 18.814 | 0.201 |
| Cholesky 0.01 | 79.45% | 79.62% | 18.938 | 0.059 |
| Cholesky 1e-4 | 80.20% | 80.33% | 18.540 | 0.059 |
| Cholesky 1e-6 | 80.75% | 80.25% | 18.736 | 0.059 |
| Cholesky 1e-8 | 79.95% | 80.28% | 18.479 | 0.059 |

0.01を弱めた全設定で、精度平均は固有値版に近い値へ戻った。
ただし単調な改善ではなく、1e-6の中央値は固有値版を0.20ポイント上回っても
平均は同じ。3 seedの小規模な探索なので最良設定・優位性の確立とはしない。
公開既定値は変えず、白色化λを独立に指定できる形で残す。

| seed | eigen | λ=0.01 | λ=1e-4 | λ=1e-6 | λ=1e-8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 79.45% | 80.10% | 80.20% | 79.00% | 81.15% |
| 29 | 80.75% | 79.45% | 80.15% | 80.75% | 79.95% |
| 43 | 80.55% | 79.30% | 80.65% | 81.00% | 79.75% |

白色化のCUDA event区間は依然としてCholeskyが短いが、step全体の差は小さい。
CUDA eventはhost発行待ちも含む。方式の実行順は固定でタイミング反復はなく、
速度差を一般化しない。warm peak allocatedはeigen 51.407 MiB、各Cholesky 51.408 MiB。

## 128 step時点

今回の学習は512 stepまで行ったが、16 stepごとの評価から128 step時点も比較できる。
128 stepで止めた独立試行ではなく、同じ実行の途中の評価値。

| 方式 / λ | 128 step精度中央値 | seed 17 / 29 / 43 |
| --- | ---: | --- |
| eigen | 76.35% | 74.40 / 77.25 / 76.35% |
| Cholesky 0.01 | 75.75% | 77.40 / 75.75 / 72.55% |
| Cholesky 1e-4 | 77.00% | 74.55 / 77.20 / 77.00% |
| Cholesky 1e-6 | 76.50% | 73.65 / 76.70 / 76.50% |
| Cholesky 1e-8 | 75.75% | 73.80 / 76.50 / 75.75% |

128 stepでは1e-4の中央値が高い。512 stepでは1e-6が高く、評価時点によって順位は異なる。
上の時間表は512 step試行のwarm集計であり、128 stepまでの時間集計とは区別する。

## 実測したRのスケール

Rは各atomの4×4 Gram。全64 atomsの256固有値を集計。
初期点と以後16 stepごとの更新後のmodel点で、計測用にのみeigvalshを実行した。
これは学習stepタイマーの外にあり、更新の基底や方向選択には再利用していない。
Cholesky更新経路に固有値分解を戻したわけではない。

初期ρ中央値はseed 17/29/43でそれぞれ **0.00142 / 0.00193 / 0.00150**。
λ=0.01未満の方向は **68.75% / 68.36% / 69.14%** だった。
したがって、この初期化で0.01は単なる微小な丸め誤差対策より強い。

| λ | 初期ρ<λの割合（seed間の範囲） |
| --- | ---: |
| 0.01 | 68.36–69.14% |
| 1e-4 | 17.19–22.66% |
| 1e-6 | 5.08–7.03% |
| 1e-8 | 2.34–4.69% |

同じJが続くとした1方向のC履歴係数は `(rho/(rho+lambda))^2`。
これは輸送による追加減衰で、さらにbeta2も掛かる。複数方向が混ざる場合や
Jが動く場合はTCT^T全体を見る必要があり、この1方向の式だけでは実際の減衰を決められない。

学習中にρのスケールも変わる。例えばeigen seed17のstep512ではρ中央値が約5.05で、
ρ<0.01は約7.42%だった。「常に0.01が大きい」という意味ではない。
今回の全保存地点の測定では負の固有値は観測しなかった。測定間の全stepまでの保証ではない。

これらは追加減衰が強い可能性と整合するが、精度差の原因を輸送だけに特定はしていない。
λはh・T・Mすべてを変え、基底選択・方向除外もeigen版と異なる。

## APIと互換性

```python
optimizer = CSTLocalAdam(
    model,
    cst=LocalAdamConfig(
        whitening="cholesky",
        first_moment_damping=0.01,
        whitening_damping=1e-6,  # 今回比較した値。一般的な推奨値ではない
        update_damping=0.01,
    ),
    dense=AdamWConfig(),  # 通常のLinear等がある場合
)
```

`whitening_damping=None` が既定で、その場合は従来どおり
`first_moment_damping` を白色化にも使う。独立指定値は有限・正を要求する。
APIの `eps` はCの平方根の外側の安定化であり、この値とは別。
Cの平方根での固有値分解と負固有値clampは引き続き残る。

有効な白色化λをcheckpoint契約に含める。旧checkpointとNone/明示0.01は互換、
異なるλへの読み替えは拒否する。独立指定しても一次の係数は変わらないことと、
小さいλでの保存・再開をテストした。

## 再現と検証

A100 80GB PCIe MIG 3g.40gb、PyTorch 2.6.0+cu126。
MNIST train8192 / held-out2000、batch128、64 atoms / 256 parameters。
lr0.05、betas(0.9,0.99)、一次・更新damping各0.01、stage timing、device_executionを使用。
seedごとにeigen→0.01→1e-4→1e-6→1e-8の順に別processで直列実行した。
同じseedの初期parameter・batch順、および全試行のソースhash一致を集計で検証。
設定探索と評価には同じheld-outを使っているため、最良値には別データでの再検証が必要。

```bash
PYTHONPATH=src:. python -m experiments.whitening_learning \
  --whitening cholesky --whitening-damping 0.000001 \
  --seed 17 --steps 512 --eval-every 16 --stage-timing --spectrum \
  --data /path/to/MNIST/raw --output results/cholesky-0.000001-s17.json
```

CPU全体 **438 passed / 73 skipped**、A100対象 **35 passed**、ruff成功。
モデル復元はoptimizer構築前に行う既存の固定kernel/chart契約に従う。

[集計JSON](whitening-damping-results.json)にseed別結果、評価曲線、スペクトル分位点、
対照の一致確認用hashを保存。rawログは `output/whitening-damping/`、GPU snapshotは
`/home/jovyan/work/srv11/cst-lab/damping-20260909`。
集計・図の再生成には `experiments/summarize_whitening_damping.py` と
`experiments/plot_whitening_damping.py` を使う。

[数式と安定化の位置](../local-adam-math.ja.md)、[最初のCholesky比較](cholesky-whitening.ja.md)。
