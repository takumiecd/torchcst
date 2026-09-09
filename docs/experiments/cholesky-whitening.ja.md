# 固有値白色化と正則化Choleskyの比較

2026-09-09。実装 `37bff16`、config位置引数の互換性修正 `ffb2cee`。
公開 `CSTLocalAdam` の一次モーメント、Cの平方根、更新solveを固定し、
RからBを構成する方法だけを比較した。正則化により計量自体も変わるので、
数学的に等価な処理の速度比較ではない。

## 結果

各seedのwarm中央値を取り、その3 seed間の中央値を集計。

| 指標 | eigen（既定） | cholesky |
| --- | ---: | ---: |
| 完走 | 3 / 3 | 3 / 3 |
| 最終精度 | 80.55% | 79.45% |
| 学習step全体 | 19.341 ms | 19.143 ms |
| 白色化区間・CUDA event | 0.205 ms | 0.060 ms |
| 白色化区間・host | 0.674 ms | 0.266 ms |
| C展開全体・CUDA event | 3.903 ms | 3.524 ms |
| optimizer全体・CUDA event | 12.024 ms | 11.724 ms |
| 更新solve・CUDA event | 0.897 ms | 0.913 ms |
| warm peak allocated | 51.407 MiB | 51.408 MiB |

白色化区間は約3.4倍速くなったが、学習全体の差は約1.0%に留まった。
この3試行で全体の安定した速度改善を確立したとは言わない。
CUDA event区間はhostの発行待ちも含み、純粋なkernel演算時間ではない。
親子の区間は重複するため足し合わせない。計測用のevent等も両方式に入る。

| seed | eigen精度 | cholesky精度 | eigen step ms | cholesky step ms |
| --- | ---: | ---: | ---: | ---: |
| 17 | 79.45% | 80.10% | 18.689 | 18.975 |
| 29 | 80.75% | 79.45% | 19.447 | 19.346 |
| 43 | 80.55% | 79.30% | 19.341 | 19.143 |

Choleskyは2/3 seedで精度が低く、中央値は1.10ポイント低かった。
追加の履歴減衰は理論上あるが、この差の原因だと実験で分離したわけではない。
全3072更新がvalidで完走。今は既定値をeigenのままにし、Choleskyを選択肢として残す。

## 数式の差分

既定版はRの固有値で方向を選び、採用方向を単位長にする。
比較版は `R + lambda I = L L^T`、`B = L^{-T}` とし、
`T = B_t^T S B_previous`、`h = B_t^T g_theta`、CのEMAとMへの変換は同じ式を使う。
`lambda = first_moment_damping = 0.01`。

正則化版では `Q^T Q = I - lambda B^T B`。Rのスペクトルによる方向の
採否判定はなく、弱い方向は滑らかに収縮する。一方で同じJが続いても
Tは一般にIではなく、Cの履歴が追加で弱まる。

CのPSD平方根には今もeighと負固有値のclampが残る。
Choleskyの成功判定・有限値検査も残る。全分岐・全同期を取り除いた実装ではない。
Bは小さい単位行列に対する三角solveで構成し、FP64で保存する。
一次再圧縮とlambdaは共通だが、Cholesky因子の再利用はまだしていない。
詳しい式は [数式ドキュメント第9節](../local-adam-math.ja.md)。

## 再現条件

- A100 80GB PCIe MIG 3g.40gb、PyTorch 2.6.0+cu126。
- MNIST train 8192、held-out 2000、batch 128、64 atoms / 256 CST parameters。
- seeds 17 / 29 / 43、各512 step、16 step毎に評価。
- lr 0.05、betas (0.9, 0.99)、eps 1e-8、first/update damping 各0.01。
- 公開CSTLocalAdam、device_execution=True、stage timing有効。
- データは計測前にGPUへ移動。warm集計は最初の2更新を除く。学習自体は全512更新。
- 同じseedの初期parameter・batch順・実行ソースのSHA256一致を集計時に検査。
- 各seedについてeigen→choleskyの順に別processで直列実行。順序の無作為化・
  同一seedのタイミング反復はしていない。
- 過去レポートの計測値は混ぜず、今回両方式を再測定した。

```bash
PYTHONPATH=src:. python -m experiments.whitening_learning \
  --whitening cholesky --seed 17 --steps 512 --eval-every 16 \
  --stage-timing --data /path/to/MNIST/raw --output results/cholesky-s17.json
```

`eigen`にも切り替え、3 seedで実行。集計は
`python -m experiments.summarize_whitening output/whitening/results --output docs/experiments/whitening-results.json`。
[集計JSON](whitening-results.json)に曲線・seed別結果・ソースhashを保存。
rawログはローカル `output/whitening/`、GPU snapshotは
`/home/jovyan/work/srv11/cst-lab/whitening-20260909`。

## 検証とAPI

- CPU全体: 433 passed / 73 skipped。ruff成功。
- A100対象: 30 passed。混在modelのFP32/FP64、通常/deferred、checkpoint続行を検証。
- dense可視空間oracleでTCT^TとMを検証。
- 正則化ゼロ・正定値・方向除外なしの場合のeigen/Cholesky計量一致を確認。
- 特異R付近のBが有限・連続で、Bの構成がeighを呼ばないことを確認。
- 既存eigen checkpointは互換、異なるwhitening方式間の読み込みは拒否。
- GPU実行後の追加変更はconfig新フィールドの配置を末尾に戻す位置引数互換性修正と
  そのCPUテストのみで、比較に使う数値処理・keyword設定は同じ。

```python
optimizer = CSTLocalAdam(
    model,
    cst=LocalAdamConfig(whitening="cholesky", first_moment_damping=0.01),
    dense=AdamWConfig(),  # 通常Linear等がある場合
)
```

次に試すなら正則化強度の比較と因子の再利用。ただし今回は設定探索まで広げていない。
