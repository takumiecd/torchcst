# MNIST CSTAdamの処理別時間

2026-09-08。[dense比較](dense-comparison.ja.md)と同じ64 atoms・256 parameters、
seed17/29/43・各512更新を、処理別のCUDAイベント計測を追加して再実行した。
公開optimizer・圧縮の式・反復予算・kernelの変更はしていない。

## 測定方法

`experiments/local_tangent_learning.py --stage-timing`で、zero_grad、forward＋loss、
backward、observation完了、一次moment展開、二次moment展開、履歴移送、更新方向求解、
再圧縮、optimizer全体の前後にCUDAイベントを置く。
各区間の間で同期せず、従来どおり更新末尾で同期してイベント時間を読む。
同時にhostの呼び出し所要時間も保存する。CUDAイベント時間はstreamがhostの投入を待つ
空白を含み得るため、純粋なGPU kernelの稼働時間とは異なる。

最初の2更新を除外し、各seedの510更新の区間平均を求め、3seedの平均を取る。
加算できるよう中央値ではなく平均を使用した。nested区間の二重加算を避け、
一次moment展開から履歴移送を引いた残りと、optimizer全体から計測子区間を引いた残りを
「その他」に入れる。区間割合の分母は重複を除いたCUDAイベント時間の合計。
CPU wall timeとCUDAイベント時間は別の時計なので、前回の中央値48.64msと厳密には一致しない。

## 結果

全3試行・1,536更新が数値検査を通過。元の計測なし試行と、同seedの全65評価地点の正解数が一致した。

| 処理 | 平均 ms/更新 | 区間合計に対する割合 |
| --- | ---: | ---: |
| 再圧縮 | 35.90 | 70.0% |
| backward（観測値収集を含む） | 6.33 | 12.3% |
| 更新方向の求解 | 3.83 | 7.5% |
| 履歴移送 | 2.93 | 5.7% |
| forward＋loss | 0.73 | 1.4% |
| その他（zero_grad・moment処理・検証/commit等） | 1.59 | 3.1% |
| 合計 | 51.31 | 100% |

| seed | 計測なし中央値 ms | 計測あり中央値 ms | 差 | 再圧縮平均 ms | 有効反復 中央値 / 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 48.22 | 48.62 | +0.83% | 34.41 | 149.0 / 358 |
| 29 | 48.64 | 48.86 | +0.45% | 36.35 | 148.5 / 475 |
| 43 | 51.34 | 51.51 | +0.31% | 36.96 | 168.5 / 395 |

計測追加による差は1%未満。ただし別process間の実行時変動も含み、純粋なinstrumentation overheadの推定ではない。


## 解釈

再圧縮が主要なボトルネックであることは区間計測で確認できた。
一方、その時間の何割が有効なGram作用で、何割が収束後の固定スケジュールや
kernel起動コストかは、この計測だけでは確定できない。
再圧縮区間にはframe準備、前処理、PCG実行、最終残差検査を含む。

現在の再圧縮PCGは1,024反復分を固定実行し、収束後の作用をmaskする。
今回の有効反復数との差は改善候補だが、反復数の比から時間短縮率を予測してはいけない。
次に改善を試すなら、この再圧縮区間を対象として、解・残差・学習軌道と時間を比較する。
backwardにはoptimizerが要求する観測値の収集も含まれるため、通常CST＋Adamの
backwardと同じ計算量とは限らない。

## 再現

```sh
for seed in 17 29 43; do
  python -m experiments.local_tangent_learning --data DATA/MNIST/raw \
    --output output/stages/s$seed.json --approximation full \
    --steps 512 --eval-every 8 --seed "$seed" --stage-timing
done
python -m experiments.summarize_stages output/stages output/dense-comparison \
  --output docs/experiments/stage-timing-results.json
```

[集約結果・各seed・ソースhash・raw SHA256](stage-timing-results.json)。
