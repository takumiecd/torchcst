# dense・CSTの同一測定手順での比較

2026-09-08。既存のCST内の高速化率をdenseとの速度差と混同しないため、
dense＋Adam、CST＋通常のAdam、CST＋現在の一次optimizerを同じrunnerで再測定した。

## 条件

A100 80GB PCIe MIG 3g.40gb、PyTorch 2.6.0+cu126、FP32、TF32無効、host 1 thread。
MNIST 8,192 train / 2,000 held-out、batch128、seed17/29/43、各512更新。
同seedのbatch順は全方式で一致、CSTの初期parameterは3方式で一致する。
denseは784→10・biasなし（7,840 parameters）、CSTは64 atoms・256 parameters、
AmplitudeBandwidthSeparable・factored backend。denseとCSTの初期関数は一致させていない。

通常のAdamはlr0.001、betas=(0.9,0.99)、eps=1e-8、foreach=True。
CST＋通常のAdamではlr0.05も追加する。現在のCSTAdamはlr0.05、spectral＋pair、
separable D、radius0.25、damping0.01、再圧縮PCG上限1,024、rtol1e-5。
通常のAdamでは再圧縮やtrust solverを使わない。JSON protocol内の該当設定はCSTAdam専用。
これは限られた設定の比較であり、各方式を十分に学習率調整した最良性能の比較ではない。

各方式・seedを独立processで直列実行し、実行順をseedごとに変えた。
各更新前後でGPU同期。学習時間はzero_grad・forward・backward・step・完了待ちの合計。
初回準備を含む累積時間と、最初の2更新を除く1更新時間中央値を区別する。
データ・model構築は計測外。評価・診断・JSON書き出しも学習時間外で、wall timeに含む。
コンパイルのディスクcacheは消していない。

8更新ごとの評価で75/80/85%の初回観測と3回連続確認を記録する。
未達成を512更新時点の時間で代用しない。連続確認は3回目の観測時点の時間。
GPUメモリは全512更新・評価中のPyTorch peak allocatedであり、
常駐データ・model・optimizer・Graph buffer・一時領域を含む。
driver全体の使用量やoptimizer stateだけのサイズではない。

## 結果

全12試行・6,144更新が完了。全方式でlossの有限性、CSTAdamではさらに既存の求解数値検査を確認した。
ソースhash・batch順・CST初期parameterの一致を集約時に検証した。
以下は各seed内の値を集約した3seedの中央値。

| 方式 | 準備後 ms/更新 | dense比 | 512更新目の精度 | peak MiB |
| --- | ---: | ---: | ---: | ---: |
| dense＋Adam（lr0.001） | 0.78 | 1.0× | 88.05% | 47.38 |
| CST＋Adam（lr0.001） | 5.11 | 6.5× | 67.80% | 48.61 |
| CST＋Adam（lr0.05） | 5.87 | 7.5× | 74.45% | 48.61 |
| CST＋現在のoptimizer（lr0.05） | 48.64 | 62.3× | 77.05% | 61.73 |

倍率は各方式の中央値の比。CSTAdamは通常のCST＋Adam（lr0.05）の約8.3倍。
512更新の累積学習時間はdense0.55秒、通常CST＋Adamのlr0.001で2.58秒、lr0.05で2.86秒、CSTAdamで30.83秒だった。

| 方式 | 75%連続確認 秒 / 達成数 | 80%初回観測 秒 / 達成数 | 85%連続確認 秒 / 達成数 |
| --- | ---: | ---: | ---: |
| dense＋Adam（lr0.001） | 0.100 / 3/3 | 0.100 / 3/3 | 0.164 / 3/3 |
| CST＋Adam（lr0.001） | 未達 / 0/3 | 未達 / 0/3 | 未達 / 0/3 |
| CST＋Adam（lr0.05） | 2.154 / 1/3 | 未達 / 0/3 | 未達 / 0/3 |
| CST＋現在のoptimizer（lr0.05） | 12.302 / 3/3 | 15.211 / 3/3 | 未達 / 0/3 |

時間は初回準備を含む学習累積時間。達成した試行だけの中央値なので、1/3と3/3の値は同等に比較しない。
80%初回観測の中央値はdense0.100秒に対してCSTAdam15.211秒で、約153倍。
80%連続確認はdense3/3、CSTAdam1/3、通常CST＋Adamは両学習率とも0/3だった。
通常CST＋Adamのlr0.05はlr0.001より改善したが、この予算では80%には届かなかった。
CSTAdamで追加計算による精度改善は見えるものの、denseの速度・精度を上回る結果ではない。

各seedの準備後時間（ms）はdenseが0.826 / 0.781 / 0.692。単発のGPU測定の変動を含むため、精密な安定レイテンシの推定とは扱わない。


## 判断と限界

この小さい層では、現状のCSTに速度・精度・総allocatedメモリの優位性はない。
256 parametersへの圧縮は約30.6倍だが、パラメータ数の減少が総GPUメモリ削減に直結していない。
CST＋通常のAdamとの差は履歴移送・再圧縮・trust solver等を含む追加コストであり、
この測定だけで個別処理の割合やcuBLASとのkernel効率差は確定できない。

次の最適化は、圧縮・復元を維持しながらsolverの作用回数・固定スケジュールの無駄を減らすことと、
CST forward/backward自体の負担の削減を分けて進める。
大きな層でのメモリ対時間の交換条件は今回未測定であり、小さいMNISTの結果から推定しない。
評価subsetを繰り返し参照しているため、最終的な独立test精度の推定ではない。

## 再現

```sh
bash experiments/run_dense_comparison_a100.sh DATA/MNIST/raw output/dense-comparison
python -m experiments.summarize_dense_comparison output/dense-comparison \
  --output docs/experiments/dense-comparison-results.json
```

[集約値・各seedの全評価checkpoint・ソースhash](dense-comparison-results.json)。

[追加測定：処理別の時間内訳](stage-timing.ja.md)で再圧縮が約70%を占めることを確認した。
