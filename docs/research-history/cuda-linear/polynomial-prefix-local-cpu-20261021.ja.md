# 小sigmaの多項式をタイル内累積から読む場合のCPU screen

2026-10-21。ユーザーの仮説は、左右の累積値だけ読むことで、小sigmaでも
入力参照量を減らせる可能性がある、というもの。GPU実装の再開ではない。

現在のP=(1-(r/(sqrt(11)*sigma))²)^4_+は支持内で8次。
**sigmaを小さくしても次数は増えず、入力momentは常に9種類。**
block中心c、幅Bでu=(x-c)/Bとし、P=sum_(k=0)^8 a_k(mu,sigma)u^k。
C_k(j)=sum_(i<j) X_i u_i^kをblock内に用意すると、
H=sum_k a_k*(C_k(hi)-C_k(lo))で区間を評価できる。
muとsigmaの微分も同じ9種類を使う。端でP=P'=0なので区間index変化の境界項は消える。

ただし「2値」は各9成分、すなわち1fragmentにつき18 scalar/batch。
2端点参照が2scalarだけという意味ではない。
small sigmaの狭い支持では、大きい係数とmoment差の相殺が精度を悪化させる可能性もある。

## 比較の境界

- 同じ8次Pの局所直接、blockでgroup化した直接、block内9moment prefixを比較。
- group化した直接も入力を共有できる。入力共有の利益をprefix固有の利益と扱わない。
- 正規化E/Em/Esは全方式とも支持内の直接geometry和を用いる。
  入力依存縮約だけの差を切り分ける。prefixでL2まで定数時間化した測定ではない。
- 幾何・係数構築・group化・prefix構築・正規化込みのCPU時間。
- query scalar数とgroup入力site数は論理モデル。GPU transaction、cache miss、帯域の実測ではない。
- 32/64/128点tileに対し、内部moment blockは4/8/16点またはtile全体でresetする。
  内部block4/8は小さいGPU launch tileを要求するという意味ではない。
- FP32 prefixに対しても係数はFP64で作ってcastする、楽観的な精度screen。
  各fragmentの結果の加算もFP64。FP32実GPUの精度保証にはならない。

精度screenはsigma .1/.15/.3/.6/1、中心33phase、tile32/64/128、block4/8/16/tile、
FP32/FP64。時間screenはA1/8/32/128、sigma .105/.3/.8、3stepでX・mu・sigmaを更新。
同じtile内に中心を集め、前処理を共有しやすい条件を意図的に使う。
広い入力全域に中心が分散する場合の勝利をこの結果から主張しない。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src:. \
python -m benchmarks.research.polynomial_prefix_local_cpu \
  --output output/polynomial-prefix-local-cpu-20261021
```

## CPU結果

source `c058f02e16023d96977eab093a59a4ef0663c013`、Apple M4 Pro、BLAS/OMP/Accelerate各1thread。
原本はignored `output/polynomial-prefix-local-cpu-20261021/`。
source snapshot/hash、artifact hash、block間fixture一致をproof.jsonで確認。
最初の実行はNumPy int64のJSON保存時に失敗。失敗logを保全し、Python intへの変換を直して再実行した。
その修正前のCPU suiteは2115 passed /5651 skipped /18 warnings(42.21s)。
修正後も新規6test pass、Ruff/diff check pass。独立TorchでH/mu/sigmaを照合。

### 項数は増えないが、精度条件がある

120精度条件、216動的時間条件を確認した。
下表は中心33phase・batch4をまとめた相対L2誤差であり、GPUの精度gateではない。

| tile点数 | sigma | moment block | dtype | H誤差 | Hsigma誤差 |
| --- | ---: | ---: | --- | ---: | ---: |
| 32 | .1 | 32 | FP32 | 1.29 | 10.43 |
| 32 | .1 | 32 | FP64 | 1.98e-9 | 2.59e-8 |
| 128 | .1 | 128 | FP32 | 9.53e4 | 1.16e6 |
| 128 | .1 | 128 | FP64 | 2.10e-4 | 2.66e-3 |
| 32 | .1 | 8 | FP32 | 2.27e-5 | 3.05e-4 |
| 32 | .1 | 4 | FP32 | 1.78e-7 | 1.98e-6 |
| 128 | .1 | 4 | FP64 | 4.21e-16 | 3.88e-15 |

小sigmaの問題は多項式の項数増加ではなく、展開係数と累積差の相殺。
tile内座標でも、幅/tile比が小さいと係数に(tile幅/R)^8等の大きい因子が入る。
大きな入力momentから狭い山を差し引くため、FP32では破綻する場合があり、
128点blockの最小sigmaはFP64でも今回の厳しい精度要求には不十分。
blockを小さくresetすると改善するが、跨ぐfragment数と参照量は増える。
これは今回のpower-moment prefix実装の制約で、全ての累積表現の不可能性を示さない。

### ユーザーの参照量削減の仮説

128点tile、A128、batch8、step0で、以下の論理参照量となった。
入力site/成分数は1batchあたり。prefix構築のmoment積和・書き込みはquery列に含まない。

| sigma中心値 | 直接のatom別入力site参照 | group入力読み込みsite数 | prefixの端点成分参照 |
| --- | ---: | ---: | ---: |
| .105 | 891 | 128 | 2304 |
| .8 | 6793 | 384 | 3114 |

sigma .8の例では端点参照量が約54%減り、仮説どおりの条件がある。
ただしgroup入力の共有はgrouped-directにも共通。prefixだけの特典ではない。
最小sigmaでは元の支持点数が9moment*2より少なく、端点を読む方が増えた。
同じtileでも、中心のphaseと支持幅でfragment数が変わる。

GPUでは各groupの入力を連続ロードし、shared memory内で直接評価またはscan/queryする構成を想定できる。
中心とsigmaを固定・量子化する必要はない。Xが変わるstepではmomentを再構築する。
このCPU測定はGPUのcache効果やcoalescing、transaction削減率を実測していない。

### CPU時間・保存容量

A128、batch8、128点tile。各step3回の中央値、さらに3step中央値のms。
prefixのdtypeはFP64。構築・係数・group・共通の直接L2込み。

| sigma | moment block | 直接 | grouped-direct | prefix | prefixの最大rel誤差(H/mu/sigma) |
| --- | ---: | ---: | ---: | ---: | ---: |
| .105 | 128 | 2.093 | 2.142 | 3.531 | 1.63e-3 |
| .105 | 8 | 2.091 | 2.808 | 5.394 | 1.33e-12 |
| .8 | 128 | 2.329 | 2.688 | 4.492 | 3.11e-10 |
| .8 | 8 | 2.295 | 7.479 | 17.869 | 4.64e-15 |

不正確な条件の時間を高速候補として採用しない。CPUではprefixの速度利益はまだない。
128点prefixはB8/FP64で72.56KiB、8点prefixは5.06KiBのprefix配列payloadを一時保持。
input、power配列、prefix構築中の一時値、metadata、結果は別。CPU/GPU peakメモリではない。

## 判断

**累積差の利用を小sigmaだからと一律に捨てる必要はない。**
必要なmomentは固定9種類で、支持がある程度広く、同じblockを多数のatomが共有する条件では
atom問い合わせの参照成分数を減らせる。
ただし今回の8次power-momentでは、最小sigmaに広いblockを使う方法は精度で不成立。
細かいblockで精度を守ると参照量とCPU時間が増えた。

小sigma側の比較軸は、同じtile入力を共有するdirectとprefix。
支持点数・moment block幅/sigma・共有atom数を用いた条件別の選択として残す。
項数の少ない別profileや別のmoment基底は候補になるが、このscreenで評価・採用したものではない。
公開dispatcherやGPUへの登録、2D完全stepの性能確認は行っていない。
