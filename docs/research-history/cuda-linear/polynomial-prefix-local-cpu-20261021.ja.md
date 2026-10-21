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

結果は検証済みcheckpointで追記する。
