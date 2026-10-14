# 最大幅からの累乗profileと幅集計の再利用

2026-10-14。形とメモリを優先する研究候補。公開CUDA dispatcherは変更しない。

## 数学と幅の呼び方

今回の利用者のσは減衰長b。以前のσ²の幅指定と混同しない。
既存`MaternMixtureSpec`の数学は変えず、宣言のwidth_knotsとPolarの幅には√bを渡す。
可変の利用者σは既存のlog幅混合率にだけ現れる。task幅stop-gradientと公開Polar更新を保つ。

```text
bmax = 10
n = 128,64,32,16,8,4,2,1
b_n = bmax/n = .078125,.15625,.3125,.625,1.25,2.5,5,10
u(r) = exp(-r/bmax)
f_n(r) = (1+n*r/bmax) u(r)^n
G_n(d) = sum_k f_n(|d+kD|) / sum_k f_n(|kD|)
g_sigma = (1-t)G_lower + t G_upper
t = log(sigma/b_lower)/log(b_upper/b_lower)
```

累乗は各周期画像の指数因子に適用し、その後に周期化・peak正規化する。
周期化済みのprofileを累乗する定義ではない。生の直線profileは頂点の傾き0、非負、単調減少。
各基底の半値半径は1.67834699b。有限circleでは画像の影響を含めて実測する。
隣接2基底の混合に対する離散L2は交差項込みの3項Gram式を使い、実site配置で評価する。
軸ノルムの積の後でfloorを一度だけ適用する。

## 実行方式

旧`research_cuda_matern_mixture_triton/v1`は変更しない。
新`research_cuda_matern_streaming_triton/v1`は幅groupを1または2に限定する。
各phaseでZ、summary、cyclic start、必要ならinjectionを一度確保し、幅groupごとに上書きする。
各atomの活性2幅がgroup境界を跨ぐ場合も、両groupの寄与をH/HpまたはYへ足し込む。
backwardも同じgroup方式でG/GpとdXを生成する。全幅分のscan Tensorは保存しない。

H/Hpのforward保存方針は同じprofileの全幅対照と揃える。
幅streamingによる削減と、H再計算による削減を混ぜない。
forwardのParameter clone、係数、index、固定scalarを保存し、更新後の値でretained backwardを計算しない。
input/outputのatom集計やoptimizer状態は別途メモリを使う。

step減衰係数とatom位置係数の指数は、最大幅のFP64指数から繰り返し二乗で作る。
状態・保存係数はFP32、norm/Gramと準備はFP64。
指数だけの累乗によって幅別の入力集計を復元できるとは仮定しない。

N1024/B32/block64でrestoreのmoment scratch payloadは全8幅8.125MiB、
stream1が1.015625MiB、stream2が2.03125MiB。
これはTensor容量の計算値で、allocator/Graph込みの完全stepピークではない。
`workspace_bound`は準備や保存値まで含む厳密な上界を出していないためNone。

## 固定比較と再現

新case `matern-dyadic-case.research.v1`。旧v7のCase/Planは変更しない。
1024×1024、52428 atoms、batch32、spacing0.1、周期102.4、利用者σ0.1..10。
FP32 IEEE、geometry FP64、実sitesのL2、AdamW＋公開Polar更新。
共有全8幅・stream1・stream2・denseを同じ入力で隔離process比較する。
独立native CPU FP64 oracleで全Y/dX/全atom dPを初期と24更新後に確認し、
max_abs/relativeL2各4e-4を保つ。Graph warmup/capture/replayをメモリ範囲に含める。

```bash
python -m benchmarks.research.dyadic_profile_shapes --output output/dyadic-profile-streaming-20261014
python -m benchmarks.research.dyadic_power_precision --output output/dyadic-profile-streaming-20261014/power-precision.json
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/matern-dyadic-1024-a52428-b32-k8.json \
  --output output/dyadic-profile-streaming-20261014/cohort1.json
```

形のCPU結果：172幅×201中心。半値半径/σは1.64028〜1.68433。
中心移動によるL2norm変動はσ0.1で0.6730%、σ1で0.00006988%。
σ1の距離10外の二乗和割合は約2.35e-6。σ10は反対側の高さ0.07309、広い裾が残る。
原本はignored `output/dyadic-profile-streaming-20261014/`へ保存する。
GPU correctness・完全stepピークの結果は、取得した原本とsource/result hashを確認して追記する。
