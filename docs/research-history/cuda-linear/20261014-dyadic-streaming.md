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
GPU結果と原本のprovenanceは以下に記録する。

## 検証済みL4結果

NVIDIA L4、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、driver580.82.07。
同じjob内の隔離workerによる2cohort。2回目は実worker順を反転した。
各cohort21 timing samplesは21独立runではない。

|方式|完全step中央値ms、run1 / run2|peak allocated MiB|peak reserved MiB|
|---|---:|---:|---:|
|全8幅共有|2.799087 / 2.780578|47.169922|82|
|1幅streaming|4.100805 / 4.084903|40.185059|62|
|2幅streaming|3.343694 / 3.338409|41.200684|62|
|dense|0.086302 / 0.086310|33.002441|86|

1幅はallocated14.81%、2幅は12.65%削減。reservedは両方24.39%削減。
時間はそれぞれ約46–47%、19–20%増えた。denseは別parameterizationの工学的対照で、
このfixtureでは時間とallocatedの両方で勝つ。公開dispatcherの採用はしていない。
対象model/optimizer/X/target/dyのwarmup・capture・replayを含むピークで、
CPU oracleと診断値は範囲外。GPU process全体のVRAMは未測定。

初期・24更新後の全shape独立FP64 oracleに対し、全方式の最大絶対誤差は
Y5.714e-6、dX4.739e-6、全atom dP9.017e-6以下。
max_abs/relativeL2各4e-4のgateは変更していない。52428全atomのlive幅が更新された。
GPU jobのテストは55 passed（31 CUDA + 24 CPU）、skip/failureなし。
ローカル全CPU suite1929 passed、2776 skipped、wheel/sdist build PASS。
CUDA unavailableによるskipをGPU検証と数えていない。

G/Gpの最終読み取り後に解放し、Polar FP64 pullbackとの同時生存をなくした。
この修正前はstream1/2とも45.372559MiBだった。数値や幅groupを変えずに
lifetimeを修正した結果が上表。同一streamの順序でallocator再利用を行う。
最初のGraphテスト失敗はretained VJPの出力graphをテストが保持したためで、
保持graphを終了して新しいinput leafでcaptureする修正後に全テストを通した。
失敗原本と修正前の成功原本も保存した。

最終検証runtime checkpoint: `6cbe1afb0a979fd9e49b5ee1771dcf2ef63c6c9c`。

|段階|job|source archive SHA256|result archive SHA256|
|---|---|---|---|
|最終|l4job-75fa59f81b1445cda79240a1be145bb2|76141a0764d3e0ba42c334df130d6a64a84a442f32601a12004978d7ce68d2af|eae7b4538b73cbce91e8f720fa01c719507ea9c7d7027f706a24011120757135|
|lifetime修正前|l4job-5d4987d576c24385b51ab36b95de873d|d088c0e9aaa31f0681e821bca4c2d735b97363aae64105e18e0ea2d33362282f|20d9a0813abae5ee758aa97be84a2ddbec120019a6ea0fa803a5cf425cf62dd8|
|Graphテスト失敗|l4job-f6e51a02ca5342f1bc0fd3d28b6a57ae|fa8892d39416f3da404304e039ee5354f8daec3a60d422e5a21ee1c68c1566ba|failed-capture-test内のreceipt参照|

原本はworktreeとprimary両方のignored `output/dyadic-profile-streaming-20261014/`。
`qualified-l4/`、`qualified-before-lifetime-fix/`、`failed-capture-test/`に
source/result archive、spec、logs、receiptを保存。copy manifestとGit recovery bundleも保存する。
今回所有したGPU sessionは停止確認済み。

## 任意の実数累乗と基準幅

`u=exp(-r/bref)`なら`u**(bref/b)=exp(-r/b)`は実数bでも厳密。
これは指数因子の恒等式で、幅別の入力集計を1本へ潰す恒等式ではない。
CPUの独立precision sweep（88幅、r<=51.2とr<=300）では、
FP64のbref1/10はいずれもprofile最大絶対差約1e-15–1e-14以下。
FP32ではr<=51.2でbref1が3.709e-7、bref10が2.986e-6。
r<=300ではbref1の底underflowによる広い裾の損失が3.476e-4。
基準1は今回のbounded rangeなら良い選択であり、最大幅が常に最も高精度ではない。
GPUで検証したrouteはbref10から固定dyadic bankを作る方式で、
bref1・任意実数幅のGPU routeは未実装・未検証。

次の資産調査と候補順位は[最適化資産の選定](20261014-dyadic-optimization-assets.md)に記録する。
