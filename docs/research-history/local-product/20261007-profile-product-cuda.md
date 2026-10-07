# 単一Chart profile productの小型CUDAコア

PR58の数学的契約を用い、Euclidean Product Chartの二つのLine軸、
Triweight、共有Polar幅、FP32/IEEE、入力/出力2..128、batch1..64を対象にする。
Strip、多次元Pattern、異なる軸spacing、別profileは最初のCUDA版では非対応。
研究Algorithmはbenchmark Registryに登録し、公開既定dispatcherへの採用は別判断。

## 正規化と微分

入力/出力のraw profileを $v_a(i),u_a(j)$、軸ノルムを
$n_v,n_u$ とし、$D=\max(n_vn_u,\varepsilon)$ とする。
CUDA内部のfactor分母は、floor非活性なら $(s_v,s_u)=(n_v,n_u)$、
活性なら $(s_v,s_u)=(\sqrt{\varepsilon},\sqrt{\varepsilon})$。
常に $s_vs_u=D$ を保つ。Torch参照のfactorの分配と異なっても、
演算子 $W_a(j,i)=\mathrm{amp}_a u_a(j)v_a(i)/D$ は同じ。

$$
\gamma_v =
\begin{cases}
\sum_i v_a(i)\partial_{c_i}v_a(i)/n_v^2 & n_vn_u\ge\varepsilon,\\
0 & n_vn_u<\varepsilon,
\end{cases}
\qquad
\partial_{c_i}W_a =
\frac{\mathrm{amp}_a u_a}{D}
\left(\partial_{c_i}v_a-v_a\gamma_v\right).
$$

出力中心も同様。幅は現在のPolar値から毎回decodeし、task VJPではdetachする。
Parameterの列順は単一Chartの $[polar_x,polar_y,c_o,c_i]$。
既存localコアのmetadataは入力/出力順で借り、cotangentの保存時にcanonical列へ戻す。
sourceと振幅上限はforwardごとにsnapshotし、後続forwardやbuffer変更で旧VJPを変えない。

## 実装と比較

- `torch`: 新定義のTorch factored参照。全factorとautograd中間値を保持する。
- `local`: 既存local contractionを再利用し、HをCTA内で再計算する。
- `saved`: 同じ定義でH[batch,atoms]を保存し、forward/source VJPで再利用する。

profile preparationの結果は9フィールド/atom。full Wや全factor配列はCUDA候補では
生成しない。正確な空支持、floor、単一支持と全中心微分を独立FP64の全site oracleで確認する。
配置/支持の移動、immutable snapshots、20 captured optimizer更新のParameter/moments/stepも確認する。
初期rho1.25/3/8、N64/N128、B32、A204/A819で完全stepとcapture/replay peakを測る。
全CST候補の初期Parameter/input/target bytesを一致させる。denseは別初期化の性能参照。
Polar更新は全候補で共通のTorch update Planを使う。

再現入口:

```bash
python -m tools.kernel_dev test --suite profile-product
python -m tools.kernel_dev prepare \
  --plans benchmarks/cuda/linear/plans-profile-product.json \
  --case benchmarks/cuda/linear/cases/profile-product-64-rho3.json \
  --candidate local --candidate saved --output output/product-comparison
python -m tools.kernel_dev check \
  --plans output/product-comparison/plans.json \
  --case output/product-comparison/case.json
python -m benchmarks.cuda.linear.run \
  --plans output/product-comparison/plans.json \
  --case output/product-comparison/case.json \
  --output output/product-comparison/result.json
```

初回CPU検証: 959 passed/1144 skipped。GPU検証・時間・メモリは未測定。
新fixture `polar_profile_product` はadapter revision4で既存形式と区別する。
生ログ・source/result archiveはignored evidenceと共有プールのjobディレクトリに保全する。

初回GPU job `l4job-84432af893d545aa9da3dbd4553f9674` はコンパイル段階で失敗。
Tritonが `1e-38` 定数をFP64として解釈してFP32専用divisionと不一致になった。
安全な分母定数をFP32最小normal値へ変更し、同じoracle/許容誤差のまま全suiteを再実行する。
失敗ログと検証済みarchiveを共有プールjobディレクトリに保持する。

FP32 guard修正job `l4job-cb6664414c55445587682ae6121f12b5` は31 passed/2 failed。
Y/dX/source、空/単一/微小/global-floor、旧sourceとscalar変更後のVJPは通過。
残る2件はテストがpublic CSTOptimizerへcapturable=Trueを渡したため既存の安全検査で拒否された。
benchmarkと同じAtomUpdate Plan付きcaptured AdamWを実行し、public CSTOptimizerは
eager参照として毎stepのParameter/moments/stepを比較する形に修正する。

Captured更新の再検証job `l4job-f46c9ed2f91c4b628f8e28979cc902a2` でも31 passed。
Graph内で毎回新しいLinearBindingを作るテストhelperが宣言cache検査で拒否された。
既存CSTLinearのbinding/state/cacheを直接Dispatcherへ渡し、構築をcapture外へ戻す。
数値比較条件は変更しない。

## 初回小型コアの検証・screen

source f2ff9122c8f8d1132715a327938c1f27899ab21b。L4 check
`l4job-5158c98f97cb40a881c3228520ef75b7` で新suite33件、既存回帰38件PASS。
L4 / Torch2.11.0+cu130 / CUDA13.0 / Triton3.6.0。
新suiteは13 metadata/CPU +20 actual GPU。Graph20 updates、Parameter/moments/step、
empty/singleton/tiny/global-floor、旧source/scalar変更後のVJPを含む。

同source/同初期Parameter/input/target、共通Torch Polar update、1独立executionでrho3を測定。
時刻計測21 samplesは21独立runではない。Graph complete-step median usとallocated peak bytes:

|N|Torch factored|local H recompute|saved H|dense|
|---|---:|---:|---:|---:|
|64|344.30 /34926080|143.48 /86528|140.63 /102400|39.10 /34179584|
|128|400.45 /40523776|374.03 /233984|323.21 /293888|45.38 /34408960|

verified source-SHA jobs:
`l4job-74e90b50229a4919ae965ba1a01c3291` (N64),
`l4job-3bfa6321b52f42a086c32bdfd707a49a` (N128)。
先行job `l4job-555bcdb527724ce08f730b347bb61122` はsource commit入力ラベルが誤っており、
生データを残し、採用値・独立run数から除外した。以後SHAはgitから自動取得する。
全workerのFP64 full-site oracle、幅更新、adapter4の整合性はverified jobsでPASS。
現段階ではCUDA候補もdenseより遅い。H保存が再計算より有利だった。

## 支持範囲による絞り込みと融合更新

次の候補 `ordered` は既存の物理的な支持順metadata、owner ranges、singleton/middle/wide
収縮を新定義の13-field metadataへ接続する。Hは保存せず、支持に沿って局所計算する。
Param VJPは2 batch partitions。配置準備、partial reductions、source snapshotを時間/peakに含める。

同じPolar CUDA更新kernelは二つのEuclidean座標に依存するので、single 2D geometryの
profile_productにも対応判定を拡張する。更新則・数値実装は変えない。
Torch/fused更新の両方を20 captured stepsでpublic eager optimizerと比較してから、
全Linear Planに同じfused更新を指定して比較する。公開既定選択器は変更しない。
