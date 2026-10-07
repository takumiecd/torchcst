# 単一Chart profile productの小型CUDAコア

更新: 最新の小型layout/launchを移植した[追加検証](20261007-profile-product-reuse.md)で、
N64 rho3は約53us、N128 rho3は約78usになった。以下は候補追加前の初回記録。

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
初回screenは全候補で共通のTorch update Plan、支持順比較は共通のfused Planを使う。

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

最終CPU検証: 965 passed/1157 skipped。GPU検証・測定結果は以下に記録する。
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

支持順/fused source `7c52eba226aa0517ee9c1607700ebbd649e8876f` のcheck job
`l4job-e44f23579c8b44cc8981d20b3abbc63a` は新suite48件、Polar更新10件、
既存回帰38件PASS。新suiteにはorderedとTorch/fused双方の20 captured updatesを含む。
CPU全suite965 passed/1157 skipped、変更PythonのRuff、diff check、
wheel/sdist buildとwheel metadata-only importもPASS。

## 共通fused Polar更新での6条件screen

source `7c52eba226aa0517ee9c1607700ebbd649e8876f`。
N64 job `l4job-827a326d2fc24fa8801f977cc1715c6e`、
N128 job `l4job-6979ec676616429fb03143c5d9859dc4`。各条件1独立execution、21 samples。
全workerのfull-site oracle、同一初期p/input/target、adapter4 projectionはPASS。
初期rhoは全atomで1.25/3/8をdecodeして確認し、stepで幅が変わることも検査する。
source snapshotの全1260 tracked filesを当該commitと照合し、不一致0を確認した。

Graph完全step median us:

|N|初期rho|Torch factored|local|saved H|ordered|dense|
|---|---:|---:|---:|---:|---:|---:|
|64|1.25|269.89|67.15|64.27|68.59|38.90|
|64|3|268.94|67.33|63.75|71.64|39.15|
|64|8|268.97|67.80|63.96|61.63|38.92|
|128|1.25|322.27|294.41|243.68|150.06|45.06|
|128|3|322.34|295.23|243.80|152.12|45.38|
|128|8|322.62|294.88|243.46|139.99|45.40|

capture/replay peak bytesは各幅で同じ。allocated / reservedを分けて記録する:

|N|Torch|local|saved H|ordered|dense|
|---|---:|---:|---:|---:|---:|
|64|34926080 / 48234496|76288 / 6291456|102400 / 6291456|122368 / 6291456|34179584 / 48234496|
|128|40523776 / 58720256|188928 / 6291456|293888 / 6291456|316416 / 6291456|34408960 / 48234496|

これはrunnerが管理するwarmed model/grad/optimizerとGraph capture/replayのpeakであり、
GPU process全体のメモリではない。denseは別のモデル/初期化の性能参照。
N64の狭い支持ではsavedがorderedより速く、N128ではorderedが3幅とも速い。
全条件でCUDAコアはTorch factoredより速いが、denseには到達していない。
N64 rho8のorderedの小さな優位性は単独screenの観測として扱う。
性能の一般化や公開既定dispatcherへの採用はしない。

長方形17x31、両軸spacing0.5、異なる原点(-3/7)、固定幅3の独立FP64全行列
Y/dX/全atom微分も3経路でPASS。job `l4job-cdfa26030e33474c9988220a8ef24e9b`、
source `ea9decd5df601252307b3d5e3a0b2222d26048dc`。全体でactual GPU83件、CPU metadata16件を検証した。

## 実行順を反転したrho3の確認

N128 job `l4job-8eb39f7140f942b4ba5eae2dc8166735`、
N64 job `l4job-0e9730393e9f4083aefd3144a9b980ed`、source `ea9decd5df601252307b3d5e3a0b2222d26048dc`。
このsourceの差分は長方形テスト・recipe拒否メッセージ・検証記録のみで、
有効なPlanのkernel/launch/math/updateは前のsourceと同じ。結果はrunごとに保持し、平均へ集約しない。
各run内で同一初期p/input/targetを確認し、候補の実行順をordered→saved→local→torchへ反転した。
初回と反転runのGraph median usは、N64 saved63.75→64.42、ordered71.64→71.91、
N128 saved243.80→243.76、ordered152.12→152.56。rho3ではサイズごとの勝敗が一致した。
allocated/reserved peakも初回と一致した。rho1.25/8は1独立executionのscreenとして残す。

小型の次段階は、N64ではsaved、N128ではorderedを有力候補としてStripのtileから使う。
直接大型化は今回の対象範囲を超えるので、支持準備・H・VJPの共有単位を再設計して測定する。
現時点の公開選択器は変更せず、既存radial/separable/strip_torus経路も維持した。

[機械可読の結果要約](20261007-profile-product-cuda-summary.json)にsource/result archive、
各結果JSON/snapshot/driverのSHA256、完全step時間・allocated/reserved peakを保存する。
8件の完成artifactはsubmission形式検査/adapter4 projection PASS。
再現時は上記runnerコマンドへ `--polar-update fused --source-commit <verified source SHA>` を付ける。
GPUは共有プールで1台を直列利用し、終了時に所有runtimeを停止する。
