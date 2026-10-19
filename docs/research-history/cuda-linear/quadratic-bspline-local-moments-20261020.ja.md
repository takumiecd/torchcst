# 局所dyadic momentのGPU候補

目標は同条件のDenseに対して完全training step時間2倍以内、allocated/reserved両peakともDense以下。可変X・atom/step可変σ、正確な離散L2/floorと全4 source勾配のmaxabs/relative L2 <4e-4を維持する。

数式は[CPU screen](quadratic-bspline-local-moment-screen-20261020.ja.md)の同じB-splineと局所離散moment。v20は、8点以上の2冪blockをσから選び、4 knot+最大2 boundary eventをFP32でquery/injectする。σと支持境界判定はFP64。log2のFP32 hintだけで支持を決めず、FP64比較でblock幅が半径を下回らないよう補正する。domain端Nのknotは最後のblockの終端prefixへ置く。

固定bankはplane/local-point/batch、基底は1/z/z²。複数尺度を一つのlaunchへまとめるが、各blockのscanは独立。Y/dXの復元は尺度ごとにFP32加算する。入力・dYの数値bankは毎step再構築する。forward時点のMeta・source順・count・event係数は不変snapshotとし、retained backwardでは更新後のPや別入力を参照しない。

FP32を使う箇所は明示的なrecipeで切り分ける：

- 基本版：局所bank/query/injectionだけFP32。座標判定・係数生成・norm・native・source縮約はFP64。
- native_float32：局所原点との差をFP64で求めてから、通常のnative支持でFP32 profile/積和。≤2-site clipped supportを含むatom groupは別launchのFP64経路へ残す。
- norm_float32：正の離散moment式をFP32で評価。tiny supportとfloor近傍はFP64へ戻す。積分近似は使わない。
- prepare_float32：FP64の局所座標から係数の多項式演算をFP32へ移す。係数snapshot自体は基本版もFP32。
- contract_float32：H/GがFP32で丸められた後の縮約をFP32とし、atomごとのsource pullbackはFP64。cold nativeでは従来のFP64縮約を保つ。

基準を通らないflagの組合せは性能の正の結果に使わない。以下の初回GPU結果は検証済み。public/default dispatcherは変えない。


## v20初回GPU結果

L4、Torch 2.11.0+cu130 / Triton 3.6、N1024 / A52428、B32/B64 mixed、IEEE FP32、CUDA Graph内の完全step。X/target/Pの初期hashが全planで一致し、全52,428 atomのσ更新を含む。source `65561523a92fa06377db85076d60fb30f5de0e08`。

| plan | B32 ms | allocated/reserved MiB | B64 ms | allocated/reserved MiB |
|---|---:|---:|---:|---:|
| query-fma | 1.616065 | 81.288 / 172 | 2.513662 | 96.791 / 188 |
| local-base | 2.023908 | 30.576 / 98 | 3.988723 | 40.250 / 100 |
| local-native32 | 1.680028 | 30.576 / 98 | 3.193050 | 40.250 / 100 |
| local-norm32 | 1.537422 | 30.616 / 98 | 2.949666 | 40.290 / 100 |
| local-all32 | 1.503326 | 30.616 / 98 | 2.913342 | 40.290 / 100 |
| local-all32-hp | 1.481997 | 40.322 / 98 | 2.991028 | 59.703 / 156 |
| Dense | 0.082701 | 49.002 / 106 | 0.104292 | 49.502 / 106 |

B32では速度とメモリを同時に改善する候補。B64はnone保存ならメモリ基準を満たすが、旧最速より遅い。Denseの2倍以内は未達。独立repeat前の差として扱う。

独立FP64 gateは全plan PASS、local対象66 tests PASS。Y/dX/全4source勾配のmaxabs上限は6.583e-5 / 7.318e-5 / 6.261e-5、relative L2上限3.320e-6 / 3.965e-6 / 3.009e-7で、既定4e-4以内。dyadic境界、retained forward、CUDA Graph動的σ、near-floor、tiny supportを含む。CPUは1980 passed / 3700 skipped。

job `l4job-40738be78cd44a04a3fa8c3f459f59a2`、source archive SHA256 `98e4425fe76761a5ffd53c5bc10902be87db22d1e7a6badcfd697ab03bf7321d`、result archive SHA256 `481801053cd853f75eea0b9444d61758a6e81628e5e0c6d8912425ebd0c11009`。1510 runtime/test/benchmark source filesをcommitとbyte照合、manifest 48 files、return0、cleanup ackを確認。原本はignored `output/bspline-local-moments-20261020/verified-jobs/`。

独立phase graphではB32/B64のstream forwardが0.249/0.851ms、backwardが0.242/0.891ms。加算してprimary時間と見なさない。broad streamはspill0。B64 native cold backwardに34 spillがあるが、これだけでprimaryの原因と断定しない。

## v21の検証仮説

全8尺度のbank/差分初期化/復元を作るv20に対し、v21はeventの整数pointから、実際に使われる尺度をdevice上で毎forward選ぶ。σやPをCPUへ戻さない。first knot pointと尺度領域の整数比較なので、支持・幅を丸めない。未使用尺度のbank/差分は未初期化でもquery/injectから参照されない。使用尺度の全prefix node（終端を含む）は毎回初期化する。

Active[axis,level]は各forwardのimmutable snapshotとしてbackwardまで保持。all-native→broad→all-native、dyadic境界跨ぎも再判定する。確保容量自体は固定のためpayload削減ではなく読み書き削減の仮説。bank/restoreのgridは変えず、各programがinactiveをskipする。v20の数学と離散L2を維持し、GPUで速度/精度/peakを再比較する。
