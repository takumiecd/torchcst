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

基準を通らないflagの組合せは性能の正の結果に使わない。まだGPU性能・精度の結果はない。public/default dispatcherは変えない。
