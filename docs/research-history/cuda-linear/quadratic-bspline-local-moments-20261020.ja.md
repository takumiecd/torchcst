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

## v22の検証仮説：境界だけをshardへ分散

各atomの4 knotは従来のpoint差分に加える。最大2つのblock境界補正だけは、next blockの始点に加える3係数を、`Border[plane,global-block,shard,batch]`へ加算する。shardはatom program IDの剰余。4 knotの支持判定・query・勾配式は不変。各blockの復元時にshard和を取り、全prefix nodeへ同じ3係数を足す。

尺度Sのblock数をt=N/Sとすると、尺度先頭のglobal block番号は`2(N/8)-2t`。境界は次block `tile+1`へ入れる。補助容量は1軸`3(2N/8-1)KB`個のFP32、Kは8/16/32/64。N1024/B64/K32では5.977 MiB。位置snapshotは2軸×2event×A個のI32（A52428で0.800 MiB）。これらをpeakへ含める。K64は速度が上がってもメモリ目標を外れる可能性があるため、容量を明示して比較する。

これは加算の順序・配置を変える実装仮説であり、atomic競合が原因だったという測定結果ではない。浮動小数点和の順序差を含め、独立FP64 gate、動的σ・全native/broad切替・retained backward・全atom勾配を同じ許容誤差で確認する。未使用尺度のborder planeも毎stepゼロ初期化し、stale値を使わない。

## v23の検証仮説：未消費の終端nodeとdomain先頭

初期mixed fixtureのCPU整数event集計では、S512の境界一つへ出力側6062 / 入力側5872 atomが加算し、knot側は最大4044 / 4127 atomが同じpointへ集まる。このlogical multiplicityはGPU性能counterではない。原本はignored `event-pressure.py/json`。

prefix bankはnode0..Sを保持するが、復元はsite0..S-1のみを読む。したがってknotが前blockの終端node Sへ加える差分は復元に使われず、省略できる。次blockの初期多項式は既存の境界補正が担う。queryでは終端prefixが必要なので、queryのpoint/係数は変えない。v23は2bit×4knotの注入modeと3bitの尺度番号を、各軸/atomのI16 snapshot（A52428、2軸で0.200 MiB）へ保存する。

domain先頭node0へ来るknotは有効な寄与なので省略しない。`shard_origin=true`では同じ尺度のfirst-block補助planeへ振り分け、境界補正と同じ復元で足す。falseでは終端skipだけを比較できる。内部block境界に一致するknotは前block終端としてskipされ、境界補正の後続pieceが次blockを初期化する。

支持、L2、query、source pullbackを変えず、空区間・domain外中心・厳密なdyadic境界・all-native/broad切替をGPUで検証してから速度を比較する。v22との加算順序差も既存4e-4 gateで判定する。

## v21検証結果：active尺度省略は不採用

source `4441b276d45ccf250143a265e8d26b5193fa7ea0`、job `l4job-ea2c9dd7ec684a38af5690d6ce4e2581`。36 tests PASS（35 CUDA +1 metadata、937 deselected）。主要2条件の全plan独立FP64 gateもPASS。source/result archiveとmanifest 42 filesを検証し、1514 runtime/test/benchmark filesがcommitとbyte一致、return0/cleanup ackを確認。

| plan | B32 ms | allocated/reserved MiB | B64 ms | allocated/reserved MiB |
|---|---:|---:|---:|---:|
| query-fma | 1.620070 | 81.288 /172 | 2.502822 | 96.791 /188 |
| local-all32 | 1.509114 | 30.616 /98 | 2.915113 | 40.290 /100 |
| local-all32-hp | 1.483581 | 40.322 /98 | 2.996097 | 59.703 /156 |
| local-active | 1.527201 | 30.616 /98 | 2.919538 | 40.291 /100 |
| local-active-hp | 1.507977 | 40.322 /98 | 2.996684 | 59.703 /156 |
| Dense | 0.082651 | 49.002 /106 | 0.105586 | 49.502 /106 |

v20のB32改善傾向は独立repeatで再現したが、v21自体には完全stepの改善がない。独立phaseではbank/restoreが約4–9µs縮んだ一方、尺度分類約29.7µsと差分初期化launchの増分が上回る。broad stream時間は変わらず、未使用bank削減だけではB64の遅さを説明・解消できなかった。これはcache容量やmiss率を測定した結果ではない。既定経路への採用は行わない。

source SHA256 `0e34f2a8c6b306d8e14bfcb6cf89dd2e36190365ea55992231980859898acc9b`、result SHA256 `e96d5f27cc3a428ca03e1a6e4d6a01dba5bc8131a96d959932a5516f8016a310`。v20/v21通算の独立gate上限はY6.583e-5、dX7.542e-5、all-source6.261e-5、relative L2はいずれも3.965e-6以下。

## v22検証結果：境界shardだけでは小幅の改善

source `fffe7584623aa0b2f4c9b256ccddd9764271993e`、job `l4job-69b92f6d28bd4750b357f5873837aaf6`。36 tests PASS（35 CUDA +1 metadata、973 deselected）。主要2条件の全plan独立FP64 gateもPASS。source/result archive、manifest 48 files、1518 runtime/test/benchmark filesのcommit byte一致、return0/cleanup ackを検証。

| plan | B32 ms | allocated/reserved MiB | B64 ms | allocated/reserved MiB |
|---|---:|---:|---:|---:|
| query-fma | 1.619552 |81.288 /172|2.494048|96.791 /188|
| local-active |1.525143|30.616 /98|2.929224|40.291 /100|
| local-sharded8 |1.540299|32.164 /100|2.855720|43.491 /100|
| local-sharded16 |1.514146|33.816 /98|2.832875|44.290 /100|
| local-sharded32 |1.527808|34.405 /98|2.872563|47.067 /140|
| local-sharded64 |1.534911|37.393 /98|2.890271|53.091 /148|
| Dense |0.083207|49.002 /106|0.107585|49.502 /106|

B64は16分散でactive対比約3.3%の候補改善。ただし旧query-fmaに届かず、Denseの約26倍。B32には明確な完全step改善がない。32分散はB64でallocatedを満たしてもreservedがDenseを上回るため、メモリ目標未達。64分散は両peak未達。部分処理の短縮だけで候補を選ばない。

独立phaseのB64 stream forward/backwardはactive 0.850/0.893ms→16分散0.796/0.834ms→32分散0.743/0.793msだが、初期化・復元と完全stepの費用が相殺する。broad streamは全てspill0、32分散の最大restoreは184 registers /shared8192 bytes。これを単純加算してprimary時間と見なさない。

source SHA256 `240bd170aaef03a2c6add186057239da8bfb62d82c812c533b1d3d8c3ad259af`、result SHA256 `6ed22ce395601312d988b1950b75e946f205da21fab8c31fc3bc3c598b69e1b5`。v20–22通算gate maxabs上限はY6.583e-5、dX7.542e-5、all-source6.261e-5。CPU v22は1982 passed /3770 skipped。public/default経路への採用は行わない。

## v24の検証仮説：準備のregister圧力を切り分ける

v20–23のprepareはPolar復号と離散L2を一つのkernelで評価し、初回compiler reportは230 registers、spill0。v24は同じFP64復号式でMetaの幾何/振幅/source微分を保存し、次のkernelで既存`_norm_pair`をそのまま評価してMetaのinvD/gammaだけを書く。Metaの型・寿命・正規化式・floor・tiny/floor guardを維持する。追加の全atom Tensorはなく、二つのkernelを足したprepare時間と完全stepで判定する。register数だけで速度向上を主張しない。

v21の負の結果に基づき、v24ではactive尺度走査を省くcontrol（split_prepare=false）も用意する。まずactive省略と二分割準備を分離して比較する。境界分散32/16を比較し、reservedもDense以下となる候補を選ぶ。デフォルト研究recipeは16分散、public/default dispatcherは変更しない。GPU未検証。
