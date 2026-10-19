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

## v23検証結果：終端skipとdomain先頭分散はB32で改善

source `be6265dc5865f5ee016c9d69097947caa2ea172f`、job `l4job-90251ded5d384b17a496fe94a966a499`。25 tests PASS（24 CUDA +1 metadata、1009 deselected）。主要2条件の全plan独立FP64 gate PASS。1522 runtime/test/benchmark filesのcommit byte一致、manifest42 files、archive hash、return0/cleanup ackを確認。

| plan | B32 ms | allocated/reserved MiB | B64 ms | allocated/reserved MiB |
|---|---:|---:|---:|---:|
| query-fma |1.622294|81.288 /172|2.494845|96.791 /188|
| local-sharded32 |1.525449|34.405 /98|2.877334|47.067 /140|
| local-drop-ends |1.475565|34.605 /98|2.752205|47.268 /142|
| local-endpoint8 |1.469794|32.364 /100|2.654509|43.691 /102|
| local-endpoint32 |1.461689|34.605 /98|2.620878|47.268 /142|
| Dense |0.082300|49.002 /106|0.106153|49.502 /106|

B32はquery-fma対比約9.9%速く、allocated/reservedともDense以下。B64は8分散なら両peakがDense以下だが、query-fmaより遅い。32分散は速度が良くてもreservedで未達。Denseの2倍以内にはいずれも届かない（B32約17.8倍、B64のメモリ適合8分散約25.0倍）。この改善率は初回run、v24比較にもcontrolを含めて再検証する。

独立phaseのstream forward/backwardはB32 sharded32 0.203/0.237ms→endpoint32 0.176/0.211ms、B64 0.742/0.792ms→0.673/0.711ms。小支持native forward/backwardはB32約0.258/0.356ms、B64約0.305/0.404msで残る。準備約0.137ms、係数約0.043msも残る。部分phaseは別graphでありprimaryとの加算一致を要求しない。

source SHA256 `412648a09c6962ba5cdc258be2b895f62865845ef297d541802dc3358c4ecf8c`、result SHA256 `697dacfb86561f31467f731898a748632ea8e05225af8a51528d4b663201494a`。通算gate maxabs上限はY6.583e-5 /dX7.542e-5 /all-source6.261e-5、relative L2はいずれも3.965e-6以下。CPU v23は1983 passed /3794 skipped。共通経路の追加GPU回帰とv24比較は別jobで実行中であり、この時点では完了扱いにしない。

## v25の検証仮説：batch行strideのpadding

v23のbroad streamはB32 forward/backward 0.176/0.211msに対しB64は0.673/0.711ms。batch2倍に対し3倍以上の費用差があるため、物理strideに依存する挙動を切り分ける。現時点ではcache missやmemory partitionのhardware counterを取っておらず、原因は確定していない。

bank/差分/border planeの物理行幅を`PB=B+padding`とし、実batchはBのまま、全load/store/query/inject/reductionは`b<B`でmaskする。X/Y/H/HPのshapeとstride、全ての演算式は変えない。bank/差分/borderのplane offsetだけをPBで計算する。paddingは4/8/16列、確保増加もpeakに含める。active尺度を使う場合のzeroも同じ物理幅で初期化し、padding列の未初期化値は読まない。比較controlはv24の16分散・split prepare。GPU gateと完全step/両peakが揃うまで有効とは判断しない。

## 共通GPU回帰とcompiler原本

v23 sourceの共通回帰job `l4job-1055ff96930845d9a71c2935d7e8d390` は384 passed /650 deselected /既存cuBLAS初期context warning1件、595.28秒。凍結testの再collectionで379 CUDA-marked +5 metadataと確認した。query/lean/native/loop/local/active/shard/endpointの共通autograd経路を含む。v24/v25がこの回帰に含まれたという意味ではない。

source SHA256 `46a76db9e6bf492c9bfb97b27346aebb88d77ccf68927d441ffed521e638ddf4`、result SHA256 `72ab864ba1bf483e2836f14bb76c90f0f27c04a72955022dde37ec099afc2671`。manifest4 files、1522 source filesのcommit byte一致、return0/cleanup ack確認。

compiler原本job `l4job-e8295e3e882744689ae9329666fc8520`、source `d3ede235`（runtimeはv23と同じ）はPTX/LLIR/TTGIR/SASSとregister/spill情報を保存した。source SHA256 `7e99b0da41b070b27ad364365289b335a28bc83177c1eed3ed75543f44b2994f`、result SHA256 `bba92968974e14ef4f0c51543ae42c7b6fccdfe00388339a3c084fd76eeb4064`、manifest184 files、source1522 filesを検証。

静的PTXのstream forwardにはB32/B64それぞれF64 multiply 3/5、backwardには12/14、F64 divide/sqrtはない。主projectionはFP32で、broad streamのspillは0。FP64が投影全体の主要演算だとする説明は現在の実装には合わない。prepareには多数のF64演算とpotential fallbackが含まれるが、静的命令数は実行された命令数やhardware counterではない。B64 native forwardは2 spill、cold backwardは34 spillを引き続き記録する。

## v24主要比較結果：分割だけでは準備時間は変わらない

source `c720ccb9a0e7882a4d2173251b47faf9bbc41a8d`、job `l4job-0fa3baffd04044f286b26a306cd136ce`。36 targeted checks PASS（35 CUDA +1 metadata、1034 deselected）、全主要plan独立FP64 full-shape gate PASS。v23 endpoint32のB32改善はcontrol再測定で再現した。

| plan | B32 ms | allocated/reserved MiB | B64 ms | allocated/reserved MiB |
|---|---:|---:|---:|---:|
| query-fma |1.618770|81.288 /172|2.506942|96.791 /188|
| local-endpoint32 |1.461635|34.605 /98|2.617734|47.268 /142|
| local-endpoint-noactive |1.452121|34.604 /98|2.641958|47.267 /142|
| local-split-prepare |1.447287|34.604 /98|2.640997|47.267 /142|
| local-split-prepare16 |1.452363|34.016 /98|2.623285|44.490 /102|
| Dense |0.083214|49.002 /106|0.105458|49.502 /106|

同一32分散/noactive control対比で、split prepareはB32約4.8µs、B64約1µs差。独立repeat前の小差で、準備を分割すれば大幅に改善するという仮説は成立しなかった。独立phaseではdecode約0.064ms（80 registers）、norm_init約0.073ms（180 registers）、合計prepare約0.142ms。fused controlは約0.145ms（230 registers）。norm_replayは4 warpsで約0.141ms、split normは1 warpであり、norm_replayをdecode時間と引き算して内訳を求めない。追加eventのあるphaseはprimaryより遅くなる場合がある。

source SHA256 `a41895bd46d6e67b8ffa169b69b0a440af8d872044b73bd657aac39c4f53094d`、result SHA256 `efccd81c29de6583877f4907240fd5e43343861f513a77461d277d14b4f6bb2f`。manifest42、source1527 filesのcommit byte一致、return0/cleanup ack確認。CPU1984 passed /3829 skipped、wheel/sdist354 runtime files byte一致。16分散はB64の両peakを満たすが、依然query-fmaより遅い。small/switchとpaddingは別jobで検証中。

## 次の数値screen（GPU未実装・未採用）

CPUで5状態×B32/B64×floor1e-6/1000の20条件を確認した。原本とコードはignored `decode32-screen.*` と `norm-ratios32-screen.py` / `norm-ratios32-cpu-step*.json`。

- 幅のlog/expだけFP32、shape/norm/contractionはFP64としたscreenは全20条件PASS。σ相対差は最大約4e-7。これはCUDA SFUの検証でもGPU高速化の結果でもない。tiny/floor近傍では正確な幅へ戻すguardを想定した。
- 既存局所moment CPU screenに、denominator/inverse/gammaのFP32計算とtiny/floorのFP64 guardを加えたscreenも全20条件PASS。Y/dX/all-source maxabs上限6.297e-5 /9.630e-5 /7.664e-5、relative L2は6.952e-6以下。native projectionとsource縮約はCPUでFP64であり、GPU候補全体の保証にはしない。

次の実装候補は、通常の正規化とFP64 guard経路を別kernelにし、必要なatomだけ正確な幅/正規化へ戻す方式。精度gateと完全step/両peakを同じ基準で検証するまで採用しない。

## v25検証結果：paddingは負の結果

source `cdf53068d29a7b63a5a4972ed8c2fdf16ae8f168`、job `l4job-6fa0a769c0ee4f8181aa474e3c9af35e`。local系回帰247 checks PASS /859 deselected、823.06秒。全主要planの独立FP64 full-shape gateもPASS。driver全体1089.45秒、timeoutなし。source1531 filesのcommit byte一致、manifest42 files、archive hash、return0/cleanup ackを検証した。

| plan | B32 ms | allocated/reserved MiB | B64 ms | allocated/reserved MiB |
|---|---:|---:|---:|---:|
| query-fma |1.615172|81.288 /172|2.483848|96.791 /188|
| local-split-prepare16 |1.449418|34.016 /98|2.620316|44.490 /102|
| local-pad4 |1.527326|35.071 /98|2.772048|45.929 /102|
| local-pad8 |1.495293|36.231 /98|2.697680|46.973 /142|
| local-pad16 |1.473581|38.550 /98|2.668978|50.463 /142|
| Dense |0.082248|49.002 /106|0.103549|49.502 /106|

全padding候補が完全stepで遅くなった。B64の8列はreserved、16列はallocated/reservedの両方でDenseを上回る。B64 stream forward/backwardもcontrol0.677/0.715ms→padding4 0.754/0.802、8 0.713/0.757、16 0.694/0.739で改善しない。単に物理strideへ余白を足す方針は採用しない。この結果だけで特定cacheやpartitionを原因から完全に除外したとは主張しない。

source SHA256 `a09afa44f435e806f6ce53940cd4ec58ffb856209f24d59de1b4016ac4bd7278`、result SHA256 `801ed6d32c68b3cd74ff34f35a16620b5651c9f4799a0fd7e1b1ec621b3e9e3e`。v20–25通算の誤差上限はY6.583e-5 /dX7.542e-5 /all-source6.261e-5、relative L2はいずれも3.965e-6以下。

## cleanup失敗runの除外

small/switch job `l4job-498058fb1bcd45c199035a92ce97ba6d` はdriver return0・timeoutなしで測定を終え、archive/manifest検証まで完了した。その後のremote cleanupが180秒でtimeoutし、Connection was lostを記録した。poolは所有VMを停止し、serverのNo active sessions foundも確認した。原本をignored `excluded-jobs/`へ保持し、このrunはprimary速度/メモリの表から除外した。source SHA256 `13ee4a3484d414c86a5cb38023a5ddb24a663ad2e71c29b6c03c7c2259596bf0`、result SHA256 `21d5041aaf2004d736b8ff0e60183315e7cba8a8b4793b4f3ba31f7093cb831f`。新L4で明示的な同条件再測定 `l4job-191754323fe74c96968dc8a74d48a495` を行う。既に検証済みの主要比較の証拠は失われていない。

## launch設定の切り分けとbatch分割screen

次は同じv24のruntime/recipeが既に受け付けるaxis_atoms/axis_warpsの(4,4)/(4,2)/(8,2)/(16,4)、およびnative_warps=8を個別に比較する。controlはaxis(8,4)、native_warps=4、padding0、16分散。FP32 projectionでB64の仕事量/プログラムが増える影響を切り分けるもので、8 warpsへ無条件に拡張する新仕様ではない。alias/Caseをcommitし、準備・L2・並べ替え・全勾配・optimizer込みの完全stepと両peakで採否を決める。

batchを32列（1状態は16列）ずつ独立に縮約し、source pullbackの部分勾配をFP32加算するCPU screenも実施した。5状態×B32/B33/B64×2floorの30条件と、tile16の1状態6条件が全て4e-4 gate内。maxabsはY6.464e-5 /dX8.962e-5 /all-source7.827e-5、relative L2は7.021e-6以下。これはGPU未実装であり、native projectionはFP64のまま。コードと原本はignored `batch-partial-screen.py` と `batch-partial-cpu-tile*-step*.json`。

## launch比較：axis_atoms=4が改善

source `68286cabadb1ba92047a657bb0788af9b8e993cf`、job `l4job-d234ba6bce2a4852aedca6938d534aae`。28 CUDA checks PASS /1104 deselected、82.13秒。全主要plan独立FP64 full-shape gate PASS。変更は既存recipeのlaunch設定で、profile/精度基準/保存量は維持した。

| plan | B32 ms | B64 ms | allocated/reserved MiB (B32; B64) |
|---|---:|---:|---|
| query-fma |1.614742|2.506577|81.288/172;96.791/188|
| axis(8,4) control |1.449141|2.622206|34.016/98;44.490/102|
| axis(4,4) |1.354232|1.976857|34.016/98;44.490/102|
| axis(4,2) |1.454083|2.629927|same as control|
| axis(8,2) |1.782447|2.624907|same as control|
| axis(16,4) |1.783516|2.644907|same as control|
| native 8 warps |1.956798|3.157575|same as control|
| Dense |0.082303|0.104018|49.002/106;49.502/106|

axis(4,4)はB32約6.5%、B64約24.6%短縮した。B64でも旧query-fmaより速く、両peakがDense以下となる候補。これは初回runで、repeat前の改善として扱う。単にwarp数を増減すれば速いという結果ではなく、native 8 warpsは負の結果。Dense対比はなおB32約16.5倍、B64約19.0倍で、2倍目標は未達。

## 小さいσ・切替の検証済み再測定

source `c6d99f4c`、job `l4job-191754323fe74c96968dc8a74d48a495` はcleanup失敗runの明示的再測定。archive/manifest/source byte/return0/cleanup ackを確認し、全planのFP64 gate PASS。

| B32 profile | query-fma ms | small W ms | endpoint32 ms | split16 ms | Dense ms |
|---|---:|---:|---:|---:|---:|
| small |1.453961|1.209235|3.228393|3.313914|0.082165|
| switch |1.729237|not selected|2.812572|2.870405|0.082578|

local系は34.016–34.605/98 MiBでメモリ条件を満たすが、小さいσの直接Hは遅い。small Wは50.231/114 MiB、Dense49.002/106 MiBを少し超える。σが小さいから直接Hへ送れば有利という前提も再検討が必要。

## v26候補：最小4-site blockによる小σの局所moment化

CPUで通常atomを全て局所momentへ送り、clipped support≤2-siteだけFP64 nativeへ残した。min block8ではmixed/smallの24条件中5条件がall-source maxabs基準を超え、最大4.596e-4となった。min block4は全24条件PASS、all-source maxabs上限1.115e-4。B32/B64、3連続状態、floor1e-6/1000を含む。これだけでGPU精度を保証しない。原本はignored `all-local-screen.py` / `all-local-cpu-{mixed,small}-min{4,8}-step*.json`。

v26は最小block4、cutoff0.05、axis(4,4)、8分散を使う。sigma_min0.1の通常atomも局所momentへ入り、tiny supportは従来の正確なnative経路へ残る。profile/R=√6σと支持、4 knot+最大2 boundary event、離散L2/floor、source pullbackは不変。

最小blockをm=2^sとすると尺度Sのlevel=log2(S)-s、prefix offset=level*N+2(N/m)-2(N/S)、全点数=(log2(N)-s+1)N+2(N/m)-1、全block数=2(N/m)-1。古いrecipeはs=3のまま、新revisionだけs=2にする。bank/restore/zero/分類/境界補正の全offsetを同じsで計算する。N1024では1 bankあたり1280 point増える。B64で1 bank約0.938 MiB増、3 banks約2.813 MiB増。分散を16→8にすることでborder容量増を抑えるが、実際の両peakはGPU測定で判定する。

係数はまずFP64で生成してFP32 snapshotへ保存し、FP32係数生成も別ablationで確認する。S4のdyadic境界、tiny/空支持、retained forward、動的σ/Xと全4 source勾配を照合後に完全stepを比較する。public/defaultへの採用はまだ行わない。
