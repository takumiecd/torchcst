# 小σもWを持たないH/G経路

目標はatom/stepごとに可変σのB32/B64混合条件で、完全step時間がDenseの2倍以内、capture/replay込みallocated/reserved両peakがDense以下。現時点で新方式の性能は未検証。

前のv17では係数再生成の遅化を確認した。またGEMM呼出箇所に帰属する常駐blockが32.5MiBあった。v18は小σ・tiny-clipped-supportをWへ集約する経路を、軸別直接H/Gへ置き換える。大σ側のFMA端点queryと差分集約は同じものを使う。

## 数式と保存

入力profileをv、出力profileをuとし、D=max(sqrt(QᵤQᵥ), floor)、s=A/Dとする。Qは全観測siteの正確な離散二乗和で、floorはprofile積の後で一度だけ適用する。

- Hₐᵦ=Σᵢ vₐᵢXᵦᵢ、Hμₐᵦ=Σᵢ (∂vₐᵢ/∂μᵥ)Xᵦᵢ。
- Gₐᵦ=Σⱼ uₐⱼdYᵦⱼ、Gμₐᵦ=Σⱼ (∂uₐⱼ/∂μᵤ)dYᵦⱼ。
- Yᵦⱼ += sₐuₐⱼHₐᵦ、dXᵦᵢ += sₐvₐᵢGₐᵦ。
- Tₐ=Σᵦ HₐᵦGₐᵦ、∂L/∂A=T/D。
- ∂L/∂μᵤ=s[Σᵦ H Gμ − T(Σⱼ u∂u/∂μᵤ)/Qᵤ]、入力中心も同様にHGμをHμGへ置き換える。floor-active時のnorm微分は既存Metaの条件をそのまま使う。

振幅のPolar source pullback、task幅stop-gradient、正規化の式は変更しない。小σ側はaxis supportを `_interval` でclipして `_shape` でprofile/中心微分をFP64評価する。site tileは8点が初期設定だが、支持が8点を超えても最後までloopする。隣のaxisがtinyでも他方の広い支持を切り捨てない。

H/Hμ/G/Gμは従来のH経路と同じFP32丸め境界を持つ。小σの旧W経路とは加算・丸め順序が異なるため、bit一致を仮定せず独立FP64 gateで判定する。Y/dXへのnative加算はFP32 atomic、広い側は従来のFP64差分atomicとscanを使う。atom勾配はnative/broadの各物理slotで一意に書き、canonical source順へ戻す。

native H/Gはregister内で利用後に破棄し、backwardで再生成する。broad Hのnone/h/hp保存方針は比較する。forward時点の不変Meta・位置順・countを保存し、別stepのσや新しい入力から古いbackwardを再計算しない。全W/dWと全体G/Gμ Tensorは作らない。Coefficients/Hの全A容量にあるnative未使用slotは現段階では残る。

## 検証予定

可変入力・可変σ・cutoff跨ぎGraph replay、retained backward、zero amplitude、empty、floor-active、個別勾配を既存queryの独立gateで照合する。maxabsとrelative L2はともに4e-4未満。混合B32/B64、小σだけ、切替Caseで、旧最速FMA・Denseと構築/L2/全勾配/更新込みの完全stepと両peakを比較する。診断は別Graphで行い、native_forward/native_backwardと広い側のstream_forward/stream_backwardを区別する。

## 最初のW-free結果

job `l4job-550cefb35e0448fa8ba9c355cae07876`、source `45d6afd7`、L4 / Torch 2.11.0+cu130 / Triton 3.6.0 / TF32 off。同じN1024/A52428・初期P/X/target・atom/step可変の混合σ。独立runは1件。

| 方法 | B32完全step ms | B32 allocated / reserved MiB | B64完全step ms | B64 allocated / reserved MiB |
|---|---:|---:|---:|---:|
| 最速W/H混合 fma | 1.616429 | 81.288 / 172 | 2.493219 | 96.791 / 188 |
| W-free、H/Hμ保存 | 2.464067 | 43.792 / 132 | 3.999791 | 60.291 / 148 |
| W-free、Hのみ保存 | 2.591712 | 38.143 / 134 | 4.256291 | 48.993 / 120 |
| W-free、H非保存 | 2.683876 | **31.743 / 94** | 4.421728 | **36.193 / 92** |
| W-free、H保存・norm FMA | 2.580499 | 38.143 / 134 | 4.253359 | 48.993 / 120 |
| Dense | .082163 | 49.002 / 106 | .104511 | 49.502 / 106 |

H非保存は両batchでDense以下の両peakを満たしたが、速度は対照から悪化。速度とメモリを両立するという目標は未達。W-freeのH/Hμ保存と対照のallocated差はB32 37.496MiB、B64 36.500MiBで、GEMM側32.5MiB等を外した効果を確認できる。reservedはpayload差から単純には予測しない。

native_forward/native_backwardの別Graph診断はB32 .293/.781ms、B64 .437/1.307ms。B64 backwardはregister255 / spill slots34（compiler報告）。PTX文字列のld.local/st.local検出はfalseでも、後段register割当によるspill報告を無視しない。PyTorchのallocated/reservedにはこの退避の全量を表すという意味はない。

独立3状態full-shape FP64 gateの最大誤差Y 1.914112e-6 / dX 1.849817e-6 / 全atom勾配3.036035e-5、relative L2 ≤1.388e-7。全CST候補で52,428 atomのσが更新された。40 CUDA +1 metadataの新経路回帰がPASS（789 deselected）。CPUは1978 passed /3595 skipped /18既存warning。これらはこのsourceの結果。

source SHA256 `7043c9440490df661bc23527deb0294f4dbc75c10fa81b13db4f693e3c46d0a0`、result SHA256 `d2890167291ed2a5882164ae2a9e0c876b7222e5c76dc9ed5b54aae1af569753`。manifest42件、runtime/test/benchmark1499ファイルのcommit一致、正常終了、cleanup acknowledgementとowned VM停止を確認した。

## batch-loop候補

既存の `16ef71a8` / `41751d40` のSmall Linear由来native batch-loopを参照した。旧cos⁴/Fourierの数学や当時の速度をB-splineへ転用しない。v19はnativeのbatch tileを8/16等に縮め、支持がsite tileに入るaxisではprofile/中心微分を一度作ってbatch間で再利用する。支持がtileを超える場合は全範囲のblock loopへ落とし、支持を切らない。batch部分ごとのHG/HGμ/HμGをFP64で足し、最後に同じsource pullbackを適用する。

旧v18の意味は保持し、v19で保存量とlaunchを比較する。幅・配置は引き続きforwardごとに更新し、前stepのprofileを使い回さない。この候補の性能・精度はこれから検証する。

## batch-loop結果

job `l4job-b0f6353dd4d34a49bcd8698c26f796e0`、source `d5ec8fa4`。全record PASS。独立runは1件。

| 方法 | B32完全step ms | B64完全step ms |
|---|---:|---:|
| 旧最速fma | 1.615691 | 2.491328 |
| v18 W-free H/Hμ保存 | 2.455866 | 3.992452 |
| v19 batch8 / atom4、H/Hμ保存 | 3.121620 | 5.349109 |
| v19 batch16 / atom4、H/Hμ保存 | 3.095024 | 5.326941 |
| v19 batch8 / atom8、H/Hμ保存 | 2.927328 | 5.152182 |
| v19 batch8 / atom4、H保存 | 3.250105 | 5.602215 |
| v19 batch8 / atom4、H非保存 | 3.344826 | 5.777541 |
| Dense | .082143 | .103211 |

保存方針ごとの両peakはv18と同じ。batch8/atom4のB64 backward registerは150、spill slots0へ減ったが、native backward診断は1.307→2.332msに増加した。batch16/atom4は245 registers / spill0、batch8/atom8は204 registers / spill0。それらも遅い。退避をなくすだけで高速になるという仮説は否定され、これ以上このlaunchの微調整を採用根拠なく続けない。40 CUDA +1 metadataがPASS。

source SHA256 `0e9f5f59c3b6a1bd09c539bbb8813a5c4aefba1b9ac5eee5a4fdd3b42b5e9597`、result SHA256 `1f68f6860a8617ac0b29bc51241e9e28065ba700f63c659583470836d76f749d`。manifest54件、source1503ファイル、正常終了とcleanupを照合した。

## 小σだけ・切替・共通経路回帰

job `l4job-5827a7360a564f838b0dad088fe93e9c`、同じ最終runtime source `d5ec8fa4`。

| B32方法 | 小σのみ ms | 切替 ms | allocated / reserved MiB |
|---|---:|---:|---:|
| 旧fma | 1.452436 | 1.730735 | 81.288 / 172 |
| W-free H/Hμ保存 | 6.595954 | 6.548019 | 43.792 / 132 |
| W-free H保存 | 6.625095 | 6.599522 | 38.143 / 134 |
| W-free H非保存 | 6.627135 | 6.622745 | 31.743 / 94 |
| 既存small W | 1.206630 | 未選択 | 50.231 / 114 |
| Dense | .082603 | .082372 | 49.002 / 106 |

小σをHで処理するだけではDense並みにならない。W経路のbatch再利用と、H経路のbatchごとの計算を区別する必要がある。小σ・切替とも数式/全勾配のgateはPASSしたが、この速度では採用しない。

共通query/lean/native/loop経路の回帰224 passed /647 deselected（220 CUDA +4 metadata）、257.69秒、cuBLAS初期context warning1件。全871件をGPUで再実行したという意味ではない。最終runtimeのCPUは1979 passed /3635 skipped /18既存warning。Ruff、全Case宣言、wheel/sdist内350 runtimeファイルのbyte一致も確認した。

source SHA256 `705a62128d895cdfac4c1f52c77d246d18ae05588d22c3df8fad5e9ab6ccf785`、result SHA256 `430a7d868de6eeba2f6162f065702f9841e17eeaf567daeae8ae720553a82073`。manifest45件、source1503ファイルを検証し、正常終了・cleanup acknowledgement・全owned L4停止を確認した。原本は `output/bspline-native-h-20261020/verified-jobs/` に保持する。

メモリ条件を満たす設定は得たが、速度条件は未達で目標を継続する。次は[局所座標momentのCPU screen](quadratic-bspline-local-moment-screen-20261020.ja.md)で数値誤差を確認した方式をGPU候補として検証する。これは同じB-splineの座標・離散和の書き換えで、Gaussian化や別kernelへの置換ではない。

3件全体のrunnerの最大誤差はY 1.113615e-5 / dX 9.283949e-6 / 全atom勾配3.036035e-5、relative L2 ≤3.595e-7。全35件のCST correctness recordで、同じsource勾配を与えた公開CSTOptimizerとのParameter/moments/step更新照合はmax/relative L2とも0。これは勾配自体の正しさを代替せず、勾配は別の独立FP64 gateで確認した。
