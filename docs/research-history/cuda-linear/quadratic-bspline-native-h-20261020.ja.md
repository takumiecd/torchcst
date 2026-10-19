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
