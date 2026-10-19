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
