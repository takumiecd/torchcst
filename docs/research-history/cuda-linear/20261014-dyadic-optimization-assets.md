# 丸い指数profileに合う既存最適化資産の選定

2026-10-14。対象は[検証済みdyadic streaming](20261014-dyadic-streaming.md)。
利用者σは減衰長b。幅0.1..10、隣接2幅混合、periodic native images、
実sitesの離散L2・中心微分、task幅stop-gradientを固定する。
以下は次の実装候補であり、新しいGPU速度・メモリ改善を実測した結果ではない。

## 選定

メモリを優先し、**stream2を基準にH再計算＋batch16、次にbatch8**を試す。
このprofile特有の速度候補は**厳密Gram圧縮＋両軸のFP64準備融合**。
次にgrouped atom VJPとPolar pullback融合を比較する。
全候補を一度に混ぜず、個別差分と組合せを測り、悪化したものを残さない。

|優先|資産 / 次の候補|この方式との適合理由|留意点|
|---|---|---|---|
|1|local_productのrecompute_hとbatch-loop|B×Aの保存・cotangentが大きい。入力Xを保存し、batchごとにHを再計算できる|scan回数・launch増加。batch8は現状16laneの半分が空く|
|2|periodic_product / grouped preparationの融合構造＋厳密Gram圧縮|Polar、幅区間、両軸norm、floor、係数のTorch中間値と重複指数を減らせる|古いTriweight式・FP32 decoderは移植しない|
|3|profile_product_globalのgrouped backwardとlocal_productの直接Polar store|1atom/CTAを8–16atoms/CTAへ変え、A方向連続readとbatch FP64 reductionを揃えられる|register spill・occupancy・全atom gradientを確認|
|4|Graph pool / H lifecycleのlifetime管理|今回もG/Gp解放だけで45.37→40.19/41.20MiBになった|arenaや全scratch常駐は逆にpeakを増やす可能性|
|低|compact dX metadata|実績があるが、今回の主要容量はH群|数KiBの削減より先に数MiBの生存区間を直す|

## 1. H保存をやめる候補

現状B32、A52428のFP32 `[B,A]` は1枚6.3999MiB。
H/Hp/G/Gpの4枚のpayloadは25.5996MiBで、live rangeが重なる。
batch16なら4枚12.7998MiB、batch8なら6.3999MiB。
これらはTensor容量の計算値で、完全stepピークの予測ではない。

forwardのH/Hp12.7998MiBを保存する代わりに、X0.125MiBと現在の
source/index/coeff snapshotを保存する。backwardで同じsnapshotに対するH/Hpを
batchごとに再計算し、G/Gpもbatch分だけ保持する。
係数・norm準備はbatchの外へ出し、全batchのcotangentをFP64で集めてから
公開Polar pullbackを行う。retained backwardやParameter更新後もforward時点の値を使う。

移植元は
`src/torchcst/_backends/cuda/algorithms/linear/local_product/executor.py`
の`keep_h`、`recompute_h`、saved inputの寿命と、同directoryの`kernels.py`の
`param_vjp_batchloop`、`_store_param_cotangent`。
現状streaming scanはbatch16 lanesなので16を先に確認する。

既存実績は[compact/batch-loop統合](../local-product/20261007-pr-integration.md)。
N64で115712→99328B、N128で303104→270336Bのallocated peak削減があったが、
latencyは悪化した。有限Triweightの結果を今回のMatérn性能へ換算しない。

## 2. profile固有の厳密Gram圧縮

D=N h、位相0<=δ<=h、λj>=λkとする。
固定profileはnative画像和をpeak正規化した `(1+λr)exp(-λr)`。
実siteのGramは次の形へ厳密に圧縮できる。

```text
Gjk(δ) = Fjk(δ) + Fjk(h-δ)
Fjk(z) = exp(-(λj+λk)z) Pplus_jk(z)
         + exp(-D λk) exp(-(λj-λk)z) Pcross_jk(z)
dGjk/dδ = Fprime_jk(δ) - Fprime_jk(h-δ)
```

Pplus/Pcrossは固定2次多項式で、有限gridの0/1/2次momentから一度だけ作る。
`exp(-D λk)`をPcross係数へ吸収し、巨大な正指数は使わない。
位相の中心微分はcell内でdδ/dcentre=-1。
混合normは `(1-t)^2 G00 + 2t(1-t)G01 + t^2 G11` のまま。
連続L2、位相補間、site間引き、support切断による近似を導入しない。

隣接dyadic幅はλnarrow=2λbroadなので、`u=exp(-λbroad z)`があれば、
3Gramはu^2/u^3/u^4とuだけで表せる。
反対側は固定step/u。h0.1、最小の選択broad0.15625ではu>=exp(-0.64)=0.52729で、
遠い裾の逆数を作る操作ではない。最大幅の1指数からの二乗でも生成できる。

独立native周期画像和（50最大減衰長より外まで）と独立中心微分で確認した。
201位相×15self/cross pairs、N1024/D102.4、N64/D6.4、N37/D3.7。
1指数＋累乗＋反対側除算版のN1024最大差はGram3.1264e-13、位相微分3.4071e-12。
既存closed formと比較するだけの循環検証ではない。
これはCPU FP64恒等式検証でありGPU高速化の証拠ではない。

原本と完全導出はignored outputの`compressed_gram_proof.py`、
`compressed-gram-note.md`、`compressed-gram.json`。
JSON SHA256: `d3c1a9637797183f63dab742be96f0ddee2a32bb75e194409eaa470418701df7`。

現在`matern_streaming/executor.py::_prepare`はTorchのPolar/幅区間計算、
軸別geometry kernel2本、coeff kernel2本、norm積/floor/ampのTorch処理を行う。
新候補では不変quadratic tablesを用意し、FP64 Polar decode、共通j/t、両軸位相、
3Gram/norm/中心微分、product floor、係数をatom block内で準備する。
source dtypeのsingular-origin guard、FP64 width/chart、floorを積の後で1回だけ適用する契約を保つ。

構造の移植元は`periodic_product/kernels.py`の準備kernelと
`profile_product_global/grouped_kernels.py`のgrouped preparation。
[fused Polar](../local-product/20261004-fused-polar.md)では有限Triweight N64/rho3で
0.249114→0.144748ms、[Sphere fused preparation](../local-product/20261009-sphere-fused-preparation.md)
ではN1024/sigma3で11.91–13.32%改善した。
両者は異なるprofile/geometryの実績で、今回の改善率を予測する値ではない。

## 3. 流用しないもの

- owner2 / support-adaptive / empty-singleton / CSRの有限support shortcut：今回のprofileは全域で非ゼロ。
- halo8のcompact derivativeやfinite-support W assembly：今回の数学のままでは同じ前提を持たない。
- 古いFP32 Polar decoder：今回の狭い幅ではnormalized中心VJPの誤差を増幅する。
- Parameter・中心・幅・normの跨step cache：live更新とforward snapshotを壊す。
- 先に全scratchを確保するarena：Graph込みallocated/reserved peakの改善を保証しない。

PR100 RegularGrid support-modeは以前1.26–8.35%遅く、H容量も減らず、rho0.25で全dP gateに失敗。
Torus on-chip routeにも全atom gradient不合格がある。これらの支持判定や融合全体を
正しさ確認なしで移植しない。`docs/h-lifecycle-kernel-design.ja.md`の生存区間設計を先に使う。

## 次の比較手順

まず今のdyadic準備・query・scatter/scan・parameter VJPを別診断として測る。
既存runnerのphase診断は旧strategy前提なので、streaming用診断をそのrunnerに追加する。
診断の時間和を未計装完全step時間の代わりにしない。

stream2 control、H-recompute/batch16、batch8、compressed/fused準備を別candidateとして比較。
初期と24更新後の全shape独立FP64 Y/dX/全atom dP、各max_abs/relativeL24e-4を維持する。
origin、seam、floor-active、retained/postmutation、optimizer moments/step、Graph live widthsを確認。
同じ初期値で隔離worker2cohortの順を反転し、warmup/capture/replay込みの
allocated/reservedと完全stepを測る。denseも同条件の対照として残す。
新候補のGPU検証はこの資産調査時点では未実施。
