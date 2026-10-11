# BM16/G16の細部ボトルネック診断（RTX 6000 Ada、2026-10-11）

**結論：単一のscatterやallocateが支配している段階ではない。Y/dX owner集約、G/Hとparameter partialの生成、forward H生成、AdamW/Polar更新に時間が分散している。**
入力側でatomを束ねる方針はXの重複アクセスを減らす可能性を保つが、同じ順序ではdYの支持が揃わず、元のatom IDのpacked fieldsにも散った参照が残る。packed fieldsの物理配置を参照順に揃える対照は両rhoで改善した。全profileのglobal cacheはrho8で悪化し、追加容量も大きい。

この文書は測定と次の実装判断を記録する。公開selectorへの採用は行わない。既存設計は[Hの寿命を軸にした設計](../../h-lifecycle-kernel-design.ja.md)、前段の実験は[研究履歴](regular-grid-h-lifecycle.ja.md)にある。

## 条件・検証・測定境界

- NVIDIA RTX 6000 Ada Generation 48GB、142 SM、L2 96MiB。Torch 2.9.1+cu128、CUDA12.8、Triton3.5.1。
- N2048/B32/A209715、初期rho3/8、seed41、RegularGrid + D2 FlatTorus + normalized profile product、FP32/TF32 off。AdamW fused/capturable、lr1e-4、weight decay0.01、24回実更新。
- BM16/G16を基準とし、forward H32/BM8、Y owner BM16、backward G16/producer BM16、G storage/dX owner BM8、BK8、producer atom group8。H/Gをforwardからbackwardへ保存しない。
- rhoごとに独立2回。2回目は5個のprimary workerと19個のmicroprobeの順序を反転。primaryは420個の時間sample、microprobeは76個の独立process。sample数を独立run数とは呼ばない。
- GPU qualification254件PASS、failures/errors/skips各0。全shape/all atomsの独立FP64 Y/dX/all-four-dP gateをinitial/post24の計32件、max abs/relative L2各4e-4でPASS。許容値・caseは変更していない。
- BM16/G16の独立repeatはpost24 P/Y/dX/all-dPがbitwise同一。matrix/factor/onchip controlsのpost24 parameterは独立repeatで異なるため、その事実をraw auditに残す。各々のFP64 gateの合格とbitwise再現性は別である。
- primaryは非計装の完全学習stepとcapture/replay peak。phase/stage/軸probeは別copyまたは固定snapshotの診断であり、相互に足したり引いたりしてprimary時間の内訳と断定しない。
- CUPTI traceは別model/optimizer copyの3回の更新Graph replay。copy clock27→30、primary post24 parameterは不変。traceのkernel合計とprimary step時間は別の測定。

## 完全stepと実測peak

| 初期rho | BM16/G16 step ms | dense step ms | dense比 | CST allocated/reserved MiB | dense allocated/reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.426–1.432 | 0.329 | 4.337–4.350× | 57.431/104.000 | 81.752/106.000 |
| 8 | 1.831–1.834 | 0.329 | 5.560–5.580× | 57.431/104.000 | 81.752/106.000 |

基準は両peakともdense以下を維持する。今回のmicroprobe改善を完全step改善と認定していない。

## 全体の時間がある場所

以下は別copyのphase/stage中央値（ms）。forward/backwardの細目は固定post24 snapshotを再利用する。

| 測定対象 | rho3 ms | rho8 ms |
| --- | ---: | ---: |
| forward + loss | 0.519–0.521 | 0.661–0.663 |
| backward | 0.547–0.550 | 0.804–0.807 |
| optimizer | 0.374–0.375 | 0.374–0.375 |
| forward: snapshot/正規化・支持準備 | 0.099–0.100 | 0.115–0.116 |
| forward: 候補index | 0.049 | 0.050 |
| forward: owner fields | 0.013 | 0.013 |
| forward: X→H | 0.187–0.189 | 0.248–0.249 |
| forward: H→Y owner | 0.162–0.163 | 0.222–0.223 |
| backward: 入力routing/fields | 0.062–0.063 | 0.061–0.066 |
| backward: G/H + parameter partial | 0.275–0.276 | 0.422–0.428 |
| backward: G→dX owner | 0.189–0.190 | 0.296–0.298 |
| backward: physical accumulation | 0.019 | 0.019–0.020 |
| backward: source VJP | 0.010 | 0.010 |
| optimizer: previous snapshot | 0.005 | 0.005 |
| optimizer: AdamW | 0.249–0.250 | 0.249–0.250 |
| optimizer: Polar/periodic retraction | 0.118 | 0.118 |

候補indexは毎step作り直す。中心・sigmaが更新されるため、固定indexを正しさの確認なしに持ち越せない。範囲を直接割り出せても、209715 atomのkey生成・sort・prefix境界・owner fieldsの準備は無料ではない。一方、この準備だけが全体を支配しているわけでもない。

## 実際のCUDA Graph trace

| kernel群 | rho3 ms/replay | rho8 ms/replay | 起動数/replay |
| --- | ---: | ---: | ---: |
| Y+dX owner | 0.347–0.349 | 0.509–0.510 | 6 |
| G/H + partial | 0.276–0.278 | 0.428 | 2 |
| forward H | 0.194–0.195 | 0.261–0.262 | 1 |
| 正規化・支持prepare | 0.095–0.096 | 0.111 | 1 |
| physical accumulation | 0.015–0.016 | 0.016 | 2 |

CSTは131 kernel/replay、denseは15 kernel/replay。CST trace上のGPU activity間のgap合計は約0.012–0.013ms/replay、denseは約0.002ms/replay。これは計装traceの観測であり、kernel数だけからhost launch overheadをprimary時間へ当てはめない。実行時間は大きなkernelにも存在し、単に小kernelを融合するだけでは主要部分を解消できない。

ownerは同じkernel名でも、最初の2起動がforward Yのnormal/guarded-fallback pair、その後4起動がbackward dXの2chunk各normal/fallback pairである。安全条件を満たすcaseではfallbackは早期returnする。batch tileごとに別kernelを起動している、という意味ではない。ソースの固定scheduleと時系列を使って区別した。RadixSortのOnesweep kernelは8起動で約0.051–0.052ms/replayを占めるが、これをrouting全体の時間とは呼ばない。

## G/H生成を19種類に分解

全probeは同じpost24 X/dY/packed fields/orderを使い、B32を16ずつ処理する。全math probeをproduction値に照合した後に測る。軸probeはmaterialized normalized per-site fieldsのFP64縮約とも照合する。この軸FP64はwhole-model FP64 oracleの代用ではない。

| probe | rho3 ms | rho8 ms | 検証する差 |
| --- | ---: | ---: | --- |
| runtime | 0.265–0.269 | 0.418–0.419 | 元producer |
| mirror | 0.262–0.269 | 0.418 | 同じ数式のdiagnostic mirror |
| sorted-P | 0.232–0.237 | 0.374–0.376 | packed Pをinput-orderに物理配置 |
| sorted-partials | 0.262–0.265 | 0.412–0.415 | partialをsorted positionへ書く |
| sorted-both | 0.225–0.230 | 0.371–0.406 | Pとpartialの両方 |
| batch-contiguous | 0.481 | 0.508–0.512 | X/dYのbatch軸を連続化 |
| constant-site | 0.213–0.219 | 0.328–0.332 | 全siteを同じ値/addressにする合成対照 |
| cached-profiles | 0.194–0.195 | 0.411 | profile value/dvを事前計算済み |
| cache-pipeline | 0.230–0.233 | 0.465 | profile生成を含めて同じGraph内で使う |
| compact-wrap | 0.255–0.261 | 0.389–0.432 | 周期wrapをguarded comparisonへ |
| G-only | 0.142 | 0.216–0.217 | Gのみ、parameter勾配なし |
| runtime-one | 0.264–0.267 | 0.415–0.421 | 元producer、1 replay/sample |
| eviction-pressure | 0.310–0.315 | 0.464–0.469 | 1 replay前に256MiBを触る合成対照 |
| G-value | 0.140 | 0.214–0.215 | Gの値だけ、1buffer |
| G-matched | 0.143 | 0.218–0.219 | Gの値を2bufferへ、書込み量対照 |
| G-derivative | 0.154–0.156 | 0.236–0.237 | Gと中心微分、2buffer |
| H-value | 0.091–0.093 | 0.136 | Hの値だけ、1buffer |
| H-matched | 0.095 | 0.137–0.140 | Hの値を2bufferへ、書込み量対照 |
| H-derivative | 0.116–0.117 | 0.172–0.174 | Hと中心微分、2buffer |

G/Hは逆方向の同型の縮約だが、同じ時間にはならない。ここではinput順にatomを束ねているためH側のsiteは重なり、G側のoutput siteは揃わない。G/H値の生成、normalization、読み取り、縮約の費用は微分を足す前から存在する。この非対称はアクセスcensusと整合するが、G/H時間差の全部をcache由来と帰属するにはoutput-order等の追加対照が必要である。

dH/dGは新しいX/dYを読み直す代わりに、同じload値から第二のaccumulatorを更新する。ただし正規化微分の補正・第二縮約・register利用は増える。2buffer同士のmatched→derivativeで、Gは約8–9%、Hは約22–25%の追加時間がある。微分の追加費用と書込みbuffer追加を分けた対照であり、融合producerから軸時間を差し引いてparameter-gradient固有時間を算出しない。

融合producer内のphysical partialは次の3つ。taskでは既存のlive sigma stop-gradientとsource VJPを維持する。

\[H_{ab}=\sum_i V_{ai}X_{bi},\quad G_{ab}=\sum_j U_{ja}\,dY_{bj}\]
\[dA_a=\sum_b H_{ab}G_{ab},\quad dc^i_a=A_a\sum_b G_{ab}\,\partial_{c^i_a}H_{ab},\quad dc^o_a=A_a\sum_b H_{ab}\,\partial_{c^o_a}G_{ab}.\]

最後にphysical partialをaccumulateして、元の4-coordinate parameterへsource VJPする。この後段は約0.019ms + 0.010msで、現時点の最大対象ではない。

## 支持とアクセスの実数

| 量 | rho3 | rho8 |
| --- | ---: | ---: |
| H有効site/atomの主な範囲 | 5: 2766 atoms / 6: 206949 atoms | 15: 21434 atoms / 16: 188281 atoms |
| G有効site/atomの主な範囲 | 5: 2680 atoms / 6: 207035 atoms | 15: 21206 atoms / 16: 188509 atoms |
| Hのpadding後lane利用率 | 74.83% | 99.36% |
| Gのpadding後lane利用率 | 74.84% | 99.37% |
| H group内 requested/union | 7.892 | 7.915 |
| G group内 requested/union | 1.010 | 1.027 |
| H論理read MiB/B32 | 153.262 | 406.983 |
| G論理read MiB/B32 | 153.273 | 407.011 |

全2048 siteの総当たりは避け、主に6/16 siteを扱う。rho3はBK8に対して約25%のmasked laneがあり、rho8のpaddingは約0.6%。しかし入力支持の範囲が8atom間で約7.9倍重複する一方、G側は約1.01–1.03倍で、ほぼ重ならない。これはunwrapped intervalのCTA単位censusであり、実cache hit rateやload transaction数ではない。

packed Pの13fieldについて32byte address sectorをgroup単位で数えると、original配置の約272万からsorted配置の約63万へ減る（約4.3倍）。partialの3fieldは約63万→13万。ただしpacked対照の改善が大きく、partialの並べ替えだけは小さい。address censusはwarp内の実transactionsではない。

owner集約では、有効atom-site pairsと候補checkの比も見る。rho3のYは約125.6万有効pairsに対して約544.7万candidate-site checks（約23.1%）、dXは約125.6万対377.6万（約33.3%）。rho8はY約333.4万対755.1万（約44.2%）、dX約333.4万対587.2万（約56.8%）。全site総当たりより大幅に減っている一方、owner tileとguarded radiusによる余分なcheckは残る。この比はbatch multiplicityとatom-group paddingを除いたCPU censusで、GPU lane利用率や時間割合そのものではない。

logical read量は同じ小さなX/dYへの反復参照も数えている。これをDRAM読取量に変換しない。256MiB pressure対照の悪化はcache状態への感度を示すが、全cacheの完全evictionや実hit率は保証しない。constant-site対照はデータ/addressだけでなくcompilerのbroadcastやregister利用も変えるため、差の全部をcache損失とは断定しない。

## 並べ替え・cacheの費用と寿命

Pのみの配置対照はproducerを約10.0–11.7%短縮した。Pとpartial両方のrho8対照は0.371–0.406msと独立run間で振れ、wrap対照も0.389–0.432msで改善方向が揃わない。最速sampleを選んで一律採用しない。Pをsorted配置へコピーする準備Graphは約0.036ms、combined対照での同じ準備は0.036–0.040ms（いずれも独立測定）。rho3でproducerが節約する時間と同程度なので、後からcopyを一つ足すだけでは完全step改善を保証できない。sorted partialは最後にraw atom IDへ戻す費用も必要になる。次の実装では「別コピーを増やす」より、canonical packed/physical layoutの作成順を設計する。

global profile cacheは4×A×S float。rho3はS8で25.600MiB、rho8はS16で51.200MiBを追加する。rho3では生成込みprobeも改善するが、rho8ではconsumer単体の改善が小さく、生成込みで悪化する。profile評価は無料ではないが、今回の両rhoへ全profile保存を一律に採用する根拠はない。局所・小batchで使い切るH/G/profileの寿命を引き続き優先する。

以下は独立microprobe自身の実測allocator peak。model/optimizer/完全stepを含まない。validation用GPU buffersを解放したことをweak referenceでcapture前に検証している。cached-profile bufferとsorted copyは意図的に残す。probe後の並べ替え費用測定はこのpeakへ含めない。

| probe | rho3 allocated/reserved MiB | rho8 allocated/reserved MiB |
| --- | ---: | ---: |
| runtime | 27.702/36.000 | 27.702/36.000 |
| mirror | 27.702/36.000 | 27.702/36.000 |
| sorted-P | 38.103/50.000 | 38.103/50.000 |
| sorted-partials | 27.702/36.000 | 27.702/36.000 |
| sorted-both | 38.103/50.000 | 38.103/50.000 |
| batch-contiguous | 28.202/36.000 | 28.202/36.000 |
| constant-site | 27.702/36.000 | 27.702/36.000 |
| cached-profiles | 53.702/62.000 | 79.702/88.000 |
| cache-pipeline | 53.702/62.000 | 79.702/88.000 |
| compact-wrap | 27.702/36.000 | 27.702/36.000 |
| G-only | 25.302/36.000 | 25.302/36.000 |
| runtime-one | 27.702/36.000 | 27.702/36.000 |
| eviction-pressure | 283.702/292.000 | 283.702/292.000 |
| G-value | 25.302/36.000 | 25.302/36.000 |
| G-matched | 38.102/50.000 | 38.102/50.000 |
| G-derivative | 38.102/50.000 | 38.102/50.000 |
| H-value | 25.302/36.000 | 25.302/36.000 |
| H-matched | 38.102/50.000 | 38.102/50.000 |
| H-derivative | 38.102/50.000 | 38.102/50.000 |

全76 microprobeでtimed Graph replay中のTorch CUDA allocator allocation/free、segment allocation/free、device allocation/freeの増分は0。少なくとも測定したGraph内の生成時間を「毎回Tensorをallocateする費用」で説明することはできない。この観測をeager全体や全optimizerのallocator挙動へ一般化しない。

runtime compilerは94 registers/thread、0 spills、512 shared bytes。CUDA driver resource APIでは最大5 CTA/SM、理論active warp上限41.7%。batch-contiguous対照は72registers、上限58.3%でも遅い。cache対照は64registers、上限66.7%でもrho8のpipelineは遅い。register数や理論occupancyだけで速度を説明できない。

hardware countersはdriverのRmProfilingAdminOnly=1で制限され、ncu/nsysは未導入。実際のL1/L2 hit率、DRAM throughput、warp stall、達成occupancyは未計測。制約は[NVIDIAの説明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)を参照。権限やdriver設定は変更していない。

## 次の実装優先順位

1. **packed fieldsとphysical partialのlayoutを揃える。** input-orderに整列したcanonical packedを、余分な全量copyなしで作れるかを試す。G16/H32のcapacityを維持し、最後のphysical→source VJPでraw ID対応を復元する。準備・復元込み完全stepと両peakで判断する。
2. **G側のdYアクセスとH側のXアクセスを別に改善する。** input順はHに有利、Gに不利。Gをoutput順に生成してinput順で使う案は再配置・buffer寿命が増えるため、mapping費用を含む対照で判断する。いま全量index adjacency/CSRを追加する根拠はない。
3. **Y/dX ownerの候補チェック・同じfield/H/G参照を細かく分ける。** 合計では最大級のkernel群。atomic回避は維持し、owner tile内で候補とvalueをどこまで共有できるかを試す。
4. **AdamWとPolar retractionを別対象として検証する。** 約0.374msのoptimizerのうち約0.249msがAdamW、約0.118msが座標更新。全体に効く可能性はあるが、AdamWがこの時間になる低水準の理由と更新kernel融合の費用は今回のproducer対照だけでは未確定。moment/step/periodic/Polar数学の独立gateを維持する。
5. **BK8 paddingとprofileの局所再利用。** rho3のmasked laneは明確だが、BKを狭めるとloop回数や縮約・layoutも変わる。cacheを全量保存する方向へ戻らず、短い寿命の対照から始める。

## 証拠・復旧

- 測定source: `16d9011ba1a7f71845977560ce8403de88a069e8`。run: `cst-fine-g16-ada-16d9011b-20261011`。
- source archive SHA256: `c207e4aac12172b8a904d6958db7df03296824651ff88d8764b09af338718fa4`。
- compact archive SHA256: `216b0ab388cb88669031dacf5f744f94a0c10eea6cd0381f43c6c76704bd9344`。390 entriesをlocalで全SHA検証。
- 独立CPU artifact/numerical audit PASS_COMPLETE: `f6a277e9b7132d0ab8080b6b528ec97a5c301dc6b46abcbb394d57d0923b68bf`。
- local: `output/regular-grid-h/fine-results/`、`fine-source-manifest.json`、`fine-operating-notes.json`、`audit-fine.py`、`fine-fixed-cpu-tests.log`。raw `.pt`はauthorized clusterに保持。
- remote: `/home/jovyan/runs/cst-fine-g16-ada-16d9011b-20261011/`。controllerと4cohortが完了し、GPU processなし・0%/39MiB。user-opened Hubはreadyのまま。
- 初回 `a53bf790` campaignはmath/primary計測/監査にPASSしたが、microprobe capture前にexpected partialやaxis用support bounds/最後のmaterialized FP64 fieldが照合用GPU storageを保持していた。個別probe peakに影響するため、`16d9011b`で明示解放とweak-reference gateを追加し、254 qualification + 4 full cohortsを再計測した。初回archive/source/results/auditは削除せず `fine-results-a53bf790/` と旧remote runへ残す。新結果を個別メモリの根拠に使う。
- CPU 1740 PASS、3806 GPU/DB skips、Ruff/diff check PASS。[測定sourceのCPU/build CI](https://github.com/takumiecd/torchcst/actions/runs/38116792413)もPASS。最終docs commit CIはPRに記録する。

再現は、新しい出力ディレクトリで次を実行し、rho8および逆順runも独立processで繰り返す。

```bash
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho3.json \
  --plans matrix-torch factor onchip-h input-site-reciprocal-bm16-g16-h32 dense \
  --isolated-oracle --phases --fine-diagnostics \
  --output output/fine-new-rho3/comparison.json
```
