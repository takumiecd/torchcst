# RegularGridの共通支持モード: 正しさPASS、完全stepは負の結果

2026-10-11、RTX 6000 Ada Generation 48 GB。
研究Plan `support-modes-bm16` を実装した。現行 `tile-owner-late-bm16` と同じ
executor / producer / ownerに、空・両軸単一サイト・一般支持の処理モードを置く。
全サイズのY/dX/全atom勾配は通ったが、完全stepでは現行より1.26–8.35%遅い。
**今回の候補は速度改善として採用せず、現行Planと公開dispatcherを維持する。**
共通実装の検証済み研究候補として残し、今後の判定・集約費用の切り分けに使う。

## 処理と寿命

- 既存13-field packedのflagsと支持区間から、両軸単一・joint floor非適用を判定。
  単一サイトを各軸sort keyの先頭band、一般支持を従来のdyadic band+1へ置く。
  空支持はownerが参照する範囲の後へ置く。毎forward再分類し、backwardはsnapshotを使う。
- 単一サイトでは、Y ownerがX、dX ownerがdYを直接参照する。
  `dA_a=sum_b X[b,i_a] dY[b,j_a]` をatom writerが計算し、中心task勾配は0。
  H/Gの書込み・読込みとprofileの再評価を省く。Polar source VJPは共通のまま。
- floor適用単一サイトと複数サイトは、既存の正規化とH/Gの小batch生成を使う。
  dH/dGは即時消費し、全量保存しない。atomicなしowner partial+reduceを維持する。
- 段階間のH/G依存のためGPU起動は複数回。一つの共通実装を使うことと、
  一回のGPU起動へ融合することを区別する。
- 初版のH/G容量は学習中の支持変化に備えてA×16を維持。バッファ読み書きを省く
  効果とallocator peak削減を混同しない。全量packedコピー、CSR、追加atom ID配列はない。

## 比較条件

D2 full-period FlatTorus / RegularGrid `((2048,),(2048,))`、spacing1、origin0、
N2048、batch32、A209715、seed41。Triweight profile product、joint discrete L2、
floor1e-6、Polar amplitude-width、FP32、TF32無効。
Torch2.9.1+cu128 / CUDA12.8 / Triton3.5.1。
全caseの幅境界はmin/birth/upper_floor=.125、max16。
以前のmin1のrho3/8比較と結果をプールしない。

単独rho=.75,3,15に加え、新たなmixはrho=.75,3,15を8:1:1、1:1:1、1:1:8で混在。
共通Polar alpha・等しい両側boundsのためinput/output幅は同じで、中心は独立。
整数quotaと固定shuffle seedで割当を凍結する。初期幅を固定して学習する方式ではない。

各caseについて2独立cohort、2回目はPlan順を反転。
各cohortはmatrix-torch / factor / onchip-h / current late / support modes / denseの
6個の独立worker。各workerはAdamWとPolar/periodic更新を24回実行し、21時間サンプル。
**72 worker / 1512 samples、CSTの120個のinitial/post24全サイズFP64 gateがPASS。**
Y/dX/全atom・全4Parameter勾配のmax_absとrelative_L2はいずれも4e-4以下を必須とする。
未計装の完全stepを主測定とし、別のphase計測は足し合わせたり差し引いたりしない。

optimizer stateを事前確保し、eager warmup勾配をcapture前に解放。native allocator。
peakはstate初期化・eager warmup・Graph capture/replayを含み、oracleと診断scratchは除外。
初期化の一回限りの遅延とGPU process総peakは測定していない。

## 完全stepの結果

単位ms。区間は2独立cohortの中央値の最小–最大。
増加率は同cohort内のcandidate/current比較であり、異なるcohort間の端点比ではない。

| 初期分布 | 現行late | 共通支持モード | 時間増加 |
| --- | ---: | ---: | ---: |
| rho0p75 | 1.328600–1.330170 | 1.373366–1.375101 | +3.25–3.50% |
| rho3 | 1.372866–1.376281 | 1.451347–1.455593 | +5.72–5.76% |
| rho15 | 2.222638–2.238482 | 2.264731–2.266685 | +1.26–1.89% |
| mix-rho0p75-narrow | 1.554624–1.555222 | 1.630389–1.630436 | +4.83–4.88% |
| mix-rho0p75-balanced | 1.756274–1.757646 | 1.876011–1.876012 | +6.73–6.82% |
| mix-rho0p75-wide | 2.082213–2.085416 | 2.218418–2.259605 | +6.54–8.35% |

全6caseでallocated/reservedピーク(MiB)は同じ:

| Plan | allocated | reserved |
| --- | ---: | ---: |
| 現行late | 62.774414 | 88 |
| 共通支持モード | 62.806641 | 88 |
| dense | 81.752441 | 104 |

allocatedは追加band/ranges分の33,792 bytes増、reservedは同じ。dense以下を満たすが、
H/Gのcapacityを縮めていないためメモリ削減にはなっていない。
denseの完全stepは.328867–.329693ms。共通支持モードは約4.2–6.9倍。

## 実際に直接参照できたatom

保存済みinitial/post24 ParameterをCPU FP64で分類した非計時の診断。
この表はGPU上のflags判定そのものやhardware traffic counterではない。
全6caseでemptyと単一floor適用は0だった。

| 分布 | 初期の両軸単一floor非適用 | post24 | 初期割合 |
| --- | ---: | ---: | ---: |
| rho0p75 | 52667 | 53673 | 25.114% |
| rho3 | 0 | 0 | 0.000% |
| rho15 | 0 | 0 | 0.000% |
| mix-rho0p75-narrow | 42117 | 42953 | 20.083% |
| mix-rho0p75-balanced | 17683 | 17939 | 8.432% |
| mix-rho0p75-wide | 5274 | 5355 | 2.515% |

単一サイトのH/Gを省けることは、poisoned NaN slabを読まない/書かないGPUテストで確認。
しかし直接参照atomが0のrho3/15でも遅く、単一サイト以外への追加費用が存在する。
判定、追加band/ranges、共通owner内の分岐は候補であり、どれが支配的かは未確定。

別Graphのphase medians（cohort1、ms）:

| 分布 | 現行forward+loss | candidate forward+loss | 現行backward | candidate backward |
| --- | ---: | ---: | ---: | ---: |
| rho0p75 | 0.415904 | 0.453152 | 0.546720 | 0.555712 |
| rho3 | 0.445376 | 0.470624 | 0.566688 | 0.617856 |
| rho15 | 0.729216 | 0.755040 | 1.087200 | 1.143232 |
| mix-rho0p75-narrow | 0.515296 | 0.559840 | 0.668096 | 0.703488 |
| mix-rho0p75-balanced | 0.584576 | 0.635296 | 0.802496 | 0.864288 |
| mix-rho0p75-wide | 0.700384 | 0.757888 | 1.015520 | 1.095840 |

forwardとbackwardの双方に増加を観測した。個別準備/owner/producerの費用、
cache hit、物理traffic、stall、occupancyは今回測っておらず、原因を確定しない。
次の候補は準備時の分類をprefixから読み、producerで再判定せず使うこと、
ownerの共通分岐をatom反復の外へ出すこと。いずれも未実装・未測定である。

## 狭いrhoの未解決のgate

rho.25はcandidateのinitial full-shape all_dPで
max_abs=.07279835641384125 / relative_L2=5.22050202746804e-6となりFAIL。
verify-only probeとして保全し、performance選択には使用しない。
これは以前のfloor適用atomの失敗と同じ値。以前の独立診断では、FP64 truthの
nearest FP32への丸めだけでmax_abs=.00053273141384となり4e-4を超える。
今回も基準を緩めず、quarter/mix含むsharp支持全般への性能改善を主張しない。
詳しくは[幅分布の比較](regular-grid-width-distributions.ja.md)を参照。

## 検証・証拠・再現

- GPU 197 PASS / 0 skip: candidate38、既存lifetime/protocol141、新mix controls18。
  floor境界・空支持・requested gradients・stride/seam/broad・retained backward・
  Graph更新・公開optimizer更新・nonfinite/zero casesを含む。
- CPU 1857 PASS / 4676 GPU/DB skip / 既存warning18。
  catalog declarations58 Plan PASS、ruff PASS。
- 測定source `efceba8eb7f889341758da897d7c15a7c4466fed`、
  frozen archive `a6039bd079a79a33442a4db3a27791e8d0288f6c996d1774c3b8c8c0d8ee01e7`。
  remote post-run auditでtracked source1698 fileの不変を確認。
- job PID318242、remote `/home/jovyan/runs/cst-support-modes-ada-efceba8e-20261011`。
  `source/`、`controller.py`、`controller.log`、`status.json`、`*-run*/comparison.json`。
  raw snapshot121個はremoteに保全。
- compact result archive `e7cb51c9c114e6842954a64ac503943e7e1137a866481f72a5a3aedb44f0a3e2`。
  local `output/regular-grid-h/support-modes-results.tar.gz` と展開先。
  228 fileのhashとsource commitを照合済み。
- CPU FP64 class診断 `73d2adb49a4e25b7ce622889f5bfde575ad351dd986132ab131e952c5673dd79`。
  post-run audit `2f46674e51364db4ec767859fb64db49afd3abcf3a81fb712b3e98af22c0746c`。
  local controller/freeze/retrieve/class drivers、source manifest、receipt、CPU logはignored outputに保全。
- 最終GPU状態: ready/idle、0%、39MiB、compute processなし。ユーザーが開いたHubは維持。

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:native python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-width-2048-mix-rho0p75-balanced.json \
  --plans matrix-torch factor onchip-h tile-owner-late-bm16 support-modes-bm16 dense \
  --isolated-oracle --phases --graph-pool-policy release-warmup \
  --optimizer-state-policy preallocate --output output/support-modes-comparison.json
# 別の独立cohortは新しい出力先に --reverse を付けて実行。
```

研究PR [#100](https://github.com/takumiecd/torchcst/pull/100) へ追加。
公開dispatcherへの採用、mainの変更、mergeは行っていない。
