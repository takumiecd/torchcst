# RegularGridの支持幅分布別比較

2026-10-11。速度の限界をrho3/8だけで判断せず、狭い・中間・広い・混在の条件で
H/Gの生成・集約・微分の費用を確認する。公開dispatcherの変更は行わない。

## 固定した比較条件

RTX 6000 Ada、D2 FlatTorus、RegularGrid `((2048,), (2048,))`、spacing=1、
B32、209715 atoms、seed41、FP32、TF32無効、全積のdiscrete L2正規化とfloor=1e-6。
新しい全caseで共通の幅境界はminimum/birth/upper_floor=0.125、maximum=16。
旧rho3/8 caseの下限1とは異なる研究条件として区別し、旧結果との速度差を改善率にしない。

| case | 初期rho | atom比率 |
| --- | --- | --- |
| rho0p25 | 0.25 | 各軸で高々1site。空支持も含む |
| rho0p5 / rho0p75 / rho1 | 0.5 / 0.75 / 1 | 各単一幅 |
| rho1p25 / rho3 | 1.25 / 3 | 各単一幅 |
| rho8 / rho15 | 8 / 15 | 各単一幅 |
| mix-narrow | 0.5, 3, 15 | 8:1:1 |
| mix-balanced | 0.5, 3, 15 | 1:1:1 |
| mix-wide | 0.5, 3, 15 | 1:1:8 |
| mix-rho1-narrow | 1, 3, 15 | 8:1:1 |
| mix-rho1-balanced | 1, 3, 15 | 1:1:1 |
| mix-rho1-wide | 1, 3, 15 | 1:1:8 |

rho0.5を含む混在caseは独立oracleの勾配gate失敗をそのまま保存する。
rho1の混在caseは空支持近傍の鋭い条件と幅の混在による実装費用を切り分けるための追加条件。
最初のcaseを置き換えず、両方を別caseとして残す。

幅のquotaは整数切り捨て後の余りを最後の群に入れ、seed+7919でatomへ割り当てる。
中心と振幅の乱数列は従来fixtureと共通。各Planに同じ初期Parameter、入力、target、dyを渡す。
幅はPolar Parameterから生成し、24更新中に変化する。固定幅に置き換えない。
現行Polarは共有alphaを使い、この研究の同じ側別boundsでは入力・出力幅が等しい。
各atomで独立した入力・出力幅の分布を与えた実験ではない。

Triweightの支持は `abs(delta) < sigma`。rho<=1でも中心の位置によって空支持・単一・複数siteとなる。
一般にone-hotと仮定せず、両軸の支持site数、空product atom数、幅分位点、実際の幅変化を記録する。
空支持を削除したり、floorを軸別に変更したりしない。

厳密にrho<0.5なら各軸で高々1siteになる。全積が空になる場合があり、
非空でもnorm積がfloor以下なら正規化weightは単純なampに縮退しない。
空支持・単一siteかつfloor非活性・単一siteかつfloor活性を別経路の候補として扱う。
rho0p25は、この境界の確認後に追加した独立caseであり、既に凍結した13条件は変更しない。

## 測定と採否

既存 `benchmarks.cuda.linear.periodic_comparison` に新しいfrozen caseを追加し、別runnerは作らない。
比較Planはmatrix-torch、factor、onchip-h、現行tile-owner-late-bm16、dense。
全Planでoptimizer state preallocate、Graph開始前warmup gradient解放、native allocatorを揃える。
準備・集約・全backward・AdamW/Polar更新を含む未計装の完全stepを主指標とする。
別Graphのforward/loss・backward・optimizer診断は主計測と非加算。
capture/warmup/replay全体のpeak allocatedとreservedを報告する。GPU process全体は未測定。

幅分布ごとにCPUと小さいCUDA回帰を通し、全shape・全atom・Y/dX/all_dPの独立FP64 gateを
initial/post24で確認する。許容誤差は従来のmax_abs/relative L2とも4e-4を変更しない。
失敗caseの時間を有効な速度比較として使わず、raw snapshotと失敗を保管する。
各有効caseは独立2cohort、2回目はPlan順を反転。21時間sampleを21独立runとは数えない。

再現例:

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:native python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-width-2048-mix-balanced.json \
  --plans matrix-torch factor onchip-h tile-owner-late-bm16 dense \
  --isolated-oracle --phases --optimizer-state-policy preallocate \
  --graph-pool-policy release-warmup --output output/width-distribution/comparison.json
```



## RTX 6000 Adaの有効な比較結果

測定source `54f99b6e37e34076301418076a8201278f177ce4`。
8条件はそれぞれ独立2cohort、Plan順反転で全5 workerがPASS。
**80 fresh workers、1680 samples、initial/post24の全shape FP64 gate128回PASS**。
CPUおよびCUDAの共通正規化・支持・全4Parameter勾配の契約は変更していない。

| 初期幅分布 | matrix-torch ms | factor ms | onchip-h ms | 現行tile-owner-late ms | dense ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| rho0p75 | 0.669835–0.675524 | 1.182494–1.188745 | 1.386696–1.387742 | 1.325969–1.328973 | 0.329411–0.329516 |
| rho1 | 0.672914–0.677388 | 1.254276–1.257813 | 1.365677–1.368131 | 1.343399–1.343903 | 0.329415–0.329497 |
| rho3 | 0.715062–0.716981 | 1.924329–1.933310 | 1.779677–1.780313 | 1.378780–1.379994 | 0.329494–0.329609 |
| rho8 | 1.368063–1.368114 | 4.206337–4.235391 | 3.790746–3.811322 | 1.715896–1.718181 | 0.329453–0.329549 |
| rho15 | 3.586016–3.597766 | 7.582266–7.623957 | 6.774405–6.842973 | 2.186293–2.193268 | 0.329475–0.329556 |
| mix-rho1-narrow | 1.467319–1.469262 | 2.430387–2.559018 | 2.689591–2.696923 | 1.560396–1.560894 | 0.329483–0.329546 |
| mix-rho1-balanced | 2.261509–2.268228 | 4.243810–4.422462 | 4.627629–4.720333 | 1.760969–1.762902 | 0.329511–0.329583 |
| mix-rho1-wide | 3.365811–3.399198 | 6.486884–6.535193 | 5.839474–5.842553 | 2.085405–2.104313 | 0.329535–0.329706 |

全8条件で以下のcapture/warmup/replay peakは一致。GPU process全体は未測定。

| Plan | allocated MiB | reserved MiB |
| --- | ---: | ---: |
| matrix-torch | 96.962891 | 126 |
| factor | 83.114258 | 106 |
| onchip-h | 42.866211 | 74 |
| tile-owner-late-bm16 | 62.774414 | 88 |
| dense | 81.752441 | 104 |

現行は全条件で両メモリ指標がdense以下。
matrix-torchはrho0.75/1/3/8と狭い群中心のmixでは現行より速いが、96.963/126 MiBでdenseを超える。
onchip-hは42.866/74 MiBと小さいが、完全stepは現行より遅い。
速度だけの最速と、dense以下のメモリ条件での最速を区別し、公開dispatcherには採用しない。

## 別Graphでの内訳

各条件の独立run1、24更新後の別instrumented copy。主stepと非加算であり、
これらを足したり主stepから差し引いたりしてexclusiveな構成比を求めない。

| 分布 | forward/loss ms | backward ms | optimizer ms |
| --- | ---: | ---: | ---: |
| rho0p75 | 0.416288 | 0.544288 | 0.373984 |
| rho1 | 0.426720 | 0.549696 | 0.374208 |
| rho3 | 0.448096 | 0.566016 | 0.375520 |
| rho8 | 0.552064 | 0.791232 | 0.375808 |
| rho15 | 0.728960 | 1.086048 | 0.375904 |
| mix-rho1-narrow | 0.518816 | 0.674208 | 0.375648 |
| mix-rho1-balanced | 0.587776 | 0.803552 | 0.375680 |
| mix-rho1-wide | 0.702560 | 1.016352 | 0.375712 |

rho0.75とrho3の現行step差は約4%。狭い支持の全処理が既に軽いとは言えない。
幅が広くなるとforwardとbackwardの診断値が増え、optimizerは約0.375 msでほぼ一定。
この内訳だけではG/H・微分係数・読み取り・所属処理のexclusiveな原因は確定しない。
次は単一siteかつfloor非活性のatomを一般H/G生成から分ける案と、
広い支持のH/G・勾配生成の共有を分けて比較する。効果は未測定。

## 正しさの失敗と保全

rho0.5、rho1.25、rho0.5を含む3つのmixは **INVALID_FOR_PERFORMANCE**。
誤差基準4e-4を維持し、失敗caseの時間を速度比較や候補選択に使わない。

小さいCUDA回帰90件は78 PASS/12 FAIL、skip0。12 FAILはrho0.5・mix-narrow・mix-balancedの
all_dP gateで、matrix/factor/onchip/現行の全4routeに共通。mix-wideは小さい回帰ではPASSしたが
全shapeでFAIL。rho1.25はinitialが通ってもpost24でFAILした。
最初の試行source32aef76aではatomic dXにビット再現を要求する不適切な追加testが20件失敗した。
source a8ff0d07で修正し、2回のVJPを同じFP64 oracleでそれぞれ検証する。
Parameter勾配のビット再現と、非atomic routeのdXのビット再現は維持。
数値基準は変更せず、source32aef76aの最初の試行・全shape診断を別に保全している。

source32aef76aの保存された現行raw snapshotを、独立FP64 oracleで再診断した結果:

| case・時点 | all_dP max_abs | relative L2 | FP64真値をFP32へ最近接丸めした最小max_abs | 最大誤差atomのnorm積 / floor活性 |
| --- | ---: | ---: | ---: | --- |
| rho0p5・initial | 0.0606459157 | 5.28552984e-06 | 0.000360223201 | 8.25551491e-07 / True |
| rho1p25・post24 | 0.00117356293 | 8.70595141e-07 | 1.32387243e-06 | 0.834996725 / False |
| mix-balanced・initial | 0.0450774289 | 5.22369473e-06 | 0.000233126464 | 7.90948196e-07 / True |
| mix-wide・initial | 0.0450774289 | 5.34068033e-06 | 0.000233126464 | 7.90948196e-07 / True |

rho0.5/mixの最大誤差atomは両側1site、norm積はfloor=1e-6未満。
単一siteを機械的にamp-onlyへ変更できない具体例である。
rho1.25のpost24最大誤差atomはfloor非活性、出力中心0.02593824の周期境界近傍。
これだけで全失敗の原因を断定せず、floorでの係数・微分と周期差の丸めを別の診断対象にする。

## 証拠と再現範囲

- 共通source archive SHA256: `3dd373513b75bf768e1c7cefc80992f9d3ea3d27a8451086938d00f6d6f68e93`。
- verified compact archive: `49d05372c7ae94f217b952cec20925c282b9f0af472cea63b08124cce88f7b66`。
- audit SHA256: `bb633d2f924d5981984ad6b1074565324e11bf1e5e6cd72e31c436ce656c8e10`。
- 1693 pinned tracked source files、280 compact evidence filesを照合。
- local evidence: `output/regular-grid-h/width-sweep-v2-results/`、source manifest、receipt。
- remote: `/home/jovyan/runs/cst-width-sweep-v2-ada-54f99b6e-20261011`、controller PID304871。
  143 raw oracle snapshotsはremoteで保全、manifestにサイズとhashを記録。再試行・条件緩和なし。
- 最初の診断remote: `/home/jovyan/runs/cst-width-sweep-ada-32aef76a-20261011`、PID301597。
  compact hash `85c07080fdf7abe9cf09f94f67eedba48808438d058ea7a56ca6b99d97a6fde4`、68 entries照合。
  追加FP64診断は `output/regular-grid-h/width-sweep-precision-diagnosis.json`、
  SHA256 `1bddd0a4f45d01e053a649aec326fa8c12731487ad77a1457ca6ac8332bff438`。
- [共通source CPU/build CI](https://github.com/takumiecd/torchcst/actions/runs/38145081419) PASS。

幅境界、初期分布、case schemaが旧rho3/8と異なるので、旧時間を混ぜた改善率は算出しない。

## rho<0.5の独立追加case

source `e9f6469f346ecc379308b32818daa73c56c316bd` のrho0.25。
共通13条件のsource54f99b6eを変更せず、新しいsourceでこのcaseだけを検証した。
小さいCPU/CUDA回帰6件PASS、skip0。N2048/B32/A209715のfull gateはmatrixと現行で
initial all_dP FAILとなり、**完全step時間の有効な比較は作っていない**。

| 独立FP64での支持区分 | atoms | 全体比率 |
| --- | ---: | ---: |
| 空product支持 | 157048 | 74.886393% |
| 両側1site、floor非活性 | 51563 | 24.587178% |
| 両側1site、floor活性 | 1104 | 0.526429% |

最大誤差atomは両側1site、norm積 `9.15274325e-07`、floor活性。
all_dP max_abs `0.0727983564138`、relative L2 `5.2205020356e-06`。
独立FP64真値を最も近いFP32値へ丸めるだけでも、全tensorの最小max_absは
**`0.000532731413841` > 4e-4**。
このsource/caseでは、どのFP32勾配実装でも現行の絶対誤差gateを満たせない。
これは実装の追加丸め誤差がないという意味ではない。実際のmax_absはその下限より大きい。
数値計算の改善と、表現dtypeに対する絶対gateの妥当性を分けて扱う。基準は変更していない。

約99.474%のatomは、taskの寄与が0またはamp-onlyに縮退する候補。
これらのH/Gの全量生成・微分を避け、1104 floor-active atomsへ一般式を残す案は有望な仮説。
空支持のParameter/AdamW stateを削除せず、幅の成長・中心移動・次forwardでの再分類を維持する。
この軽量経路は未実装・未計測であり、速度やメモリ削減率は約束しない。

- source archive SHA256 `0f4cde7bc55bbe00b3b0b63da8f2ea0d1a15c416b1dc29dda4d0ffde456d5ab9`。
- result archive `088adf4a720183fbe74f687fc8f47ff7d92ed6211b651f341f4b87b56c6d1172`。
- audit `84a8e3bb1abfebee8d39fb228cda6df20a088e19c987a7ba027031c2484f8ab4`。
- 1694 pinned source files、20 compact evidence files照合。3 raw oracle snapshotsはremoteで保全。
- remote `/home/jovyan/runs/cst-width-single-ada-e9f6469f-20261011`、PID317701。
- local `output/regular-grid-h/width-single-results/` とreceipt、source manifest。
- CPU全suite **1847 PASS/4630 GPU-or-DB skips/18既存warnings**。
- [rho0.25追加source CPU/build CI](https://github.com/takumiecd/torchcst/actions/runs/38145633754) PASS。
- 最終確認でuser-opened RTX 6000 Ada Hubはready、GPU0%、39 MiB、compute processなし。停止・再起動なし。
