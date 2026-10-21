# 階層Fourierの最終版切り分けと再利用 2026-10-21

branch: `kernel/hierarchical-fourier-tiles`、開始source `d76bd667`。
[前回](hierarchical-fourier-gpu-20261021.ja.md)の研究用live-I契約を維持する。
公開CSTLinear、旧Polar更新、profileの数式を変更する研究ではない。

## 約1.19秒の内訳

Colab L4、Torch2.11.0+cu130、Triton3.6.0、IEEE FP32（TF32なし）。
N=1024、A=52,428、B=32、spacing=.1、atomごと・stepごとのlive I。
M8 mixed/direct-tableを既存Linear runnerで再測定した完全stepは1,186.430ms、
M4 small/direct-tableは158.529ms。Denseは0.079872ms。
両Caseとも独立FP64全site・全atom gate（3動的状態＋Graph変更）がPASS。

別の未計装workerとは分けて、1 eager学習stepをprofilerで観測した。
下表はkernelのdevice時間であり、完全stepの代替・差引き計算には使わない。

| 処理 | M8 mixed/direct ms | M4 small/direct ms |
| --- | ---: | ---: |
| H/G生成の合計 | 931.384 | 52.802 |
| 係数・周期準備 | 80.468 | 72.294 |
| L2と中心/I微分 | 47.898 | 8.873 |
| phase係数pack | 6.800 | 7.882 |
| Y/dXの直接書き込み | 115.833 | 12.334 |
| atom VJPの最終合計 | 4.529 | 4.528 |

M8のH/Gはforward H 198.944ms、backward Hとその微分366.159ms、
backward Gとその微分366.281ms。H/Gが約79%を占めた。
M8 shared64/tableもH/G 973.749ms、aggregate11.258ms、moments/reconstructは小さい。
ソート・集約準備だけが1.19秒の主因という結果ではない。
ハードウェアDRAM/spill counterは未取得であり、物理的な帯域・cache原因は断定しない。

KinetoはCUDA user annotationの`_prepare`/`_contract...`等もdevice eventとして返した。
これらはkernelと重なるため、kernel集計から除外した。二重加算したstage合計を性能に使わない。
元の全event集計と、除外後の集計は両方保存する。

baseline job: `l4job-2a2c465b3ce34ef0b9ecf49ce38bd048`。
source archive SHA256 `ebf7731250eea09a6d74ea6f29c3d2494387231fed9c1590ba28ac58113954df`、
result archive SHA256 `fbb52712d625293e0ab25b05d62098c5eb6d4842c0d9b80a3708d138a3cb1f97`。

## 実装候補と数学的な範囲

- 4 atom×4 batchのblockでH/Gを縮約し、同じprofile評価を小batch内で共有する。
  atomの中心/Iは自由で、異なるatomに同一profileを仮定しない。
- `reuse=h`: forwardでH・中心微分・I微分をFP32で保存し、backwardのH生成を省く。
  A=52,428、B=32なら生payloadは20,132,352 byte。完全peakは実測する。
- `reuse=all`: 係数・norm・routing・phase係数・小支持cacheとHをforwardから保存する対照。
  再計算が減っても、全atom分の保持は大きい。容量目標を免除しない。
- `reuse=chunk`: 最初の1 chunkだけ準備情報とHを残し、残りを再計算する容量上限付き対照。
  supportを捨てたり古いstepの配置を無検証で使う方法ではない。
- `moment_policy=demand`: liveな中心/Iから完全内側タイルをGPU上でmarkし、不要なZ生成・C集約を省く。
  Graphの形を固定するため、Z/Cのaddress空間そのものは最大候補数のまま保持する。
- `moment_policy=hierarchical`: demandを子へ伝播し、Lが2倍、タイル幅も2倍である親子bankに限り、
  親の偶数周波数を二つの子の集約から合計する。奇数周波数は入力から新規計算する。
  現M8/64点には6 bankの親子関係、M4/32点には対応親子がない。

部分タイルと小支持は前回と同じFourier Fを直接評価する。追加窓、支持切り捨て、
幅や中心の丸めは導入しない。L2は実site二乗和のFP64計算で、product全体でfloorを一度だけ。
P/Xのversion検査を通してから保存値を使う。retained backwardでも保存値を破壊しない。
current I依存の保存は同じforward/backward内だけであり、次stepへ持ち越さない。
M4ではI≤10に対して最小64点の共有bankが存在しないため、共有候補は32点で評価する。

## 証拠の保存

raw source/results/receipts、CPU/build logsはignored
`output/hierarchical-fourier-optimization-20261021/`。
測定は既存`benchmarks.cuda.linear.run`に登録済みPlan/Caseを渡す。
診断driverはignored output内に置き、runtimeからbenchmarkへ依存させない。

## ブロック化・H保存・全準備保存の比較

`l4job-e71c4000188848f28f10cff93d09d65f`、source `0168935c`。
GPU pytest27件、M4 small/M8 mixed×5 Plan×4状態の全atom oracle40照合がPASS。
最大絶対誤差3.20e-5以下。Graph完全stepの7標本中央値:

| 条件 | M4 small ms | M8 mixed ms | M4 allocated B | M8 allocated B |
| --- | ---: | ---: | ---: | ---: |
| direct-table | 158.537 | 1186.236 | 19,929,600 | 31,597,056 |
| grouped / chunk4096 / 保存なし | 121.412 | 798.603 | 19,929,600 | 31,597,056 |
| grouped / chunk1024 / H保存 | 123.175 | 732.256 | 28,265,472 | 30,626,816 |
| grouped / chunk1024 / 全準備＋H保存 | 76.939 | 655.412 | 102,819,328 | 165,053,440 |
| shared64 / chunk1024 / 全準備＋H保存 | 77.619 | 564.956 | 102,819,328 | 168,165,888 |
| Dense | 0.080896 | 0.084992 | 51,382,784 | 51,382,784 |

H保存の行はchunkも変えているため、その差全てをH保存単独の効果としない。
M8のH保存行のreservedは58,720,256Bで、Dense111,149,056B以下。
全保存のreservedはM4 109,051,904B、M8 174,063,616B（共有178,257,920B）。
全保存はallocatedがDenseを超えるので、今回の容量条件では採用しない。
M4 smallではH保存が速くならず、保存なしのブロック化が容量・時間で有利だった。

Plan名`shared64-grouped`はこのsnapshotのrecipeで`reuse=all, atom_chunk=1024`を
指定している。名称だけから保存なしと読まない。後続では保存方針を明示した別IDを用意した。
M4のshared64は共有bankがなく、全保存直接評価の対照である。

source archive SHA256 `d15c180eca524fcf76cd5f520262c56910eb084d40c62089981f6061e07daff0`、
result archive SHA256 `ca66a8332ec504a1f2c80b286a78cc7452c943418f3786970664b5dabc94eeea`。

## 必要タイル判定と親子再利用

`l4job-d40429ac0591433880a6e0f3b68e1b07`、source `4ea6a3d7`。
GPU pytest40件、M4/M8 mixed×6 Plan×4状態の48独立oracle照合PASS。
最大絶対誤差3.06e-5以下。同一run内のGraph中央値:

| Plan | M4 mixed ms | M8 mixed ms |
| --- | ---: | ---: |
| direct-table | 725.060 | 1186.427 |
| grouped / 保存なし / chunk4096 | 442.636 | 798.478 |
| grouped / 最初の1 chunkのみ保存 | 434.238 | 784.245 |
| shared / 全保存 / 全tile | 361.721 | 564.630 |
| shared / 全保存 / demand | 362.653 | 565.884 |
| shared / 全保存 / hierarchical | 362.452 | 565.516 |
| Dense | 0.079872 | 0.080800 |

M4は最小32点、M8は64点。sharedの3行は全てchunk1024・全保存の対照であり、
省メモリ候補の値として扱わない。allocatedはM4約105.35MB、M8約167.94MBでDenseを超える。
最初の1 chunk保存のallocated/reservedはM4 27,531,776/37,748,736B、
M8 43,393,536/54,525,952B。約1.8%の差は1独立runなので採用根拠にはしない。

初期状態の論理的な必要node数（入力/出力は同数）:

| M / 分布 | 全候補 | demand | 親子依存を閉じたhierarchical |
| --- | ---: | ---: | ---: |
| 4 / small | 224 | 0 | 0 |
| 4 / mixed, large | 224 | 192 | 192 |
| 8 / small | 104 | 0 | 0 |
| 8 / mixed | 104 | 76 | 76 |
| 8 / large | 104 | 44 | 76 |

M8 largeでは親の偶数周波数を作るため、直接には不要だった子nodeも必要になる。
需要を閉じた後の費用で評価する。node数はDRAM読取byteやcache hit率の実測ではない。
mixedでは約14%/27%のZ候補を省けるが、判定・依存伝播の追加もあり全stepの有意な短縮はない。
M4にはこの候補集合で親子対応がなく、hierarchicalの特別な利点はない。

source archive SHA256 `6492eab28459dac00df1084c198f3cc3c0d7559745b7b92df31bb33552b2e2b5`、
result archive SHA256 `10f541b3b87e9e19762e4ec4748efd913bc8b210ab4a555e24ee46413c628942`。

## FP64のprofile値をchunk内で再利用

`l4job-91a16dd152824b998a6b03a251bc34cc`、source `80fbc653`。
`sample_cache=512`はL2で得たF・中心微分・I微分をFP64のまま保存する。
従来の16/32点用FP32 cacheを広い支持へそのまま拡張して精度を下げる方法ではない。
512点を超えた場合は同じFourierの直接評価へ戻り、支持を切らない。
cacheはchunk=256内で上書き再利用し、H/G・Y/dXが32点ずつ読む。
raw profile cacheの容量は`256*2*3*512*8=6,291,456B`。
全atom×全siteを保持しない。学習stepをまたぐlive値の保存も行わない。

GPU pytest44件、M4/M8 mixed×6 Plan×4状態の48候補照合がPASS。
独立FP64参照は8状態を計算し、同じX/P/dY・spec・period・oracle実装hashが
完全一致する場合だけ40回再利用した。参照tensorのSHA256・shape・dtypeも検査する。
参照cacheはbenchmark専用で、測定workerとは別process。学習stepの高速化には数えない。
許容値`atol=rtol=4e-4`を維持し、最大絶対誤差3.06e-5以下。
新sample候補の最大絶対誤差はM4 2.23e-5、M8 2.14e-5以下だった。

N1024/A52,428/B32、Graph完全step、7標本中央値。allocated/reservedはcapture/replay込み。

| Plan | M4 mixed ms | M8 mixed ms | M4 allocated / reserved B | M8 allocated / reserved B |
| --- | ---: | ---: | ---: | ---: |
| direct-table | 725.103 | 1186.356 | 19,929,600 / 35,651,584 | 31,597,056 / 56,623,104 |
| direct-samples | 244.790 | 402.656 | 18,061,824 / 31,457,280 | 18,457,088 / 33,554,432 |
| direct-samples-h | 232.236 | 382.526 | 38,095,872 / 56,623,104 | 38,491,136 / 58,720,256 |
| shared-all-h | 418.500 | 643.198 | 29,240,832 / 37,748,736 | 31,956,992 / 58,720,256 |
| shared-demand-h | 419.022 | 644.324 | 29,242,880 / 35,651,584 | 31,958,016 / 58,720,256 |
| shared-hierarchical-h | 419.114 | 644.648 | 29,242,880 / 35,651,584 | 31,958,016 / 58,720,256 |
| Dense | 0.081920 | 0.079872 | 51,382,784 / 111,149,056 | 51,382,784 / 111,149,056 |

ここで`shared-all-h`のallは全候補タイルを計算する意味であり、`reuse=all`ではない。
3つとも`reuse=h, atom_chunk=1024`。M4最小32点、M8最小64点。
全Planで52,428個の幅logitの更新を確認した。

省メモリのdirect-samplesは基準より約3倍速く、allocatedもM4約9%、M8約42%減る。
H保存を追加すると約5%短縮する一方、約20MB増える。両方のPareto候補を保持する。
需要判定・階層化には、この省メモリ条件でも完全stepの改善は見られなかった。

別の診断stepでのM8 mixed/direct-samples-hのkernel時間:

| 処理 | 改良前 direct-table ms | direct-samples-h ms |
| --- | ---: | ---: |
| H/G生成 | 931.384 | 108.464 |
| 係数・周期準備 | 80.468 | 94.045 |
| L2・profile評価とcache保存 | 47.898 | 116.833 |
| phase pack | 6.800 | 7.400 |
| Y/dXの直接書き込み | 115.833 | 49.953 |
| atom VJP最終合計 | 4.529 | 5.538 |

H/Gの負担を大きく減らしたが、準備とL2側の時間は増えた。chunk幅・保持範囲・
書き込みも変わっているため、116.8msの増加を特定のhardware要因だけに帰属させない。
現在はH/G、L2/profile cache構築、係数準備の三つに費用が分散している。
これらはprofilerの診断値であり、合計を完全stepの測定値に置き換えない。

Denseの2倍以内という速度目標は未達。新候補もDense比では数千倍であり、
公開dispatcherへの採用やDense相当速度は主張しない。
memoryは単層・独立processの比較であり、ライブラリ作業領域を共有するnetwork全体の
削減率やGPU process全体のdevice使用量を示すものではない。

source archive SHA256 `cb6119ab1ba4e6af1e5ffe512a7da13fac4eaea7b984a71f9f6f18554db60a46`、
result archive SHA256 `aa9941910c78dadf0cf349416526ba8bbef1c888e065dc24527992db97dd0063`。

## 独立jobでの再測定と判断

`l4job-31b2faf05a5f4375ad91895f51fb6ebf`で新しいL4 sessionを使って再測定した。
数値sourceは同じ`80fbc653`、初期P/X/targetのhashも前jobと一致。
全atomの24候補照合がPASS（8参照状態を独立に生成、16回は完全一致する参照を再利用）。
最大絶対誤差3.20e-5以下。各jobで7個の時間標本を測り、独立run数は2である。

| 条件 | 1回目 ms | 独立2回目 ms | allocated peak B |
| --- | ---: | ---: | ---: |
| M4 mixed / 基準 | 725.103 | 725.069 | 19,929,600 |
| M4 mixed / 値再利用 | 244.790 | 245.123 | 18,061,824 |
| M4 mixed / 値再利用＋H保存 | 232.236 | 232.539 | 38,095,872 |
| M8 mixed / 基準 | 1186.356 | 1186.291 | 31,597,056 |
| M8 mixed / 値再利用 | 402.656 | 403.081 | 18,457,088 |
| M8 mixed / 値再利用＋H保存 | 382.526 | 382.515 | 38,491,136 |

reserved peakも両runで一致。H保存による追加の約5%短縮が再現した。
同一jobのDenseはM4 0.080896ms、M8 0.080800ms、allocated51,382,784B、
reserved111,149,056B。新候補はallocated/reservedともDense以下だが速度目標は未達。
今回のmixed/N1024条件では、容量優先なら`direct-samples`、Dense以下の容量で
時間を優先するなら`direct-samples-h`を研究用の候補とする。他の分布・shapeへの自動採用はしない。
M4 smallは先の実測で保存なしgroupedが有利であり、mixedの結論をそのまま流用しない。

source archive SHA256 `0b85a5287766074c13f58def5b404c9fee29b0ddad0cb60a5340b5e426718caa`、
result archive SHA256 `e4d46f3d39cee6e3c4dd35bf0eddbb9e39f7670b37709b48948d6aca11dd23fd`。

## 最終確認と再現

- CPU suite: 2,237 passed / 5,695 skipped / 18 warnings。GPU skipを実GPU検証とは数えない。
- 最終数値sourceのL4 GPU専用test: 44 passed。出力・dX・全P、retained backward、
  空支持・非空floor・片軸floor付近、dynamic Graphを確認した。
- wheel/sdist build、Ruff、diff check、最終Case/Plan宣言検査がPASS。
- 5つの完了jobのsource/result archive SHA256をreceiptと照合し、385個の回収ファイルの
  manifestを保存した。最終2jobと現在のruntime/benchmark/case/test 1,476ファイルがbyte一致。
- queuedのまま設定を置き換えた3job（b365d8ad、1504519a、722b2baa）にはGPU結果がなく、
  実行回数や性能の証拠に数えない。source snapshotは保全した。
- Colabのowned slotは停止確認済み。Kyutechは使っていない。

`result-summary.json`と`artifact-sha256.json`、全receiptと個別JSONは
`output/hierarchical-fourier-optimization-20261021/`に保持する。
計測driverも同directoryの`baseline-profile.py`、`grouped-gate.py`、`hierarchy-gate.py`、
`final-campaign.py`、`repeat-best.py`。既存runnerを呼ぶdriverであり別測定runnerではない。
再測定はこのcheckoutからColab共有キューへ提出する。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source "$PWD" \
  --script "$PWD/output/hierarchical-fourier-optimization-20261021/repeat-best.py" \
  --label hierarchical-fourier-reproduce --timeout 1200
```

driverは宣言されたCaseとPlanを読み、既存`benchmarks.cuda.linear.run`に
完全step・独立oracle・Dense比較を委ねる。raw archiveはignored出力に保全し、
公開dispatcherやDBの既定を更新する操作とは分離する。
