# 支持幅の生成粒度とdH/dG寿命を別々に比較

2026-10-11。比較完了。支持幅版は固定tile対照より速くなったが、現行より遅く両peakも悪化。
dH/dG寿命版は保存量を減らしたが、完全stepを悪化させた。両候補は採用せず、現行
BM16/G16/H32を維持する。公開defaultの変更はない。

対照は[支持タイルBM16](regular-grid-tile-dot.ja.md)と現行BM16/G16/H32。
2候補を組み合わせず、同じ32-site所属routing、8 owner partial、IEEE FP32/TF32 off、
全operator-sitesの正規化・床・singleton・source VJPを維持する。
全支持の縮約後にparameter勾配を作り、タイル間の交差項を落とさない。

`tile-window16-bm16`は生成粒度だけを変える。同じ支持開始タイルのatom束について
liveな支持区間の和集合を求め、16-site panelでH/Gを縮約する。空支持は0 writerを持つ。
H/dH/G/dGの全atom×小batch slab、生成回数と後段のowner集約は対照のまま。
rhoによる固定分岐ではなく現在のpacked supportsを使用する。広支持は既存の正確なfallback。

`tile-derivative-bm16`は32-site panelを保ち、dH/dGをproducer内で消費する。
1. Hのvalueだけを全支持で縮約し、小batchのH slabへ書く。
2. G/dGを全支持で縮約し、Hと掛けてdAと出力中心勾配を作る。G valueだけをslabへ書く。
3. H/dHを再計算し、Gと掛けて入力中心勾配を作る。H/dHはここではglobalへ書かない。
4. Gを既存dX ownerで集約し、physical→source VJPを行う。

数式は `dA=sum_b H G`、`dc_in=A sum_b dH G`、`dc_out=A sum_b H dG`。
batch chunkをまたぐphysical勾配は元と同じ順序で加算する。H/Gの2 slabは残り、dH/dG slabは
作らない。H valueを先行させ、入力profile/微分を再び生成する別passと融合勾配の費用も完全stepに含める。
forwardとneed_x-onlyは元の固定panelを使う。retained forward snapshot/routingを維持し、
次forwardで更新後の支持に基づき再構築する。gradientsはatomicなしで唯一writerが書く。

比較はRTX 6000 Ada N2048/B32/A209715/rho_initial3/8/seed41、同じMSEとAdamW/Polar。
現行、固定tile BM16、各独立候補、matrix/factor/onchip/dense対照をfresh workerで測る。
rhoごとに独立2cohort（2回目reverse）、24実更新、21 primary samples。
GPU資格検証を先行し、初期/post24全shape・全atomのFP64 Y/dX/全4 dP gate（4e-4）を維持する。
準備・所属sort・集約込みの未計装完全stepとcapture/replay allocated/reserved peakを判断に使う。
別Graph診断は加算せず、実cache hit/DRAM/stall/達成occupancyを未測定のまま断定しない。


## 実測条件・正しさ

計測source `aff01d671e76687380a90bdb1e4cddfcaa9aab81`。
NVIDIA RTX 6000 Ada Generation 48GB、Torch2.9.1+cu128/CUDA12.8/Triton3.5.1。
同じRegularGrid/FlatTorus/D2/full-period、N2048/B32/A209715、rho_initial3/8、seed41、
MSE、IEEE FP32/TF32 off、実際のAdamWとPolar/periodic更新24回。
各rhoで独立2cohort、2回目はplan順を逆転。8 fresh workers/cohort、21 primary samples/worker。
前回campaignの値を混ぜず、今回の同cohort対照を使う。

- GPU資格検証194 PASS、0 failures/errors/skips、50.62秒。
- 32 workers/672 primary samples。dense以外の7 CST workersについて初期/post24、
  計56の全shape・全atom FP64 Y/dX/全4 dP gates PASS。max_absとrelative_l2の4e-4基準を維持。
- 6つの固定post24診断PASS。primaryと未計装版のY/dX/dPがbitwise一致し、Parameterは不変。
- floor/singleton/empty/full-axis、周期seam、stride/tail、grad mask、retained snapshot、
  live幅更新とCUDA Graph/public optimizerを検証。backward中の実際のempty_like要求を数え、
  windowのH/dH/dG 3 slabに対し寿命版はH 1 slabのみ（Gは双方new_empty）を確認した。
  これは割当要求の契約検査でありreserved peakの測定とは異なる。
- CPU1789 PASS/4190 GPU・DB skips/18既存warnings、34.83秒。catalog46 plans PASS、
  SHA256 `1d5b9e29bb0e32f1646b632c7e701680aeec62c9138b513bd5e881f5f1ebffbf`。
  Ruff/diff PASS、[計測sourceのCPU/build CI](https://github.com/takumiecd/torchcst/actions/runs/38130447196) PASS。

## 完全stepとcapture/replay peak

準備・source/amp snapshot・2軸sort/bounds・H/G生成・Y/dX集約・physical/source勾配・
optimizer更新を含む未計装step中央値（ms）。

| cohort | 現行BM16/G16/H32 | 固定tile BM16 | 支持幅16-panel | dH/dG寿命 | dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| rho3-run1 | 1.421618 | 3.204491 | 1.992323 | 3.551129 | 0.329472 |
| rho3-run2 | 1.434180 | 3.202397 | 1.994941 | 3.555183 | 0.329213 |
| rho8-run1 | 1.819815 | 3.902928 | 2.344919 | 4.305288 | 0.329399 |
| rho8-run2 | 1.831354 | 3.911128 | 2.363092 | 4.311712 | 0.329452 |

| 全cohortで同じpeak (MiB) | allocated | reserved |
| --- | ---: | ---: |
| 現行BM16/G16/H32 | 57.430664 | 104 |
| 固定tile BM16 | 93.717285 | 150 |
| 支持幅16-panel | 93.717285 | 150 |
| dH/dG寿命 | 68.116699 | 122 |
| dense | 81.752441 | 106 |

支持幅版は固定tile対照より37.70–39.92%短縮したが、現行よりrho3で39.10–40.14%、
rho8で28.85–29.04%遅い。生成のみの変更なので4 slabのpayloadとpeakは固定tile対照と同じ。
寿命版は固定tileより10.24–11.02%遅く、allocatedを25.600586MiB（27.32%）、
reservedを28MiB（18.67%）削減した。allocatedはdense未満になったが、reservedは122MiBで
106MiBのdenseを超え、現行の両peakも超える。GPU process全体のpeakは未測定。

同じcohortの明示的対照も保存した（ms、最後の列は全cohort同じpeak）：

| route | rho3 run1 | rho3 run2 | rho8 run1 | rho8 run2 | allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| matrix-torch | 0.707749 | 0.712758 | 1.361260 | 1.370990 | 96.163086 / 142 |
| factor | 1.953430 | 1.972663 | 4.292641 | 4.348965 | 83.114258 / 130 |
| onchip-h | 1.794350 | 1.810636 | 3.959812 | 3.889929 | 42.866211 / 114 |

## 別Graphの細部診断

post24の固定snapshot、5回の別Graph診断。各replayで同stageのchunkを合計し中央値を取る。
primaryと足し合わせたり差し引いたりしない。producer eventは狭支持とguarded fallbackの
両launchを含む。profile/norm準備にはsource/amp cloneを含まず、primaryには含む。
寿命版はG/dGとdHのstageにphysical勾配への縮約を融合しているため、独立physical stageはない。

| stage (ms) | rho3 固定 | rho3 支持幅 | rho3 寿命 | rho8 固定 | rho8 支持幅 | rho8 寿命 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| support/norm準備 | 0.100416 | 0.101664 | 0.099648 | 0.118144 | 0.118208 | 0.116096 |
| 所属sort/bounds | 0.110752 | 0.110368 | 0.110336 | 0.110688 | 0.110496 | 0.110432 |
| forward H | 0.576832 | 0.233504 | 0.574016 | 0.716832 | 0.291200 | 0.715936 |
| forward Y局所行列積 | 0.223392 | 0.221184 | 0.221888 | 0.274688 | 0.273856 | 0.273920 |
| forward partial reduction | 0.006496 | 0.006464 | 0.006464 | 0.006464 | 0.006432 | 0.006432 |
| backward先行H value | — | — | 0.574528 | — | — | 0.715872 |
| backward G/dG (+寿命版dA/dc_out) | 0.917760 | 0.425152 | 0.926112 | 1.141344 | 0.530912 | 1.151040 |
| backward H/dH (+寿命版dc_in) | 0.919520 | 0.423520 | 0.743360 | 1.140448 | 0.528992 | 0.927712 |
| backward dX局所行列積 | 0.222944 | 0.221696 | 0.222592 | 0.274752 | 0.273056 | 0.274592 |
| backward partial reduction | 0.006432 | 0.006592 | 0.006400 | 0.006336 | 0.006336 | 0.006496 |
| 独立physical勾配 | 0.028128 | 0.028128 | — | 0.028128 | 0.028064 | — |
| source VJP | 0.010272 | 0.010240 | 0.010176 | 0.010208 | 0.010240 | 0.010208 |

支持幅版はforward Hが約59%、backwardのG/dG・H/dHが約54%短縮した。
Y/dX局所積、sortとpartial reductionは対照と近い時間。4 slabの論理payloadは51.2MiBのまま。
寿命版は2 slabで25.6MiBだが、先行H valueの独立passが0.575/0.716ms必要。
続く入力微分とdc_in生成は0.743/0.928msへ減ったが、別passを加えた完全stepは悪化した。
この別Graph値からprimaryの差分を計算しない。

狭支持producerのcompiler資源は固定tile=168 registers/0 spills、forward shared9472B、
微分shared14080B。支持幅版はforward80 registers、微分142、双方0 spills/shared4096B。
寿命版のG/dG/dA/dc_outは168/0/shared14080B、dH/dc_inは168/0/shared9984B。
ownerはすべて153/0/shared9728B。広支持fallbackは別kernelでcompiled spillsを持つが、
6つの診断のwide flagはすべてfalseであり、この比較では広支持の計算はinactive。

post24平均の単一atom支持長はrho3約5.987、rho8約15.898だが、32-entry束の支持和集合は
rho3入力35.526/出力35.521sites、rho8入力44.913/出力44.896sitesまで広がる。
16-site panel数は大半が3（48sites分）で、単一atomの支持長だけの仕事量にはならない。
論理coefficients評価数の比較（masked/empty-only境界束を除く）は、固定64sites分に対し
支持幅版は約25.0–26.2%少ない。粒度変更は演算域とTriton layout/compiler資源も変えるため、
速度改善の全量をpadding削減だけへ帰属させない。実cache hit/DRAM traffic/stall/達成occupancyは未測定。

## 採否と復旧

2候補を混ぜた版は試していない。支持幅版は生成粒度の有効な改善として研究対照に残すが、
現行を上回らずメモリも増えるため採用しない。dH/dG寿命版はメモリ削減と複数passの費用を
示す対照として残す。保存をなくすだけで速くなるとは判断しない。

branch `kernel/regular-grid-h-lifecycle`、draft [PR #100](https://github.com/takumiecd/torchcst/pull/100)。
final documentation commit以降も計測sourceとの差分は文書のみとする。

| evidence | SHA256 |
| --- | --- |
| frozen source tar | `b31e0398cfd8572e8e1e1842b80dc7e38b57ef48e2ec37da0f1771d196cd36eb` |
| compact results tar | `ff2e423b1e6ded268086f842996c868b5424a2801cbd6ea9b79b60c20541b508` |
| audit.json | `d71342ee190347cab9421a0248ac4cece6ffc3b2f547335014664ae5b4a42108` |
| compiler IR supplement tar | `3fb70ecf6fa9698b121ddecb0fca5e2bf72893aaa1baa800f1b131feef13d6e6` |

remote `/home/jovyan/runs/cst-tile-schedule-ada-aff01d67-20261011`、controller PID218514、
固定診断postprocess PID225335、read-only audit PID225787。全primary・診断はPASS。
source tarを展開後1664 tracked filesをmanifestと照合。audit PASS_COMPLETE後、compact archiveの
131 entriesをdownloadしてhash確認し、1664 filesをpinned Git blobsとも照合した。
56 raw snapshotsはhash/shape/全勾配付きでremote保管し、bulk tensor転送はしていない。
localは研究worktreeのignored `output/regular-grid-h/tile-schedule-results/`、対応tar、
source-manifest、controller/validation/diagnosis/audit/operating notes。
追加のcompiler IR確認はrho3寿命版の保存snapshotを1回評価しただけで、timed replayやoptimizer
更新を行わず、primary Y/dX/dPとbitwise一致・Parameter不変を確認した。script/JSON/8 TTIR filesの
10 entriesを別supplementへ保存した。前処理IRのdot数をGPU実行時間やtrafficと同一視しない。
ユーザーが開いたHubはreadyのまま残し、所有したGPU処理だけを終了した。

最初のread-only auditはFP64 gateの件数を64と誤って期待した。8 workersの1つはdenseであり、
7 CST ×初期/post24×4cohort =56が正しい。全raw gateをそのまま保持・検査し、expected countを
宣言済みORDERから計算するよう修正した。失敗audit script/logもcompact archiveに保存した。
CUDA source、case、数値基準、更新回数、primary timingを変更・再実行していない。

再現はpinned source/runtimeを復元し、新しい出力先で保存済み194件qualification driverを
先行した上で既存runnerを使う：

```bash
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho3.json \
  --plans matrix-torch factor onchip-h input-site-reciprocal-bm16-g16-h32 \
    tile-dot32-bm16 tile-window16-bm16 tile-derivative-bm16 dense \
  --isolated-oracle --phases --output output/tile-schedule-reproduction/rho3/comparison.json
```

rho8 caseでも実施し、各rhoの別fresh cohortを `--reverse` で実行する。
controller・固定snapshot診断・auditの実command/logは上記evidenceへ保存した。
