# RegularGrid: 支持幅bucketと可変タイル数

2026-10-11。比較完了。幅16は既存組み合わせ版より1.37–6.40%速く、allocatedを減らしたが、
現行より27.31–35.17%遅い。幅32は集約がさらに悪化。両候補は採用せず現行を維持する。

## 実装と仮説

各軸の支持タイル数Kを1/2/3–4/5–8…のdyadic区分へ分類し、
区分と開始タイルをkeyとしてatomを一度だけsortする。
保存するのは各軸A個のint32 IDと、小さい区分×タイルのint32 bounds。
支持所属の全展開、入力×出力タイル組合せ、全tile×atomのH/G partialは作らない。
次forwardで現在の幅と中心から準備し、同じforwardのbackwardへindexを保存・再利用する。

各タイルownerは区分上限K_upperに基づく開始タイル範囲を読む。
周期端では最大2区間へ分け、全周期区分は全タイルを一度だけ読む。
実際のKで所属を絞る。各側1次元では全タイル合計のlive候補参照数は
準備済み支持区間の所属数の2倍未満。これは候補数の上界であり速度比の保証ではない。
末尾の短いタイル、全周期・seam・空支持でも同じsite/tileを二重に数えない。

H/G producerは幅区分と開始タイルが近いatom束を担当する。
16-site共有panelを読み、支持区間全体をregister内で縮約して唯一writerがH/Gを保存する。
区分境界を跨ぐ束は区分ごとに処理し、広いatomで狭いatomのpanel loopを長くしない。
入力・出力側は独立してsortし、H/Gは元atom順の小batch slabへ書く。
全量packedコピーやH/Gの並び替えpassは追加しない。

H/G寿命は既存 `tile-combined-bm16` に揃える。
H value → G/dGをHと即時消費 → dHをGと即時消費 → dX集約。
dH/dG slabはなくH/Gの2 slabを保持する。入力consume2ではH valueのdotを生成しない。
H/Gを全支持で完成してからパラメータ勾配を作り、交差項を維持する。
全operator-sitesの正規化・floor・singleton・source VJPは既存の準備とVJPを使う。
H/Gとphysical gradientsはatom唯一writer、Y/dXはtile×8partsと別reductionでatomicを使わない。
広支持を2タイルの制限によるall-atom fallbackへ切り替えず、そのまま処理する。

研究plan `tile-bucket16-bm16` / `tile-bucket32-bm16` は同じAlgorithmのtile幅だけを変える。
今回の対応範囲はD2/full-period RegularGrid/FlatTorus/Triweight、各側1次元である。
一般Chart/APIや公開defaultは変えない。

## 固定比較

RTX 6000 Ada、N2048/B32/A209715、rho_initial3/8、seed41、MSE、
IEEE FP32/TF32 off、実AdamW/Polar更新24回、primary21 samples。
現行、固定tile、支持幅単独、組み合わせ、bucket16/32とmatrix/factor/onchip/denseを
fresh workersで比較する。独立2cohort/rho、2回目はplan順を反転。
準備・sort・IDのint32変換・生成・集約・backward・optimizerの費用を全て含める。
capture/replay peak allocated/reservedを同じdense対照と比較する。
10 workers × 4 cohorts =40 workers/840 samples、72 full FP64 gates。
初期/post24全shape・全atom Y/dX/全4 dPを既存4e-4基準で検証し、失敗した計測を再選択しない。
GPU qualification251件（bucket/schedule/tile-dot）を全PASS後にprimaryへ進む。
支持が3個以上のタイルを跨ぐ場合、非整除N、周期端、空・全周期、保存状態と可変幅更新を検証する。
固定post24の10診断は別Graphで行い、primaryへ足し合わせない。

## 正しさと実行条件

source `ea98ba4b4100a98d675c7ec63fb7c766a17634f9`。NVIDIA RTX 6000 Ada Generation 48GB、
Torch2.9.1+cu128/CUDA12.8/Triton3.5.1。上述の固定プロトコルを変更せず、
GPU qualification251 PASS/0 failures/errors/skips、76.25秒。
40 fresh workers/840 primary samples、初期/post24の72 full FP64 gates PASS。
10固定snapshot診断はprimary/未計装Y/dX/dPとbitwise一致し、Parameter不変。
現行と各候補の独立repeatのpost24 Parameter/Y/dX/all_dPもbitwise一致。
GPU試験には支持3個以上のtile、非整除N=33/65/257、任意interval・全周期・seamの
inverse coverage/重複回避、floor/singleton/empty、stride/tail/gradient mask、
retained forward、live幅・Graph/public optimizer更新を含む。大形性能測定はrho_initial3/8のみ。
CPU1799 PASS/4299 GPU・DB skips/18既存warnings、37.17秒。catalog49 plans、Ruff/diff PASS。
[計測sourceのCPU/build CI](https://github.com/takumiecd/torchcst/actions/runs/38134525108) PASS。

準備済みintervalにはprecision guardによる保守的なsite範囲が含まれうる。
候補数の2倍未満という上界はそのintervalのtile所属に対するもの。
真の非零profile数・実traffic・時間比を意味しない。正規化/VJP契約は近似しない。

## 完全stepとメモリ

全て今回の同cohort対照。過去campaignとpoolしない。未計装完全step中央値（ms）。

| cohort | 現行 | 固定tile | 支持幅のみ | 組み合わせ | bucket16 | bucket32 | dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rho3-run1 | 1.423667 | 3.212668 | 1.997266 | 2.052834 | 1.924361 | 2.636353 | 0.329237 |
| rho8-run1 | 1.852922 | 3.916265 | 2.380374 | 2.456931 | 2.358921 | 2.966549 | 0.329374 |
| rho3-run2 | 1.439619 | 3.265531 | 2.002818 | 2.072085 | 1.939519 | 2.650731 | 0.328704 |
| rho8-run2 | 1.875236 | 3.939032 | 2.360850 | 2.428762 | 2.395432 | 2.929585 | 0.329364 |

| 全cohortで同じcapture/replay peak (MiB) | allocated | reserved |
| --- | ---: | ---: |
| 現行 | 57.430664 | 104 |
| 固定tile / 支持幅のみ | 93.717285 | 150 |
| 組み合わせ | 68.116699 | 122 |
| tile-bucket16-bm16 | 62.523438 | 124 |
| tile-bucket32-bm16 | 62.518555 | 124 |
| dense | 81.752441 | 106 |

bucket16は組み合わせ対照より1.37–6.40%短縮したが、現行より27.31–35.17%遅い。
支持幅単独よりrho3で3.16–3.65%速い。rho8ではrun1は0.90%速くrun2は1.46%遅く、
一貫した速度の優位はない。bucket32は組み合わせ対照より20.62–28.43%遅い。
bucket16のallocatedは組み合わせから5.593262MiB減ったが、reservedは2MiB増えた。
両候補のreserved124MiBはdense106MiBより18MiB大きい。全cohortで最速の両peak適合CSTは現行。
新planは研究用に留め、公開defaultは変更しない。GPU process全体のpeakは未測定。

| control | rho3-run1 | rho3-run2 | rho8-run1 | rho8-run2 | allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| matrix-torch | 0.707932 | 0.715478 | 1.362912 | 1.370286 | 96.163086 / 142 |
| factor | 1.963398 | 1.981898 | 4.327485 | 4.338453 | 83.114258 / 130 |
| onchip-h | 1.803570 | 1.818819 | 3.899885 | 3.803552 | 42.866211 / 114 |

## 固定post24の生成・集約診断

5 replayの別Graph。各stageのchunkを合計した中央値。primaryへ足したり差し引いたりしない。
準備stageにはsource/amp cloneを含まないが、primaryには含む。
index stageにはkey生成、sort、int64→int32 ID変換とboundsを含む。

| stage (ms) | rho3 組合せ | rho3 bucket16 | rho3 bucket32 | rho8 組合せ | rho8 bucket16 | rho8 bucket32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prepare_support_norm | 0.099936 | 0.099296 | 0.099136 | 0.116416 | 0.116192 | 0.116000 |
| support_membership_routing | 0.110464 | 0.098144 | 0.095904 | 0.110784 | 0.097888 | 0.096096 |
| forward_H | 0.233216 | 0.104672 | 0.144032 | 0.290336 | 0.153696 | 0.155456 |
| forward_Y_local_products | 0.221312 | 0.466816 | 0.756480 | 0.273504 | 0.545664 | 0.872736 |
| backward_H_value_recompute | 0.235008 | 0.105152 | 0.144256 | 0.290976 | 0.154560 | 0.155936 |
| backward_G_dG | 0.434656 | 0.201664 | 0.269792 | 0.541056 | 0.287488 | 0.289056 |
| backward_H_dH | 0.273312 | 0.150560 | 0.197760 | 0.336512 | 0.207456 | 0.211584 |
| backward_dX_local_products | 0.221952 | 0.464384 | 0.763776 | 0.274176 | 0.545344 | 0.873952 |
| source_vjp | 0.010240 | 0.010176 | 0.010176 | 0.010240 | 0.010240 | 0.010208 |

生成は改善した。bucket16のH valueは約47–55%、G/dGは約47–54%短縮した。
一方Y/dX ownerは約2.0–2.1倍、bucket32では約3.2–3.4倍へ悪化した。
最後のowner partial reductionは各約0.0064msのまま。atomic競合による悪化ではない。

全4 bucket診断で候補参照数は準備済みintervalの所属数と一致しており、幅区分による
候補過剰が今回の主因とは言えない。各tile×part kernelは全幅区分と最大2範囲を処理し、
atom束ごとにfirst/countをpacked boundsから再計算して判定する。
この汎用処理とrangeの分割/padding、register圧力を次の切り分け対象とする。
個々のALU・メモリ・分岐・cache/stall寄与は未測定で、合算して原因を断定しない。

| compiled kernel | registers | spills | shared bytes |
| --- | ---: | ---: | ---: |
| bucket H value | 96 | 0 | 4096 |
| bucket G/dG消費 | 148 | 0 | 4096 |
| bucket dH消費 | 142 | 0 | 4096 |
| bucket16 Y/dX owner | 160 | 0 | 6144 |
| bucket32 Y/dX owner | 168 | 4 | 8192 |
| 組み合わせ Y/dX owner | 153 | 0 | 9728 |

## index保持量と次の方針

各軸の保存済みIDはA=209715個/int32、838860bytes。2軸で約1.600MiB。
以前の2A個/int64のorderは2軸で約6.400MiB。
bucket16のboundsは各4100bytes、bucket32は各1796bytes。
保存index減少とallocator peakは同一ではなく、sortのint64一時ID・変換とGraph poolも測定に含む。
reserved増加のallocator内訳は未切り分け。

今回のrho3/8 post24のtile数histogramは全て1/2個。3個以上はGPU qualificationで検証したが、
大形の3個以上の支持について速度の勝ちを主張しない。
H/G生成とcompact indexは次の研究材料として残し、次の改善対象はY/dX ownerとする。
まず非空候補範囲の処理を準備へ移せるか、反復する所属判定を安いmetadataへできるかを
別々に測る。全tile所属の展開や全量コピーを直ちに追加しない。今回の比較にはその未実装案を混ぜない。

## 証拠と復旧

- source archive SHA256 `2c13ae9af67d86aa47706ca50b6c2fe22fc7826e285f08fac8289eeabd3a863e`。
- compact archive SHA256 `c29866941e10ccf6c44ed6fd72f3028a12b0225123ebb0c4157a2629890bc2c8`。
- read-only audit SHA256 `93722c94953ef40f909e8feadd26ad34440dae991e56a664c027bc6abef5b95f`。
- compact167 entriesをローカルでhash照合。1669 source filesはpinned Git blobsと照合済み。
- ローカルignored evidence `output/regular-grid-h/tile-bucket-*`。
- remote `/home/jovyan/runs/cst-tile-bucket-ada-ea98ba4b-20261011`。
  全72個のraw oracle .ptをremoteに保存。compactは全JSON/log/scripts/manifestを保持。
- controller PID232715、followup PID234697。command/log/output/PIDは各pid.jsonと運用メモへ保存。
- source/runtime/input・cohort順・全更新/sample・全FP64 gateとsnapshot hash・両peak・
  診断bitwise不変をauditで検証。retry/tolerance変更なし。
- ユーザー起動済みHubをreadyのまま残し、終了時0%/39MiB、GPU計算processなしを確認。
