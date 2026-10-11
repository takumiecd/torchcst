# タイル16のY/dX集約: 準備へ移す処理の分離

2026-10-11。比較完了。空の幅区分を飛ばす効果が最大だった。3つを組み合わせた版は
元bucket16より25.96–29.03%、現行より3.32–6.49%速い。reserved124MiBが
dense106MiBを超えるため採用は保留し、公開defaultと両peak適合の研究baselineは維持する。

## 比較の目的と固定条件

`tile-bucket16-bm16` のownerで、幅区分の走査、周期境界を含む候補範囲の
構築・8分割、atomの所属情報の再構築を別々に準備へ移す。
H/Gのproducer、16行のbatch slab、32 atomの局所積、ownerの8分割と最終reduce、
正規化・floor・source VJPは維持する。新しい研究Algorithm
`research_cuda_regular_grid_tile_owner_prep_h` は公開dispatcherへ接続しない。

| Plan | 準備側へ移す処理 | 保存する追加情報 |
| --- | --- | --- |
| `tile-owner-bands-bm16` | 非空の幅区分の抽出 | 各軸 `LEVELS+1` 個のint32: 個数と昇順の区分 |
| `tile-owner-ranges-bm16` | 候補区間の構築とowner partへの分割 | 各軸 `[tiles,8,LEVELS,2,2]` int32: 開始・終了位置 |
| `tile-owner-membership-bm16` | 周期先頭タイルと所属タイル数の構築 | 各軸 `[2,atoms]` int32 |
| `tile-owner-all-bm16` | 上記3つ | 上記すべて |

幅区分は実際の支持幅から毎forwardで作る。GPU→CPUのcount同期を挟まない。
範囲は既存のsorted order/boundsに対する位置であり、tile×atomのIDを展開しない。
所属情報を保存してもownerでの `distance < count` 判定は残す。任意の幅でこの判定を
省くと、幅区分の上限が実際のタイル数を上回る場合に候補を誤って採用するため。

準備した情報はforwardのpacked snapshotと共にbackwardまで保持し、次のforwardで
再構築する。古いParameter・幅のindexを再利用しない。H/Gを保存する寿命は変えず、
dH/dGは局所積の直後に物理勾配へ消費する。H/G全支持の縮約後に勾配を積算するので、
タイル間の交差項も維持する。

N=2048、A=209715では、両軸の追加tensorの合計はbands 72 B、ranges 262144 B、
membership 3355440 B、all 3617656 B。これはtensorの論理量であり、capture/replayを
含むallocated/reservedピークの実測とは区別する。sortのtemporaryの上界は未知のため
`workspace_bound=None`を維持する。

## 検証・計測

CPUの宣言・catalog往復を確認する。GPUでは任意区間・周期wrap・非divisible形状・
全周期支持のmetadataを独立に列挙し、Y/dX/全4成分のatom勾配を独立FP64 oracleと
比較する。元のbucket16とのbitwise一致、floor、retained snapshots、20回のGraph
更新、勾配mask、dH/dGの保存量も検査する。

RTX 6000 Ada、N=2048/B=32/A=209715、rho_initial=3/8、IEEE FP32・TF32無効、
seed=41、AdamW/Polarの24回更新、21サンプル。現行の候補削減版、bucket16、4つの
準備版、denseを同じcohortのfresh workersで比較し、各rhoで独立2cohort・2回目は
逆順とする。各CST workerは初期とpost24の完全形状FP64 gateを測定前に通す。
失敗した場合は性能比較に使わず、同じ条件をretryせず証拠を残す。

固定post24の別Graphでは、準備全体とbands/ranges/membershipの内訳、Y/dX owner、
最終reduceをCUDA eventsで計測する。内訳は準備全体と重複し、診断時間を完全stepへ
足さない。単独版との差は実装変更の効果であり、特定命令の排他的な実行時間とは
呼ばない。完全stepのallocated/reservedピークと合わせて判断する。

## 性能測定前の共通精度修正

最初のsource `d62a7fd6060114319c0b2da23b1c3991e00bb7c9` の405回帰で、
sigma0.7、shape(257,65)、513 atom/17 batchの全dP誤差が0.000472860604026で
4e-4を超えた。性能測定には進まず、404 PASS/1 FAILとsource/logを保存した。
固定したCUDA RNG streamで入力を再現し、元bucket16と4候補のY/dX/all_dPが
bitwise一致することを確認した。失敗は所属range処理ではなく共通のprofile計算にある。

複数周期ずれた代表座標を一周期内へ戻してからnorm/profileを計算する診断では、
最大all_dP誤差が0.000012331545051へ下がった。source/optimizer/VJPに使う元座標は
変えない。bucketとowner準備版は、既存のprepare kernel内で必要なatom groupだけ
FP64周期reductionを行う。中心のpacked保存はFP32、norm/floor/gradの定義は維持し、
全量source copyや新しいglobal scratchは増やさない。他のPeriodic経路の既定は維持する。

失敗入力を固定した5経路の回帰を追加し、未変更の4e-4で再検証する。最初のcampaignは
性能結果なしのFAILとして残し、修正後は別source・別remote directoryへfreezeする。
完全stepは修正後のbucket16を対照とし、過去のbucket campaignと時間をpoolしない。

source `59d058fd` はlibdevice.fmodの定数型がFP32だったためFP64とのdispatchで
compile failureになった。229 PASS/181 compile FAIL、性能測定なしで保存し、
周期定数も明示的にFP64へ揃えて別sourceで再検証する。

## 完了した検証と比較条件

計算source `c0c656f266da71ed89389acaa87820ceec211c3a`、GPU RTX 6000 Ada Generation 48GB、
Torch2.9.1+cu128/CUDA12.8/Triton3.5.1。CPU1809 PASS/4448 GPU・DB skips/
18既存warnings、35.21秒。catalog53 plans、Ruff/diff、
[計測sourceのCPU/build CI](https://github.com/takumiecd/torchcst/actions/runs/38137524463) PASS。
固定失敗入力5件のpreflight PASS。GPU回帰410 PASS/0 errors/failures/skips、120.60秒。

N2048/B32/A209715、rho_initial3/8、seed41、MSE、IEEE FP32/TF32無効、AdamW/Polar。
各rhoで独立2cohort、2回目は逆順。各cohortのfresh workersはmatrix-torch、factor、
onchip-h、現行BM16/G16/H32、bucket16、4準備版、denseの10個。
40 workers、各24実更新/21サンプル、計840サンプル。
初期/post24の72 full-shape/all-atom FP64 Y/dX/全4dP gates PASS、4e-4を維持。
実験用の7-worker指定は必須対照が不足してCLIで拒否され、worker/gate/timingは0だった。
必須対照を補った10-worker protocolを計測前に固定し、新しいdirectoryへ実行した。
同じfrozen sourceの410件の資格検証を再利用し、旧・新1673 tracked filesの一致と
資格検証logs/XML/completed.jsonのhashを確認してから計測した。性能retryはない。

10固定post24診断はprimary/未計装Y/dX/all_dPとbitwise一致、Parameter不変。
独立repeatのpost24 Parameter/Y/dX/all_dPもbitwise一致。
さらに全4cohortで4準備版とbucket16のpost24 Parameter/Y/dX/all_dPがbitwise一致。
同じ状態・同じ候補・同じH/G生成でownerの準備移動効果を比較できる。
大形性能の支持タイル数は1/2個。3個以上・端数タイル・全周期は資格検証で扱い、
大形の広支持性能には外挿しない。

## 完全stepとcapture/replayピーク

下記は今回の同cohort対照だけ。過去campaignとpoolしない。中央値ms。

| cohort | 現行 | 元bucket16 | 幅区分 | 範囲 | 所属情報 | 全部 | dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

| rho3-run1 | 1.423444 | 1.927678 | 1.525066 | 1.737231 | 1.704028 | 1.376190 | 0.329278 |
| rho8-run1 | 1.828889 | 2.345455 | 1.912907 | 2.104604 | 2.060324 | 1.718625 | 0.329221 |
| rho3-run2 | 1.441718 | 1.942603 | 1.536107 | 1.757238 | 1.709347 | 1.378570 | 0.329153 |
| rho8-run2 | 1.843762 | 2.328454 | 1.919325 | 2.108800 | 2.067274 | 1.724032 | 0.329185 |

| 全cohortで同じピークMiB | allocated | reserved |
| --- | ---: | ---: |
| 現行 | 57.430664 | 104 |
| 元bucket16 | 62.523438 | 124 |
| 幅区分 | 62.524414 | 124 |
| 範囲 | 62.773438 | 124 |
| 所属情報 | 65.723633 | 124 |
| 全部 | 65.974609 | 124 |
| dense | 81.752441 | 106 |

幅区分単独は17.57–20.93%、範囲は9.43–10.27%、所属情報は11.22–12.16%、
全部は25.96–29.03%、元bucket16から完全stepを短縮した。単独版は現行より遅い。
全部は現行よりrho3で3.32–4.38%、rho8で6.03–6.49%速い。
allocatedは現行から8.543945MiB増え、denseより15.777832MiB少ない。
reservedは現行より20MiB、denseより18MiB大きい。元bucket16もreserved124MiBで、
4準備版を通じて同値だった。総GPU process peakは未測定。
両allocator peakをdense以下にする選択gateでは現行が最速のまま。全部版は研究候補として
保持し、採用せずメモリ削減を次に検討する。

| 追加対照 | rho3-run1 | rho3-run2 | rho8-run1 | rho8-run2 | allocated/reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| matrix-torch | 0.708384 | 0.715252 | 1.363399 | 1.372093 | 96.163086/142 |
| factor | 1.969443 | 1.981126 | 4.320406 | 4.352305 | 83.114258/130 |
| onchip-h | 1.800828 | 1.818563 | 3.954358 | 3.938559 | 42.866211/114 |

## 固定post24で分けた費用

別Graphの5 replay、各stageのchunk/軸を合計した中央値ms。optimizer更新を含まない。
準備全体の中に追加準備stageが含まれ、二重加算しない。完全stepへ足したり引いたりしない。
「所属情報」ではfirst/countの再構築を消すが、distance<countとprofile自体の支持maskは残る。
単独版の差はcompiler/resource・アクセスも含む変更効果であり、排他的な命令実行費用ではない。

| rho/版 | sort+追加準備全体 | 追加bands | 追加ranges | 追加membership | Y owner | dX owner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3/元bucket16 | 0.098784 | 0.000000 | 0.000000 | 0.000000 | 0.468448 | 0.466112 |
| 3/幅区分 | 0.108416 | 0.007424 | 0.000000 | 0.000000 | 0.244800 | 0.243840 |
| 3/範囲 | 0.107136 | 0.000000 | 0.006080 | 0.000000 | 0.362976 | 0.358240 |
| 3/所属情報 | 0.108544 | 0.000000 | 0.000000 | 0.007712 | 0.337536 | 0.337472 |
| 3/全部 | 0.127296 | 0.007328 | 0.006016 | 0.007904 | 0.153312 | 0.152224 |
| 8/元bucket16 | 0.098176 | 0.000000 | 0.000000 | 0.000000 | 0.546464 | 0.547904 |
| 8/幅区分 | 0.107840 | 0.007328 | 0.000000 | 0.000000 | 0.322432 | 0.322560 |
| 8/範囲 | 0.106624 | 0.000000 | 0.005984 | 0.000000 | 0.423584 | 0.424256 |
| 8/所属情報 | 0.108256 | 0.000000 | 0.000000 | 0.007680 | 0.397120 | 0.396960 |
| 8/全部 | 0.127008 | 0.007328 | 0.006048 | 0.007872 | 0.204512 | 0.205120 |

幅区分を飛ばす単独変更がY/dX ownerを約41–48%短縮した。今回存在する幅区分は2個で、
元ownerは全8区分を毎tile×part、forward/backwardの各小batchごとに処理していた。
全部版ではownerがrho3で約67%、rho8で約63%短縮し、sort+追加準備は約0.0285ms増えた。
最後のpartial reduceは全版で各約0.0064ms。支持候補数とH/G生成は同じ。
空区分・範囲構築を短い準備寿命へ移し、保存済みfirst/countをbackwardにも再利用する方針は
完全stepでも有効だった。準備を増やした効果は単純加算できない。

| owner compiled resources | registers | spills | shared bytes |
| --- | ---: | ---: | ---: |
| 元bucket16 | 160 | 0 | 6144 |
| 幅区分 | 166 | 0 | 6144 |
| 範囲 | 167 | 0 | 6144 |
| 所属情報 | 115 | 0 | 5632 |
| 全部 | 125 | 0 | 5632 |

所属情報版はregister160→115、全部版は125。幅区分・範囲の単独版はそれぞれ166/167であり、
今回の短縮をregister減少だけで説明しない。Nsight Compute/SystemsはこのHubに未配置。
cache hit、実traffic、stall、achieved occupancyは未測定。

全部版の残る主要stageは以下。Y/dXだけが突出する状態は改善したが、生成・微分も残る。

| stage | rho3 ms | rho8 ms |
| --- | ---: | ---: |
| prepare_support_norm | 0.102624 | 0.118848 |
| support_membership_routing | 0.127296 | 0.127008 |
| forward_H | 0.104480 | 0.152992 |
| forward_Y_local_products | 0.153312 | 0.204512 |
| backward_H_value_recompute | 0.103488 | 0.152032 |
| backward_G_dG | 0.202208 | 0.288832 |
| backward_H_dH | 0.151136 | 0.207680 |
| backward_dX_local_products | 0.152224 | 0.205120 |
| source_vjp | 0.010112 | 0.010208 |

## 次の判断

優先する準備移動は非空幅区分の列挙。論理保存量は72Bで、今回最大の単独効果を得た。
範囲と所属情報も含めると速度はさらに改善し、現行を上回った。一方reservedの超過は
元bucket16にもあり、保持tensor量だけから原因を決めない。
次はcaptureのallocator/Graph poolとbuffer寿命を分け、124MiBの内訳を調べる。
所属情報を既存key生成と同時に保存する融合やfirst/countのcompact encodingは、準備passと
論理保存量を減らす次の候補であり、今回の比較には混ぜていない。
Y/dX ownerの更なるALU・cache最適化は、残る生成・勾配stageとの比率を見てから選ぶ。

## 証拠と復旧

- source archive SHA256 `7fe94738e0545351c3fa16d31baa09114233945e1b3c18d7d30085852f281f72`。
- compact archive SHA256 `e7602d20b323e53cab7b25bee6b6c2ccd5cdd682db0a5b7131f0cc107703fd63`。
- read-only audit SHA256 `c8c0a6cef19e48a9266b6180a38a86351f4345e0e34c19cbb816deaa056b24de`。
- paired post24 equality receipt SHA256 `5f69bf3fb35e50fcdfb468a5449aaa9124116d484364ab1d90bb73e6da87e2d3`。
- 171 compact entriesをローカルでhash照合、1673 source filesをpinned Git blobsと照合。
- 初回精度FAIL25 compact entriesとraw失敗fixture、型dispatch FAIL19 entriesもhash照合して保全。
- local ignored `output/regular-grid-h/tile-owner-prep-*`、成功比較は`tile-owner-prep-controls-*`。
- remote `/home/jovyan/runs/cst-tile-owner-prep-ada-c0c656f2-20261011-controls`。
  全72 raw gate `.pt`をremoteへ保持。controller PID256451、followup PID256454。
  command/log/output/PIDと資格検証再利用のreceiptは各JSONと運用メモへ記録。
- 最初の精度FAIL remote `.../cst-tile-owner-prep-ada-d62a7fd6-20261011`、
  型dispatch FAIL `.../cst-tile-owner-prep-ada-59d058fd-20261011`、
  410資格検証/CLI拒否のみ `.../cst-tile-owner-prep-ada-c0c656f2-20261011` を保持。
- mainへpushしない。draft PR #100へ研究source/tests/記録を追加。公開default変更なし。
- ユーザー起動済みHubはreadyのまま、終了時0%/39MiB、GPU計算processなしを確認。
