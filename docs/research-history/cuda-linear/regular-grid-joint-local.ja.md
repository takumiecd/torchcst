# RegularGrid joint cell内でH/Gを使い切る比較

RegularGrid + D2 FlatTorus + normalized profile-productの数学を維持し、入力と出力の
両方の中心binでatomを束ねる。公開chart/defaultは変更しない。
`joint-local32-bm8` / `joint-local64-bm8` は研究対照で、現行BM16/G16を置き換えない。

## 実行と準備の再利用

既存 `_prepare` によるforward source/amp snapshot、whole-chart norm、支持、
singleton/floor、live sigma stop-gradientを使う。keyは
`output_center_bin * input_bin_count + input_center_bin`。
一度のsortとjoint cell prefixで元atom IDを保持する。forwardのindex・距離上界を
そのautograd状態へ保存し、backwardで同じものを使う。更新後の次forwardでは作り直す。
異なるforwardや更新後のindexを無検証で使い回さない。全siteの支持表やCSRは作らない。

通常の狭い支持ではcell×BM8の担当がそのcellの全atomをgroup8ずつ処理する。
forwardはHを生成して局所Y部分和へ消費する。backwardはG/dGを生成し、
H/dHを再計算してParameter partialと局所dX部分和へ消費する。
H/Gのglobal bufferやbackwardまでのH保存はない。

中心cellから両側にcell幅のhaloを取り、`[cells,8,4*cell_sites]` の一つのscratchを
batchごとに再利用する。有効panel幅は3cell、4cellへのpaddingはTritonのpower-of-two。
最終Y/dX ownerが隣接cellと反対側binの部分和を合計する。Y/dX・atom勾配にscatter atomicを
用いない。Parameter partialは元atom IDの `[1,3,A]`、physicalは `[3,A]`。
支持距離がhalo容量を超えるか、短い最終bin等では正確なowner再計算へ切り替える。
そのfallbackも同じscratch容量を使い、支持・atomを切り捨てない。

## 最初の比較: `ff9531b7`

RTX 6000 Ada 48GB、Torch2.9.1+cu128/CUDA12.8/Triton3.5.1。
N2048/B32/A209715、rho3/8、seed41、FP32/TF32 off、MSE、24 actual AdamW/Polar updates。
各rho独立2cohort、2回目逆順、matrix/factor/onchip/BM16-G16/joint32/joint64/denseのfresh workers。
準備・sort・部分和reduction・勾配・optimizer込みの未計装完全stepを判断に使う。

| route | rho3 step ms | rho8 step ms | allocated / reserved MiB |
| --- | ---: | ---: | ---: |
| BM16/G16 | 1.422–1.438 | 1.825–1.870 | 57.431 / 104 |
| joint32 | 9.145–9.237 | 9.600–9.602 | 54.346 / 122 |
| joint64 | 21.794–22.045 | 22.483–22.554 | 45.524 / 114 |
| dense | 0.308–0.329 | 0.310–0.329 | 81.752 / 106 |

両候補は完全stepで大きく悪化し、allocatedは減るがreservedがdenseを上回る。不採用。
全4cohortで通常halo経路が動き、距離上界はrho3=5、rho8=10。
32cellは4096個/atom中央値51/最大79、64cellは1024個/中央値205/最大265。
logical scratchは16/8MiB。これらは実測総peakやphysical DRAM trafficとは別の指標。

別Graphで固定post24 snapshotを使った実route診断（各5 samplesのphase合計中央値）:

| phase ms | rho3 joint32 | rho3 joint64 | rho8 joint32 | rho8 joint64 |
| --- | ---: | ---: | ---: | ---: |
| joint routing一回 | 0.0543 | 0.0534 | 0.0542 | 0.0547 |
| forward H/local Y | 4.2183 | 10.3983 | 4.3908 | 10.7471 |
| forward部分和reduction | 0.1148 | 0.0665 | 0.1152 | 0.0680 |
| backward G/H/dH/local dX/Parameter | 4.8782 | 10.7818 | 5.1356 | 11.2489 |
| backward部分和reduction | 0.1148 | 0.0656 | 0.1150 | 0.0669 |

これらを完全stepへ足したり差し引いたりしない。準備と最終集約ではなく局所計算が重い。
compilerは両方255 registers、joint32 forward/backward spills=6/28、joint64=264/284。
atom×batch×panelの大きな一時配列が保持量を増やし、狭い支持でも広いpanelを評価している。
spillは確認できたが、cache hit/DRAM/stall countersは未測定で、全悪化をspillだけへ帰属しない。

## 修正版: 大きな一時配列を作らない

`joint-compact32-bm8` / `joint-compact64-bm8` は独立Algorithm
`research_cuda_regular_grid_joint_compact_h` として登録する。
元のjoint-local対照は保持し、意味を上書きしない。
atomのouter productを一つずつBM×panelへ加え、GROUP×BM×panelを作らない。
haloは16sitesのguard付き、scratchは `[cells,8,2*cell_sites]`。
距離上界が16を超えれば正確なowner再計算へ戻す。全支持を切り捨てない。
準備・index再利用・norm/VJP・更新則は同じ。FP32の加算順は変わるため再検証する。
修正版もsource `8415dd9f` で比較を完了した。完全stepと両peakの条件では不採用。
外積の順次累積とhalo縮小を同時に変更しており、その寄与を単独には分離していない。

## 検証・証拠

最初の測定は127 GPU資格検証（zero failures/errors/skips）、全48 initial/post24
full-shape/all-atom FP64 Y/dX/all-four-dP gatesがmaxabs/relL2=4e-4でPASS。
6 fixed snapshot probesは未計装結果・primary snapshotとbitwise一致し、Pを変更しない。
28 primary workers / 588 samples。21 samplesを21独立runとは数えない。
CPU1762 passed/4020 GPU-or-DB skips/18既存warnings、40 Plan宣言、Ruff/diff、wheel/sdist PASS。

最初の資格検証は64cellのindex長を32cellと同じとしたテストで126pass/1fail、性能未実行。
テストとdriverの期待件数を修正した。次の起動はrunnerが要求するmatrix/factor/dense対照を
省いたためparserで拒否され、workerも性能測定も始まらなかった。両失敗のsource/logsは保存。
その後、全対照を含む新campaignで上述の完全比較を行い、誤差基準・caseは変えていない。

Remote: `/home/jovyan/runs/cst-joint-local-ada-ff9531b7-v2-20261011`。
controller PID177818、diagnostic/audit PID182446、全完了。前の二つの失敗directoryも残す。
Local: ignored `output/regular-grid-h/joint-local-results/` とsource/result archives。

- source archive SHA256 `c892c2a42a5ef94db16603b67d66482ae966e12bcac0cfe6357daacf598ca1db`
- compact archive `edafdddbdb0282d4e2fcf2e07d914b43d0425b6f9fd2bd62ca094429d8268fda`
- PASS_COMPLETE audit `c98eefc9272659e62499d13d4239bcade34dcf8019c9ead5121644aa4d0932ba`
- 117 compact entriesと1654 tracked sourceをlocalで固定Git blobと照合。
  48 raw snapshotsはremoteでhash/shape/全gradient coverageを監査し、tensorはremoteに保持。
- 失敗logs archive `e7a7c3710ebeb705fdbfe7252c14be01033dc84b636765bd460213271e7920ab`

この結果はStripが必要という証明ではない。今回の局所実行の具体的な保持量と分割を
評価するもので、入力・出力のjoint grouping一般の性能保証も行わない。


## 修正版の完全比較と採否: `8415dd9f`

同じhardware/runtime・N2048/B32/A209715/rho3+8/seed41/24更新で、元の2候補も同一cohortへ含めた。
9 fresh workers×4cohort、各21 primary samples。最初のstudyと今回を同じrunとしてpoolしない。

| route | rho3 step ms | rho8 step ms | allocated / reserved MiB |
| --- | ---: | ---: | ---: |
| BM16/G16 | 1.439–1.441 | 1.840–1.842 | 57.431 / 104 |
| original joint32 | 9.069–9.073 | 9.501–9.504 | 54.346 / 122 |
| original joint64 | 22.057–22.117 | 22.667–22.674 | 45.524 / 114 |
| compact32 | 4.520–4.523 | 4.747–4.911 | 45.547 / 114 |
| compact64 | 2.804–2.909 | 3.054–3.262 | 42.866 / 114 |
| dense | 0.329–0.329 | 0.329–0.330 | 81.752 / 106 |

compact64は元のjoint64より85.6–87.3%短縮したが、現行BM16/G16よりrho3で94.5–102.2%、
rho8で65.8–77.3%遅い。allocatedは25.4%減るが、reserved=114MiBは現行104/dense106を上回る。
全4cohortで最速かつdenseの両peak以下のCSTは現行BM16/G16。
全joint候補を測定済み研究対照として保持し、公開defaultは変更しない。

準備reuseは1sort/forward+backwardで成立した。固定post24で通常haloが動き、支持距離5/10は
guard16以内。compact32/64のscratch論理容量は8/4MiB。H/Gのglobal bufferは0。
ただしscratchはjoint cell数に比例し、任意の大shapeで小さいメモリを保証しない。
`ceil(Ni/tile)*ceil(No/tile)*BM*panel*4` bytesであり、chart宣言のO(D)容量とは別である。

別Graphの実route phase中央値（ms、5 samples、非加算）:

| phase | rho3 compact32 | rho3 compact64 | rho8 compact32 | rho8 compact64 |
| --- | ---: | ---: | ---: | ---: |
| joint routing一回 | 0.0543 | 0.0547 | 0.0542 | 0.0545 |
| forward H/local Y | 1.8937 | 0.9470 | 2.0102 | 1.0221 |
| forward部分和reduction | 0.0388 | 0.0696 | 0.0387 | 0.0697 |
| backward G/H/dH/local dX/Parameter | 2.2333 | 1.3146 | 2.4413 | 1.4944 |
| backward部分和reduction | 0.0385 | 0.0685 | 0.0384 | 0.0685 |
| physical accumulation | 0.0399 | 0.0399 | 0.0396 | 0.0401 |

全compact specializationsのspillsは0。compact32 forward/backwardは203/237registers、
compact64は94/126registers。shared memoryは2048/4096bytes。
それでも主な遅延は生成と局所集約を融合したkernelに残り、index構築や最後のreductionではない。
profile panelには支持外のsiteも含まれ、atomのouter-product順次累積とcell内のgroup loopがある。
これらの費用は残るが、今回の計測で個々の寄与や達成occupancy/cache/stallを確定していない。
`prepared norm/support`のeventは `_prepare` の範囲で、source/amp cloneを含まない。
全準備・clone・集約・optimizer費用はprimary完全stepに含まれている。

192 GPU資格検証はzero errors/failures/skips。全64 initial/post24 full-shape/all-atom
FP64 Y/dX/all-four-dP gatesがmaxabs/relL2=4e-4でPASS。
36 workers/756 samples、6 fixed probesがprimary snapshot/未計装結果とbitwise一致し、P不変。
基準と全4候補はそれぞれ独立repeatのpost24 P/Y/dX/dPがbitwise一致。
異なるrouteの加算順は違い、route間のbitwise一致を要求せず個別FP64 gateで判定する。
CPU1771 passed/4076 GPU-or-DB skips/18既存warnings、42 Plan宣言/Ruff/diff/wheel/sdist PASS。
Measured-source CI: https://github.com/takumiecd/torchcst/actions/runs/38124951069

Remote: `/home/jovyan/runs/cst-joint-compact-ada-8415dd9f-20261011`。
controller PID182882、diagnostic/audit PID194877、すべて完了。user-opened Hubはreadyのまま。
Local: ignored `output/regular-grid-h/joint-compact-results/` とsource/result archives。

- source archive `a270fa61db18401c01a7c8f235fe7bac7a340fd8b1a2d4d2d8aaed92839cb23c`
- compact archive `457706cebf14a4a9876397049175b7fb02aad35a00bff8001a78b3918462a477`
- PASS_COMPLETE audit `5a05a50952342bbd0709d660d6c91a1f91cdb28221d30325126f045620addb04`
- 141 compact entries/1656 tracked sourceをlocalでhash/Git blob照合。
  64 raw snapshotsをremoteでhash/shape/全gradient coverage監査し、tensorはremoteに保持。

再現はこの固定source snapshot内で既存runnerを呼ぶ。2回目は `--reverse` を追加する:

```bash
python -m pytest -v tests/test_regular_grid_joint_local_cuda.py tests/test_regular_grid_joint_compact_cuda.py tests/test_regular_grid_batch_producer_cuda.py
python -m benchmarks.cuda.linear.periodic_comparison --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho3.json --plans matrix-torch factor onchip-h input-site-reciprocal-bm16-g16-h32 joint-local32-bm8 joint-local64-bm8 joint-compact32-bm8 joint-compact64-bm8 dense --isolated-oracle --phases --output output/comparison.json
```

rho8は対応するfrozen Caseを使う。実行時に `CST_FROZEN_SOURCE_COMMIT` を測定commitへ設定する。
診断driverはignored evidence内に置き、source/result manifestを保存する。原本を上書きしない。
この比較からStripの必要性や、joint grouping全般の限界を断定しない。
