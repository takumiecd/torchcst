# RegularGridの支持タイルと小行列積

2026-10-11。RTX 6000 Adaで比較完了。支持タイルでatomをまとめ、IEEE FP32の
小行列積でH/Gを生成・消費する方式は、現行BM16/G16/H32より遅く両メモリピークも増えた。
BM8/BM16とも採用せず、現行研究基準を維持する。公開defaultの変更はない。

## 数学と寿命

`W = U diag(A) V` の全operator-sites L2正規化と床、source VJPを維持する。
入力I、出力J、atom束S、小batchTで
`H[T,S] = sum_I X[T,I] V[S,I]^T`、
`Y[T,J] += H[T,S] (U[J,S] diag(A[S]))^T`。
backwardは `G[T,S] = sum_J dY[T,J] U[J,S]`、
`dX[T,I] += (G[T,S] diag(A[S])) V[S,I]`。
中心のセル所属と違い、各軸の支持区間が重なるタイルを一覧化する。
1つのatomが複数タイルへ所属しても、各siteは所有タイルから1度だけ出力する。
H/Gの全支持の集約後にdA/dcenterを作り、入力・出力タイル間の交差項を残す。
タイル単位で再正規化しない。

## 実行

`TileDotHAlgorithm` はD2 full-period Torusの既存数学契約に限定する研究候補。
32-siteタイル、32-atom束、8/16-row batch、8 owner partialを使用する。
支持長<=32かつ各軸が32の倍数なら各atom/軸の所属は最大2個。
packed支持区間から入力・出力それぞれ2A個のkeyをsortし、boundsを作る。
forwardのroutingをbackwardで再利用し、次forwardでは現在の値から再構築する。
空支持もH/G writerを持ち、全atomの結果を初期化する。
幅が広い／短い端タイルではGPU上で全支持の正確なfallbackを選ぶ。
容量超過による切り捨てはない。
負の支持座標のタイル所属にはfloorを使用し、周期末尾の所属も残す。

H/G producerは同じ支持開始タイルのatomでX/dYの連続panelを共有し、
最大2枚のpanelに対して `tl.dot(input_precision='ieee')` を行う。
TF32や低精度のTensor Coreへ置き換えない。matrix row shapeは16で、BM8は半分をmaskする。
タイル一覧の第1entryが唯一のH/G writerとなる。
ownerは支持が重なるatomのH/Gを行列積で消費し、8 partialを別kernelで集約する。
Y/dX/dPへのscatter atomicは使用しない。

Hはforwardから保存せず、backwardで再計算する。
H/dH/G/dGは小batch分だけの `[A,BM]` slabを繰り返し利用する。
したがって寿命はbatch chunk単位であり、全atomを跨ぐslabは存在する。
完全にregister内でatom束ごとに使い切る実装ではない。
入力×出力タイル積のscratchや全量Wは作らない。

## 比較条件と数値検証

最終計測sourceは `2fbeeabeecdc21d08615f5d63fea627747915269`。
`TileDotHAlgorithm` / `research_cuda_regular_grid_tile_dot_h` v1をbenchmark-localの
`tile-dot32-bm8`、`tile-dot32-bm16`として登録した。tile_sites32、owner_parts8は固定。
library sortのworkspaceを静的に保証できないためworkspace_boundはNone。

NVIDIA RTX 6000 Ada Generation 48GB、Torch2.9.1+cu128、CUDA12.8、Triton3.5.1。
N2048/B32/A209715、rho_initial3/8、seed41、MSE、IEEE FP32/TF32 off。
同じAdamWとPolar/periodic更新を含む24 actual updates、21 primary samples。
rhoごとに独立2cohort、2回目はplan順を逆転した。各cohortは現行BM16/G16/H32、
tile-dot BM8/BM16、matrix-torch、factor、onchip-h、denseの7 fresh workers。
初期・post24の全site/all-atom FP64 Y/dX/全4種類のdPについて、max_absとrelative_l2の
既存4e-4 gateを維持した。source修正後に資格検証から実施し、timing retryや条件緩和はない。

- GPU資格検証127 PASS、0 failures/errors/skips（136.80秒）。
- 28 primary workers、588 samples、48 full-shape/all-atom FP64 gates PASS。
- post24固定snapshotの4診断はPASS。primaryと未計装版のY/dX/dPがbitwise一致し、Parameter不変。
- CPU全体1780 PASS /4132 GPU・DB skips /18既存warnings。CUDA修正後の宣言テスト9 PASS /56 GPU skips。
- catalog44 plans、Ruff、diff check、wheel/sdist build PASS。計測sourceの
  [CPU/build CI](https://github.com/takumiecd/torchcst/actions/runs/38127687971)もPASS。

## 最終sourceの完全stepとメモリ

準備、2軸のsort/bounds構築、batch chunk処理、Y/dX集約、勾配・optimizer更新を含む
未計装Graphの中央値（ms）。異なるsourceのcampaignは混ぜない。

| cohort | 現行BM16/G16/H32 | tile BM8 | tile BM16 | dense |
| --- | ---: | ---: | ---: | ---: |
| rho3 run1 | 1.435793 | 5.800962 | 3.249550 | 0.329249 |
| rho3 run2 | 1.443315 | 5.809945 | 3.248550 | 0.329360 |
| rho8 run1 | 1.839287 | 7.125148 | 3.884935 | 0.329536 |
| rho8 run2 | 1.838444 | 7.160792 | 3.945252 | 0.328875 |

| capture/replay peak (MiB) | allocated | reserved |
| --- | ---: | ---: |
| 現行BM16/G16/H32 | 57.430664 | 104 |
| tile BM8 | 68.415039 | 134 |
| tile BM16 | 93.717285 | 150 |
| dense | 81.752441 | 106 |

各peakは4cohortで同じ。GPU process全体のpeakは未測定。
BM16は現行の2.11–2.26倍、BM8は3.87–4.04倍かかる。
BM16は両peakがdenseを超え、BM8もreservedがdenseを超える。
H/dH/G/dGの論理payloadだけでBM8=25.6MiB、BM16=51.2MiB、owner partialは0.5/1MiB。
これはtensor shapeの費用であり、上表の実測peakと同一ではない。

同cohortの対照（ms）も保持した：

| cohort | matrix-torch | factor | onchip-h |
| --- | ---: | ---: | ---: |
| rho3 run1 | 0.712083 | 1.972350 | 1.812632 |
| rho3 run2 | 0.715070 | 1.990113 | 1.834624 |
| rho8 run1 | 1.362186 | 4.338756 | 3.964581 |
| rho8 run2 | 1.369409 | 4.350404 | 3.929599 |

| 対照のcapture/replay peak (MiB) | allocated | reserved |
| --- | ---: | ---: |
| matrix-torch | 96.163086 | 142 |
| factor | 83.114258 | 130 |
| onchip-h | 42.866211 | 114 |

これらの採否も同一scopeの両peakで判断し、時間だけでmatrix方式へ切り替えない。

## 生成kernelのfallback分離

最初の有効source `47ae68de88df38f6363c09e1cccd8f2405487063`では、広支持fallbackと
狭支持の小行列積を同じproducerに置いた。資格検証127 PASS、48 full FP64 gates、
28 workers/588 samples、4固定snapshot診断はPASSしたが、BM16 producerは255 registers、
forward46 spills、中心微分324 spillsだった。

最終sourceでは狭支持producerと広支持fallbackを別kernelにした。両方をlaunchし、GPUの
wide flagで一方だけ計算する。広支持の数学・容量・正確なfallbackは維持した。
追加のguarded launchもprimaryとproducer診断に含めた。4診断のwide flagはすべてfalse。
実行された狭支持producerはBM8/BM16とも168 registers /0 spillsになった。
inactive fallbackはBM16 forward255 registers/44 spills、微分255/318のままであり、
これを狭支持で実行されたspillと解釈しない。ownerは153 registers /0 spills。

| 初回source47の完全step (ms) | tile BM8 | tile BM16 |
| --- | ---: | ---: |
| rho3 run1 | 6.745783 | 3.678826 |
| rho3 run2 | 6.750090 | 3.676554 |
| rho8 run1 | 8.284097 | 4.448937 |
| rho8 run2 | 8.287789 | 4.453215 |

最終版は対応する初回cohortより約11.4–14.1%短縮したが、peakは変わらず現行にも届かない。
別sourceの独立campaignとして保存し、平均をpoolしない。介入はkernel分離とlaunchを含むため、
短縮の全量をspill除去だけへ帰属させない。

## 固定snapshotの細部診断

各stageについて5回の別Graph診断を行い、1 replay内の同stage各chunkの時間を合計して
中央値を取った（ms）。source/amp snapshot cloneはprepare_support_normの外、primaryの中。
producer eventは狭支持とguarded fallbackの両launchを含む。
stageは計装で実行条件が変わるため、表を足したりprimaryから差し引いたりしない。

| stage | rho3 BM8 | rho3 BM16 | rho8 BM8 | rho8 BM16 |
| --- | ---: | ---: | ---: | ---: |
| support/norm準備 | 0.100544 | 0.100320 | 0.116800 | 0.119040 |
| 所属sort/bounds構築（1回） | 0.110624 | 0.110656 | 0.110272 | 0.111040 |
| forward H生成 | 1.150848 | 0.574304 | 1.425632 | 0.715680 |
| forward Y局所行列積 | 0.446752 | 0.221664 | 0.550112 | 0.274080 |
| forward partial reduction | 0.012128 | 0.006528 | 0.012128 | 0.006464 |
| backward G/dG生成 | 1.836384 | 0.914368 | 2.282400 | 1.140512 |
| backward H/dH生成 | 1.834976 | 0.913824 | 2.277760 | 1.138848 |
| backward dX局所行列積 | 0.448064 | 0.222144 | 0.551168 | 0.275008 |
| backward partial reduction | 0.012128 | 0.006528 | 0.012192 | 0.006528 |
| physical parameter gradient | 0.036288 | 0.028288 | 0.036320 | 0.028032 |
| source VJP | 0.010272 | 0.010208 | 0.010176 | 0.010208 |

主な費用はH/Gと中心微分のproducerで、最後のpartial reductionやsortではない。
32-site panelに対しpost24の平均支持長はrho3で入力5.98681/出力5.98722、
rho8で15.89779/15.89888。入力/出力所属数はrho3=242462/242341、rho8=307358/307402、
平均所属数は約1.156/1.156、1.466/1.466。64タイルのatom数はrho3で3623–3928/3655–3919、
rho8で4643–4947/4624–4952。容量切捨てやfallback実行による結果ではない。
支持外ゼロの演算、BM8のrow padding、中心微分を含むprofile評価とslab書込みは残る。
これは構造上の費用だが、個別の因果寄与を分離計測したとは言わない。
実cache hit、DRAM traffic、stall、達成occupancyは未測定。

次の仮説は、支持幅に合うproducer粒度と、全atomのdH/dG slabを残す寿命を再検討すること。
atom束ごとに微分まで消費すると小さな保持量にできる一方、Y/dX ownerの参照順と
再計算が増える可能性がある。今回の結果は、その方式やStripへの変更を検証したものではない。

## 証拠・復旧・再現

既存draft [PR #100](https://github.com/takumiecd/torchcst/pull/100)、
branch `kernel/regular-grid-h-lifecycle`。ソース・tests・研究記録をcommitし、rawはignored保存。

| evidence | SHA256 |
| --- | --- |
| 最終source tar | `5c69fa6e13f45d62d6261ee473b35efb0429bddea58ab39a8f880439a6ae0097` |
| 最終compact results tar | `8e559150c9c0ba3d971e8fcd1ce0200bbf96b4961c701d7d9aed0fa2496761da` |
| 最終audit.json | `1c7956cb66250bd5f37663182fd98a1038ceb3b40c703b964426950238ecd9e0` |
| 初回source47 tar | `adc68e645d6b0d2d4fbdad00735376e12b6d26943a477d3d2093255e6001329b` |
| 初回compact results tar | `cf6cdd17323559c5da1912428cad7213e5c0502d66e3be6cf872ea64958f1a49` |
| 初回audit.json | `d652021025c8a1d864473d212c382e2393c7dc6c65c5bab5e4dd87f54e200f1a` |
| 失敗資格検証tar | `05fb2549b6f10af65340cad321355627df4f4c4706edd8eef9e9008a74f77493` |

最終remote directoryは `/home/jovyan/runs/cst-tile-dot-split-ada-2fbeeabe-20261011`、
controller PID208483。初回は `/home/jovyan/runs/cst-tile-dot-ada-47ae68de-20261011`、PID199553。
各sourceは1660 tracked filesをmanifestと照合した。remote auditはPASS_COMPLETE。
最終compact archiveの110 entriesをdownload後にhash確認し、1660 filesをpinned Git blobとも照合。
初回archiveも110 entriesを確認した。48 full snapshots/campaignはremoteにhash・shape・勾配付きで保管。
localは研究worktreeの `output/regular-grid-h/tile-dot-split-results/`、
`tile-dot-initial-results/`、対応tar/source manifests/operating notes。
大きなtensor snapshotはtransferせずremote保管。ユーザーが開いたHubはreadyのまま残す。

失敗した資格検証もsourceごと保存した。`af3a6347`（PID195334）はruntime countへの
constexpr cdiv呼出しがcompile失敗し、runtime整数演算へ修正。
`75ce6ec6`（PID195970）は123 PASS/4 FAIL、負の支持座標でtruncationが周期末尾の所属を
落としたためfloorへ修正。いずれも性能測定には進まず、基準を緩和しなかった。
remote失敗directoryは `cst-tile-dot-ada-af3a6347-20261011` と
`cst-tile-dot-ada-75ce6ec6-20261011`、まとめたtarもlocal/remoteに保持した。

再現はpinned sourceとruntimeを復元し、既存runnerを使う。新しい出力先を用意する：

```bash
python -m pytest -q tests/test_regular_grid_tile_dot_cuda.py
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho3.json \
  --plans matrix-torch factor onchip-h input-site-reciprocal-bm16-g16-h32 \
    tile-dot32-bm8 tile-dot32-bm16 dense \
  --isolated-oracle --phases --output output/tile-dot-reproduction/rho3/comparison.json
```

rho8 caseでも実施し、それぞれ別fresh cohortを `--reverse` で実行する。
controller、qualification driver、fixed snapshot診断、audit scriptと実際のcommand/logは
上記archive/remote/local evidenceに保存した。GPU再現時は完全qualification127件と既存の
full-shape gatesを含む保存driverを使用し、上記単独pytestだけで全検証完了とはしない。
