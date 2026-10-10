# RegularGridでH/Gを全保存しない最初の試作

2026-10-10。[Hの寿命を軸にした設計](../../h-lifecycle-kernel-design.ja.md)の最初の研究候補。
公開dispatcherへの採用は別判断で、当面は研究branchに置く。

## 最新checkpoint（2026-10-11）

計算source `547d1dcf`、L4/B32/N2048/A209715、FP32、joint discrete-L2、
24実更新の完全step。各Case12独立workers、21 replay samples、
11 CST×initial/post24×2Caseの44 full oracleをPASSした。
全562 GPU環境tests、CPU1685 tests、GitHub CPU validationをPASS。

| Case | prepared H32 [ms] | 両peak条件を満たす最速CST | 完全step [ms] | allocated/reserved [MiB] | dense [ms; MiB] |
| --- | ---: | --- | ---: | ---: | ---: |
| N2048/rho3 | 7.9262 | group32 H32 | 5.6655 | 57.42/104 | 0.5923; 81.75/106 |
| N2048/rho8 | 10.6602 | stream G8/H32 | 7.9083 | 57.42/104 | 0.5907; 81.75/106 |

rho3はgroup32で約28.5%短縮。rho8はgroup32の8.6506msからstreamでさらに約8.6%、
prepared controlから約25.8%短縮した。stream G8はrho3では6.4045msと遅いため選ばない。
旧input-owner G16/32はreservedがdenseを超えるため、両Caseで不採用。
stream G8でそのmemory問題を解消し、H/G・parameter partialの寿命を短くした。

これは各Case1 cohortでの研究Plan選択であり、任意shape/batch/rhoのdispatch閾値ではない。
公開dispatcherへの統合は行っていない。N1024の選択は下記のsource `7fcf7fe3` の4条件cohortを参照する。
上のN2048結果をN1024や別GPU、高次元CUDA、別profileへ外挿しない。
denseは依然として約0.59msと速い。残る重い区間はH→Yの集約とrho8のbackwardである。
粗いbinによる無効site検査は前のcensusで約8倍（rho3）／3倍（rho8）だった。
次は支持の正値範囲を使う候補選別とprofile積和を、準備・routing費用込みで比較する。
物理的なatomic stallやL2 hit率の原因は未測定であり、allocatorや静的compiler情報から断定しない。

詳細・不採用結果・source/result hashes・停止と復旧の証跡は末尾に保全する。

## 実装範囲と契約

RegularGridの二つのweight軸に各一つの座標を対応させ、FlatTorus、Triweight、
Polarの共有live幅、task幅stop-gradient、全operatorのdiscrete-L2を維持する。
単純なsite番号wrapを使うため、各座標で`period / n == spacing`を確認する。
Euclidean、部分周期、各側複数座標はこのCUDA Algorithmの初期対応範囲外。
Chartそのものの宣言能力は制限しない。

`research_cuda_regular_grid_onchip_h`は連続する元atom8個×batch8個を一CTAで扱う。
支持候補・norm・normの中心微分は既存PeriodicProductと同じ準備処理で全atomについて作る。
Xから生成したHはCTA内に保持し、正規化したUとampを評価してYへatomic加算する。
Hの全体Tensorは作らず、入力支持の重なりによる並べ替えはまだ行わない。

backwardはforward時点のParameter snapshot、amp scalar、準備済み幅・中心・norm・
norm微分・支持区間と、forward時点のplacementを使う。更新後のlive値を読まない。
GとH／各中心方向の縮約をCTA内で再計算し、dXにはatomic加算する。
batch部分ごとの物理dA/dci/dcoを`[ceil(B/b),3,K]`に置き、別のreductionで
全batchを合計してPolarのParameter VJPへ戻す。Parameter勾配へのatomicは使わない。
ampゼロのatomもdAの対象であり、ampで割ってGを復元しない。

H/Gは論理的な中間量として短い寿命を持つ。registerの使用やspillの実際の費用は
GPU測定前には確定しない。この試作はatom担当でatomicを残す。
出力区間担当／重なりによるgrouping／容量上限付き部分保存は次の比較候補とする。

保存するのはX、Parameter clone`[K,4]`、amp scalar、packed情報`[13,K]`。
H/Gは保存しない。FP32で管理scratchの上界は
`4 * ((17 + 3*ceil(B/b))*K + 1)` byte。これはsnapshot・準備・batch勾配部分和を含み、
返すY/dX/dP、元X、optimizer、allocator/Graph poolは含まない。
完全stepの総peakとこの式を区別する。

## 比較方法

既存の`benchmarks.cuda.linear.periodic_comparison`を`--regular-grid`へ拡張する。
旧PeriodicGridのPlan、recipe、既定cohort／JSON／採否gateは維持する。
新候補とRegularGrid版の既存matrix／全H保存factor controlをbenchmark Registryへ登録し、
公開Registryの既定を変更しない。研究専用JSONは提出schema／DB取り込みの対象外。

固定Caseは`cases/regular-grid-h-{1024,2048}-rho{3,8}.json`。
B32、K=floor(.05N²)、seed41、FP32 IEEE/TF32 off、同じ初期値とAdamW＋Polar更新。
2 eager warmup＋1 replay＋21 timed replayで24実更新を確認する。
全axis・全atomの独立FP64 oracleで初期／24更新後のY/dX/全4dPを検査し、
max_abs／relative_l2各4e-4のgateを固定する。
候補を独立processで測り、norm、snapshot/prep、更新、Graph memoryも含める。

全H保存factor、grouped W＋Torch/Triton GEMM、denseを同条件で再測定する。
allocatedとreservedの総peakが両方ともdense以下の候補から最速を選ぶ。
診断Graphは主測定後の別状態copy。fused H→Yの内部をCUDA eventで分けて測ったとは
主張せず、既存factorのH／scatter時間と、融合kernel全体、全backwardを比較する。
allocated/reservedから物理L2 hit率やatomic競合を断定しない。

```bash
PYTHONPATH=src:. python -m tools.kernel_dev check --plans benchmarks/cuda/linear/plans-regular-grid-h.json
PYTHONPATH=src:. python -m pytest -q tests/test_regular_grid_h_cuda.py tests/test_periodic_cuda.py
PYTHONPATH=src:. python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-1024-rho3.json \
  --phases --output output/regular-grid-h-1024-rho3.json
```

## 検証状況

CPU全体1644 passed / 2731 skipped（CUDA・実DBを含むskipはGPU検証ではない）。
Plan宣言の往復、changed-file Ruff、diff check PASS。
以下のL4検証と測定を完了した。


## 2026-10-10 L4結果

全H/Gの保存をなくしてallocatedを減らせたが、採用条件は未達である。
2048²のallocatedは99.36→59.12MiB（約40.5%減）、reservedは170→154MiB。
**denseのreserved106MiBを超えるため、allocatedだけを見て採用しない。**
完全stepは同サイズの全H保存factorより約12〜14%短縮したが、grouped Wやdenseより遅い。
1024/rho8の約1.1%差は1cohortでの観測で、速度改善が確立したとは扱わない。
各caseは独立processごとに21sampleを保存した1cohortで、独立run数は各1。

各セルは **完全step中央値ms / allocated MiB / reserved MiB**。

| N / 初期rho | W＋Torch GEMM | W＋Triton GEMM | 全H保存factor | CTA内H/G試作 | dense |
| --- | --- | --- | --- | --- | --- |
| 1024 / 3 | 0.396 / 48.67 / 96 | 0.470 / 32.42 / 56 | 1.282 / 37.21 / 56 | 1.187 / 27.09 / 56 | 0.089 / 33.00 / 86 |
| 1024 / 8 | 0.879 / 48.67 / 96 | 0.947 / 32.42 / 56 | 2.749 / 37.21 / 56 | 2.718 / 27.09 / 56 | 0.086 / 33.00 / 86 |
| 2048 / 3 | 1.339 / 96.16 / 162 | 1.515 / 79.36 / 162 | 5.334 / 99.36 / 170 | 4.674 / 59.12 / 154 | 0.601 / 81.75 / 106 |
| 2048 / 8 | 3.433 / 96.16 / 162 | 3.651 / 79.36 / 162 | 11.656 / 99.36 / 170 | 10.023 / 59.12 / 154 | 0.600 / 81.75 / 106 |

1024では試作とTriton Wがdenseの両peak以下で、この制約内ではTriton Wが速い。
2048では測ったCST候補すべてがreservedの制約を満たさず、採用候補はなし。
公開dispatcherは変更せず、試作は研究branchとdraft PRに保持する。

### 次に優先する費用

2048/rho8の別状態copyで、試作のforward＋lossは4.371ms、backwardは4.485ms、
optimizerは0.316ms。固定状態のsnapshot/prepは0.354ms、Y初期化＋融合H→Yは4.423ms。
これらは別の診断Graphの中央値であり、足して主測定時間にしない。
融合の内部H／Y時間を別々に測ったとは主張しない。

同じcaseの全H保存controlでは、H縮約0.616ms、Y scatter3.906ms、
全backward6.036msだった。Hの寿命と融合を変えるとbackwardは減ったが、
出力への加算を束ねる課題は残る。準備だけを優先しても大きい費用は消えない。

次の候補は**出力支持の重なるatomを集め、Yへの局所部分和を合計してから書く**方式。
まず入力／出力担当を全面的に変える前に、現行H生成の中で同じYへの寄与をまとめる。
支持範囲のunion・周期境界・無効laneを正確に扱い、幅が広いときにも切り捨てない。
別group間にはatomicが残るため、完全な出力所有方式と区別する。
並べ替え・mapping・norm・backwardの元atomへのscatter・保存量を完全stepに含める。
dX側の加算についても対応する入力支持groupingを別に評価する。
この候補の性能はまだ測っていない。

もう一つは**reservedの内訳**。59.12MiBのallocatedに対してreserved154MiBが残る。
現結果だけではallocatorの未使用領域、Graph pool、断片化、workspaceの寄与を特定できない。
同じ測定境界でmemory stats/snapshotを別途取り、どの割当と寿命がreservedを増やすか確認する。
メモリ方針を変更して基準を通ったことにせず、両peakのdense以下条件を維持する。
物理L2 missやatomic競合の内訳はcounter未取得なので未確定。

### 正しさと失敗の記録

CUDA環境で**109 passed**（CPUで実行できる宣言検査22件、実GPU検査87件）、skipなし。
全atomのY/dX/dP、strided X/dY、batch1/3/32/64、別tile、周期境界、広い支持、
ゼロ振幅、空／単一支持、joint floorの下／一致／上、必要な勾配だけの分岐、
retained backwardのforward snapshot、H/G非保存、20 eager/Graph更新を確認した。
公開fused AdamWとの同一cotangent更新20回はParameter／exp_avg／exp_avg_sqの最大誤差0。

大きい4caseは全axis・全atomのFP64 oracleを初期・24更新後に通した。
全対照を含む最大max_abs=3.3565138820357276e-4、最大relative_l2=1.0062414776722231e-6。
各上限4e-4は変更していない。各workerの実更新counter24、試作の全atomのlive幅変化も確認。

保存した失敗は以下。数値FAILと通信／セットアップ失敗を区別する。

- `l4job-aab7323172a144ccba1cfe31e5d387ab`: Colab Pythonのensurepip失敗。GPU検証前。
  job内`pip --target`へ変更し、system packageは変更していない。
- `l4job-499a6a852c3243c39e0a0793bd8e615c`: CLIの作業場所設定でConnection was lost。
  実験コードの起動／完了は確認できず、結果は採用しない。pool recoverと所有VM停止確認済み。
- `l4job-e1b7690408724a378c8c18b2833d4b41`: 108 passed / 1 failed。
  新grid用に周期を大きくしたGraph試験で、候補fused AdamWと参照foreach=Falseの
  異なる実装を比較し、Parameter差2.8610e-6が固定atol2e-6を超えた。
  今回の完全step契約に合わせ、参照もfused AdamWに揃えた。
  grid・入力・cotangent・lr・更新回数・atolは変えず、全109件を再検証してPASS。
  旧PeriodicGrid試験の既定cross-implementation比較は残した。

### Provenanceと保全先

測定sourceは`81761251dc53c968f27072e259938e370812cc91`。
GPUはNVIDIA L4、driver580.82.07、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、Python3.13.15。

| scope | job | source archive SHA256 | result archive SHA256 |
| --- | --- | --- | --- |
| correctness/state | `l4job-388ebd88e3ce450ca5797597164aaeb5` | `e73a32abe033b359fc91f34d576ec89f1b44af23a5b48d4a42ce995900a7ce5f` | `fbbf5446f37bb052a9fa22fbd296f51f554cb564fc9b954e0c7fcf6c88703757` |
| 4case comparison | `l4job-bc485a14ce3d4daca098f2677f7b560a` | `86618f44bd5ab2e8222600e6543250ab83efd0678f2504120a632c37baaddc2f` | `c0925c7a36ed6033661c62a40b636ad82af9ed22685ff14ab328e34a7ee4ff8b` |

全receipt/manifest hashと、報告source file hashをlocal sourceに再照合した。
raw原本・失敗logsは共有poolの各job directoryに保管する。
成功2jobのsource/results/receipt copyとdriverは、このworktreeのignored
`output/regular-grid-h/measured-81761251/`と`output/regular-grid-h/driver.py`に保存する。
所有L4は停止確認済み。branch・worktree・失敗原本は削除しない。


## Kernel時間とallocator内訳の追加診断

2048/rho8、同じsource/runtime/初期値/24更新/FP64 gateで、試作・全H保存factor・denseを
独立processに分けて診断した。allocator historyはprimary capture開始から測定境界まで。
主測定後の別状態copyで1 warm replay＋3 replayをTorch profilerへ記録し、
別に1 eager traceも保存した。この計装runのstep時間は主測定の代替に使わない。

Graph traceのCUDA kernel時間では、試作の`forward`が45.01%、`backward`が47.78%、
`prepare`が3.51%、`reduce_parameters`が0.31%だった。
3 replayのkernel一回平均はそれぞれ4.009ms、4.256ms、0.312ms、0.028ms。
これはdevice kernel時間に対する割合で、CPU同期・launch gapを含む完全step時間の割合ではない。
全H保存factorではYとdXの`axis_scatter`合計が68.85%、`factor_vjp`が14.11%、
HとGの`axis_contract`合計が11.33%、`prepare`が2.86%だった。

**次の速度候補は加算の集約から比較する。**
全H保存controlでscatterの費用が大きいこと、試作でも縮約＋scatterを融合した
二kernelに約93%のdevice時間が集中することが根拠である。
ただしatomic命令のstall、非連続load、register spillのどれが支配的かは未測定。
「atomic競合が93%」とは解釈しない。まず出力支持groupingと局所Y部分和を一候補として
完全stepで比較し、入力共有／dX側のgroupingは次の独立候補とする。

primary測定境界のsnapshotは以下。MiB、currentはsnapshot時点、peakはcapture/replay込み。

| 方式 | allocated peak | reserved peak | default pool reserved / active | Graph pool reserved / active |
| --- | --- | --- | --- | --- |
| CTA内H/G | 59.12 | 154 | 104 / 29.81 | 50 / 3.70 |
| 全H保存factor | 99.36 | 170 | 84 / 29.81 | 82 / 3.70 |
| dense | 81.75 | 106 | 88 / 65.00 | 18 / 16.50 |

試作のsnapshot時のinactive領域はdefault 74.19MiB、Graph 46.30MiB。
defaultでは完全にinactiveなsegmentが20MiB、active blockと同居するinactiveが54.19MiB。
Graphのinactiveにはreplayで再利用する一時bufferが含まれ、単純なリーク／不要領域ではない。
全H保存factorはcurrent reserved166MiB、peak170MiBで、表のpool内訳はcurrentの合計。

H/Gの非保存によってGraph poolは82→50MiBへ減った。一方defaultは84→104MiBで、
完全にinactiveな20MiB segmentが試作側に残る。したがって次のメモリ候補は、
warmupからcaptureまでの割当順序、固定形状bufferの再利用、optimizer stateとの同居・
一時bufferの寿命を確認する。`empty_cache`だけではactive blockに挟まれた領域は回収できず、
Graphが再利用する一時領域を削除する対策にもならない。
比較gateは変更せず、allocator設定／capture手順を変える候補はdenseを含め同条件で再測定する。

診断sourceは`82e2f830f9ecefff6e451283dc324b356cfaa1ba`（測定source81761251とruntimeは同じ）。
job `l4job-3963762d8d694cf29f132c964b054021`、三controlとも初期／24更新後gate PASS。
source archive SHA256 `16dae9d74817fc5a3cdf963f0a08aceafb0a8b66f08c4f10ec4996dde706e083`、
result archive SHA256 `2a30d4914eeda232a72f821b9962a599472aa6c04dd8d9f51b48bde4bfa532d0`。
再現driverはignored `output/regular-grid-h/diagnostic.py`、集計は
`output/regular-grid-h/analyze-diagnostic.py`。Chrome trace、stats/snapshot、原本JSONを保存する。


### 測定境界の注意と次の比較順序

allocator historyをfixture作成前から有効にした追跡では、試作のdefault poolに残る
8.125MiB blockの一つの確保元は初期FP64 `oracle_vjp`のGEMM
（`periodic_profile_product.py:108`）だった。同じサイズの別blockにはPython frameがなく、
その確保元は未特定。denseにも8.125MiB blockが二つあり、一つはeager warmupのLinear
（`torch/nn/modules/linear.py:134`）に由来する。
試作側は二つの20MiB segment、dense側は同じ20MiB segmentに両blockがある。
この差は検証側の割当順序がreservedへ残る影響を示す。
これらがGEMMのlibrary workspaceであることはサイズ・確保箇所からの推定で、
全blockの所有者をstackだけで確定したとは扱わない。

元結果の`memory_scope`は「oracle/diagnostic scratch excluded」と記すが、
**検証が作った持続的割当まで除外できているわけではなかった。**
表とraw JSONは固定した境界で得た事実として保管し、値を書き換えない。
現境界のreserved gateは未達だが、これだけで本番kernel固有の必要量とは断定しない。
次のメモリ比較はoracleを別processに分離し、同一source・初期状態・24更新後snapshotを
独立oracleで検査し、性能processへ検証の割当を持ち込まない方式を全controlへ適用する。
正規化・数値gate・optimizer・更新回数・dense以下条件は維持し、旧測定と別protocolの結果として記録する。
その後に必要量が残れば、warmup/captureの割当寿命と固定bufferの再利用を変える候補を比較する。

速度の次の比較は出力支持grouping＋局所Y集約を一つずつ加え、
並べ替え・index mapping・全backward・更新・両memory peak込みで評価する。
register spill／atomic stall／DRAM transactionのcounterは未取得なので、
この候補が勝つことを先取りしない。今の試作を比較基準として保持する。

追跡job `l4job-f88b5554d81a468aa337a9034ee40b94`、試作とdenseの二processともPASS。
source archive SHA256 `303753e3307bbbbb44690472451c945642f8c9a99919ec958f27f7f5fd3a5894`、
result archive SHA256 `a366e47d2b898fdba3024cbac82ee7b3fe56dbb09d594e1a5b5a80a27a3ca89f`。
再現driverはignored `output/regular-grid-h/allocation-origin.py`。
二diagnostic jobのsource/result archive・全source file・全result manifestを再照合し、
raw原本とlocal copyを`output/regular-grid-h/diagnostic-82e2f830/`へ保存した。
runtimeとbenchmark sourceのhashはlocalに一致する。所有L4は停止確認済み。

## 次の候補：出力区間を所有するforward

`research_cuda_regular_grid_output_owned_h` / `OutputOwnedHRecipe`を追加する。
batch8×出力16siteを一CTAが所有し、区間への全寄与をFP32で累積してYへ一度storeする。
forwardのY初期化・Y atomicは不要。Hはatom8×batch8ずつ生成し、使ったら上書きする。
支持が複数の出力区間へ重なるatomは各区間でHを再生成する。全Hは保存しない。

毎forward、準備snapshotの出力中心を粗い区間のkeyへ変換し、`torch.sort`で並べる。
`searchsorted`で各中心区間の開始／終了を作る。siteごとのCSR／支持index表や固定容量bucketは作らない。
候補探索の半径は、実際の準備済み支持区間の端から中心keyまでの距離の全atom最大値を
GPU上で求める。live幅の定数上限を仮定せず、広い／fallback支持でも全寄与を拾う。
半径が全区間へ及べば全atomを一度ずつ処理する。周期wrapで同じ区間を重複しない。
末尾の短い出力区間には追加の候補区間を含め、全候補で実profileの正値を検査してHを生成する。
全正規化・norm微分は元のprepareを使う。

routingは毎forward再構築し、retained backwardへ保存しない。元atomのidentityは保ったまま、
backwardは既存のatom担当H/G再計算・dX atomic・batch別Parameter部分和を使う。
これはforwardの所有方式だけの比較であり、dXまでatomicを消した候補ではない。

routingの明示Tensorはkeys4K、sorted keys4K、order8K、区間境界8(T+1)、
距離部分和4ceil(K/256)、最大距離4byte。sort/searchsortedの内部workspaceは別であり、
Algorithmのworkspace_boundは`None`とする。routing費用とallocator/Graph poolを測定へ含める。
検証と性能processを分離した比較protocolで全controlを測り直し、旧表の数値とは混ぜない。

### 出力所有候補の検証と完全step比較

出力所有forwardのCUDA検査を含め150 passed / skipなし
（実GPU116件、CPUで実行可能な宣言／protocol検査34件）。
原子数集中257、末尾の短いtile、tile8/16/32/64、周期境界、広い支持、
空／単一支持、joint floor、ゼロ振幅、strides、必要な勾配の分岐、
forward snapshot、H/G非保存、20 eager／Graph更新を検査した。
GPU検査sourceは`a7518e44ee1b3b48bc78b3c0fa4f7abcd64952c7`。
後続のoracle snapshot復元修正はbenchmarkのatom count復元とCPU検査だけで、
全runtime source hashはGPU検査時と一致する。
CPU全体1650 passed / 2760 skipped、Plan宣言往復、Ruff、diff check、
wheel/sdist build、CPU CI PASS。CUDA・実DBのskipをGPU／DB検証と扱わない。

`isolated-oracle-v2`では、候補のY/dX/全dPとlive model state/X/dYをCPU snapshotへ保存し、
FP64全axis・全atom oracleを別processで実行する。初期と24更新後の同じsnapshotを検査し、
性能processへFP64 GEMMの割当を持ち込まない。候補自身の初期VJP、warmup、capture/replayと
その持続的library割当は測定側に含む。denseにも同じ測定protocolを適用する。
全40 oracle snapshot hash、24 workerの更新counter24、source/result manifestを確認した。
全対照の最大max_abs=3.3532883353792897e-4、最大relative_l2=1.0063840142777506e-6。
各4e-4 gate、B32、5% atoms、seed41、FP32 IEEE、optimizer／更新則は変更しない。

各セルは **完全step中央値ms / allocated MiB / reserved MiB**。
全controlをv2で再測定した。v1の表とは別protocolとして扱う。

| N / 初期rho | W＋Torch | W＋Triton | 全H保存factor | atom担当H/G | 出力所有H | dense |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 / 3 | 0.391 / 48.67 / 116 | 0.463 / 16.17 / 36 | 1.278 / 20.96 / 36 | 1.172 / 10.84 / 36 | 2.013 / 10.84 / 36 | 0.085 / 33.00 / 86 |
| 1024 / 8 | 0.871 / 48.67 / 116 | 0.942 / 16.17 / 36 | 2.740 / 20.96 / 36 | 2.710 / 10.84 / 36 | 3.270 / 10.84 / 36 | 0.085 / 33.00 / 86 |
| 2048 / 3 | 1.319 / 96.16 / 142 | 1.491 / 63.11 / 122 | 5.213 / 83.11 / 130 | 4.401 / 42.87 / 114 | 8.812 / 42.87 / 114 | 0.593 / 81.75 / 106 |
| 2048 / 8 | 3.331 / 96.16 / 142 | 3.534 / 63.11 / 122 | 11.725 / 83.11 / 130 | 10.241 / 42.87 / 114 | 13.641 / 42.87 / 114 | 0.592 / 81.75 / 106 |

出力所有候補はこの1cohortではatom担当より約21〜100%遅く、両memory peakは同じ。
Y atomicの除去だけで速くなるという仮説は、この実行方式では支持されなかった。
1024ではmemory条件内でTriton Wが最速。2048はv2でも全CST候補がdenseの
両peak以下条件を満たさない。H非保存のreservedはv1の154からv2の114MiBとなったが、
依然dense106MiBを超える。公開dispatcherへの採用はなし。

主測定後の別copyでは、2048/rho3のforward＋lossはatom担当1.810ms、出力所有5.078ms、
backwardは1.912ms／1.918msで、forward側の増加が大きい。
2048/rho8の固定forwardではprepの後の融合／routing＋所有集約が4.333ms／7.407ms。
これらを足して主測定にしない。診断sampleには時系列変動があり、raw値を保存する。

索引作成は「H→Yの支持区間を使い、中心区間の並びからY→atom候補を引く」方式。
明示的な全site→atom対応を持たないため、出力tile内の無効候補確認と
複数tileでのH再生成が残る。全支持indexの反転表を先に採用せず、
まずrouting構築と集約kernelを分けて診断し、Hの再計算と逐次loopも確認する。
出力所有を維持しつつ、小batchのHを一度だけ生成して再利用する有界bufferは次の比較候補であり、
性能／memory改善は未検証。現在の出力所有候補は比較対照として研究branchへ保持する。

| scope | job | source archive SHA256 | result archive SHA256 |
| --- | --- | --- | --- |
| 150 checks | `l4job-84350cfdc1404ee9b70423e82f95c811` | `d1db2d239c19db5d9b1a94bd7ce52d4ff18ed345c3320a9d97e13f0d5f993319` | `f18091474c61988f6b7054129de07e6a8cdefdc41d8d89aecaaa5d10397b5983` |
| 4 v2 cohorts / 24 workers | `l4job-82226d60e9704b588b95f300119776a0` | `791af3a1dbf18a5181e5bded656e16653a64dcdbe70468dbed4bc51159f1033b` | `56a727dd2c574da35d3190b6caac384e4bad4d0d59331b44c5c2583f9006b9c7` |

比較sourceは`5f979468`。NVIDIA L4、driver580.82.07、Torch2.11.0+cu130、
CUDA13.0、Triton3.6.0、Python3.13.15。各case/controlは独立process、各21sampleの1cohort。
source/result archive、全source/result file、snapshotを照合し、ignored
`output/regular-grid-h/output-owned-evidence/`へraw copyを保管した。
再現driverは`owner-validate-driver.py`／`owner-measure-driver.py`、例えば以下を使う。

```bash
PYTHONPATH=src:. python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho8.json \
  --isolated-oracle --phases --output output/regular-grid-h-v2.json
```

追加のtrace専用job `l4job-4ce0482f863f45a2b3ed8246f7660fd8`は、
Colab execの外側が420秒でtimeoutし、pool状態は`interrupted`となった。
trace／結果を回収できておらず、GPU実行の完了も確認できない。
この失敗からsort、候補探索、H再生成の個別費用やkernel停止を推定しない。
成功済みの150検査と4cohort比較は別jobの検証済み結果として維持する。
自動再実行は行わない。終了後、slot1のlifecycleでsession terminatedと
server上のactive sessionなしを確認し、全pool slotは`stopped`。
失敗job、transport／lifecycle logもignored evidenceへ保存した。

## 次の比較：小batchのHを一度生成して出力所有で再利用

`research_cuda_regular_grid_reused_h` / `ReusedHRecipe`は、既存の粗い出力索引を使い、
各batch8ごとに全atomのHを一度生成する。Hはsorted atom position × batch8の
FP32 bufferへ置く。各Y所有区間が同じbufferを読み、全Y所有区間の処理終了後に
次のbatch8で上書きする。振幅はY集約時にUと掛け、Hへ入れない。
H scratchは4K×8 byteで、全batch32分を保存せず、backwardへも残さない。
元のsource/prepared snapshotとbackwardのH/G再計算、dX atomicは維持する。

比較は同じ4case・isolated-oracle-v2・24更新・dense両peak条件で行う。
出力所有／再利用候補は同一sourceで粗い索引構築を独立CUDA Event区間として計測する。
再利用候補では各batch区間のH生成とY集約にもEventを置く。
支持正値の最終候補確認はY集約へ含む。元の融合候補のH/Yは分割できないため、
その融合区間を一つの指標とし、他方式の時間の差し引きでHの費用を推定しない。
Eventのある固定post24 forwardは診断用の別copyであり、主測定の完全stepにはEventを入れない。
性能／memoryの採否は実測後に記録する。

### 小batch H再利用の検証・採否

実装sourceは`8d424a6ec9cf1bc6e63017c5ec72d4eb8fcad285`。L4で182 passed / skipなし
（CUDA検査とCPUで実行可能な宣言／protocol検査を含む）。CPU全体は1651 passed / 2790 skipped。
Plan 6件の宣言往復、Ruff、diff check、CPU CI（wheel/sdist含む）PASS。
H bufferをYへ使用した直後にNaNで埋める検査で、batch開始0/8/16が同じ129×8 bufferを
完全に上書きして端数batchと全勾配を処理することを確認した。既存の正規化／joint floor、
zero amp、strides、retained snapshot、20 eager／Graph更新も通過。

4caseそれぞれ7controlを独立processで同一sourceから再測定。各case/controlは21sampleの1cohort、
同じB32・seed41・5% atoms・IEEE FP32・fused AdamW＋Polar更新・実更新counter24。
全28workerと初期／post24の全48独立FP64 snapshotがPASS。数値gateは変更しない。
最大max_abs=0.0003356502955487173、最大relative_l2=1.006376675101177e-06。

各セルは **完全step中央値ms / allocated MiB / reserved MiB**。旧cohortとの時系列差を
kernelの改善とは扱わず、この表の同じcohort内で比較する。

| N / rho | W＋Torch | W＋Triton | 全H保存factor | atom担当H/G | 出力所有・H再計算 | 小batch H再利用 | dense |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 / 3 | 0.393 / 48.67 / 116 | 0.463 / 16.17 / 36 | 1.279 / 20.96 / 36 | 1.175 / 10.84 / 36 | 2.017 / 10.84 / 36 | 5.422 / 10.84 / 36 | 0.085 / 33.00 / 86 |
| 1024 / 8 | 0.871 / 48.67 / 116 | 0.942 / 16.17 / 36 | 2.740 / 20.96 / 36 | 2.711 / 10.84 / 36 | 3.178 / 10.84 / 36 | 6.282 / 10.84 / 36 | 0.085 / 33.00 / 86 |
| 2048 / 3 | 1.311 / 96.16 / 142 | 1.494 / 63.11 / 122 | 5.211 / 83.11 / 130 | 4.008 / 42.87 / 114 | 8.622 / 42.87 / 114 | 13.348 / 42.87 / 114 | 0.596 / 81.75 / 106 |
| 2048 / 8 | 3.275 / 96.16 / 142 | 3.466 / 63.11 / 122 | 11.335 / 83.11 / 130 | 9.915 / 42.87 / 114 | 13.145 / 42.87 / 114 | 16.459 / 42.87 / 114 | 0.594 / 81.75 / 106 |

次は固定post24 forward copyに外部CUDA Eventを入れた診断の中央値ms。
H生成／Y集約はbatch8の4区間を各sample内で合計してから、その5sampleの中央値を取る。
主測定にはEventを入れず、以下を足し合わせて完全stepの代わりにしない。
候補索引はkeys生成・sort・searchsorted・半径reduction。候補loopと支持正値確認はY側へ含む。
このため「候補探索全体が軽い」とは断定しない。計装forwardは同じsnapshotの
未計装forwardと照合し、Parameter不変も確認した。

| N / rho | 再利用：準備 | 再利用：候補索引 | 再利用：H生成 | 再利用：Y集約・支持確認 | 従来：候補索引 | 従来：融合H＋Y・支持確認 |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 / 3 | 0.072 | 0.045 | 0.116 | 4.549 | 0.045 | 1.244 |
| 1024 / 8 | 0.083 | 0.045 | 0.164 | 4.569 | 0.046 | 1.605 |
| 2048 / 3 | 0.266 | 0.061 | 0.458 | 9.981 | 0.068 | 5.451 |
| 2048 / 8 | 0.313 | 0.062 | 0.655 | 9.992 | 0.069 | 7.013 |

**再利用候補は不採用、研究対照として保持。** 従来の出力所有方式より約25〜169%遅い。
Hの明示scratchは1024で約1.6MiB、2048で約6.4MiBだが、capture/replay込みの両peakは
従来と同じ。1024は10.84／36MiB、2048は42.87／114MiB。2048のreservedはdense106MiBを
依然超える。全CSTはdenseより遅く、1024でdense両peak内の最速CSTはTriton W、
2048は全CSTが両peak gateを満たさない。公開dispatchへの採用やmainへの統合はなし。

確定したのは、今回の再利用scheduleではH生成よりY集約・候補走査が重いこと。
幅3と8で再利用Y集約の時間がほぼ同じで、H生成の時間は変化する。
構造上、1024のY gridは256 CTA一括から64 CTA×4逐次、2048は512から128×4へ変わった。
並列度低下、依存のあるH／Parameter参照、候補loopを原因候補として残す。
物理的なcache miss／stall／spill counterは未取得で、どれが原因かはまだ未確定。
次の比較候補はH bufferのbatch容量とY kernelのbatch tileを分離し、
H再利用を保ってYの同時実行数を戻すablation。未実装・未測定であり、勝つとは扱わない。

初回の測定driver4本はcommit文字列の置換が環境変数名にも及ぶSyntaxErrorで、
benchmark import前に失敗した。性能sampleは得られていない。修正後は4本とも構文検査し、
同じ条件を明示的に再提出した。kernel・oracle・gate・時間budgetは変更しない。
失敗jobは`l4job-a0666e8a4c4343799e4a5307c2c084d4`、
`l4job-299f61a5386849269e76dfa958d874a2`、`l4job-eade8364672a416c8bcc17c237b98d3e`、
`l4job-144915324e144a57888a653cd51439ac`。rawログとarchive／manifest検証を残す。

| scope | job | source archive SHA256 | result archive SHA256 |
| --- | --- | --- | --- |
| 182 checks | `l4job-0be2ce2557e64767887c0b8fd222398b` | `3eb54976cc4a8970b8efe5646a86bb573100a039e6e17c85eee226be51e345ae` | `ddd43eeebd9be553c76a302015ad6904f35bafa64ded5172850a3ee14ec30d63` |
| n1024-rho3 / 7 workers | `l4job-610127c546c24348926bed7e3e5e3f2e` | `b0967d96b558e28c1ce57090d085b2f4894069f44fb16c89cb020e3eda4f4a2e` | `ed14eaf275cd3d716731a8e045b5663f2a0bbb12a596e0de85d25d630bcf6c8a` |
| n2048-rho3 / 7 workers | `l4job-1067a4e4e0044a559267c6067354aaed` | `3338f1e0bbfdcfb028d4be5749236a74e1fb27b9c20531478471b568bd762cc9` | `0de104e2c4c62a7ed20c929bd7a1b28097d82734525a785fcf181282ba5f22a4` |
| n1024-rho8 / 7 workers | `l4job-5f71176cc8cb483b95f29c857f802f5b` | `d8f025277878d41a1297a284aa4a78324192673fd646b3f307af556246efd315` | `e498a4ee3231c458934a23882913f6c11d7e68d4712c739f4762e76ee29d9529` |
| n2048-rho8 / 7 workers | `l4job-3f4bd4c2219c48ecbef66f50ce35dc5b` | `958fe6e78a6402fdd566a1cf17bc6ae106a46a6cf7719533634880b705bffba1` | `9ba49d623eb455b00ec6e09f27e91381f4258cd3845cf216db21db49232a13ba` |

NVIDIA L4、driver580.82.07、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、Python3.13.15。
検証／比較の全source fileはdriverを除き同一。source/result archive・全source file・
全result manifest・48snapshot hash・runtime/benchmarkのlocal source hashを照合した。
ignored `output/regular-grid-h/reused-h-evidence/`へ全job、失敗証拠、集計、停止ログを保存。
再現driverは`reuse-validate-driver.py`と`reuse-measure-{N}-rho{rho}.py`。
既存runnerの前節の`--isolated-oracle --phases`コマンドは今回の7controlを実行する。
比較終了後は全pool slotがstoppedで、lifecycleにはsession terminatedと
serverのactive sessionなしを確認した。実DB取込／別GPU／高次元CUDAは未検証。


## H容量とY計算tileを分離する比較（事前方針）

前節のH8再利用は4chunkを直列に実行し、Y側の並列度が小さくなった。
この説明は仮説であり、次の比較ではYのbatch tile=8、output tile=16、atom group=8、
支持・正規化・backwardを固定したまま、H容量だけ16／32batchへ増やす。
Hは `(capacity / batch_tile, atoms, batch_tile)` のtile順配置にして、各tile内の
atom/batch連続アクセスをH8と揃える。H生成・Y集約とも容量内のbatch tileを
同じlaunchで並列に実行し、次chunkでは同じbufferを上書きする。backwardへは保存しない。

新候補は `reused-h16` / `reused-h32`、新Algorithmは
`research_cuda_regular_grid_parallel_reused_h`、recipeは `h_batch=16/32`。
従来のReusedHRecipeとPlanは維持する。sort/searchsortedの内部workspace上限は未確定なので
workspace_boundはNoneのままとする。明示H scratchは `4 * atoms * h_batch` bytes。

測定条件は既存4Case（N=1024/2048、初期rho=3/8、B=32）とisolated-oracle-v2を維持し、
各Caseで従来7control＋新2候補を独立processで同じsourceから測る。
初期・24更新後のY/dX/全dP oracle、21個の未計装Graph完全step sample、capture/replayの
allocated/reserved peakを判定に使う。post24コピーで候補index、H生成、支持検査込みY集約を
各5sample記録する。これら診断時間は完全stepと加算しない。
まずGPU回帰検証を1job（600秒上限）、その後各Case1cohortずつ（480秒上限）実行する。
失敗・timeoutを自動retryせず、仕様・source・結果archive・oracle snapshotを保存する。
小さい改善だけなら独立確認が必要で、公開dispatchへの採用はこの比較と分けて判断する。


### H容量分離の測定結果・採否

測定sourceは `d14fdaf48dba884c090e67bb7ac1b6a8c6c14764`。
CPU 1659 passed / 2850 skipped、Plan 8件往復、Ruff、diff check、
CPU CI（wheel/sdist含む）PASS。L4は250 passed / skipなし。
追加したH16／H32はstrides、全勾配、joint floor、zero amp、空atom、保持snapshot、
20 eager／Graph更新を通過。B37の端数chunk検査では使用後のbufferをNaNで埋め、
H8は開始0/8/16/24/32、H16は0/16/32、H32は0/32で同じbufferを上書きして
Y/dX/全dPがoracleと一致することを確認した。Hはbackwardへ保存しない。

4Case × 9 control = 36 workerは全PASS。初期とpost24の64個の全軸・全atom FP64
snapshotも変更していない4e-4 gateをPASSした。
全誤差の最大max_abs=0.00033597983959943178、最大relative_l2=1.0063558715220024e-06。
各Case/controlは1cohort、21timing sampleであり、21回の独立runではない。
source、fixture、B32、5% atoms、seed41、FP32設定、optimizer、24実更新を揃えている。
旧7controlの過去cohortとは混ぜず、以下の新しい同一cohort内で比較する。

各セルは **完全step中央値ms / allocated MiB / reserved MiB**。

| N / rho | W＋Torch | W＋Triton | 全H保存factor | atom担当H/G | 出力所有・H再計算 | H8再利用 | H16再利用 | H32再利用 | dense |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 / 3 | 0.395 / 48.67 / 116 | 0.468 / 16.17 / 36 | 1.286 / 20.96 / 36 | 1.186 / 10.84 / 36 | 2.030 / 10.84 / 36 | 5.427 / 10.84 / 36 | 3.346 / 10.84 / 36 | 3.291 / 13.92 / 36 | 0.086 / 33.00 / 86 |
| 1024 / 8 | 0.876 / 48.67 / 116 | 0.943 / 16.17 / 36 | 2.745 / 20.96 / 36 | 2.717 / 10.84 / 36 | 3.227 / 10.84 / 36 | 6.281 / 10.84 / 36 | 4.220 / 10.84 / 36 | 4.190 / 13.92 / 36 | 0.085 / 33.00 / 86 |
| 2048 / 3 | 1.317 / 96.16 / 142 | 1.502 / 63.11 / 122 | 5.252 / 83.11 / 130 | 4.636 / 42.87 / 114 | 8.488 / 42.87 / 114 | 13.578 / 42.87 / 114 | 13.375 / 43.26 / 102 | 12.118 / 55.02 / 104 | 0.594 / 81.75 / 106 |
| 2048 / 8 | 3.313 / 96.16 / 142 | 3.544 / 63.11 / 122 | 11.661 / 83.11 / 130 | 10.214 / 42.87 / 114 | 13.486 / 42.87 / 114 | 16.745 / 42.87 / 114 | 16.642 / 43.26 / 102 | 15.165 / 55.02 / 104 | 0.594 / 81.75 / 106 |

次は固定post24コピー・外部CUDA Eventの5sample中央値ms。
各sample内のchunk時間を合計してから中央値を取る。H8/16/32でYのbatch tileは8、
output tileは16、atom groupは8のまま。H生成／Y集約を実行するchunk数は4/2/1。
粗い索引構築は別区間で、候補loopと支持正値チェックはY集約に含む。
主測定にEventはなく、以下の時間を足し合わせて完全step時間にしない。
計装したforwardと未計装forwardの一致、Parameter不変も全variantで確認した。

| N / rho | H容量 | 準備 | 粗い候補索引 | H生成 | Y集約・候補走査・支持確認 |
| --- | --- | --- | --- | --- | --- |
| 1024 / 3 | 8 | 0.074 | 0.045 | 0.117 | 4.543 |
| 1024 / 3 | 16 | 0.073 | 0.045 | 0.116 | 2.452 |
| 1024 / 3 | 32 | 0.074 | 0.045 | 0.110 | 2.344 |
| 1024 / 8 | 8 | 0.085 | 0.045 | 0.171 | 4.551 |
| 1024 / 8 | 16 | 0.084 | 0.045 | 0.158 | 2.469 |
| 1024 / 8 | 32 | 0.084 | 0.045 | 0.148 | 2.356 |
| 2048 / 3 | 8 | 0.270 | 0.061 | 0.469 | 9.954 |
| 2048 / 3 | 16 | 0.276 | 0.062 | 0.479 | 9.713 |
| 2048 / 3 | 32 | 0.288 | 0.070 | 0.515 | 8.081 |
| 2048 / 8 | 8 | 0.315 | 0.061 | 0.663 | 10.102 |
| 2048 / 8 | 16 | 0.328 | 0.065 | 0.693 | 9.891 |
| 2048 / 8 | 32 | 0.336 | 0.072 | 0.727 | 8.222 |

N1024ではH16がH8より約33〜38%速く、両peakは10.84／36MiBで同じ。
H32のH16に対する上積みは約0.7〜1.7%だけで、allocatedは13.92MiBへ増える。
小差を確定した勝者とはせず、追加の独立確認なしにH32を採用しない。
N2048ではH32の短縮幅がH16より大きいが、Y集約は依然支配的。
小shapeのH生成はほぼ同じで、大shapeではやや増えた。並列batch tile／launch数を
変えたscheduleの主な短縮はY区間にあった。cache miss／stall／spill counterは取得しておらず、機構の内訳は未確定。

明示H scratchはH8/16/32で、N1024が約1.6/3.2/6.4MiB、N2048が6.4/12.8/25.6MiB。
H16／H32のN2048 allocatedは43.26／55.02MiBに増える一方、reservedは102／104MiBと、
H8の114MiBより小さかった。これはGraph capture/replayを含むallocator実測であり、
H tensor自体の容量が減ったという意味ではない。allocator配置の原因は未検証。
新2候補は全4Caseでdense両peak gateを満たした。ただし全CaseでH再計算controlより
遅く、denseよりも遅い。メモリ条件内で最速のCSTはN1024がTriton W、N2048がH32。
この条件依存の候補としてH16／H32を研究branchへ保持する。公開dispatchへの採用・
main統合は行わず、Draft PR #100を維持する。

次の優先対象はH容量をさらに増やすことではなく、Y側の候補走査・U評価・H参照の構造。
小shapeではH16を、largeではH32を対照として使えるが、実行時自動dispatchは実装していない。
H寿命を軸にした設計は保ち、候補indexとY内の探索費用を区別して次のablationを決める。

| scope | job | source archive SHA256 | result archive SHA256 |
| --- | --- | --- | --- |
| 250 checks | `l4job-6c417b7df62c427d802d789bb2983ed9` | `93287869ef3eeca4728f0777c63ee64a30f1958ae283c1d5d352d88b76e2ab93` | `368c1bf8e67bc3e57c34ca2585714f87767d95cb8811846c746944abc300eb27` |
| n1024-rho3 / 9 workers | `l4job-51877bc13edb45caa58c87ff4194ec1a` | `3b1859483fe476d24efa6634e25ad186251589e0bc949a9d2cceb0ff38834685` | `e7ea2af6e624842ce9ac5944da70a825df0d81fae9b348da00ce9dd576745d29` |
| n2048-rho3 / 9 workers | `l4job-88a5e9fa6022447c8960cca4fe050995` | `65a842c16d17811f17b3ea0c703facfdf8251aee9771ac48a51fa1adbfd1c512` | `2c367cd58377d90126cfabf49387b7007678b286e5d91f6575b299c183ce8555` |
| n1024-rho8 / 9 workers | `l4job-52747b918e564940a18ccb386e708a76` | `6eec6dfde2606ed6df7ca41d93d844166b1fcc6c26c23579ce90e5bae3c7e921` | `ea56085b16e755401d1cb78569efa7ad8b29db57fad294cb4393d02213f2f5c7` |
| n2048-rho8 / 9 workers | `l4job-dc790d9d866c456fbe30104ae9788ead` | `a003cf91fefaad5a9f47c290537856545e22c44af2449d2b2443872d66f16706` | `18acba578a591858c790e430437b35e340ed8f0acf043ff92c18fe0ee3122e75` |

NVIDIA L4、driver580.82.07、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、Python3.13.15。
検証／4比較jobの全sourceはdriverを除き同一で、全source/result archive・file manifest・
64 snapshot・runtime/benchmark source hashを照合した。失敗・timeout・retryなし。
ignored `output/regular-grid-h/parallel-h-evidence/`へ全jobとverified-summary.jsonを保存した。
再現driverは`parallel-validate-driver.py`、`parallel-measure-{N}-rho{rho}.py`。
source commit上の以下のコマンドは新9controlを実行する。

```bash
PYTHONPATH=src:. python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho8.json \
  --isolated-oracle --phases --output output/regular-grid-h-parallel.json
```

CPU CI: https://github.com/takumiecd/torchcst/actions/runs/38052982366/job/114215711769
実DB、別GPU、高次元CUDA、独立confirmatory run、physical counterは未検証。

全pool slotのstopped、所有sessionのterminated、serverのactive sessionなしを確認した。
停止ログとpool-final-status.jsonも同じignored evidence directoryへ保存した。


## Y集約の内訳調査（事前方針）

H容量比較に続き、kernelを変更せず、既存workerの初期／post24 snapshotに対する診断を追加する。
既存periodic_comparisonの `--aggregation-diagnostics` がH16／H32のoracle完了後に実行する。
主測定の完全step・Graphメモリpeakには診断を含めない。新runner／公開Planは追加しない。
診断GPUコードはregular_grid_h/diagnostic_kernels.py、host orchestrationはbenchmark側に置く。

候補bin数、8atom group反復数、検査atom数、Uが正のatom数／site pair数、振幅fieldの
groupごとの32byte address-sector数を数える。全ownerを独立CPU FP32列挙と照合する。
このsector数は論理addressの数で、実transaction数やL2 miss counterとは区別する。

同一のH、routing、prepared metadataを固定し、H生成をY-onlyの測定区間から外す。
容量16ではchunkごとに同じH bufferを生成／上書きしてからYのみをGraph計測し、
各sample内のchunk時間を合計する。buffer配置、batch tile=8、output tile=16、atom group=8を固定。
9probe（runtimeそのもの、同じ式のclone、amp/normを先に求めるscale-first、
Hの順に13-field metadataを並べるsorted-P、両方、binary support coefficient、
synthetic H、synthetic coefficientでのgather/reduce、index walk checksum）を比較する。
数学を保つ4probeは初期とpost24の全Yを独立FP64と4e-4 gateで照合し、clone／sorted-Pは
runtimeとbitwise一致も要求する。残りは出力を変える診断で、採用候補／完全step改善とは扱わない。

21回の外部Event測定は順序をrotation／reverseして、Eventをaggregation Graph内へ入れない。
sorted metadata作成のindex_select時間と明示scratch bytesを別途記録する。
PTX、compiler registers／spills／shared memory、static instruction数も保存する。
static codeとruntime resource metadataはphysical stall／cache counterの代わりにはしない。
処理を除いた時間の差はcompiler／register／parallelismの相互作用を含むため、内訳として加算しない。

GPU診断テストを1job（600秒）、その後既存4Case × 全9controlを同一sourceで各1cohort、
各job600秒の事前上限で測定する。主controlの全勾配・正規化gateは従来どおり。
GPU測定の成功／失敗とarchive hashesを保存し、retryで都合のよい結果を選ばない。

初回診断検証 `l4job-a0fe4fd4902a4c1d806b1469ebaacb09` はGPU YとCPU FP64 oracleのdevice不一致で4件失敗した。
clone／sorted-Pのbitwise照合とCPU census照合は通過したが、診断timingには到達していない。
比較Tensorをdetachして同じdeviceへ転送するhost比較だけを修正し、kernel・gate・budgetは維持する。
修正sourceを別commitで検証に明示的に提出する。失敗archive／manifestは照合して保全した。
source archive `bbc774d075e98e48e666990b401d7d004621b128f0acb273b65810f6a081d648`、result archive `261c2c105cd6f57efc43cbe88c7ec7b7c46ea299e9ad16e095f2da6398fb58f1`。


## Y集約の内訳調査（2026-10-10結果）

測定sourceは `8b26df2ba50f87a10445eb266773031c78b935dd`。L4、B32、N1024/2048、
rho3/8、5% atoms、seed41、FP32/TF32 off、fused AdamW＋Polar update、24実更新を固定。
各Case/controlは1cohort、21sampleであり、独立21runではない。
CPU suiteは1659 passed / 2854 skipped（18既存warning）、追加GPU診断テストは4 passed / 170 deselected。
前節の250 GPU checksは前のsourceでの検証で、今回の新4checkと区別する。
CPU CIもSUCCESS: https://github.com/takumiecd/torchcst/actions/runs/38055509930/job/114223015387

36主workerは全PASS。初期／post24の64 FP64 snapshotでY、dX、全atom勾配、正規化を
従来の4e-4 gateで検証。診断の16 snapshot（4Case×2容量×2状態）でも全ownerの
CPU censusが一致。数学を保つruntime含む5variantの全Y、計80比較が独立FP64 gateをPASSした。
診断max_abs最大0.00032577739691497243、relative_l2最大7.394209486711291e-7。
cloneとsorted-Pはruntimeとbitwise一致。診断snapshot hashも主worker snapshotと照合した。
変更した係数順序／並び替えについて、candidate backward・optimizerは未実装／未検証。

### 完全step対照（今回のcohort）

各セルは完全step中央値ms / capture-replay allocated MiB / reserved MiB。
主測定は未計装Graphであり、診断kernelや並び替えscratchを含まない。

| N / rho | W＋Torch | W＋Triton | 全H factor | atom担当H/G | 出力所有H再計算 | H8 | H16 | H32 | dense |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 / 3 | 0.393 / 48.67 / 116 | 0.462 / 16.17 / 36 | 1.276 / 20.96 / 36 | 1.172 / 10.84 / 36 | 2.014 / 10.84 / 36 | 5.413 / 10.84 / 36 | 3.354 / 10.84 / 36 | 3.220 / 13.92 / 36 | 0.085 / 33.00 / 86 |
| 2048 / 3 | 1.295 / 96.16 / 142 | 1.483 / 63.11 / 122 | 5.131 / 83.11 / 130 | 4.415 / 42.87 / 114 | 8.531 / 42.87 / 114 | 13.190 / 42.87 / 114 | 13.235 / 43.26 / 102 | 11.681 / 55.02 / 104 | 0.591 / 81.75 / 106 |
| 1024 / 8 | 0.866 / 48.67 / 116 | 0.941 / 16.17 / 36 | 2.719 / 20.96 / 36 | 2.703 / 10.84 / 36 | 3.210 / 10.84 / 36 | 6.274 / 10.84 / 36 | 4.200 / 10.84 / 36 | 4.119 / 13.92 / 36 | 0.085 / 33.00 / 86 |
| 2048 / 8 | 3.267 / 96.16 / 142 | 3.475 / 63.11 / 122 | 11.217 / 83.11 / 130 | 9.911 / 42.87 / 114 | 12.933 / 42.87 / 114 | 16.048 / 42.87 / 114 | 16.040 / 43.26 / 102 | 14.581 / 55.02 / 104 | 0.593 / 81.75 / 106 |

### 固定post24・Y-only probe

H生成を除外し、同じH/routing/metadataを使う。BM8 / BO16 / GROUP8。
chunk別Graphを外部Eventで計測し、sample内のchunk時間を合計して中央値を取った。
probe順序をrotation/reverseしている。値はms、完全stepとは加算しない。

| N / rho | H容量 | runtime | 同式clone | amp/norm先行 | sorted-P | 両方 | binary支持 | synthetic H | gather/reduce | index walk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n1024-rho3 | 16 | 2.482 | 2.467 | 1.621 | 2.012 | 1.309 | 2.431 | 2.489 | 0.249 | 0.147 |
| n1024-rho3 | 32 | 2.360 | 2.324 | 1.474 | 1.822 | 1.160 | 2.285 | 2.368 | 0.187 | 0.072 |
| n2048-rho3 | 16 | 9.499 | 9.383 | 5.926 | 7.287 | 4.623 | 9.201 | 9.566 | 0.733 | 0.279 |
| n2048-rho3 | 32 | 7.910 | 7.777 | 5.028 | 6.253 | 4.164 | 7.668 | 7.954 | 0.663 | 0.156 |
| n1024-rho8 | 16 | 2.490 | 2.478 | 1.625 | 2.017 | 1.313 | 2.440 | 2.484 | 0.246 | 0.145 |
| n1024-rho8 | 32 | 2.359 | 2.347 | 1.489 | 1.841 | 1.171 | 2.309 | 2.365 | 0.188 | 0.072 |
| n2048-rho8 | 16 | 9.546 | 9.417 | 6.195 | 7.329 | 4.663 | 9.239 | 9.532 | 0.736 | 0.279 |
| n2048-rho8 | 32 | 7.968 | 7.810 | 5.095 | 6.266 | 4.168 | 7.706 | 7.896 | 0.664 | 0.157 |

- amp/norm先行は約35〜38%短縮、sorted-P単独は約19〜23%短縮、両方は約47〜51%短縮。
  係数 `q_u / D * amp` を `q_u * (amp / D)` とした数学を保つ診断である。
  Dは既存の全site joint L2 normalizer＋一度だけのfloorをそのまま使う。
- H読込をsynthetic値で置換しても差は約1%以内で、今回のscheduleでH読込単独を
  第一候補にする根拠は弱い。置換は出力を変え、演算／compilerにも影響する。
  H帯域のphysical上限や他scheduleについての証明ではない。
- binary支持でも支持距離・正値maskの計算は残るため、U評価全体を除いた診断ではない。
  gather/reduceはP読込・U評価を除き、軽いsynthetic係数で全候補のHを読む。
  index walkはOrder ID checksumを出力へ残す下限診断。これらは出力を変え、shared memory、
  register数、layoutも変わる。時間差を排他的な「各部分の費用」に分解／加算できない。
- 今のYはownerごとに一度storeし、atomicを使わない。この測定の遅さはY atomic競合ではない。

### 候補とmetadataアクセス

下表はpost24全ownerの論理件数。batch tileによる重複を除いたcensusであり、
U-positiveは出力支持が正という意味（Hやampが非ゼロという意味ではない）。
site検査は候補atom×16。sectorは振幅fieldのgroup内32byte address区画数で、実transactionではない。

| N / rho | bins / groups | 候補atom | U-positive atom | U-positive pair | 候補/positive atom | site検査/positive pair | 原順sector / sorted sector |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n1024-rho3 | 192 / 19740 | 157284 | 68821 | 313913 | 2.285 | 8.017 | 149733 / 35712 |
| n2048-rho3 | 384 / 78816 | 629145 | 274796 | 1255610 | 2.289 | 8.017 | 614061 / 145161 |
| n1024-rho8 | 192 / 19743 | 157284 | 101332 | 833593 | 1.552 | 3.019 | 149769 / 35715 |
| n2048-rho8 | 384 / 78810 | 629145 | 405054 | 3334234 | 1.553 | 3.019 | 614112 / 146433 |

rho3/8ともBO16単位で3binを訪問し、候補数は同じ。MaxDistanceはそれぞれ5/10だが、
ceil(distance/16)=1のため粗いbinが差を吸収する。rho3は約56%の候補atomがownerの
全16siteでU=0、site検査は正値pairの約8倍。rho8は約36%／約3倍。
規則格子で支持範囲を直接割り出せても、Yでの粗い候補走査が残っている。

sorted-PはHのsorted positionと同じ順にPをコピーする。振幅fieldの論理sector数は約4.2分の1。
ただし、original atom IDのOrder gatherも省いているため、短縮をcache改善だけに帰属しない。
コピーした全13fieldの明示scratchとGraph index_select時間は以下。事前確保bufferへの書込であり、
allocation費用を含まず、candidate完全step peakは計測していない。

| N / rho | sorted P MiB | H16 copy ms | H32 copy ms |
| --- | --- | --- | --- |
| n1024-rho3 | 2.60 | 0.021 | 0.021 |
| n2048-rho3 | 10.40 | 0.072 | 0.075 |
| n1024-rho8 | 2.60 | 0.022 | 0.020 |
| n2048-rho8 | 10.40 | 0.073 | 0.076 |

### compilerとphysical counterの限界

valid probeのstatic `div.rn.f32` はruntime、clone、amp/norm先行、sorted-P、両方すべて32で同じ。
「先行化でGPU除算命令を16分の1にした」とは言えない。依存順／compiler schedule／layoutの
影響が考えられるが、dynamic instruction数とstall原因は未確定。
N2048/rho8/H32のcompiler registers / spills / shared bytesは、runtime115/0/128、
clone114/0/128、先行128/2/128、sorted113/0/128、両方134/0/128。
static ld.globalは原順75／sorted59。PTXとhashを各oracle directoryへ保存した。
compiler spills=2でもPTX ld.local/st.localは0であり、実spill trafficの根拠にはしない。
register数だけからoccupancy／速度を推定しない。

物理counterの取得も既存sourceの別固定job（240秒上限、retryなし）で一度試した。
Nsight Computeは利用可能でmetric queryも通過。N2048/rho3/H32 post24を再構成し、
runtime／先行／sorted／両方のYを確認してからcudaProfilerStart/Stop区間を指定した。
CLI exit=0だがCSVにMetric Name行がなく、counterはUNAVAILABLE。空のreportと全logを保持する。
permission拒否は記録されず、kernelのcapture/match原因は未調査。cache miss／stall／
physical bandwidthは依然不明。Nsight時間を主測定へ代入せず、この結果で追加retryは行わない。

### 次の実装候補と判定

1. 係数の寿命をHから分離し、同一forward中で再利用できる `beta_a=amp_a/D_a` を準備段階で
   一度計算する。今回の局所先行化より本当にdynamic重複が減るかを別Algorithmとして測る。
2. H順のhot metadata（beta、inv、center等）だけを小さく用意する。
   現在の全13fieldコピーはN2048で10.4MiB増えるため、そのまま採用しない。
   3/4fieldなら明示容量は約2.4/3.2MiBだが、完全step両peakと全勾配の検証が必要。
3. centerの整数site順に細分化し、No+1長のprefix境界からownerの候補区間を参照する。
   torus seamでは2区間、全周支持は全atomを一度だけ訪問し、既存guardとexact U評価を保持する。
   per-output CSR edge listを作らず、候補索引の寿命はParameter更新までとする。
   BO16＋MaxDistance5/10ならcenter候補windowは26/36site相当で、粗い3binの48siteより
   小さくできる可能性がある。ただし削減率／速度は未実装・未測定の仮説。

まず1＋2をforward/backward/optimizerの真のcandidateにし、同条件の完全stepとdense両peakで
比較するのが妥当。Hの小batch寿命とY所有は維持する。候補細分化は次の独立変更とし、
効果を切り分ける。公開dispatchへ採用せず、研究候補と診断をDraft PR #100に保持する。

### evidenceと運用結果

| scope | job | source archive SHA256 | result archive SHA256 |
| --- | --- | --- | --- |
| 4 GPU tests | `l4job-c2be2564a1ee4bb7b5d9c585b41be91a` | `a93eced882f19401b7de2f0504b7dab394bb6105a8bba0996c7dee63f18628ec` | `880e71efa6f4b2db8f646748db9a79ee3c3a2d3967e42ecc96130c0fbf11d407` |
| n1024-rho3 | `l4job-883b1a6013df4edc9034f1307673416d` | `250798849246875097d5e68ca7d104d8f09a5b361c08491315782ba7df92cd8b` | `91ec8eba58d7a9d7d0b7e9bf362d11008b7388bd965f619c556f5086e4d1f2f8` |
| n2048-rho3 | `l4job-bf87537f954f450892cc5d046f7397e6` | `49bec144cfd8b70b8e1bd9926cc721271df6b872deaca664ae46cd606f96cb77` | `4aeefc5d7d8774958b1261b30f5eb92149a811ab87b1a0b2f3e0aeb747b82810` |
| n1024-rho8 | `l4job-a8ae652f83804eef9ea87d4b9cca3048` | `6d3e323e716e6973729028b5c566bc8f2b276ed001f62d61a2eee2f99aff62bd` | `d87ee27d727354468c487f37416fe7962799e060ed0bfa7fea69b0704d3336b4` |
| n2048-rho8 | `l4job-b8b53e503a224f27810ac2d6a9571d85` | `761cdf97f025a09a52ab6e2e67e313bbbfb3d2fe7ffdc165b274f15e9d9932a3` | `0dc1539fc3aab08de03bd1e0d2ce485a0ff5bb0dbfbc57477fe9133930d1973c` |
| counter unavailable | `l4job-f8f8297634ae48c3bb5105145000fca0` | `5fad2623d1dd88edf89fdc9b32aedf67032d9e4c2e232a7db8118ac9fe93b132` | `0126affe0e8dfd4db61c35454dbc598cee4b119e49cc365155e5c61b67dda6da` |

成功unit／4比較／counter attemptの6sourceはdriver以外同一。全source/result archive hash、
全file manifest、64主snapshot＋counter anchorの2snapshot、runtime/benchmark source hashを照合。
ignored `output/regular-grid-h/aggregation-evidence/`に全job、PTX、verified-summary、ncu-verifiedを保全。
初回検証失敗も前節のとおり保全し、host device比較だけを別commitで修正した。
成功検証の結果回収後にremote cleanup RPCがtimeoutしたが、結果hashは一致し、旧所有VMを停止。
terminated／server active sessionなしを確認してから新VMで4比較を実行した。測定のretryではない。
最終的にも全slot stopped／所有VM terminated／server active sessionなしを確認。
lifecycle.logとpool-final-status.jsonをevidenceへ保存した。

再現は測定source commitで以下（4Caseを各一度。追加診断はH16/H32 workerだけ）。

```bash
PYTHONPATH=src:. python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho3.json \
  --isolated-oracle --phases --aggregation-diagnostics \
  --output output/regular-grid-h-aggregation.json
```

高次元CUDA、別GPU、実DB、独立confirmatory cohort、candidate完全step／backward、
physical counterによる因果確定は未検証。

## 原因確定を優先する追加診断（事前方針）

2026-10-10のユーザー指示により、実行candidateの最適化に進む前に、Y集約の原因を分ける。
前回の空NCU reportは取り消さず、今回の追加調査ではProfilerStart/Stopとkernel名filterを
外す。準備済みP/routing/HをCPUファイルへ保存し、profile子プロセスはそれをGPUへ転送して
対象kernelを2回だけ起動する。profiling-from-start on、launch-skip 1 / count 1で2回目を取得。
clock/cache control none、単一sampleのhardware診断であり、主測定／完全stepには使わない。
既存sourceのN2048/rho3/H32 post24・4valid variantで取得経路を検査する（480秒固定job）。
これは新しいユーザー指示に基づく、区間指定を変えた追加調査で、同protocolの自動retryではない。

新sourceではsorted-P＋identity ID配列の2variantを追加する。original Orderと同じdtype／
contiguous ID loadを残して、metadataアドレスの並びとID load除去を分離する。両方とも
FP64 Y gateと対応するdirect-index variantとのbitwise一致が必要。
`norm-one`／`sorted-p-norm-one`はnormを1に置換し、variable正規化除算とnorm loadを省く。
支持判定・H・group/reductionは残るが、出力とcompiler resourceは変わる下限診断で、
採用候補にはしない。norm load除去と除算処理の影響が含まれ、厳密な単独component費用ではない。

CPU suite、既存4 GPU診断テスト（狭支持／全周支持、端数chunk、strided X、CPU census）を
先に検証し、測定sourceをcommitする。新診断は既存runnerの1worker、N2048/rho3/H32、
初期／24実更新、同一FP64 gate、21外部Event sampleで一度測る。GPU job上限600秒、retryなし。
取得経路の結果に応じて、同じpost24状態で新controlled variantのhardware counterを一度採取する。
physical counter、SASS、時間の3種類を照合し、断定できる範囲と残る仮説を分ける。

最初のcontrolled probe sourceでSASSのMUFU.RCP、FCHK、精度例外helperへのCALLを確認した。
除算命令の数だけでなく、U=0への精度維持除算が例外経路へ入る仮説を追加する。
`safe-numerator`では、finite positive normかつU=0の入力を1へ置換して同じexact divideを
実行し、商を0へ戻してampを掛ける。U非ゼロとinvalid normは元の式を保つ。
数学を保つprobeとしてruntimeとのbitwise一致とFP64 gateを要求する。
元P／sorted Pの2variantを、新sourceで既存4 GPU checks後にN2048/rho3/H32の既存workerで
初期／post24・21sample、一度だけ測定する（600秒固定job）。SASSも保存する。
前の診断値と混ぜず、新cohort内のclone／先行計算／IDあり／IDなしと比較する。
この追加はzero入力仮説を検証するためで、失敗／測定値を選ぶretryではない。

最後にzero operand仮説を、CSTと独立の同一cubinで検査する。backend診断kernelで
GPU入力のnumerator 0 / 1 / warp内0・1混在と、positive denominator 2を使用する。
non-pure inline exact `div.rn.f32`を256反復し、loop invariant hoistingを禁止。
結果をstoreしてからclock64を読む。BLOCK128、128CTA、同一launch／cubin、外部Eventの
21 Graph sample（回転順）、各sampleのSM cyclesも記録する。出力は厳密な0/128を要求。
SASSのloop内にdivisionを残したことを確認し、PTX／cubin／SASSと全operand hashを保存。
clock64はこのmicrokernelの経過SM cycleであり、CSTのstall counterや完全step時間ではない。
1固定job180秒、retryなし。この結果でCSTへのゼロ精密除算回避の影響を説明し、
physical cache/stall counterを取得したとは扱わない。


## 原因を切り分けた結果（2026-10-10）

最適化実装へ進む前に、metadataの並び／ID参照、精密除算のoperand、compiler codeを分けて調査した。
L4・N2048/rho3/B32/H32・post24、seed41、5% atoms、FP32/TF32 off、既存fused AdamW＋Polar更新。
今回の大きな知見は、**ゼロnumeratorへのexact divideに、命令数では説明できない実行費用がある**こと。
支持範囲外の大量のU=0へ `U / norm` を実行していたことが、Y集約の大きな原因の一つだった。
全費用がこの一因だけで説明できた、hardware stallの割合まで確定した、という意味ではない。

### metadataの並びとID参照の分離

source `e2513cf4` の1cohortで、同式clone7.752ms → sorted-P＋identity ID7.176ms →
ID参照を省くsorted-P6.143ms。前二者は**cubin hashもSASSも完全同一**で、
input metadata／ID値とアドレス配置だけが違う。約7.4%の差は同じ命令列で生じる。
並び替えの効果を全部cacheへ帰属したり、全部ID load除去へ帰属したりしない。
ID load除去の追加短縮は約14.4%だが、load命令数／依存鎖／compiler編成の変化を含む。
先行計算では5.002ms → sorted＋ID保持5.032ms → ID除去4.077msだった。
ID保持での0.6%差は勝敗を確定する小差として扱わない。

### zero numerator回避の対照

source `c51d1dfbbe75cd05950f597997c307e08e071c4d` の別1cohortで、下記を同時測定した。
前cohortと混ぜず、同式cloneを比較基準とする。runtime kernelそのものは8.068msで、
cloneは7.877ms。差を隠さず、probe間の機構比較には同じprobe関数のcloneを使う。
値は固定post24のY-only外部Event中央値ms。H生成／sortingを含まず、完全stepへ加算しない。

| probe | ms | scope |
| --- | --- | --- |
| full | 7.877 | 同式clone |
| scale-first | 5.117 | amp/norm先行 |
| sorted-p-with-id | 7.280 | 並び替え・ID保持 |
| sorted-p | 6.289 | 並び替え・ID除去 |
| sorted-p-scale-first-with-id | 5.137 | 先行＋並び替え・ID保持 |
| sorted-p-scale-first | 4.173 | 先行＋並び替え・ID除去 |
| norm-one | 7.405 | norm=1、出力を変える下限 |
| sorted-p-norm-one | 6.126 | norm=1＋並び替え、出力を変える下限 |
| safe-numerator | 6.063 | zero divide入力を安全値に置換・商を0へ復元 |
| sorted-p-safe-numerator | 4.449 | zero入力回避＋並び替え・ID除去 |
| synthetic-h | 8.098 | H readを置換、出力を変える診断 |

`safe-numerator`はfinite positive normかつU=0の場合だけ、divide入力を1へ置換し、
商を0へ戻してampを掛ける。非ゼロU／invalid normは元の演算を保つ。
このcaseと4 GPU testsの全Yがruntimeとbitwise一致し、独立FP64 gateもPASS。
これだけで7.877→6.063ms、約23.0%短縮。先行計算は約35.0%短縮で、
zero入力回避だけでその全効果を説明できたとは言えない。追加mask／register／layoutと
依存順も変わるため、残差を単独要因の時間として引き算しない。

元のcloneと先行計算とsafe-numeratorはPTX div32、SASS MUFU.RCP9、FCHK32、LDG74で同じ。
norm=1はPTX div32を残したが、SASS MUFU.RCPが9→1、LDGが74→66になった。
その時間は7.405msで、約6.0%だけ短縮。精密divideの命令数／reciprocal数／norm loadの
数だけが支配因という説明は不十分。先行／safeのregistersは128/127で、元の114より多い。
先行はcompiler spills2、SASS STL/LDLが各2あるが速い。static spillコードから実traffic量は推定しない。
SASSに `FCHK` と `__cuda_sm3x_div_rn_noftz_f32_slowpath` の条件付きCALLを確認した。
zero値でどのdynamic pathが何回実行されたかはcounter未取得であり、静的存在だけで断定しない。

### 同一cubin・operandだけを変えるmicro対照

source `397f8f59`。128CTA×128thread、positive denominator2をGPU loadし、
non-pure inline exact divideを256回反復。同じcubin hashを3operandで要求し、
結果0/128は誤差0で一致。外部Event Graph21sampleを回転／逆順で計測した。
SM clock64は256反復＋結果store発行までの経過cycleで、純粋なdivide1個のlatencyやstall counterではない。

| numerator | kernel ms | median SM cycles |
| --- | --- | --- |
| one | 0.025856 | 18734 |
| zero | 0.051904 | 74298 |
| mixed | 0.054336 | 79167 |

zeroは非ゼロの約3.97倍のSM cycles、kernel時間約2.01倍。warp内0/1混在も約4.23倍で遅い。
SASSを手動確認: UR4=0x100で反復初期化、loop `.L_x_2` のMUFU.RCPがPC00f0、
FCHKが0130、slowpath CALLが01b0、FADDが01d0、01e0からloopへbackward branch。
begin/end clockは00e0/0220で、divisionはhoistされず反復区間に残る。
同一registers22/spills0/shared0の同一命令列でoperandだけを変え、zero入力の実行費用を確認した。
精度例外分岐／helper処理という説明と整合するが、FCHKの非公開predicateの真偽を直接計測したわけではない。
Graph時間とSM cycleの比が違うため、microの4倍をそのままCST全体へ適用しない。

これを、実CSTのzero入力回避23%短縮・bitwise同一Y・大量の支持外zeroという観測と合わせると、
zeroへの精密除算は実装上の確定した改善対象になる。前節の「依存順の変更が有力」という説明を
更新し、operandに依存する精度処理の費用も大きな要因として含める。まだ35%の先行計算効果の
全てをzero slowpathで説明したとは扱わない。

### counter取得の境界

前回ProfilerStart/Stop＋name filterで空reportだったため、source22971b63の追加jobで
準備tensorをCPUファイルへexportし、子プロセスはtransferと対象2launchだけにした。
profile-from-start on、filterなし、skip1/count1でもCLI exit0・counter0行だった。
さらにsourcee2513cf4のgeneric torch.sin対照は、config off／filter・skipなし／basic metricsで
2launchの出力PASSだったが、同様にcounter0行。研究kernelの特性だけでは説明できず、
profiling環境／interceptionの問題が残る。permission拒否logはなく原因は未確定。
これ以上同じ空reportを繰り返さず、counter取得を止め、SASSと同一cubin operand対照を使った。
NVIDIA公式CLIのlaunch filter／開始区間の意味も確認した:
https://docs.nvidia.com/nsight-compute/NsightComputeCli/
cache hit率、bandwidth、warp stall、dynamic helper実行回数は依然未取得。

### 検証と次の優先順位

CPU suite1659 passed / 2854 skipped、18既存warning。sourcee2513cf4の追加GPU checksは
4 passed / 170 deselected、sourcec51d1dfbでも4 passed / 170 deselected。
各focused workerは既存baseline完全step／Y,dX,全atom勾配のFP64 gate・24実更新がPASS。
追加Y gateは初期／post24で14＋18件PASS、全owner CPU census一致。新safe variantは
full runtimeとbitwise一致。microは3operandの正確性／同一cubin gateをPASS。
別sourceの同じ4 checksは独立な新test case数へ加算せず、各sourceでの検証と記載する。
各cohortの21sampleは独立21runではない。

sourcec51d1dfbでのbaseline完全stepは11.514ms、allocated 55.02MiB / reserved 104MiB。

これはsafe／sorted candidateの完全step／peakではない。今回candidateのbackward／optimizer／
メモリpeakは未実装・未検証で、公開dispatchへの採用はしない。

次の第一候補は、`beta=amp/D` をprepared metadataの寿命で一度計算し、
Y内では `raw U * beta * H` とすること。大量のzero Uへdivideを繰り返さず、
beta/center/invだけをH順に置いてID依存も省く。元のjoint L2 norm＋一度のfloorは保つ。
ampが0の場合も勾配を捨てず、dX／全atom／norm微分を既存数学契約で検証する。
全13fieldコピーの10.4MiBをそのまま持たず、必要なfieldsの約2.4MiBというtensor案を、
実際のcapture/replay両peakで検査する（reservedの余裕は小さく、容量だけでは合格を保証しない）。
候補indexを細かくする変更は別に検査し、支持外zeroの検査自体を減らす。
Hの小batch寿命・Y ownership・正規化を軸にする方針は維持し、Draft PR #100へ研究診断として保持する。

### frozen sourceとevidence

| scope | job | source archive SHA256 | result archive SHA256 |
| --- | --- | --- | --- |
| isolated ncu unavailable | `l4job-e8bb18e1a95a4fa0ba75269496db3167` | `9b42671a1e321da22ba96017036736e98167b30d0c49265772f96394bd43d42b` | `e1fe698bacead9bc8d8b1aa8a5716208a0de198422364f11accee0378db1bab7` |
| ID/norm controlled probes | `l4job-f8d578a27a964093b1df1996a1d5abba` | `052160de31bacd3624c3865d76ada37ecbe293d7f0377d1e8c3274fdd5015245` | `d8757a08a674d78e1da47ad6156a06ebba853916c0ba536a4508be35f64c75d6` |
| zero input exact probes | `l4job-3c9fe2e892fd4cfc8ea0b69097b6dec1` | `729e23772365ec0bc6520b8c2b33c68513439e9daefe8367e86d69c39b06d49d` | `2bea62e8b254c747a8dbf2b86c4ccdd73cf66620eef05dc05f6db90459a8158f` |
| same-cubin division operands | `l4job-0e5c2c25935e4cc6befdda5906a686aa` | `5f18721b90d50ca5e9fed1638fb0906af180db0db84aaac40964fe90c6fe9861` | `5bf10c6289a9800b7eb3aa9a5b4dfd949a9626eb383c19a2e058a3e239a2ce04` |

全source archive／result archive／全file manifest／worker source hashes／全6 worker snapshot／
全PTX,cubin,SASS hashを照合した。sourceは診断追加ごとに別commitとし、同一sourceと偽らない。
各job失敗／timeout／自動retryなし。NCUはdriver自体成功でも「counter unavailable」と記載する。
ignored `output/regular-grid-h/cause-evidence/`へ全jobとverified-summary、SASS summary、
predeclared protocol、停止ログを保存した。全slot stopped／所有VM terminated／server active sessionなしを確認済み。

既存runnerでfocused probeを再現するコマンド（sourcec51d1dfb以降）は以下。

```bash
PYTHONPATH=src:. python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho3.json \
  --worker reused-h32 --isolated-oracle --phases --aggregation-diagnostics \
  --output output/regular-grid-y-cause.json
```

同一cubin operand対照はignored `division-operand-driver.py` とbackend `division_latency_probe`。
raw exact divideは`is_pure=False`で固定し、cubin一致、SASS loop review、exact出力gateを省かない。
別GPU、別precision、独立confirmatory run、high-D chart CUDA、実DB、candidate完全stepは未検証。

診断source397f8f59のCPU CIもSUCCESS:
https://github.com/takumiecd/torchcst/actions/runs/38058918855/job/114232988065

## β先行計算と3項目整列の実行candidate（事前方針）

2026-10-10、原因調査を受けたユーザー指示により実行経路を追加する。
`PreparedReusedHRecipe(h_batch=16/32)` は一度だけ `β=amp/Su` をexact FP32除算し、
β・inverse width・output centreだけをHと同じsorted positionへ置く（12A bytes）。
Y ownerはoriginal IDを読まず、`coefficient=raw_U*β` で集約する。
既存preparationは `Sv*Su=max(||raw V||₂||raw U||₂, floor)` に分母を配分しており、
Hは `(raw V/Sv)X` のまま小batchで上書き。新3項目・routing・Hはforwardだけの寿命で、
backwardはforward時の元13項目snapshotから再計算する。joint L2単一floorは維持する。
数学的には同一だがFP32演算順序が変わるため、数値一致を仮定せず検査する。

測定前にsourceをcommit固定する。共有L4一台でGPU検証jobを一度（600秒）実行し、
regular-grid全経路の独立FP64 Y/dX/all dP、空・単一・floor、狭/広支持・seam・stride、
B37端数buffer再利用、retained backward、20 eager/Graph公開optimizer更新を要求する。
振幅0／X=0でも残る非zero導関数を追加検査する。失敗時はtimingに進まない。

検証成功後、既存4Case（N1024/2048×rho3/8、B32、5%atoms、seed41、FP32/TF32off）を
各一度の独立job（各900秒）で測る。各Caseの同一source・入力に対し、matrix-torch、factor、
onchip-h、reused-h16/32、prepared-h16/32、denseを別worker processで実行する。
初期／24 real updates後の独立CPU FP64 oracle、21完全step samples、capture/replay込みの
allocated/reserved peaksを要求する。既存runnerの`--isolated-oracle --phases`を使い、
forward診断ではcandidate index、3項目準備、H生成、Y集約を分離する。
別Graphのphase時間は完全stepへ加減算しない。denseの両allocator peak以内を採用前提とする。
既存診断の旧数値とは混ぜず、新cohort内で比較する。raw evidenceはignored outputに保存し、
source/result/file/snapshot hashを検証する。NCU counter追加、fine site routing、CSR化は今回に含めない。
公開dispatchへの採用は別判断、retry・gate緩和・有利なsampleの選択はしない。

### 先行βの浮動小数点範囲guard（追検証方針）

最初のsource `dc635793` は通常4Caseと312 testsを通ったが、追加範囲probeで
floor=1e-40、amplitude_max=1e30、空支持の組合せを検査すると、元H経路は独立FP64の
Y/dX/all dPと一致し、新経路だけYがNaNとなった（`amp/Su` がoverflow）。
これは数学契約の追加ではなく、新演算順序の数値regressionなので修正する。
極端に狭い非空支持では元経路も従来の絶対FP64誤差gateに失敗したが有限だった。
そのprecision failureは保持し、gateを緩和した採用主張は行わない。
初回の小floor/通常振幅probeは全PASS。追加の大振幅probeはbaseline非空精度gateの
失敗を含みdriver exit1となった。いずれもarchive、log、4通常Caseを保存する。

準備kernelで非finite βを検出した場合のみforward flagを立てる。通常βのkernelと
元演算順序のkernelを別specializationで起動し、flagに合う一方だけがYへ書く。
これによりnormal pathのregister scheduleにoriginal ID／除算を含めず、host同期も不要。
flagは4 bytes、新情報の値は引き続き3項目12A bytes、すべてforward寿命。
通常時の余分なflag初期化／空launchもstep費用とピークに含めて測る。

sourceを再commit固定し、同じ共有L4／600秒の全検証jobを一度追加する。
追加4checksは空支持の独立FP64 Y/dX/all dP、元演算順序とのbitwise一致、
非空極端値の有限性／元経路bitwise一致、Graph replay、live振幅を通常値へ戻したforwardを検査。
通常precision gateは変更しない。成功後、同じ4Case／8worker比較を各一度900秒で再実行し、
修正版と旧H経路を同じ新source cohort内で比較する。前sourceの速度を修正版へ流用しない。
旧結果の選別retryではなく、範囲regression修正による新sourceの検証である。
記録の式も実装へ合わせる：βはamp/D全体ではなくamp/Su、Hはraw V Xではなく
(raw V/Sv)Xである。joint floor一つという数学契約は同じで、両側のfloor独立適用はしていない。

修正版source `bb379a27` の全GPU検証は315 PASS／追加capture test 1 FAIL。
overflow時の空/非空支持Y/dX/all dPの有限性・旧順序とのbitwise一致は通り、既存の20回
公開Graph更新もすべてPASS。失敗は新testがcapture前のeager autograd graphを保持していた
`cudaErrorStreamCaptureImplicit`。runtimeを変えず、期待値と実測をdetachしてgraphを解放し、
warmup/captureを同じ明示streamへ統一する。新sourceで600秒の全GPU検証を一度実施し、
全部PASS後に前述4Caseを測る。失敗job/log/source/result hashは保持する。


## β先行計算・整列情報・範囲guardの結果（2026-10-11）

実装・検証・通常4Caseの比較を完了した。最終計算sourceは `3dc762d8639caec388f0de3ed0baf52023faa5be`。
実行kernelは `bb379a27` と同一で、`3dc762d8` は追加capture testの修正だけ。
初回 `dc635793` の通常4Case／範囲probe、失敗した `bb379a27` のcapture jobも別cohortとして保持し、
以下の最終数値へ混ぜていない。βはamp/Su、Hは(qv/Sv)Xで、Sv Suはjoint L2／単一floorのD。
3項目12A bytesと4-byte flagはforwardだけに保持する。backward保存TensorはX、元Parameter、
amplitude_max、元13項目PのままでH／新情報を保存しない。

L4（Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、driver580.82.07）一台を共有poolで使用した。
N1024/2048×rho3/8、B32、5%atoms、seed41、FP32/TF32off、公開fused AdamW＋Polar、24 real updates。
各Case一度の独立job内で8方式を別process測定。各方式21時間sampleを21独立runとは数えない。
初期／post24のY・dX・全atom dPをCPU FP64別processで検査し、32 workers／56 snapshotsすべてPASS。
すべてのworker source hash、Case/Plan、初期入力／Parameter、runtime一致をassessで確認した。
メモリはcapture/replay込みのallocated/reservedで、GPU process usage／physical DRAM trafficは未測定。

GPU環境の全検証は316 PASS、2 warnings（初回cuBLAS context、追加testのcapture外stream警告）。
全勾配、floor、空／単一／狭／広支持、seam、stride、端数H上書き、retained backward、
20 eager／Graph公開optimizer更新、非finite β fallback、同じGraphで通常振幅へ戻す検査を含む。
極端な非空・大振幅で元経路も絶対FP64誤差gateに失敗する既存precision limitは解決したと扱わない。
その領域の追加checkは元順序とのbitwise一致・有限性であり、通常FP64 gateは一切変更していない。
CPU全suite1661 PASS／2922 skip／18 warnings。計算sourceのGitHub CPU validation（wheel/sdist含む）成功：
https://github.com/takumiecd/torchcst/actions/runs/38062440077

### 完全step（ms）

以下は最終cohortの未計装capture/replay中央値。diagnostic stageを足し引きした値ではない。

| N / rho | W＋torch GEMM | saved factor | onchip H | H16旧 | H16新 | H32旧 | H32新 | dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 / 3 | 0.4004 | 1.2939 | 1.1840 | 3.3775 | 2.0708 | 3.3287 | 1.9535 | 0.0877 |
| 2048 / 3 | 1.3152 | 5.1602 | 4.5359 | 13.3422 | 8.6758 | 12.1451 | 8.1090 | 0.5944 |
| 1024 / 8 | 0.8807 | 2.7511 | 2.7178 | 4.2539 | 3.0237 | 4.1305 | 2.7970 | 0.0874 |
| 2048 / 8 | 3.3805 | 11.6054 | 9.9586 | 16.5095 | 11.8624 | 15.0133 | 11.1010 | 0.5961 |

| N / rho | H16短縮 | H32短縮 | メモリ内で最速のCST |
| --- | ---: | ---: | --- |
| 1024 / 3 | 38.7% | 41.3% | onchip-h |
| 2048 / 3 | 35.0% | 33.2% | prepared-h32 |
| 1024 / 8 | 28.9% | 32.3% | onchip-h |
| 2048 / 8 | 28.1% | 26.1% | prepared-h32 |

### 完全step peak（MiB、allocated / reserved）

同一Nではrho3/8とも同じpeakを観測した。新H16/H32の全8比較でdenseの両peak以下。
新情報を追加したためallocatedは旧Hより増えており、「メモリ無料の高速化」とは扱わない。
H16のpeakはbackwardとの寿命重なりにも左右され、12A bytesをpeak差分と同一視しない。

| N | W＋GEMM | saved factor | onchip H | H16旧 | H16新 | H32旧 | H32新 | dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 48.67 / 116 | 20.96 / 36 | 10.84 / 36 | 10.84 / 36 | 11.32 / 36 | 13.92 / 36 | 14.52 / 36 | 33.00 / 86 |
| 2048 | 96.16 / 142 | 83.11 / 130 | 42.87 / 114 | 43.26 / 102 | 44.22 / 102 | 55.02 / 104 | 57.42 / 104 | 81.75 / 106 |

### 分離したforward診断（H32、ms）

固定post24 copyの別Graph、5 samplesの中央値。候補indexは粗いbin構築だけで、
正確な支持checkはY集約に含む。field準備にはflag初期化・非finite検出も含め、
Y集約にはguardの両launchを含む。各phaseは完全stepと非加算的。

| N / rho | candidate index | 3項目＋flag準備 | H生成 | Y旧 | Y新 | forward/loss新 | backward新 | optimizer新 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 / 3 | 0.0461 | 0.0102 | 0.1085 | 2.3542 | 1.0465 | 1.2841 | 0.5151 | 0.1475 |
| 2048 / 3 | 0.0717 | 0.0307 | 0.5192 | 7.9319 | 4.0151 | 4.2322 | 1.9487 | 0.3410 |
| 1024 / 8 | 0.0461 | 0.0092 | 0.1505 | 2.3634 | 1.0783 | 1.3548 | 1.2892 | 0.1475 |
| 2048 / 8 | 0.0737 | 0.0307 | 0.7332 | 8.0292 | 4.2322 | 4.7800 | 4.4237 | 0.3523 |

Y集約は旧H32より約47〜56%短縮し、候補index／H生成／backwardは大筋同じ。
rho3ではYが依然forward最大の処理、rho8ではforwardとbackwardが近い大きさになった。
次の切り分けはY内の候補走査・U評価・H load・積和/reductionと、rho8 backward内訳。
物理cache/stall counterが取得できたとは扱わず、今回のallocator peaksから推測しない。
fine routingのindex容量・構築費用と支持check減少を比較する余地はあるが、今回CSRは追加していない。

### 判断と保全

このsourceのprepared H32は、N2048の両rhoでdense両peak以下のCST候補中最速。
N1024はonchip Hが最速。H16も低メモリ選択肢として残す。W＋GEMMは速いが両Nでdenseを
メモリ超過するので、今回の低メモリ選択から外れる。denseの速度には全Caseで未到達。
一つのL4 cohort／D2 FlatTorus Triweightの結果で、別GPU、高次元実行、実DB、
独立confirmatory cohort、公開dispatcherへの採用は未実施。Draft PR #100に記録しresearch経路に留める。

最終5 jobsのsource/result archive、全1627 source file hashes、result manifest、56 snapshot hashes、
実行workerのsource hashesとlocal runtime bytesを検証し、各job一式をignored
`output/regular-grid-h/prepared-guard-evidence/`へ保存した。初回通常比較と2範囲probeは
`prepared-evidence/`、capture failureは`prepared-guard-evidence/`内の別jobとして保持する。
すべてbounded job、timeoutなし、source修正前後の結果を上書きしていない。
最終的に全slot stopped、所有VM terminated、server active sessionなしをpoolで確認し、
`pool-final-status.json`と`final-stop-lifecycle.log`を保存した。

| 最終job | scope | source archive SHA256 | result archive SHA256 |
| --- | --- | --- | --- |
| l4job-1433115f757c44eb8566bdf75587874c | 316 tests | 4eb10ae2b277d2e7089bcd651ab054ef22e370d3cc39d20f2e4aaf33566afd46 | b8041af1c48085e4e4d10281c30fcc7fc161b0129828a1a90da10ba77c49295a |
| l4job-2ec72a5d8e9543818332207ef070e7a2 | N1024/rho3 | 06886e58d4173af4141d70006143c08f69542c713bcf9e2005c8220d20f47931 | 17e512238b8d51804ee8f0ff46cb84c4a247efd4a113d861433840d3621ea142 |
| l4job-392cc69de78a4d78933f0a8472ceab8b | N2048/rho3 | dcfbb76d633915dc505cd73a8cb1f5164c64e2ea0f754c21a56aefde399b7159 | bbb2e8881d007caee53400c8b6ce0abe9159ecba273c3ea176352c37cb69008d |
| l4job-8b89a14501b3449d9f4736750d86bf01 | N1024/rho8 | 878cd90553e0ff924800b561da257af705b9a13b9f78e75696078b9bd5cc301a | 9332e506771c6b4988080a129bbb69d100004641f37d070be5f94660e7a2c85c |
| l4job-4a63256dcbe641199109b718dbb1e210 | N2048/rho8 | c4e8d0a06d05c03cff836da545503633954afb19eff9a3b2e59e844d088f266b | 41319e34ecdb79e1311e4025fa36fb9aa1b1504a4a4dd8c51d7e8d4b0eddb4fc |

範囲probe jobs：`l4job-7707cd6f969b4b0ca9042e16b2f6915a`（通常振幅全PASS）、
`l4job-40d06253d10f4c69a5036ba60038c010`（旧順序の非空極端値precision FAILを含むexit1）。
修正前capture failure：`l4job-132f9d1b2b1a49079766918bd3cfd8cd`（315 PASS／1 harness FAIL）。
初回通常4 jobs：`l4job-98dcae3f557e479eadd1ff1f349b1532`、
`l4job-28c9b44194cf447d97cc38cb4a77818f`、`l4job-a4e1bdcf996a4dd093bfec7bfb54836b`、
`l4job-634846cd09b9426c8c431f90ec6ee6ed`。hashは各evidenceのverified JSONへ保存。

再現は計算sourceで（各Caseを一度、rhoとNだけ変更）：

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_regular_grid_h_cuda.py tests/test_periodic_cuda.py
PYTHONPATH=src:. python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho3.json \
  --plans matrix-torch factor onchip-h reused-h16 reused-h32 prepared-h16 prepared-h32 dense \
  --isolated-oracle --phases --output output/regular-grid-h-prepared-final.json
```

## 残る集約・backwardの切り分け（2026-10-11）

ユーザーの継続依頼を受け、guard付きprepared Hを基準として夜間調査を続ける。
出発点はcommit `f206a24d` の結果であり、denseとの速度差はまだ残る。
Hの小batch寿命、単一joint normalization floor、全atom勾配を維持する。

最初の固定診断はN2048、B32、rho3/8、prepared H32の2条件とする。
既存runnerのfull oracle、24実更新、未計装完全stepの後で、initial/post24 snapshotを調べる。
診断用Graphは主測定・capture/replay memoryから分離し、時間を足し引きしない。

- prepared式のcloneを実際のguard付きruntimeとbitwise照合する。
- profile評価を簡単な係数に置換、H読取りを合成値に置換、index-onlyの3条件は
  出力を変える下限診断。選択可能なPlanや採用根拠にはしない。
- exact式でoutput atom groupを8から32へ増やす。また各候補列を4/8分割し、
  atomicなしの部分Yを最後に加算する。group32+split4も固定候補に含める。
  reduction順序の変更はFP64 oracleの既存4e-4 gateで検査する。
  全Hは保持せず、同じ小batchのHを全分割で再利用する。
- 既存backwardのfull/input-only/parameter-only specializationを比較する。
  input-onlyはG計算+dX zero/scatter、parameter-onlyはG/dG/H/dH+partial store。
  specializationによるcompiler scheduleの違いも含むため、差をatomic stallなどの
  物理的な原因に直結させない。共通dXの一致、parameter partialのbitwise一致を確認する。

有望なexact候補のみ明示的な研究Planへ実装し、GPU契約テストの後、同一条件で
基準・candidate・denseの完全stepとallocated/reserved総peakを測る。
メモリは両peakがdense以下という既存方針を維持する。CSRは導入しない。
GPUは共有poolのL4を1台使用し、NCUの空counter報告を繰り返さない。

初回診断source `5fde5748` のjob `l4job-e29b995705e6420697ce4c57077fb7fb` は、
動的Tensor長にconstexpr用 `tr.cdiv` を使ったcompile errorで小さい2検証が失敗した。
大規模性能測定へは進んでいない。修正sourceは `3be51957`。
誤ったsource hintを指定したqueued job `l4job-2c8fcb06b2e646928ae3d7a3e52e7551` は
実行前にcancelし、正しいhintで別jobを提出した。失敗archiveは
ignored `output/regular-grid-h/overnight-evidence/` にhash照合して保全した。

修正後job `l4job-78d759e05dfe4cdc95dce0bb34933680` は小fixture 2検証と
N2048/rho3・8のfull oracle、24更新、initial/post24診断をPASSした。
source archive SHA256 `d580e562577cf1b385148c268dfe7c9e49326ee780cfe06be0ce0a3648a8c251`、
result archive SHA256 `851249f2e0ec72a3a6bc96fe9768f343ae153a1b5c19e2b8772f4d39140d47c4`。
1627 source files、result manifest、worker source hashes、snapshot hashesを照合済み。
実機はL4 / driver580.82.07 / Torch2.11.0+cu130 / Triton3.6.0。

| post24 Y-only診断 [ms] | rho3 | rho8 |
| --- | ---: | ---: |
| 実runtime | 4.088 | 4.156 |
| prepared clone | 4.018 | 4.087 |
| Hを合成値へ置換（出力変更） | 3.816 | 3.857 |
| profileを簡単な係数へ置換（出力変更） | 0.629 | 0.637 |
| index-only（出力変更） | 0.093 | 0.095 |
| output group32 | 2.085 | 2.107 |
| split4＋最終加算 | 3.564 | 3.614 |
| split8＋最終加算 | 3.507 | 3.554 |
| group32＋split4＋最終加算 | 1.912 | 1.932 |

prepared cloneは実runtimeとbitwise一致。5つのexact式は全Yの既存oracle gateをPASS。
各時間は21サンプル中央値で1独立job。profile置換はmetadata load、support mask、
compiler配置も変えるため「profile命令だけの費用」やcache miss率ではない。
H読取りを外す差は小さい。group32はgroup8よりregistersが128→217、shared128→1024 bytesへ
増えたがspillsはいずれも0。occupancy/stall counterは測っていない。

追加scratchなしで約半減したgroup32を研究候補へ進める。split4/8のみは今回不採用。
group32+split4の追加効果は小さく、部分Y・加算launchが増えるので今回の候補から外す。
H生成/backwardのatom_group8は維持し、output_group32のみ別Recipe/Algorithm/Planにする。
β overflow fallbackは元のgroup8・演算順序を使う。

| post24 backward診断 [ms] | rho3 | rho8 |
| --- | ---: | ---: |
| full | 2.222 | 4.836 |
| input-only（G＋dX zero/scatter） | 1.884 | 4.637 |
| parameter-only（G/dG/H/dH＋partial） | 1.474 | 1.908 |
| parameter reduction | 0.044 | 0.042 |

共通dXは既存gateで一致、parameter partialはbitwise一致、full dX/all_dPもoracle PASS。
rho8ではinput-onlyがfullに近く、dXを作る経路の改善が次の構造的候補になる。
atomic競合の物理stallを確認したわけではなく、G・profile評価・scatterを含む経路の診断。
これらの時間は加減算しない。

### 入力site担当のbackward候補

group32実装source `7fcf7fe3` はGPU394 testsをPASS（272.10s、2 warnings）。
CPU1669 passed /2992 skipped。検証job `l4job-409ac26d0c7344cbba5a13c1abb87842`、
source `ab894e81257df2a74db3be8a40d050e969d0ff69bcb210b3723ea39f0bd4723e`、
result `5c36b462953cb8a4f6cf74d664a5c9e4e990f3975fe164093ce2638f74cc80e7`。
この固定sourceで4条件の完全step比較を提出した後、別の候補としてinput-ownerを追加する。

input-ownerは既存G contractionをinput中心順に実行し、Gとparameter partialを作る。
そのGからdXを入力site担当で集約する。G容量16/32、input tile8、output group32を固定する。
新規support ID list/CSR、forward H/G保存、正規化やoracle条件の変更は行わない。
parameter-onlyの場合は従来のkernelを使う。partialは元のatom IDへ書く。

全GPU契約テストとGのchunk上書き・部分入力tileを検証してから、N2048 rho3/8に対し
同じsource内でgroup32 control・input-owner candidate・既存方式・denseを比較する。
時間は21サンプル、24実更新、全atom/全siteのinitial/post24 oracle、両allocator peak<=dense。
試作が遅い／メモリ条件を外れる場合も結果を保全し、条件を変えて採用扱いにしない。

3つ目のgroup32比較（N1024 rho8）は結果archive・全manifest・oracle snapshotを取得照合後、
remote cleanup要求の接続が切れた。supervisorはcleanup CLIの180s timeout後に終了1となり、
所有VM `cst-pool-0baa66a66ecb-1` を停止した。lifecycleの `Session terminated` と
`No active sessions found on server` を確認した。測定job自体は成功済み・結果保全済みであり、
再測定は行わない。残るqueued N2048 rho8比較とinput-owner検証を新supervisorへ渡した。
状態DBの編集・pool所有sessionへの直接操作は行っていない。
停止記録は `output/regular-grid-h/overnight-evidence/interrupted-stop-lifecycle.log`。
`interrupted-supervisor-status.json` は再開直後のsnapshotで、停止確認そのものはlifecycleを使う。

### group32の4条件完全step結果

計算source `7fcf7fe3`。各Caseは8独立worker、各workerは24実更新と21 replay samples。
全7 CST route×initial/post24×4Case =56 full oracleをPASS。denseは別parameterizationのengineering control。
これは各Caseにつき1 cohortで、source hintだけでなくarchive・worker source bytes・snapshotを照合した。
N2048 rho8は上記transport切断後の別L4 VMで測った。Case内比較は同じGPU/runtime。

| N/rho | Wtorch | factor | onchip | H16 | g32 H16 | H32 | g32 H32 | dense [ms] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024/3 | 0.3974 | 1.2860 | 1.1790 | 2.0666 | 1.5461 | 1.9356 | 1.4039 | 0.0867 |
| 2048/3 | 1.3027 | 5.2011 | 4.6211 | 8.5514 | 6.1451 | 8.0059 | 5.6803 | 0.5921 |
| 1024/8 | 0.8752 | 2.7412 | 2.7114 | 2.9662 | 2.3792 | 2.7875 | 2.2477 | 0.0865 |
| 2048/8 | 3.2286 | 10.9636 | 9.7287 | 11.1455 | 8.7670 | 10.5733 | 8.4900 | 0.5933 |

H32の追加改善は27.5% /29.0% /19.4% /19.7%。group32は追加Tensorを作らないため、
両allocator peakはmatched prepared controlとbyte単位で同じだった。
H16も改善し、H32より低いscratchを必要とする代替として保全する。

| N | H16/g32 H16 allocated/reserved | H32/g32 H32 allocated/reserved | dense allocated/reserved [MiB] |
| --- | ---: | ---: | ---: |
| 1024 | 11.32/36 | 14.52/36 | 33.00/86 |
| 2048 | 44.22/102 | 57.42/104 | 81.75/106 |

両peak<=denseのCSTから選ぶと、N1024 rho3はonchipを維持し、残り3条件はg32 H32を選ぶ。
Wtorchは速いが今回のmemory条件を外れる。N2048のfactor/onchipも条件を外れる。
公開dispatcherは変更していない。denseはすべてのCaseでまだ速い。

| post24 H32 Y-only stage [ms] | prepared | g32 |
| --- | ---: | ---: |
| 1024/3 | 1.0435 | 0.5161 |
| 2048/3 | 3.9885 | 2.0285 |
| 1024/8 | 1.0588 | 0.5151 |
| 2048/8 | 3.8973 | 2.0224 |

Y stageは約48–51%短縮。H生成・routing・backwardを削減した結果ではない。
各stageは別Graphの固定snapshot診断で、完全stepの内訳として足し合わせない。
次のinput-owner候補は別計算source `e2a4984d` として検証する。

| job / Case | source archive SHA256 | result archive SHA256 |
| --- | --- | --- |
| l4job-805d8c75be824407ada3a87f79b6246b / 1024/3 | `8b842de87a57d8c77b3d0da839cc2f4f15b04dedc92aba84e0d0f704e64d2a1a` | `468f4ec3f74d449eb072cf1b4866866df7924781e176ae9a098bf569022d44b8` |
| l4job-e7f3f86298a84617b5ad2da6feb141ba / 2048/3 | `8b842de87a57d8c77b3d0da839cc2f4f15b04dedc92aba84e0d0f704e64d2a1a` | `8246277d73d85b84af18f5aba19d1769da215d4a6a7e116ba8b24b538656e941` |
| l4job-5a816127611243d1a8580e1cfb350a00 / 1024/8 | `79c26b105a77f30c3b77e68cf0264a4c852b918492e6b8cfe1816a3e1b8e7e00` | `3d79f668a27de50b00e6d592a485097dfba82c63424845632c017255dec654f6` |
| l4job-1645726de64149a9908034d7e6e34f8e / 2048/8 | `007b332dad43a2b79c1c24e6e6d5344120a6b3eca056aee49d9678fff461b21d` | `cd32c405be7713dbd72bdfbce2020bae2b638c7711d3e214d48cb88d0452c3cb` |

再現は既存runnerを使う（N/rhoを4つのCaseに置換）。

```bash
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho8.json \
  --plans matrix-torch factor onchip-h prepared-h16 prepared-h32 \
          prepared-g32-h16 prepared-g32-h32 dense \
  --isolated-oracle --phases --output output/regular-grid-h/grouped-comparison.json
```

GPU回帰testは `python -m pytest -q tests/test_regular_grid_h_cuda.py tests/test_periodic_cuda.py`。
raw evidenceは全jobの凍結source・結果archive・manifest付きで
`output/regular-grid-h/overnight-evidence/` に保全した。

input-owner計算source `e2a4984d` はGPU環境で全478 testsをPASS（324.30s、2 warnings）。
CPUは1677 passed /3068 skipped /18 warnings。GPUではGの同一buffer上書き、input tile8/16/32/64、
partial batch、dX-only/parameter-only分岐、retained snapshots、empty/single/floor/広い支持、
β overflow fallback、20 eager/Graph更新を含む既存全契約を確認した。
検証job `l4job-09b05c9b286941d1abf98c2b336cad51`、source archive
`042dc58fc7d6a36fd2ba65a4fd709927e5544f5c03d9733207f530ae1a3f474a`、result archive
`cb71347441e2bbd2e2815477eb222029ef2f0b82382d458efadd15a6b138b311`。
1629 source filesと全result manifestを照合済み。GitHub CPU validationも`e2a4984d`をPASS：
https://github.com/takumiecd/torchcst/actions/runs/38067291396 。

この検証後にN2048/rho3・8の2 comparison jobを提出した。source hintは計算checkpoint
`e2a4984d` とし、現在の追加commitは研究ノートだけでruntime/runner bytesは同じ。
各Caseで10 workers（Wtorch/factor/onchip、prepared H16/32、group32 H16/32、input-owner16/32、dense）を
独立processで比較する。GPU correctness PASSだけで速度・memoryの採用判断はしない。

### input-ownerのpaired結果とmemory不採用

計算source `e2a4984d`、各Caseは10独立workers・24実更新・21 replay samples。
9 CST×initial/post24×2Case =36 full oracleをPASS。測定直後にruntime/runner bytesと凍結sourceの一致を確認した。
全workerの入力・初期Parameter・Case・source・declaration・runtimeを同じcohort内で照合する。

| N2048/B32 [ms] | rho3 | rho8 |
| --- | ---: | ---: |
| Wtorch | 1.2995 | 3.2491 |
| factor | 5.1651 | 11.1742 |
| onchip | 4.4713 | 9.9355 |
| prepared H16 | 8.3304 | 11.3550 |
| prepared H32 | 7.5250 | 10.6973 |
| group32 H16 | 5.8278 | 8.9054 |
| group32 H32 | 5.7090 | 8.5980 |
| input-owner G16/H16 | 6.2938 | 7.9252 |
| input-owner G32/H32 | 6.1292 | 7.5581 |
| dense | 0.5924 | 0.5942 |

| peak allocated/reserved [MiB] | rho3/8共通 |
| --- | ---: |
| group32 H16 | 44.22/102 |
| group32 H32 | 57.42/104 |
| input-owner G16/H16 | 58.72/116 |
| input-owner G32/H32 | 70.72/126 |
| dense | 81.75/106 |

rho3のinput-ownerは遅い。rho8はG32で約12.1%短縮したが、両容量ともreservedがdenseを上回る。
したがって現版input-ownerは両Caseで不採用。研究用の明示Planと失敗条件を保全し、
メモリ基準を緩めたり、allocatedだけで通った扱いにしない。
post24の独立backward診断はrho8でgroup32 H32の4.250ms→input-owner G32の3.121ms。
これも完全stepへ加減算せず、atomic stallのhardware計測とは呼ばない。

N2048 rho3 job `l4job-636f6720aedf4e0485aa6254150179df`、result
`f1cec8a68c61e082eca12915476d82b931d2c5917708eb94a9eb41843626f114`。
rho8 job `l4job-c5200f75d6004a2aac046340f29b55f1`、result
`726ce0702f9c03de79c403a793bbb60d89cdb38f0a99d99c87a002899f08d03b`。
両source archiveは `fd303b44e45281851a3455461f50b5339d2a28404f5a369b7a0c7659c2b2f9a7`。
全1629 source files、result manifest、worker source hashes、全36 snapshot hashesを照合済み。

### Gとparameter partialをchunk内で畳む次候補

次は別Recipe/Algorithm/PlanでG容量8を独立に指定する。forward H容量16/32は維持する。
各G chunkのparameter partialを3Aのphysical sumへ畳み、同じpartial bufferを上書きする。
全chunkを終えてから元のPolar source VJPを一度適用する。G・partialの同時寿命を縮める。
数式は同じで、FP32の加算順序変更を全勾配oracleで検証する。

再び全GPU契約検証を通してからN2048 rho3/8のpaired cohortを測る。
input-owner旧版の不採用結果は変更しない。Case・24更新・21 samples・4e-4 gate・
両peak<=denseは前と同じで、batch/atom数/支持を減らして比較しない。

streaming計算source `547d1dcf` のCPU検証は1685 passed /3144 skipped /18 warnings（30.62s）。
L4では全562 testsをPASS（389.76s、2 warnings）。BM4/8/16、partial batch、
同一G/partial buffer上書き、dX-only/parameter-only、floor/単一/広い支持/周期境界、
β overflow fallback、retained snapshots、20 eager/Graph更新を含む。
GPU検証job `l4job-f658cb5fcef04329ae540f17923e424e`、source archive
`804d00fc279f60b4a35a38bdb3219df489999b04be87e8a71a280f504095c6f9`、result archive
`4f2412380d4499c83a212af5846cfc2a7768f77a300d78a649beccea4862d28f`。
全1629 source filesとresult manifestを照合済み。GitHub CPU検証も同commitでPASS：
https://github.com/takumiecd/torchcst/actions/runs/38069619720 。

このPASS後にN2048 rho3/8の12方式paired cohortを提出した。前の10 controlsに
stream G8/H16・G8/H32を加える。Case・独立process・24更新・21 replay samples・全atom oracle・
allocated/reserved両peak<=denseは同じ。結果を確認するまで採用とはしない。

streaming rho3 jobは全12 workersとresult download/manifest照合を完了した後、
poolのremote cleanup `exec --timeout30` がhost側180sでtimeoutした。
supervisorは以後のdispatchを止め、所有VM `cst-pool-4f11afbc85b7-1` を停止し、
serverの `No active sessions found` を確認して終了した。rho3 jobはsucceededのまま保全する。
`streaming-interrupted-stop-lifecycle.log` と、再開前に取った `streaming-interrupted-status.json` を
ignored evidenceへ保存した。queuedのrho8のみ新しいsupervisorへ引き継ぐ。
計算のretry・再budget・好ましい結果の選択は行わない。

### G8とphysical部分和のpaired結果

計算source `547d1dcf`。N2048/B32/A209715、各Case12独立workers・24更新・21 replay samples。
11 CST×initial/post24×2Case=44 full oracleをPASS。Case・入力・初期Parameter・全declaration・
runtime・source hashを同cohort内で照合し、現在のruntime/runner bytesも凍結sourceと照合した。

| 完全step [ms] | rho3 | rho8 |
| --- | ---: | ---: |
| matrix-torch | 1.3155 | 3.2730 |
| factor | 5.1648 | 11.1084 |
| onchip-h | 4.4553 | 9.8378 |
| prepared-h16 | 8.0490 | 11.3693 |
| prepared-h32 | 7.9262 | 10.6602 |
| prepared-g32-h16 | 6.1404 | 8.9497 |
| prepared-g32-h32 | 5.6655 | 8.6506 |
| input-owned-h16 | 6.4177 | 7.8512 |
| input-owned-h32 | 6.2003 | 7.6426 |
| stream-g8-h16 | 6.4463 | 8.0981 |
| stream-g8-h32 | 6.4045 | 7.9083 |
| dense | 0.5923 | 0.5907 |

rho3の総peakと採否：

| route | allocated / reserved [MiB] | dense以内 |
| --- | ---: | --- |
| matrix-torch | 96.16 / 142 | FAIL |
| factor | 83.11 / 130 | FAIL |
| onchip-h | 42.87 / 114 | FAIL |
| prepared-h16 | 44.22 / 102 | PASS |
| prepared-h32 | 57.42 / 104 | PASS |
| prepared-g32-h16 | 44.22 / 102 | PASS |
| prepared-g32-h32 | 57.42 / 104 | PASS |
| input-owned-h16 | 58.72 / 116 | FAIL |
| input-owned-h32 | 70.72 / 126 | FAIL |
| stream-g8-h16 | 47.12 / 102 | PASS |
| stream-g8-h32 | 57.42 / 104 | PASS |
| dense | 81.75 / 106 | PASS |

両peak条件を満たすCST内の最速は `prepared-g32-h32`、5.6655ms。denseは0.5923msで引き続き速い。
段階診断（別Graph、完全stepへ加減算しない）：

- prepared-g32-h32: forward_loss=2.5713ms, backward=1.9282ms, optimizer=0.3072ms。
- input-owned-h32: forward_loss=2.5743ms, backward=2.2651ms, optimizer=0.3062ms。
- stream-g8-h32: forward_loss=2.5784ms, backward=2.4433ms, optimizer=0.3113ms。

rho8の総peakと採否：

| route | allocated / reserved [MiB] | dense以内 |
| --- | ---: | --- |
| matrix-torch | 96.16 / 142 | FAIL |
| factor | 83.11 / 130 | FAIL |
| onchip-h | 42.87 / 114 | FAIL |
| prepared-h16 | 44.22 / 102 | PASS |
| prepared-h32 | 57.42 / 104 | PASS |
| prepared-g32-h16 | 44.22 / 102 | PASS |
| prepared-g32-h32 | 57.42 / 104 | PASS |
| input-owned-h16 | 58.72 / 116 | FAIL |
| input-owned-h32 | 70.72 / 126 | FAIL |
| stream-g8-h16 | 47.12 / 102 | PASS |
| stream-g8-h32 | 57.42 / 104 | PASS |
| dense | 81.75 / 106 | PASS |

両peak条件を満たすCST内の最速は `stream-g8-h32`、7.9083ms。denseは0.5907msで引き続き速い。
段階診断（別Graph、完全stepへ加減算しない）：

- prepared-g32-h32: forward_loss=2.7976ms, backward=4.2168ms, optimizer=0.3082ms。
- input-owned-h32: forward_loss=2.8938ms, backward=3.1048ms, optimizer=0.3103ms。
- stream-g8-h32: forward_loss=2.8017ms, backward=3.4068ms, optimizer=0.3092ms。

source/result archive：

| job | source SHA256 | result SHA256 |
| --- | --- | --- |
| l4job-f021c54d21d74decabcfe6a6f33d225e | `037358aed05a2a35bd459c79c5196712cd5234423d98c05181e8955479b3f0e9` | `065799a53ecc270937940dc7893e34d6e5870e142647b301ab2a40b971b23ca0` |
| l4job-5d8e77998bb14d59abc1b6353f4cd81f | `037358aed05a2a35bd459c79c5196712cd5234423d98c05181e8955479b3f0e9` | `79f6711ebc2871aa51d0b943d681dfe43774e07291a3977e9165595239df9f1b` |

全source files、result manifest、worker source hashes、initial/post24 snapshotsを照合し、
raw evidenceを `output/regular-grid-h/overnight-evidence/` に保全した。

再現は既存runnerのCaseをrho3/8に置換する。

```bash
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho8.json \
  --plans matrix-torch factor onchip-h prepared-h16 prepared-h32 \
          prepared-g32-h16 prepared-g32-h32 input-owned-h16 input-owned-h32 \
          stream-g8-h16 stream-g8-h32 dense \
  --isolated-oracle --phases --output output/regular-grid-h/streaming-comparison.json
```

全測定・結果照合後、pool supervisorは正常終了した。全3 slots stopped、待機／実行jobなし、
所有VM `cst-pool-c051b85cbf04-1` のterminatedとserverの `No active sessions found` を確認した。
最終statusと停止ログはignored evidenceの `final-pool-status.json` と `final-stop-lifecycle.log` に保全した。
成功した比較は各Case一度だけで、transport失敗後にも測定を再実行していない。

## 中心site prefixで候補範囲を絞る（事前方針、2026-10-11）

coarse binの候補を減らす別Algorithmを追加する。centreのphase/site整数計算は元と同じで、
BOで割らずsite keyをsortし、N+1 prefixを作る。実在owner区間を既存MaxDistance Dで左右に
広げ、周期境界で最大2区間を走査する。Dの+2 guard、raw profile、norm/VJP、β overflow時の
原式とGROUP8 fallbackは維持する。input-ownerは引き続きcoarse routingを使う。

新Planはsite-routed H16/32とsite-stream G8/H16/32。H/G寿命と元の4 saved tensorsは同じ。
BO16/GROUP32/BM8/atom producer8を固定し、探索精度だけを変える。
支持を近似したり短く切ったりしない。full-axis/broadでは全候補を一度ずつ処理する。
小さいNO<BO、partial owner、seam、empty/single、large-coordinate支持のcoverageを確認する。
GPUでは既存全契約と、中心key/prefix/最大距離・独立円周距離predicateとの候補集合一致、
prepared span包含、input prefixがcoarseであることを追加検証する。

全GPU契約PASS後にN2048/rho3・8を同じCaseで比較する。cohortはWtorch/factor/onchip、
group32 H16/32、stream G8/H16/32、新site H16/32、新site stream G8/H16/32、denseの12独立workers。
旧prepared/旧input-ownerの不採用記録は残し、今回再実行しない。
24実更新、21 samples、全atom FP64 initial/post24 gate4e-4、allocated/reserved両peak<=denseは不変更。
診断はsnapshot/prep、candidate index、3項目prep、H生成、Y集約を分け、完全stepへ加減算しない。

### Centre-site prefixのpaired結果（2026-10-11）

計算source `93ffcb2b`。Astraが実装し、独立レビューでcircular interval・
H layout・入力側のcoarse routing維持を確認した。CPU1690 passed /3298 skipped、
L4全721 contract tests、GitHub CPU validationをPASS。
GPU検証job `l4job-50c3bd96dd95480dbabd8ad1fb4f07ca`、source `80d68d53afafec83474d8a1095cdd22dc42dc95831d4c0a73b18d844c6a8f66f`、
result `4182cece4a2af9cb00714503c31b47a90b1bcc48f9eae62e607ba9c21256afd0`。

rho3/8の12-worker比較で44 full initial/post24 FP64 oracleをPASS。
同cohortの入力・初期Parameter・Case/catalog・runtime/sourceと、実測時の
current runtime/runner bytesを照合してから次の変更へ進んだ。


N2048 / B32 / rho3.0 / A209715、24更新。

| route | 完全step median [ms] | allocated / reserved [MiB] | 両peak≤dense |
| --- | ---: | ---: | --- |
| matrix-torch | 1.2978 | 96.16 / 142 | FAIL |
| factor | 5.1366 | 83.11 / 130 | FAIL |
| onchip-h | 4.3900 | 42.87 / 114 | FAIL |
| prepared-g32-h16 | 5.4835 | 44.22 / 102 | PASS |
| prepared-g32-h32 | 5.7074 | 57.42 / 104 | PASS |
| stream-g8-h16 | 6.5210 | 47.12 / 102 | PASS |
| stream-g8-h32 | 6.2027 | 57.42 / 104 | PASS |
| site-routed-h16 | 4.8192 | 44.23 / 102 | PASS |
| site-routed-h32 | 4.8714 | 57.43 / 104 | PASS |
| site-stream-g8-h16 | 5.2653 | 47.12 / 102 | PASS |
| site-stream-g8-h32 | 5.4040 | 57.43 / 104 | PASS |
| dense | 0.5932 | 81.75 / 106 | PASS |

このcohortで条件を満たすCSTの最速は `site-routed-h16`。独立runは各Case1回、各route21 replay samples。denseは依然比較基準。

別Graphによる段階診断。完全step時間へ加減算しない。
| route | 候補index | H生成 | Y集約 | forward/loss | backward | optimizer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prepared-g32-h16 | 0.0645 | 0.4803 | 2.2292 | 2.8764 | 1.8749 | 0.2990 |
| prepared-g32-h32 | 0.0666 | 0.4987 | 1.8985 | 2.5559 | 1.9087 | 0.3144 |
| stream-g8-h16 | 0.0655 | 0.4813 | 2.2620 | 2.8743 | 2.3644 | 0.2959 |
| stream-g8-h32 | 0.0655 | 0.4987 | 1.8842 | 2.5600 | 2.3921 | 0.3123 |
| site-routed-h16 | 0.0768 | 0.4997 | 1.2390 | 1.9558 | 1.8780 | 0.2939 |
| site-routed-h32 | 0.0809 | 0.5233 | 1.1684 | 1.8012 | 1.9077 | 0.3082 |
| site-stream-g8-h16 | 0.0696 | 0.4915 | 1.1971 | 1.9558 | 2.3675 | 0.2980 |
| site-stream-g8-h32 | 0.0809 | 0.5192 | 1.1356 | 1.8074 | 2.3951 | 0.3092 |

job `l4job-7e7dee9222694e7092b697f08eb49553`
source archive `4682964312adb670453797912aad87276a206f07d7bd01b99dcb7ae41d5c9701`
result archive `4896a3731d811767b0275759fec695c2c5fc60e538d435fccfa44e3e37474977`
全source 1629 files、result manifest、worker source hashes、initial/post24 snapshotsを検証し、raw archivesをignored `output/regular-grid-h/overnight-evidence/`へ保全。


N2048 / B32 / rho8.0 / A209715、24更新。

| route | 完全step median [ms] | allocated / reserved [MiB] | 両peak≤dense |
| --- | ---: | ---: | --- |
| matrix-torch | 3.2550 | 96.16 / 142 | FAIL |
| factor | 11.4027 | 83.11 / 130 | FAIL |
| onchip-h | 9.8300 | 42.87 / 114 | FAIL |
| prepared-g32-h16 | 9.1094 | 44.22 / 102 | PASS |
| prepared-g32-h32 | 8.8334 | 57.42 / 104 | PASS |
| stream-g8-h16 | 7.7099 | 47.12 / 102 | PASS |
| stream-g8-h32 | 8.0402 | 57.42 / 104 | PASS |
| site-routed-h16 | 8.6029 | 44.23 / 102 | PASS |
| site-routed-h32 | 8.3209 | 57.43 / 104 | PASS |
| site-stream-g8-h16 | 7.8260 | 47.12 / 102 | PASS |
| site-stream-g8-h32 | 6.8709 | 57.43 / 104 | PASS |
| dense | 0.5920 | 81.75 / 106 | PASS |

このcohortで条件を満たすCSTの最速は `site-stream-g8-h32`。独立runは各Case1回、各route21 replay samples。denseは依然比較基準。

別Graphによる段階診断。完全step時間へ加減算しない。
| route | 候補index | H生成 | Y集約 | forward/loss | backward | optimizer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prepared-g32-h16 | 0.0655 | 0.6932 | 2.2774 | 3.1293 | 4.3807 | 0.3195 |
| prepared-g32-h32 | 0.0696 | 0.7137 | 2.0296 | 3.1150 | 4.5588 | 0.3267 |
| stream-g8-h16 | 0.0707 | 0.7076 | 2.5027 | 3.2686 | 3.5062 | 0.3041 |
| stream-g8-h32 | 0.0645 | 0.7066 | 1.8237 | 2.9604 | 3.5441 | 0.3195 |
| site-routed-h16 | 0.0799 | 0.7332 | 1.8893 | 2.6378 | 4.2189 | 0.2949 |
| site-routed-h32 | 0.0778 | 0.7332 | 1.5135 | 2.6225 | 4.5588 | 0.3287 |
| site-stream-g8-h16 | 0.0829 | 0.7270 | 1.8340 | 2.6368 | 3.6465 | 0.3215 |
| site-stream-g8-h32 | 0.0840 | 0.7475 | 1.6548 | 2.4228 | 3.4970 | 0.3103 |

job `l4job-c1c751b8527d423ebf62f90f66363810`
source archive `54245e87ea891f134afbbe62ab7ea4d6cd55b2e05080b81b66b9c63d0c12e5c1`
result archive `ec1ab0637f5e26bc728393641298c023b5a55d2238070b79303d7fffac536a88`
全source 1629 files、result manifest、worker source hashes、initial/post24 snapshotsを検証し、raw archivesをignored `output/regular-grid-h/overnight-evidence/`へ保全。

同H32のcoarse→site比較は、rho3が5.7074→4.8714ms、rho8のG8が8.0402→6.8709ms。
後者の総peakは57.43/104MiB、dense81.75/106MiB。
rho3のH16/32は4.8192/4.8714msで近く、保存幅の順位は確定させない。
rho8のG8/H16は7.7099→7.8260msで改善しなかった。
方式全体の普遍的な勝ちや公開dispatcherへの採用は主張しない。
Y集約の段階時間の減少は確認できるが、hardware cache/stallは測っていない。

両jobの回収・SHA検証後にpool全slotのstoppedを確認した。証拠は
`site-routing-final-pool-status.json`と`site-routing-final-stop-lifecycle.log`。
次の候補はH producer BM8を保ち、Y ownerだけBM16へ広げる。
H16/H32と通常/G8を同じ条件で比較し、未実施の性能を改善と呼ばない。
