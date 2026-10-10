# RegularGridでH/Gを全保存しない最初の試作

2026-10-10。[Hの寿命を軸にした設計](../../h-lifecycle-kernel-design.ja.md)の最初の研究候補。
公開dispatcherへの採用は別判断で、当面は研究branchに置く。

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
