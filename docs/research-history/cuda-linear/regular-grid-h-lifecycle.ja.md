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
