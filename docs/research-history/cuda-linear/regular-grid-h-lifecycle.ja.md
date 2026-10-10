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
