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
GPU独立oracle、Graph、完全step時間／総peakは測定結果を取得後に追記する。
