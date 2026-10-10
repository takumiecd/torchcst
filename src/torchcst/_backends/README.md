# 宣言と実行の配置規約

数学的な意味を宣言する型と、それを Tensor 上で計算する実装を分ける。
宣言には backend の選択、forward / backward、GPU の launch 設定を入れない。
backend は宣言の意味を変えず、対応する Algorithm と Recipe で実行する。

共通Algorithmの処理/state lifecycleは`_backends/algorithm.py`と`state.py`に置く。
`atoms/state.py`のAtomStateはModule所有の学習状態の正本で、optimizer状態の
保持と配置への追従宣言は`atoms/optimizer_state.py`、Torchとの接続は`optim/binding.py`。
具体的な計算・配置・bufferは各Algorithm内に置く。
[状態所有と配置変更の契約](../../../docs/atom-state.ja.md)を参照。
Inputs・Context・Registry・Dispatcherの境界は
[Algorithm実行の設計契約](../../../docs/algorithm-dispatch-boundaries.ja.md)に記録する。
下記は現在の実装を示す。

## 現在の配置

```text
torchcst/
  geometry/
    spec.py / presets.py       空間・metric の不変な宣言と構築
    state.py                   GeometryState の scalar/vector と checkpoint
  patterns/
    spec.py / presets.py       軸の Line / Grid / Points 宣言と構築
    state.py                   PatternState の bounded な Tensor 状態
  charts/
    base.py                    ChartSpec / ChartState の ABC
    explicit.py                ExplicitChartSpec / ExplicitChartState
    product.py                 ProductChartSpec / ProductChartState
    periodic_grid.py           PeriodicGridChartSpec / PeriodicGridChartState
    strip.py                   StripChartSpec / StripChartState
    presets.py / __init__.py    純粋な宣言構築・compile_chart
  kernels/
    profiles/                  Triweight / Gaussian など関数の形の宣言
    parameterizations/         振幅・幅・activity の座標解釈の宣言
    normalization.py           正規化の規則・領域・床
    spec.py                    ProfileBinding / KernelSpec / 状態方針
    presets.py                 利用者向けの Spec 合成
    options.py                 数学的な Spec と分離した実行設定
    state.py                   共通の固定 Tensor 状態・checkpoint
  operators/
    spec.py                    単一 Chart / Chart の組 / linear Operator の宣言
    binding.py                 Module の live state を参照する Operator
    context.py / execution.py  Linear metadata・typed Inputs・直接実行のbinding
    atom_update.py            全Atom共通の更新Inputs・Context・live binding
    metadata.py               演算bindingで共有する宣言cache
  _backends/
    algorithm.py / state.py    backend・演算共通のAlgorithmと所有者ごとの実行状態
    schema.py                  Context protocol / Plan / support / decision
    registry.py                共通登録とPlan宣言/JSON検証
    dispatch/                  runtime・候補検証・差し替え可能な選択器
    serialization.py           宣言metadataの厳密なJSON codec
    catalog.py                 組込みTorch/CUDA Algorithmの遅延登録
    cuda/
      algorithms/
        linear/
          normalized_euclidean_strip/
            _shared/           full / window が実際に共用する計算
            full/              algorithm / recipe / executor / kernels
            window/            algorithm / recipe / executor / provider / kernels
          strip_torus/fused/   現行の融合計算・準備・autograd・schedule
          local_product/      小さい積演算・支持・tile配置・autograd
        polar_update/         Atom更新のCUDA最適化候補・executor・kernel
    torch/
      algorithms/
        linear/               materialized / factored / tiled / normalized_radial
        atom_update/          全Kernel・Geometryの座標更新の参照Algorithm
        polar_update/         Polar更新のAlgorithm・座標法則・Graph対応経路
      geometry/                埋め込み・距離・射影・更新・輸送
      patterns/                格子展開・点選択・bounds
      charts/                  Chart の座標生成・部分領域・支持域
      profiles/                Profile の評価・微分
      parameterizations/       振幅・幅・activity の実際の解釈
      kernels/                 atom 合成・重み生成・初期化・factor 微分
      updates/                 Kernel 座標の更新と optimizer 状態輸送
                               Polarの法則はpolar_update/executor.pyに置く
      operators/               Operator を受け取る Torch linear 実行
  nn/
    公開 Module                Parameter / buffer / checkpoint の所有
```

本体から `benchmarks/`、`tests/` へ依存しない。Torch backend は
CPU 専用という意味ではなく、Torch Tensor の対応 device 上で計算する参照経路。
NVIDIA 向け Triton の最適化をここへ混ぜない。

## 宣言の区分

| 宣言 | 固定する意味 | 実行設定に移すもの |
| --- | --- | --- |
| Geometry | 空間、次元、metric、半径、center の保存表現 | 座標変換・距離・射影・更新の計算手順 |
| Chart / Pattern | shape、点配置、flatten 順序、原点、間隔、Strip の物理 pitch | 座標展開、GPU 上の配置・routing の手順 |
| Profile | 無次元距離に対する関数の形 | 関数評価・微分の実装 |
| Parameterization | atom 座標から振幅・幅・中心を解釈する規則と微分契約 | 解釈の計算・Tensor の pack |
| Normalization | 対象領域と L2 等の規則、ノルム床 | reduction、保存・再計算、作業領域 |
| Kernel | radial / separable / amplitude の合成 | materialize / full / window / fused の方式 |
| 初期化・更新方針 | 初期分布や座標更新の意味 | sampler、射影・retraction・状態輸送の計算 |

Sphere / Torus は現行の chord metric を維持する。正規化領域や amplitude-width の
結合微分を Recipe によって変えない。数値的な契約を変える場合は別の意味の版を
付け、既存 checkpoint や同じ演算の fallback と混ぜない。

## 共有処理の配置

一つの Algorithm で使う準備・計算は、その Algorithm 内に置く。同じ演算の複数
Algorithm が実際に共用するものは、演算グループ内の `_shared/` に置く。
別演算でも使う CUDA 専用処理は、必要性を確認して CUDA の中に共通化する。
将来の MPS / Radeon との共有を理由に、先回りして backend 共通の `shared/` を
作らない。共通の数学的意味は宣言で共有し、計算実装の共有とは区別する。

## 現在の接続

CSTLinear / CSTConv2d は ChartSpec / KernelSpec から具体的な ChartState / 共通 KernelState を作る。
Geometry / Pattern / Kernel / Profile は共通の State を使い、Chart の State は
Explicit / Product / Strip に分ける。互換メソッドや旧 checkpoint loader はない。Operator は Module が所有する ChartState / KernelState / Atoms を参照する。単一 Chart と入出力 Chart の組は
layout の型で区別する。Torch の評価・更新は backend の関数で実行する。
宣言 snapshot を forward ごとに作らず、現在の Tensor / buffer を使う。
詳細は [Operator](../operators/README.md)を参照。

Registryはbackend・演算のどちらにも依存せず、登録・取得・Planの宣言検証だけを扱う。
共通Dispatcher.run(binding, inputs, plan=...)が入力検証、Context構築、選択/対応判定、
State取得、Algorithm.runを接続する。強制Planも同じ経路を使う。Context protocolは
operation_idとworkspace_limit_bytesだけで、Tensor検証や学習状態を持たない。
Selector.selectのmetadata検証もdispatch/validation.pyとruntime.pyの同じ処理を使う。
Registry.validate/executeは削除し、実行の互換wrapperを置かない。

CSTLinearはLinearInputs(x)とbinding interfaceを提供し、AlgorithmStateを引き続き所有する。
LinearBindingはModule外で既存Operator、または宣言とParameterを結び付ける。具体的な
演算入力はAlgorithm.input_typeで宣言し、PolarやoptimizerにLinearの入力を要求しない。
Algorithmはbinding/RecipeからStateを作り、execute(state, inputs)で演算する。
Atomsを使わない演算のStateも同じ経路に乗る。CSTLinearのbackend名は既存Planへのaliasで、
実行にはselector自身のRegistryを使う。登録はcompute codeを読み込まない。

CSTOptimizerは全siteにAtomUpdateBindingを作り、AtomUpdateInputs(previous, step_size)を
共通Dispatcherへ渡す。operation_idは全候補でatom_update。optimizerにPolarの型判定や
専用dispatcherを置かない。既定はtorch_atom_updateで、Kernelの座標法則とGeometryの
retractionを通す。update_selectorはLinearのselectorと独立して差し替えられる。
torch_polar_updateとcuda_polar_update_fusedは同じ演算の候補で、対応条件はAlgorithmに置く。
勾配射影とOptimizerStateAdapterのベクトル輸送は既存Torch参照処理を維持する。
AdamW等の提案・moment更新・step計数はbase optimizer、状態の所有はAtomStateが担当する。

Chart/Geometry/Kernelを取得する入口は既存Operatorだけにする。normalized専用Operatorは
追加しない。regular siteの配置抽出と対応判定はTorch normalized Algorithmの`layout.py`にあり、
CUDA FULL/WINDOWもその純粋metadata処理を再利用する。具体的な実行・準備・kernelは各方式内に
維持する。Moduleはbuffer versionで宣言を更新し、数値的なatom準備は毎回現在値を使う。

演算の契約は backend の外に置く。Algorithm / Recipe の設定と kernel 技術の名前を
区別し、Triton というだけの理由で複数の計算方式を一つのファイルへ集めない。
未実装のディレクトリや空の雛形は作らない。

検証は `tests/`、継続する正しさ・時間・メモリ測定は `benchmarks/` に置く。
未採用方式は独立した研究 branch で扱い、本体ツリーへ試作ディレクトリを残さない。
過去の採否は `docs/research-history/` に保存する。登録、対応判定、承認、既定 dispatch への採用は別の段階とする。

詳細は [Kernel](../kernels/README.md)、[Geometry](../geometry/README.md)、
[Pattern](../patterns/README.md)、[Chart](../charts/README.md) を参照。


## Plan の JSON export / import

`ExecutionPlan` は実行設定の宣言。registry は登録済み Algorithm の recipe 型を使って
復元し、schema・ID・revision・全 recipe フィールドとその値を検証する。

| Registry API | 変換 |
| --- | --- |
| `dump_plan(plan)` / `load_plan(data)` | Plan ↔ JSON-compatible dict |
| `dumps_plan(plan)` / `loads_plan(text)` | Plan ↔ JSON text（UTF-8 bytes の読み込みも可） |

```python
from pathlib import Path
from torchcst._backends.catalog import REGISTRY
from torchcst._backends.cuda.algorithms.linear.normalized_euclidean_strip.plans import WINDOW

path = Path("plan.json")
path.write_text(REGISTRY.dumps_plan(WINDOW), encoding="utf-8")
restored = REGISTRY.loads_plan(path.read_text(encoding="utf-8"))
assert restored == WINDOW
```

dispatcher の `decision.plan` も同じ API で保存できる。復元したPlanは
`Dispatcher(registry=REGISTRY).run(binding, inputs, plan=restored)`へ渡す。
Linearでは`LinearBinding(operator, p)`と`LinearInputs(x)`を使える。
実機・tensor・数学契約の適合性は実行時に検証する。

保存対象は `schema_version`、`algorithm_id`、`algorithm_revision`、`recipe`。
export/import は GPU 不要。recipe は JSON-native な scalar / list / string-keyed dict を
使い、非有限数、重複 JSON key、型の暗黙変換、未登録・未検証の設定は拒否する。
export は元の recipe と独立した辞書を返し、往復で元の Plan と一致することを確認する。
JSON text はキーをソートして決定的に出力する。
`algorithm_revision` は実装契約の版。再現実験では benchmark と同様に source commit / hash
と実行条件も記録する。

## 選択器

[Selector の契約と完全一致 artifact](dispatch/README.md)を参照。
full / window の専用選択 API は廃止し、両者を通常の Algorithm 候補にする。
選択器は `CSTLinear(selector=...)` に渡し、数学的宣言・checkpoint から分離する。
