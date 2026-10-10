# カーネル追加と改善の手順

2026-10-05時点。小さいCST線形変換の研究は既存のLinear runner、PostgreSQL保存、
ディスパッチ生成を使う。main統合はGitHub PRで行う。

開発の入口・PR手順は [CONTRIBUTING.md](../CONTRIBUTING.md)、
所有者のGPU・研究worktreeの運用規則は [研究環境の運用規則](research-operations.ja.md)に置く。
この文書は実装・登録・検証の詳細を扱う。

RegularGrid上のprofile-product Linearを新規実装・最適化する場合は、
[Hの生成・再利用・寿命を軸にしたkernel設計](h-lifecycle-kernel-design.ja.md)を参照する。
X→Hの読み取り共有、H→Yの局所集約、backwardまでの保存／再計算を同じ実行計画で扱い、
候補の処理単位・保存量・完全step時間をその方針に沿って説明する。

## 追加する種類と変更箇所

ここでのCUDA kernelは計算実装を指す。数学的なKernelSpecを追加する場合は、
宣言・Torch参照実装・演算の契約も変更対象になる。

| 変更 | 実装・宣言 | benchmarkと検証 |
| --- | --- | --- |
| 既存Algorithmのrecipe/launch調整 | 対象の `recipe.py`、`validate_recipe`、executor/kernels。local研究routeは `benchmarks/cuda/linear/local_product.py` | 新Planと比較Case、JSON往復、同一契約の独立oracle・GPU測定 |
| 同じ数学契約の新CUDA Algorithm | `algorithms/<方式>/` にAlgorithm/Recipe、必要なcontract、executor/kernels | benchmark Registryへ登録、catalog/Case、oracle・対応/非対応条件・autograd・Graph検証 |
| 新しい数学的なKernel/演算 | `kernels/` の宣言とTorch参照実装、必要なOperator/Chart契約、対応Algorithm | 対応fixture、独立oracle、protocol、結果adapterと提出/dispatch側の検証を追加 |

Linearの計算は各backendのalgorithms/linear/へ置く。Atomの座標更新は別演算の
atom_updateで、Torch参照はalgorithms/atom_update/、Polarの具体的な更新方式は
algorithms/polar_update/へ置く。optimizerにKernelの種類ごとのdispatcherを追加しない。
全更新が同じAtomUpdateInputs/Contextと共通Dispatcherを使い、Sphere/Torusを含む
一般の更新はTorch参照へ落とす。LinearのPlan、更新Plan、base optimizerを別に扱う。
既存runnerの--polar-updateは更新候補の選択だけを行い、Linear Planを変更しない。

数学的な意味が同じ候補は同じsemanticsを保つ。正規化・支持・微分の意味を変更したら
新しい意味の版として区別する。既存Planのaliasだけを変えて同じPlanを重複登録しない。
登録済みrecipeの意味を変える場合は実装契約のrevisionを更新し、古い結果との比較範囲を明示する。

### 新Algorithmの実装先とインターフェース

実際に動いている最小の見本は
[`window/algorithm.py`](../src/torchcst/_backends/cuda/algorithms/linear/normalized_euclidean_strip/window/algorithm.py)と
[`window/recipe.py`](../src/torchcst/_backends/cuda/algorithms/linear/normalized_euclidean_strip/window/recipe.py)。
共通interfaceは [`Algorithm`](../src/torchcst/_backends/algorithm.py)で定義する。

| 要素 | 実装する内容 |
| --- | --- |
| `id` / `revision` | 他候補と区別できる実装のID・契約の版 |
| `operation_id` / `semantics_id` | 対応する演算・数学的意味 |
| `recipe_type` / `validate_recipe` | immutable dataclassと、実装が受け付ける設定の厳密な検査 |
| `supports(context, recipe)` | Kernel/Chart契約、shape、stride、dtype、GPU、精度、勾配などの適合性。非対応理由を返す |
| `workspace_bound` | 実装が管理するscratchの上界。分からなければ `None`。実測GPUピークとは区別する |
| `input_type` | 演算共通のtyped Inputs。必須fieldと任意fieldを入力型で定める |
| `execute(state, inputs)` | binding/Recipeと実入力からexecutorを遅延importし、forwardの保存状態に対応したbackwardを接続する |

Algorithm構築、recipe検査、support判定はGPUコードのimportやTensor値の読み出しを行わない。
Algorithmインスタンスへ入力・支持・勾配bufferを保存しない。forwardごとの保存状態を
autograd呼び出しに持たせ、Parameterの更新はoptimizerが行う。
`supports`はメタデータだけで判定し、現在のsigmaに依存する支持分類は実行時に更新する。

### Registry、runner、adapterの接続

新しい研究Algorithmはまず
[`benchmarks/cuda/linear/manifest.py`](../benchmarks/cuda/linear/manifest.py)のbenchmark-local
`REGISTRY`へ登録する。metadataをimportするだけでTritonを読み込まないことを確認する。
本番Registry・公開CSTLinearの選択器へ接続するかは、実機検証後の別判断にする。

既存のnormalized Windowを独立Registryへ登録してJSON往復を確認する実行例：

```bash
python - <<'PY'
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
from torchcst._backends.cuda.algorithms.linear.normalized_euclidean_strip.window.algorithm import NormalizedWindowAlgorithm
from torchcst._backends.cuda.algorithms.linear.normalized_euclidean_strip.window.recipe import WindowRecipe

registry = Registry()
algorithm = NormalizedWindowAlgorithm()
registry.register(algorithm)
plan = ExecutionPlan(algorithm.id, algorithm.revision, WindowRecipe())
assert registry.loads_plan(registry.dumps_plan(plan)) == plan
print(registry.dumps_plan(plan))
PY
```

自分のAlgorithm/Recipeもこの形で検査し、拒否する設定・未知revision・非対応contextの
テストを追加する。normalized full/windowは登録済みの既定recipeだけを許可するので、
JSONの数値だけを書き換えても新候補として実行できない。

同じ数学契約の候補でも、runnerのfixture構築、Plan実行、source hash収集、
correctness/measure workerの全経路を確認する。
既存Linear runnerが対応するfixtureはnormalized Stripと小さいlocal polar product。
任意の新演算が自動で通る入口ではない。
新fixtureや結果指標を追加する場合は
[`fixtures.py`](../benchmarks/cuda/linear/fixtures.py)、
[`protocol.py`](../benchmarks/cuda/linear/protocol.py)、
[`run.py`](../benchmarks/cuda/linear/run.py)、
[`database/adapters/linear.py`](../benchmarks/database/adapters/linear.py)、
提出とdispatchのdecoderまで対応を揃える。既存fixtureの契約に押し込まず、
新しい結果形式には明示的なadapter/取り込み入口を設ける。
指標の意味が変わる場合はadapter revisionを上げ、過去のprojectionを保持する。

## 登録済みAlgorithmで手順を試す

この例は実装済みのnormalized fullをbaseline、window512をcandidateにして、
登録→比較条件→検証→測定→提出前検査の流れを試す。
新しいkernelの性能結果を作る例ではない。最初の二つはCPUだけで実行できる。

```bash
python -m tools.kernel_dev prepare \
  --case benchmarks/cuda/linear/cases/normalized-1024-broad.json \
  --candidate window512 --output output/window-comparison
python -m tools.kernel_dev check \
  --plans output/window-comparison/plans.json \
  --case output/window-comparison/case.json
```

出力は選択したbaseline/candidateだけの `plans.json`、同じ入力・optimizer・dense参照を
維持した `case.json`、元Case/catalogのパスとSHA256を持つ `source.json`。
出力先は新しいディレクトリにする。複数候補は `--candidate` を繰り返す。
同じ数学契約の新Planをcatalogへ登録したら、そのaliasも候補に指定できる。
runtimeでの適合性と数値の正しさは次のGPU検証で確認する。

対応CUDA GPUで、repoルートから実行する：

```bash
python -m tools.kernel_dev test --suite normalized-strip
python -m benchmarks.cuda.linear.run \
  --plans output/window-comparison/plans.json \
  --case output/window-comparison/case.json \
  --source-commit "$(git rev-parse HEAD)" --correctness-only \
  --output output/window-comparison/correctness.json
python -m benchmarks.cuda.linear.run \
  --plans output/window-comparison/plans.json \
  --case output/window-comparison/case.json \
  --source-commit "$(git rev-parse HEAD)" \
  --output output/window-comparison/benchmark.json
python -m benchmarks.submissions check output/window-comparison/benchmark.json
```

correctnessが失敗したら修正してから測定へ進む。測定runner自身も先にcorrectnessを実行する。
新しい実装はcommitした状態で測定し、実際のsource hashも結果に保持する。
完成した結果JSONの提出・PRへの証拠の記載は [CONTRIBUTING.md](../CONTRIBUTING.md)に従う。

## 開発環境と確認コマンド

すべてrepoルートから実行する。`tools.kernel_dev`はrepo内の開発用入口で、
配布wheelには含めない。GPUの割当・DBへの保存・dispatchの更新は行わない。

```bash
python -m pip install -e '.[dev]'
python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-local-contraction.json \
  --case benchmarks/cuda/linear/cases/local-contraction-64-sigma3.json \
  --case benchmarks/cuda/linear/cases/local-contraction-64-middle.json \
  --case benchmarks/cuda/linear/cases/local-contraction-128-early.json
python -m tools.kernel_dev test --suite cpu
```

`check`は既存runnerのmanifest / Registryを使って、全Planの登録・JSON往復、
各Caseの候補・baseline・数学的契約、worker用snapshotの往復を検査する。
結果には入力ファイルのSHA256を含める。GPU実装をimportせず、GPUもDBも不要。
`--case`は繰り返し指定でき、省略時はcatalogのみ検査する。
ここでのPASSは宣言の整合性だけで、勾配や性能を検証したことにはならない。

`test --suite cpu`はCUDAのない環境で全pytestを実行する。GPUホストでCPU検証する場合は
`CUDA_VISIBLE_DEVICES='' python -m tools.kernel_dev test --suite cpu`とする。
GPU・実DBのテストは条件に応じてskipされるので、pytestのskip件数も記録する。

GPU開発環境では`python -m pip install -e '.[dev,cuda]'`を実行し、対象に応じて
`python -m tools.kernel_dev test --suite local-product`または
`python -m tools.kernel_dev test --suite normalized-strip`を使う。
Atom更新は`python -m tools.kernel_dev test --suite atom-update`で確認する。
これらはCUDA / Tritonがない環境では失敗し、GPU検証をskipだけで通さない。
テスト群は入口の回帰検証であり、変更対象に応じて境界・勾配・配置などの追加テストを選ぶ。
性能測定は引き続き既存Linear runnerを使う。

## ソースと比較条件

1. mainから名前付き`kernel/<目的>`研究ブランチとworktreeを作る。
   新規kernel・高速化・recipe/launch調整のブランチ命名は
   [CONTRIBUTING.md](../CONTRIBUTING.md)に従う。
2. 実装は`src/torchcst/_backends/cuda/algorithms/`へ置く。数学的な仕様と実行設定を分ける。
   local productは`linear/local_product/`、研究Planの登録は`benchmarks/cuda/linear/local_product.py`。
   既存Planの設定と意味を保ち、新しい候補には明示的なrouteを与える。
3. `benchmarks/cuda/linear/plans-*.json`へ候補を追加し、`cases/`で比較対象、baseline、
   dense参照を明示する。固定σを導入せず、初期幅と各stepの幅更新を区別する。
4. 既存runnerの`--validate-only`で宣言を確認する。同じ初期Parameter・入力・dtype・
   optimizer契約で、時間とメモリを比較する。

`tools.kernel_dev check`で複数Caseをまとめて確認できる。新しいrouteを追加したら、
宣言をregistryから復元できることと、候補を含むCaseでbaseline / denseが明示されている
ことを確認する。幅・shapeを変える比較は別Caseとして記録する。

新方式の実装をbenchmarkフォルダへ埋め込まず、別runnerを追加しない。
既存の`research_local_product`は研究用registryであり、公開CSTLinearの既定ではない。

## 正しさとGPU測定

独立したFP64 scalar oracleでY、dX、全atom勾配を確認する。正規化、支持境界、
単一サイト、空支持、切り出したdomain、retained backwardを含める。
更新を変える場合は公開CSTOptimizerとParameter、AdamW moments、stepを比較する。
可変σ、支持・配置の移動、Graph replayを確認する。

L4は[共有プール](../tools/colab-l4-pool/SKILL.md)へ提出する。提出前に`git status`
を確認し、raw evidenceがsnapshotへ入らないことを確かめる。独立のdriverはignored
`evidence/`に置き、既存runnerと検証を呼び出す。driverは新しい測定runnerではない。

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-contraction.json \
  --case benchmarks/cuda/linear/cases/local-contraction-64-middle.json \
  --polar-update fused --phase-diagnostics --kernel-diagnostics \
  --output benchmarks/cuda/linear/evidence/comparison.json
```

未計装の完全学習stepを採用判断の時間とする。診断は別Graphで計測し、足し合わせたり
主測定から差し引いたりしない。メモリはcapture/replay込みのpeak allocatedとreservedを
区別し、GPU process usageを測っていなければ未測定とする。21時間サンプルを21独立runと
数えない。改善が小さい場合は独立jobで再測定する。負の結果も残す。

## DB保存とディスパッチ候補

完成runner JSONを既存[DB入口](../benchmarks/database/README.md)へ追記する。
元のバイト列のexport照合と同じprovenanceでの再送を確認する。owned poolのjob ID、
source snapshot SHA256、結果archive SHA256を付け、資格情報はGPUに転送しない。
研究local productのadapter revisionは3、normalized Stripは2。

[生成依頼JSON](../benchmarks/dispatch/README.md)でsource、GPU/runtime、case、時間目的、
メモリ方針、最小独立run数を指定する。dataset・leaderboard・dispatch JSONを生成し、
未知の条件でfallbackすることを確認する。DB保存、候補生成、公開の既定採用は別の操作。

現在のlocal productの実行条件keyには初期rho分布が含まれない。同じshapeで幅分布だけ
異なるcaseを混ぜた自動選択は完成していないため、候補はcaseごとに生成する。
`baseline-peak`は比較方針であり、ユーザーが指定したメモリ上限ではない。

## 記録と統合

開発・提出・PRの検証項目は [CONTRIBUTING.md](../CONTRIBUTING.md)、
所有者の共有GPUの制約・研究worktreeの証拠保全は
[研究環境の運用規則](research-operations.ja.md)を正本とする。
local productの簡潔な記録は `docs/research-history/local-product/`、
CUDA Linearの記録は `docs/research-history/cuda-linear/` に置く。
採否、誤差、完全step時間とピーク、独立run数、job ID、source/result hash、
再現コマンドと保全先を残す。実機未検証の候補は研究ブランチで扱う。
