# Plan の選択器

入力条件から Plan を返す境界を `Selector.select(context) -> DispatchDecision` に統一する。
full / window を特別な分岐にせず、Registry の Algorithm ID / revision / Recipe で表す。
`LinearOptions` と `memory="full" | "window"` は削除した。互換 loader は用意しない。

## データから実行まで

```text
追記された観測データ
  → 評価関数で候補を順位付け
  → 各条件の候補と証拠を得る
  → 選択器を生成・検証し artifact を保存
  → 読み込み時に Selector を構築
  → select(context) → Registry の適合性検証 → Plan の実行
```

評価関数は速度・メモリ等の任意の指標を使う。実行時には評価関数を呼ばない。
`score_policy` はその関数の ID / revision / parameters を記録する metadata。
関数本体を JSON に埋め込んだり、forward で任意コードを評価したりしない。
DBからの固定snapshot抽出・評価関数・leaderboard・exact artifact生成は
[生成ツール](../../../../benchmarks/dispatch/README.md)に用意する。
評価関数は生成ツール側で実行する。DB内のPython実行や既定policyへの自動採用は含めない。

| 選択器 | 現在の状態 | 選択方法 |
| --- | --- | --- |
| FixedSelector | 実装済み | 明示した一つの Plan。bootstrap・直接比較用 |
| ExactSelector | 実装済み | 観測された条件の完全一致表 |
| OrderedSelector | 実装済み | 明示したPlan順でAlgorithmのrouting条件を確認 |
| ルール / 決定木 | 将来の候補 | 同じ `_match(context)` 境界から Plan または未選択を返す |
| MLP 等 | 将来の候補 | 同じ境界から Plan または未選択を返す |

`Selector` ABC の `_match` が候補を返し、共通の `select` が Registry の対応条件・
workspace 上限を検証する。未選択や非対応時は、方針に明示された fallback Plan を検証する。
fallback も非対応なら例外にする。Algorithm 名による特別扱いはしない。
実行時にも `Registry.execute` が実機・Tensor metadata・数学契約を検証する。
backward は forward の Plan と保存状態を使い、再選択しない。

## Artifact v1

| フィールド | 意味 |
| --- | --- |
| schema_version | 外側の形式と完全一致条件の版。現在は 1 |
| selector.kind / revision | 現在は `exact_table`。選択方針の版 |
| score_policy | 評価関数の `id`, `revision`, `parameters` |
| dataset_snapshot | 順位付けに使った固定データ集合の識別子 |
| runtime_versions | PyTorch の完全な版、任意で Triton の版。読み込み時に照合 |
| plans | ID → Registry が復元・検証できる Plan |
| entries | `condition`, `plan_id`, `evidence_ids` の一覧 |
| fallback_plan_id | 未観測・非対応時に検証する Plan |

条件は Operator の型付き宣言を丸ごと保存する。Kernel / Chart / Geometry / Pattern と
入出力 layout の型・設定・revision を含む。shape / stride / dtype / atom 数 / 座標幅、
GPU の種類・名前・compute capability・SM 数、必要勾配、eager / Graph、精度・決定性、
workspace 上限も区別する。GPU のローカル index は条件に含めない。
atom の Tensor 値、初期幅の broad / sharp ラベル、optimizer 状態は読み取らない。

宣言の型名は比較する文字列であり、JSON から型を import・構築しない。
重複条件・未知の Plan / revision・不正 recipe・型の違う flags・非有限値・重複 JSON key は拒否する。
未知の kind や schema は明示的に拒否し、MLP を読み込めると見せかけない。
条件の網羅性、証拠の信頼性、GPU driver 等の測定条件の比較可能性は生成・承認側の責任。
読み込み成功は性能の認証を意味しない。

## 利用方法

```python
from pathlib import Path
from torchcst import CSTLinear, presets
from torchcst._backends.catalog import REGISTRY
from torchcst._backends.dispatch import load_selector

selector = load_selector(Path("dispatch.json").read_bytes(), registry=REGISTRY)
layer = CSTLinear(chart=chart, atoms=p,
                  kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
                  selector=selector)
```

`selector`はすべてのCSTLinear経路に適用する。CPUのTorch参照、CUDA、
Strip/Torus fusedも同じRegistry/Plan/Algorithmを使う。実行にはselector自身のRegistryを使い、
custom Registryも渡せる。selectorは数学的宣言やstate_dictに含めず、別artifactとして保持する。
未指定時はModuleに明示したOrderedSelectorを使う。FULL、Torch normalized、
factored、materializedの順で各Algorithmのrouting条件を確認する。性能による自動採用ではない。

FixedSelectorとOrderedSelectorはoperation固有のContextを受け取る。ExactSelectorも
Linear以外のContextを扱える。その場合、frozen dataclassのContextを型名とscalar/tuple/
不変なdataclassのfieldとして保存する。conditionは`{"context": typed_declaration}`になり、
Tensorや可変な値は拒否する。Linearの既存artifact v1条件形式は維持する。
汎用Contextでは全fieldを比較し、GPU indexを無視する既存Linear条件の規則を暗黙に適用しない。
型名は比較に使う文字列であり、読み込み時にimportや任意コードの実行を行わない。

生成側は、順位付け済みの条件と Plan から artifact を構築できる。

```python
from torchcst._backends.dispatch import ExactEntry, ExactSelector

selector = ExactSelector.from_entries(
    [ExactEntry(context, winning_plan, (observation_id,))],
    registry=REGISTRY,
    fallback_plan=baseline_plan,
    revision="policy-v1",
    score_policy={"id": "my-score", "revision": "v1", "parameters": {}},
    dataset_snapshot=snapshot_id,
)
Path("dispatch.json").write_text(selector.dumps(), encoding="utf-8")
assert load_selector(selector.dumps(), registry=REGISTRY).select(context).plan == winning_plan
```

JSON の解析と辞書の構築は読み込み時だけ。forward は metadata の key で辞書を引く。
固定 Operator の key は最大128宣言までキャッシュする。Tensor や勾配状態を保持しない。
entry 数に比例した走査、DB / network / file I/O は実行時に行わない。
辞書検索は平均 O(1) だが、metadata key と適合性検証のコストは別途存在する。
選択器を差し替えた後は CUDA Graph を再 capture する。既存 Graph の replay は保存された実行を使う。

## 将来の切り替え

観測点が増えて表が大きくなったら、まず領域分割・ルールを生成する。
複数の山という理由だけで自動的に MLP へ移行せず、未知 shape / 別 session を使った
検証で実際の速度損失、メモリ、推論時間、artifact サイズを比較して採否を決める。
ルールで必要な性能を保てなければ MLP 等を検討する。

変更するのは選択器の本体と kind ごとの loader。Context / Plan / Registry / 実行の境界を維持する。
完全一致表や raw 観測を捨てず、近似の参照と未対応領域の判断に利用する。
近似器にも未選択を返せるようにし、外挿で数学契約や workspace 検証を回避しない。
メモリの測定 peak と Algorithm の workspace 上界は区別する。

## 検証

```bash
python -m pytest -q tests/test_dispatch_selectors.py tests/test_cuda_dispatch.py tests/test_linear_parameters.py
python -m pytest -q
```

2026-10-02 のローカル CPU suite は 602 passed / 133 skipped。
JSON 往復、条件の全フィールド、未観測条件、明示的 fallback、workspace 制限、
別 Algorithm ID、custom rule、二回の forward と autograd、選択器を替えた checkpoint を検証する。
CUDA と PostgreSQL の専用テストは環境未接続のため skip。
L4 上で実ファイルからの公開経路、独立 oracle、Graph replay と完全 step の時間・peak を確認した。
対象4ファイルは108 passed / 0 failed / 0 skipped。
[実機検証記録](../../../../docs/dispatch-json-verification.ja.md)を参照。
