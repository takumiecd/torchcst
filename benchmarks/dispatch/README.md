# 観測からdispatcherを生成する

```text
PostgreSQLの固定snapshot（READ ONLY / REPEATABLE READ）
    → 生成依頼JSONの条件で観測を抽出
    → 比較条件ごと・Planごとに独立した実行を集計
    → 評価関数で順位付け
    → leaderboard.json と dispatch.json を生成
    → 使用する環境でload_selector → 条件に一致するPlanを実行
```

DB接続・集計・評価は生成時に行う。ライブラリのforwardからDBへ接続しない。
生成adapterは `cuda.linear.complete-step`（normalized Strip は revision 2、
研究用 local product は revision 3）、選択器は `exact_table`。
既存の [Selector artifact v1](../../src/torchcst/_backends/cuda/dispatch/README.md)を出力する。
GPUでの再測定、kernel承認、既定dispatchへの自動採用は行わない。

## 生成依頼JSON

[request.example.json](request.example.json)をコピーして、実際のデータを指定する。
例の全ゼロ `source_id` は置き換え必須のplaceholder。3実行という例の下限も用途に応じて明示的に選ぶ。

| フィールド | 意味 |
| --- | --- |
| schema_version | 生成依頼の形式。現在1 |
| revision | この生成方針の識別子 |
| dataset | adapter / revision、GPU名、正確なsource_id。任意でcase_id・environment_id・protocol_id |
| execution_mode | eagerまたはcuda_graph。モードごとに生成する |
| min_runs | 各Planに必要な独立実行数。時間サンプル数ではない |
| excluded_run_ids | 集計対象から除外する観測run ID。元データは削除しない |
| score_policy | 評価関数のid・revision・parameters |
| fallback_plan | 未観測条件で検証して使う、完全なPlan宣言 |

対象source_idやcase_idなどは次の照会で取得できる。
`source_id` はGit commitだけでなく、記録されたsource hashesも含む内容ID。

```bash
python -m benchmarks.database records --gpu 'NVIDIA L4' --kind measure --adapter-revision 2
```

GPU・runtime・source・Case・protocol・初期入力/Parameter hashesが揃う比較集合で集計する。
GPUのローカルdevice_indexだけは比較条件から除く。
開始時刻はUTC表記に正規化し、DBの表示timezoneによってdataset IDが変わらないようにする。
FAIL、明示的に除外した実行、必要な観測数に達しないPlanは勝者にならない。
同じrun / Planの複数projectionを二重に集計しない。

同じ実行時条件へ異なる比較集合が対応した場合は生成を止める。
例えばbroad / sharpやseed・optimizer・CUDA版などは、現在のruntime keyから区別できない場合がある。
case_id / environment_id / protocol_idで対象を絞るか、除外対象を指定して別の生成物を作る。
勝者を選ぶために、この違いを暗黙に混ぜたり、後から出た行で上書きしたりしない。

## 評価とleaderboard

同じPlanの各実行の指標値を1票ずつ集計する。時間なら各実行のmedianからmedianを求める。
ある実行だけサンプルを増やしても、集計時の票数は増えない。
指標はname / scope / unit / statisticごとに件数・中央値・平均・標本分散・標本標準偏差・最小・最大を残す。
1実行では分散と標準偏差をnullにする。指標が一部の実行に欠ける場合、完全な集合の指標として補完しない。

| プリセット | 小さいほど良いscore | 任意の制約 |
| --- | --- | --- |
| speed / v1 | 指定モードのstep.timeの中央値 | max_peak_allocated_bytes：観測された最大capture/replay allocated peakの上限 |
| memory / v1 | capture/replay allocated peakの中央値 | max_time_ms：指定モードの時間中央値の上限 |

メモリはGraph capture/replayを含む測定範囲。eagerのpeakやGPUプロセス全体の使用量に読み替えない。
同点はPlan内容IDの辞書順で決める。同点の勝者に統計的な優位性を主張しない。
自動外れ値除去、信頼ポイントの自動重み付け、認証の閾値はこのプリセットに含めない。
提出者・ポイント・全指標・原workerを含む観測を独自評価関数から参照できる。

任意のPython関数は信頼できる生成側のコードとして明示的に渡す。
JSONに式・関数本体・import先を書いて実行する方式にはしない。
関数はCandidateを受け取り、有限の数値（小さいほど良い）またはNone（候補から外す）を返す。
関数の意味を変更した場合はid / revision / parametersも更新する。

```python
from benchmarks.dispatch import ScorePolicy, generate
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip import REGISTRY


def my_score(candidate):
    time = candidate.metric("step.time", "graph", "ms", "median")
    memory = candidate.metric("memory.allocated", "capture_replay", "bytes", "peak")
    if time is None or memory is None:
        return None
    return time.mean + memory.median / 1024**2 * 0.02


policy = ScorePolicy("my-score", "v1", {"ms_per_mib": 0.02}, my_score)
request["score_policy"] = policy.declaration()
result = generate(dataset, request, registry=REGISTRY, policy=policy)
```

初版ではこれらの関数を生成ツール側で実行する。PostgreSQLにユーザーのPython関数を配置する仕組みは含めない。
データ抽出・評価関数・選択器の生成を分離しているので、将来DB側で集計する経路やルール生成も追加できる。

## ローカルで生成する

schema migration 2まで適用されたDBと、SELECT可能な接続URLを環境変数に設定する。
`.env`は自動では読み込まない。URLをCLI引数やGitへ記録しない。

```bash
python -m benchmarks.dispatch \
  --request output/generation-request.json \
  --output output/generated-dispatch
```

既存の出力ディレクトリは上書きしない。生成物はignored `output/`に保存する。

| 出力 | 内容 |
| --- | --- |
| dispatch.json | 完全一致条件→勝者Plan、fallback、評価方針、dataset ID、証拠ID |
| leaderboard.json | 全候補のscore・順位・統計・証拠ID、除外/棄権/件数不足、生成依頼 |
| dataset.json | 固定した抽出結果と内容hash。ネットワークなしでの再生成に使う |

```bash
python -m benchmarks.dispatch \
  --request output/generation-request.json \
  --dataset output/generated-dispatch/dataset.json \
  --output output/replayed-dispatch
```

生成機のTorchを対象GPUのTorch版として記録しない。出力は観測に記録されたTorch / Triton版を対象にする。
CPU生成時は構造とRegistryを検査し、実行用 `load_selector` は引き続き実際の導入済み版を照合する。
runtime版の違う環境では読み込みを拒否する。CUDA版などJSONが照合しない条件の扱いは生成・配布側で管理する。

## Local product の研究候補

CLI は benchmark-local Registry を使うため `research_local_product@v2` の Plan も生成できる。
local の条件は入出力各1次元、共有幅・PolarAmpWidth・normalized Triweight product、
16/32/64/128 sites、batch 1..64、parameter_dim=4、CUDA FP32 / TF32 off。
Operator の width bounds を Case から復元し、polar optimizer 契約を照合する。
生成 artifact の読み込みにも `benchmarks.cuda.linear.manifest.REGISTRY` を指定する。
研究候補の本番登録や公開 CSTLinear 経路への接続は行わない。
persistent route の実行には、既存研究 runner と同じ model-owned layout が必要。

初期 rho mixture は実行時の完全一致条件には入らない。混合・narrow-only のように
同じ runtime key を持つ Case は `case_id` で分けて artifact を生成する。
複数 Case を暗黙にまとめると衝突検査が拒否する。各 artifact は測定 Case に限定した
開発候補であり、現在の rho 分布を読み取る自動選択器ではない。

[2026-10-05 の保存・候補登録記録](../../docs/research-history/local-product/20261005-database-and-dispatch.md)
と [`requests/local-product-l4-20261005/`](requests/local-product-l4-20261005/) に、
最終 L4 source の Case ごとの生成依頼を保存する。速度優先と、従来 control の
allocated peak 以内で最速を選ぶ候補を分ける。後者の ceiling は比較方針であり、
ユーザー指定のメモリ上限ではない。各 Case は1独立実行なので `min_runs=1` を明記し、
21時間サンプルを21独立実行に読み替えない。

## GitHub Actions

[Generate dispatch artifact](../../.github/workflows/dispatch-generate.yml)をmainから手動実行する。
実際のsource_idなどを記載してコミットした生成依頼JSONを `request_file` に指定する。
既存の `benchmark-database` Environmentの `NEON_DATABASE_URL` を使い、読み取り専用transactionで抽出する。
URLを参加者へ配布する必要はなく、所有者URLをこのSecretへ置き換える必要もない。

3つの生成ファイルは `dispatch-candidate` Artifactとして14日間保存する。
使用・保管する生成物はダウンロードして保存する。結果原本の永続保存先はNeon。
このworkflowは既定policyの更新、commit、PR、公開releaseを自動作成しない。

## 接続URLの役割

| 保存場所 | 使用するURL | 用途 |
| --- | --- | --- |
| ローカルのignored `.env`のDATABASE_URL | 所有者用 | 管理者によるmigrationとrole権限の設定 |
| GitHub EnvironmentのNEON_DATABASE_URL Secret | 専用SELECT / INSERT role | Issue受付と、この生成workflowの読み取り |
| ローカル生成時の環境変数 | SELECT可能なrole | ローカル抽出。生成に所有者権限は不要 |

`.env`を作ることはGitHub Secretを更新することではない。両者は独立している。
所有者URLを共有せずに、管理者が [migration手順](../../docs/benchmark-contributions.ja.md#中央actionsとdbの設定)を実行する。

## 検証

```bash
TORCHCST_TEST_DATABASE_URL='dbname=postgres' python -m pytest -q \
  tests/test_dispatch_generation.py tests/test_dispatch_generation_postgres.py \
  tests/test_dispatch_selectors.py tests/test_benchmark_database_postgres.py
```

構造検証fixtureは実GPU性能の証拠として提出しない。生成・DBテストはGPUを割り当てずに実行できる。
生成物の性能と、既定採用してよいかの確認は別途実機で行う。
[実DB・既存L4観測・再生成の確認記録](../../docs/dispatch-generation-verification.ja.md)を参照。
