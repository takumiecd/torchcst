# Benchmark database

Neon / PostgreSQL に、**Plan と、その Plan を実行して得た複数の観測結果**を追記する。
ライブラリの実行時依存にはせず、checkout 内の benchmark tooling として扱う。
SQLite の旧試作や互換 API は含めない。

[公式測定経路](../automation/README.md)は、固定 request と Colab の回収結果を照合し、
この保存 API に中央の出所情報を付けて取り込む。GPU 端末へ DB の認証情報を渡さない。

## 保存するもの

| テーブル | 主なフィールド | 役割 |
| --- | --- | --- |
| `plans` | `id`, `algorithm_id`, `algorithm_revision`, `declaration` | registry が検証した全 recipe を含む Plan JSON。内容の SHA256 が ID |
| `cases` | `id`, `declaration` | shape、dtype、atoms、profile、seed、optimizer、warmup、rounds。表示名を除く内容で識別 |
| `runs` | `id`, `execution_id`, `started_at`, `artifact`, `artifact_sha256`, `artifact_schema_version`, `status`, `provenance`, `received_at` | １回の実行結果。元の JSON **バイト列**を保存。測定開始時刻と DB 受信時刻を区別 |
| `projections` | `run_id`, `adapter`, `adapter_revision`, `case_id`, `protocol_id`, `protocol` | 元の結果をどの読み取り処理で指標化したか。読み取り処理の改訂も追記できる |
| `run_plans` | `projection_id`, `alias`, `plan_id`, `is_baseline` | 実行に含めた Plan の名前と baseline。名前は実行内の alias |
| `records` | `projection_id`, `ordinal`, `plan_alias`, `kind`, `status`, `environment_id`, `environment`, `source_id`, `source`, `payload` | worker ごとの観測。正しさ、性能、dense 参照を区別。GPU / runtime、source commit / hashes、元の worker JSON |
| `metrics` | `name`, `scope`, `unit`, `statistic`, `value`, `samples`, `details` | 指標ごとの値と各サンプル。時間・メモリ専用の固定列を作らない |

たとえば同じ Plan を３回測れば、`plans` は１件を共有し、実行・worker の観測を３回分残す。
同じ JSON の再送は増やさない。キー順や空白だけの変更も同じ run として扱い、最初に受け付けたバイト列を保持する。
`run.py` は実行ごとの UUID / UTC 開始時刻を付ける。同じ数値を得た別実行も区別でき、
同じ UUID に異なる結果を登録すると拒否する。これらを持たない既存の schema v1 artifact は内容の ID で識別する。
UUID は再送・衝突を検知する識別子であり、別 UUID を付けた虚偽の投稿を証明するものではない。

`received_at` は測定時刻の代用にしない。元の結果には Plan、固定 Case / catalog hash、
snapshot hash、初期値 hash、計測結果、追加フィールドも残る。
JSONB は検索用であり、元のバイト列を復元する役割は `artifact` が持つ。

## 現在の指標と検証

初期 adapter は `cuda.linear.complete-step` revision 1、入力は
[`benchmarks.cuda.linear.run`](../cuda/linear/README.md) の完成した schema v1 JSON。
Strip/Torus 診断、wheel driver の別形式はこの入口では受け付けない。

| 指標名 | 範囲 | 単位 / 統計 |
| --- | --- | --- |
| `step.time` | `eager`, `graph` | ms / median、全 samples |
| `memory.allocated` | `before_capture` | bytes / current |
| `memory.allocated` | `capture_replay` | bytes / peak |
| `memory.reserved` | `capture_replay` | bytes / peak |
| `error.w`, `error.y`, `error.dx`, `error.dp` | `small_oracle` | dimensionless / max, rel_l2 |

メモリは warmed model・勾配・AdamW 状態を含む Graph capture/replay の測定。
eager の peak として流用しない。GPU process usage は未測定なので指標を作らない。
dense は Plan を持たない別の性能参照として保存する。

PASS の取り込みでは、登録 recipe、snapshot、worker の揃い、Case、source / runtime / GPU、
精度、初期 Parameter / input hash、sample 数・中央値、メモリの整合性を検査する。
FAIL も途中の worker・エラーを保存するが、集計用指標は作らない。
correctness-only PASS は誤差だけを保存する。
小さい独立 oracle の結果を、完全 shape の全勾配の証明に読み替えない。

これらは **内部整合性の検査**。投稿者の真正性・kernel 認証・採用判断は別の工程。
`source_commit` が短い SHA / `unrecorded` の記録は `commit_is_full_sha: false` として区別する。
現在の runner が収集しない GPU UUID、driver、clocks、power は推測して補わない。
将来の集計では Case だけで束ねず、protocol、source、environment、adapter revision、
測定範囲・unit・statistic と採用判断を揃える。
baseline と dense が入っているかも考慮する。

## 将来の拡張

forward / backward の個別時間、p95、新しいメモリ指標などは、`metrics` の行として追加できる。
指標の意味や取り出し方を変えたら adapter revision を上げる。
同じ原本から新しい projection を作り、以前の projection を上書きしない。
現時点で原本に含まれない値は再解析だけでは得られず、benchmark 側の追加計測が必要。

DB の構造変更は `migrations/002_*.sql` 以降の順序付き migration として追加する。
適用済み SQL を書き換えると checksum 不一致で拒否する。
DB schema version、artifact schema version、adapter revision、測定 protocol revision を分ける。
新しい benchmark 形式は明示的な adapter と取り込み入口を追加し、DB 接続処理と切り離す。

認証・除外・leaderboard・dispatch 規則は今回のテーブルから自動生成しない。
今後、判断の理由と対象の run / Plan ID を持つ追記専用の別テーブルを追加する。

## 使い方

```bash
python -m pip install -e '.[dev,benchmark-db]'
# Neon の接続文字列を環境変数 DATABASE_URL に設定してから実行する。
python -m benchmarks.database migrate
python -m benchmarks.database import-linear output/normalized-step.json
python -m benchmarks.database status
python -m benchmarks.database records --gpu 'NVIDIA L4' --kind measure --adapter-revision 1 --limit 100
python -m benchmarks.database records --plan-id PLAN_SHA256 --case-id CASE_SHA256
python -m benchmarks.database export-run RUN_SHA256 output/restored.json
```

`--database-env NAME` で別の環境変数を使える。接続文字列を CLI 引数に渡さない。
Neon の接続文字列は提供される TLS 設定を維持する。
初期化は owner で行い、通常の取り込みは SELECT / INSERT のみの専用 role を使う。
read-only role は SELECT のみとする。

```sql
-- owner により一度設定する。role の接続認証は別途設定する。
CREATE ROLE benchmark_ingest NOLOGIN;
CREATE ROLE benchmark_reader NOLOGIN;
GRANT USAGE ON SCHEMA benchmark TO benchmark_ingest, benchmark_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA benchmark TO benchmark_ingest, benchmark_reader;
GRANT INSERT ON benchmark.plans, benchmark.cases, benchmark.runs,
    benchmark.projections, benchmark.run_plans, benchmark.records, benchmark.metrics
    TO benchmark_ingest;
```

実際の LOGIN role にそれぞれを付与する。migration 後に新テーブルの権限も設定する。
取り込み role に schema owner・UPDATE・DELETE・TRUNCATE 権限を渡さない。
証拠テーブルの trigger も UPDATE / DELETE / TRUNCATE を拒否する。
owner 自体の改ざんを防ぐ仕組みではないため、owner と取り込み role を分ける。

１ artifact の Plan・Case・run・worker・指標の保存は１ transaction。
同時に同じ artifact を送っても１件にまとまる。
取り込み途中の失敗は全体を rollback する。
`provenance` は Python API の信頼できる呼び出し側が渡す JSON object であり、
artifact の自己申告フィールドから取り出さない。再送時に異なる provenance を渡すと拒否する。

CLI の `records` は最大1000 workerまで。`next_after` を `--after PROJECTION_ID ORDINAL` に渡して次を読む。
複数ページを固定した状態で読む場合は、Python API の `with database.snapshot():` 内で
`list_records()` を繰り返す。PostgreSQL の REPEATABLE READ / READ ONLY を使う。
別の adapter revision の同じ run を二重に集計しないよう `--adapter-revision` を指定する。

`export-run` は元の JSON を検証付きで復元し、既存ファイルを上書きしない。
別 DB に import すると同じ run / Plan / Case ID と指標を再構成できる。
provenance・受信時刻や全テーブルのバックアップではない。運用バックアップには PostgreSQL の backup を使う。

今回はコードとローカル PostgreSQL での検証まで。
Neon project 作成・接続、中央 GitHub Actions だけに書き込み credential を与える ingestion 経路、
Colab 等の測定実行、認証、leaderboard / dispatch の生成は次の段階で接続する。
生成物の実行時形式は [Selector artifact](../../src/torchcst/_backends/cuda/dispatch/README.md) に定義する。
DB の保存、任意の評価関数による順位付け、選択器の生成、実行時の選択を分離する。
中央 credential を持たない contributor は結果 JSON を提出する。

## テスト

```bash
python -m pytest -q tests/test_benchmark_database_adapter.py
# 専用のローカル PostgreSQL に接続する環境変数を設定する。
python -m pytest -q tests/test_benchmark_database_postgres.py
```

後者は `TORCHCST_TEST_DATABASE_URL` の CREATE DATABASE / CREATE ROLE 権限を持つ接続を使用。
test ごとに UUID で命名した専用 DB を作成・削除する。対象の既存 DB のテーブルは変更しない。
環境変数または psycopg が無い場合は integration test を skip する。
実際の PostgreSQL 上で再送、並行 import、失敗の rollback、追記専用制約、権限、
snapshot、pagination、migration history、復元、新指標の追加を確認する。

[初回の実 DB 検証と既存 L4 artifact の保存結果](results/README.md)を参照。
