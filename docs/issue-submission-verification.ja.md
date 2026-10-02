# Issue経由の結果受付の確認（2026-10-02）

提出先を同じtorchcstのIssueに統一し、各自のGPU / Colabから得た完成JSONを
中央Actionsで受け付ける実装へ変更した。旧fork artifact importとColab測定からの
自動DB登録は削除し、Colab workflowは測定補助として維持した。

一般はUTCで1日10実行・1点、認定は件数上限なし・既定10点。
認定IDと任意のポイント変更は公開設定で管理し、自動加点・ランキング計算は含めない。
投稿者のGitHub ID・ユーザー名と、受付時点のpolicy・区分・ポイントを原本に紐付けて追記する。
同じファイルの再送や他アカウントによるコピーは観測・帰属を増やさない。
別のexecution UUIDは別観測として残し、過去の測定結果やprovenanceを変更しない。

## 確認結果

ローカルPython 3.11.15 / Torch 2.13.0 / psycopg 3.3.6 / PostgreSQL 14.19で確認した。
実DBテストは、テストごとに固有名のDB・必要に応じて固有roleを作成し、終了時に削除する。
ユーザーの既存Neonや他のDBを変更していない。

```bash
TORCHCST_TEST_DATABASE_URL='dbname=postgres' .venv/bin/python -m pytest -q \
  tests/test_benchmark_submissions.py tests/test_benchmark_submission_postgres.py \
  tests/test_benchmark_database_postgres.py tests/test_benchmark_database_adapter.py \
  tests/test_benchmark_automation.py

TORCHCST_TEST_DATABASE_URL='dbname=postgres' .venv/bin/python -m pytest -q

output/benchmark-tools/actionlint \
  .github/workflows/benchmark.yml .github/workflows/benchmark-submit.yml
```

- 関連130テストが成功。
- 全体702 passed / 117 skipped。skipはGPUなどこのCPU環境で実行できない条件。
- 既存TorchScript由来の非推奨warningが18件。今回の受付処理のfailureはない。
- 新規/変更コードのRuff、Actionsのactionlint、diffの空白検査、変更文書のローカルリンク確認が成功。
- 実際の公開GitHub JSON添付を、GitHub tokenを添付先へ送らずに取得できた。
  これは添付ダウンロード経路の確認で、benchmarkやNeonへの実投稿の確認ではない。

実DBで確認した境界：

- 一般10件の上限、11件目の拒否、認定の免除、1点/10点のpolicy snapshot。
- 同一ユーザーの最後の枠を競合提出しても１件だけ保存。
- 同じ原本が別ユーザーから同時提出されても１観測・１帰属。
- 上限到達後の再送でも追加枠を消費しない。
- self-reported開始日ではなくDB受信UTC日付で制限。
- 同じexecution UUIDに異なる原本を付けると全体rollback。
- SELECT / INSERT専用roleで受付可能、UPDATEは禁止。
- migration 1から2へ移行しても既存原本・件数・出所を保持。
- 通常ユーザーは未登録でも受付可能。認定はユーザー名ではなく数字のIDで判定。
- 管理者が再実行しても提出者は元のIssue作者。JSON内のpoints/role/userは権限に使わない。

## 本番反映

この確認時点では新しいworkflowは開発branch上にあり、mainのIssueイベント実行と
Neonへの本番migration 2はまだ行っていない。新コードへの切り替え時にownerでmigrationと
新テーブルの専用roleへのSELECT / INSERT grantを適用し、実際の計測JSONの提出を確認する。
手順は [参加・運用ガイド](benchmark-contributions.ja.md) を参照。

今回のテストデータはCPUの構造検証fixtureであり、GPU計測結果として本番DBへ提出していない。
受付結果は常に `consistency_checked` / `self_reported` とし、kernel承認や測定値の真正性と区別する。

後続作業でPR #18をmainにマージし、所有者URLの再設定後にNeonのmigration 2と専用roleへの
新テーブルのSELECT / INSERT grantを適用した。既存原本・件数・provenanceは維持された。
Issueへの実投稿から保存までの本番確認は別途残る。
[本番移行と生成ツールの確認](dispatch-generation-verification.ja.md)を参照。
