# dispatch生成の確認（2026-10-02）

branch `codex/dispatch-artifact-generation`で、PostgreSQLの観測抽出から
比較集合の集計、任意の評価関数、leaderboard、完全一致Selector JSONまで接続した。
新しいGPU計測や既定policyの採用は行っていない。

## 検証

- 新規32テスト：速度/メモリ/独自評価、汎用の追加指標、欠損指標、独立実行数、
  平均・中央値・標本分散、重複、除外、比較集合の衝突、複数batch shape、runtime境界、CLI再生成。
- 実PostgreSQLの一時DBで、ページング中の別接続からの追加が固定snapshotに入らないことと、
  生成前後で原本・テーブル件数が変わらないことを確認した。
- 全体734 passed / 117 skipped。GPUなど環境依存のテストをskipし、既存TorchScriptのwarningが18件。
- 変更コードのRuff、新旧3workflowのactionlint、diffの空白検査、変更文書のローカルリンク検査が成功。

```bash
TORCHCST_TEST_DATABASE_URL='dbname=postgres' .venv/bin/python -m pytest -q
output/benchmark-tools/actionlint .github/workflows/dispatch-generate.yml \
  .github/workflows/benchmark-submit.yml .github/workflows/benchmark.yml
```

## 既存の実測観測による生成

取り込み用roleで既存Neonのrun原本4件をREAD ONLY / REPEATABLE READで取得し、
原本のSHA256を照合した。固有名のローカル一時PostgreSQL DBへschema 2として複製し、
検証後にその一時DBだけを削除した。Neonのschema・観測は変更していない。

生成依頼は過去のL4データのsource / Case / protocol / environmentを明示的に絞り、
`min_runs=1`とした。選択されたmeasureは2件、候補Planは2つ、出力条件は1つ。
speed / v1、cuda_graphでnormalized_fullを選んだ。
この1実行分の比較は生成経路の確認であり、統計的な採用判断や新たな性能向上の証拠ではない。

- GPU：NVIDIA L4
- 対象runtime：Torch 2.11.0+cu128 / Triton 3.6.0
- source_id：`d5f637d45c8bb0fbdd646e4fa1f55c031c7cbd61bb6d4857bea2f2a7770cfd72`
- 最終dataset snapshot：`e6be9c617dbdf24db8c80ff1d127316e323fe90dc1a2d6f7010622282b5dad56`
- 最終dispatch.jsonのSHA256：`a950988cf9eebfab48e9f1cc98cfabcb8f8e51fd1e1802b4160f670d2d721b0e`

初回の原本・依頼・生成物はignored `output/dispatch-generation-neon-evidence/`に保存した。
後続の本番移行後、Neonから直接生成した結果とローカル複製のdataset IDが異なった。
差は開始時刻のDB表示timezoneだけだったため、抽出時にUTC表記へ正規化した。
UTC / Asia/Tokyoで同じdataset IDになる実DBテストを追加した。

最終生成物はignored `output/dispatch-generation-neon-utc-evidence/`に保存した。
Neon直接抽出、ローカル複製、ネットワークなしのCLI再生成の3経路で、
dispatch.json・leaderboard.json・dataset.jsonすべてがバイト単位で一致した。

```bash
PYTHONPATH=src .venv/bin/python -m benchmarks.dispatch \
  --request output/dispatch-generation-neon-utc-evidence/request.json \
  --dataset output/dispatch-generation-neon-utc-evidence/live-generated/dataset.json \
  --output output/dispatch-generation-neon-utc-evidence/replay-new
```

実行環境が異なるCPU機でも対象runtimeを保った構造検査ができるようにした。
`validate_selector_artifact`はデータだけを返す。実行用の`load_selector`は引き続き
導入済みTorch / Triton版を照合し、不一致を拒否する。構造検証テストでは対象版を模擬して
条件→勝者Planと、未観測条件→fallbackの両方を確認した。新規CUDA実行はしていない。

## 本番の状態

所有者DATABASE_URLの再設定を確認し、Neonのmigration 2を適用した。
新テーブルへの専用取り込みroleのSELECT / INSERTも追加し、UPDATE / DELETE権限がないことを確認した。
移行前後でrun原本のhash・provenanceと既存7テーブルの件数が一致した。
件数はplans 2、cases 1、runs 4、projections 4、run_plans 8、records 20、metrics 124。
submissionは0件で、テスト用の偽観測は登録していない。

専用roleで本番Neonから生成するCLIは確認済み。GitHub Secret自体は変更していない。
PR #19をmainへマージし、commit `99f4e52f2c47be631a3a45e8f92d47d78c241dfa`で
[生成Actions（37021155265）](https://github.com/takumiecd/torchcst/actions/runs/37021155265)が成功した。
Environment `benchmark-database`の既存Secret `NEON_DATABASE_URL`でNeonへ接続し、
観測抽出・ランキング・JSON生成・artifact保存まで完了した。
ダウンロードしたdispatch.json・leaderboard.json・dataset.jsonは、上記のローカル生成物と
すべてバイト単位で一致した。Actions生成物はignored
`output/dispatch-generation-actions-evidence/`に保存した。
生成したartifactは候補であり、既定dispatcherの自動更新は行わない。

Issue受付の本番確認は[受付の検証記録](issue-submission-verification.ja.md)を参照。

使い方は [生成ガイド](../benchmarks/dispatch/README.md)を参照。

本番Actionsの再現用依頼は
[l4-archived-verification.json](../benchmarks/dispatch/requests/l4-archived-verification.json)。
過去の固定された観測を用いる経路確認用で、通常の採用判断用プリセットではない。
