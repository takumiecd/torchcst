# Neon / Actions 初回設定（2026-10-02）

`takumiecd/torchcst` の実行環境を設定した。接続文字列は ignored `.env` から読み取り、ログ・この記録には保存しない。

- Neon `neondb` に schema migration 1 を適用した。実接続の PostgreSQL は18.6。
- SQL で Actions 専用 LOGIN role `torchcst_actions_ingest` を作成した。benchmark schema の USAGE、テーブルの SELECT と evidence テーブルの INSERT を付与した。
- UPDATE / DELETE / TRUNCATE 権限がなく、UPDATE・DELETE・benchmark schema 内の CREATE TABLE が拒否されることを実接続で確認した。
- `benchmark-database` Environment の `NEON_DATABASE_URL` に専用 role の接続情報を登録した。owner の接続情報は GitHub に渡していない。
- `benchmark-colab` / `benchmark-database` Environment は branch `main` のみ許可する。
- この Mac に `colab-client` label の一時 runner `torchcst-colab-mac` を登録し、online 接続を確認して停止した。runner はv2.337.0、公開 asset の SHA256 を検証済み。Python3.11 と認証済み Colab CLI を使用する。常駐サービスは作らない。

## 実データによる保存確認

既に回収・hash 検証した L4 / G4 の観測を、専用 role で Neon に登録した。
出所は `colab-neon-integration-verification` と明記し、GitHub workflow の観測とは区別する。
2 runs / 10 worker records / 62 metrics を保存し、再送による重複防止と元 JSON の byte-exact export を確認した。

[実機測定記録](measurement-json-verification.ja.md)に source・job・archive hash・全測定値を保存している。
今回の保存対象の run ID:

- `6293283de9fc23af1d237473c099b7b2084952cc6088de92d3580d8d8a4fa88e`
- `4f906c4d2b23199c71217f56a441a3722ee90f0d54d0d7b21723f9e36758fb73`

## 次の確認

[PR #15](https://github.com/takumiecd/torchcst/pull/15) の main への反映後、
[測定 JSON](../benchmarks/requests/colab-linear.json) を指定して実際の Actions を１回実行する。
まだ GitHub 上で prepare → measure → ingest が通ったという確認はしていない。

この runner は ephemeral のため1 jobを終えると GitHub の登録が解除される。
次の batch は runner の再登録・起動が必要。GPU VM の確保・実行・回収・停止は共有 Colab queue が管理する。

汎用の初回設定は [automation README](../benchmarks/automation/README.md) を参照。
ローカルの非公開設定・詳細 receipt は ignored `output/actions-setup/` とホストの private state に保持している。
