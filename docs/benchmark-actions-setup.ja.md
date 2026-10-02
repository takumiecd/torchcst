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

## Actions の通し確認

[Actions run 36995360838](https://github.com/takumiecd/torchcst/actions/runs/36995360838) で
`prepare → measure → ingest` が全て成功した。
[PR #15](https://github.com/takumiecd/torchcst/pull/15) の merge commit
`1f9f1774242b1df9fe256de87c89ceef29d4b092` を、
[測定 JSON](../benchmarks/requests/colab-linear.json) から固定して測定した。
L4・独立実行1回で、full / window512 と dense 参照の正しさ・完全 step・capture/replay peak memory を確認した。

Neon に今回の5 worker records / 31 metrics が保存され、
`github-actions-colab-bridge` の出所と実際の workflow run ID / source SHA を照合した。
元 JSON の byte-exact export と同じ provenance での再送が重複を増やさないことも確認した。
DB 全体は3 runs / 15 worker records / 93 metrics（先の接続確認の2 runs を含む）。

[機械記録](../benchmarks/automation/results/actions-neon-20261002.json)に request、job、archive hash、誤差、全時間サンプル、runtime と保存・停止確認を残した。
raw source / result archive、GitHub artifacts、workflow logs は ignored `benchmarks/automation/evidence/actions-20261002/` に保持した。
owned GPU slots は全て停止、一時 runner は measure job 完了後に自動登録解除された。
kernel の認証・採用や leaderboard / dispatch 規則生成は行っていない。

## この記録の実行方式

上記は Mac の一時 runner を使った初回の検証記録。
以後の運用は [参加・運用ガイド](benchmark-contributions.ja.md) の GitHub-hosted runner 方式に移す。
初回検証時のローカル runner 設定・raw evidence は削除せず private state / ignored directory に保持する。

