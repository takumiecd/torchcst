# Actions から Colab で測定し、共有 Neon へ提出する

通常の入口は、既存の測定 JSON を選ぶだけ。
GitHub の CPU runner が Colab の GPU を確保・測定・回収・停止する。
初回の Colab ログイン以降は、測定中に利用者の PC を動かしておく必要はない。

```text
参加者の fork の Actions → 参加者の Colab GPU → benchmark artifacts
                                                  ↓ 実行番号を提出
中央の Actions → upstream source / JSON / 結果を検証 → 共有 Neon
```

## 参加者の初回設定

1. 公開 repository を fork し、Actions を有効にする。
   upstream の main をそのまま同期する。Fork 固有の source 変更は中央取り込みで拒否する。
2. 自分の端末で `google-colab-cli==0.6.0` をインストールし、`colab login` と
   `colab --auth oauth2 whoami` で自分の Colab アカウントを確認する。
3. Fork の Settings → Environments で `benchmark-colab` を作り、deployment branch を
   default branch（通常 `main`）だけに制限する。
4. その Environment に Secret `COLAB_AUTH_JSON` と variable `COLAB_ACCOUNT` を登録する。
   Secret は CLI の `~/.config/colab-cli/token.json` の authorized-user JSON 全体、variable は
   確認したメールアドレス。接続情報をチャット・issue・repository に貼らない。

CLI が既に認証済みなら、GitHub CLI で Secret に直接渡せる。

```bash
gh secret set COLAB_AUTH_JSON --repo YOUR_NAME/torchcst \
  --env benchmark-colab < ~/.config/colab-cli/token.json
gh variable set COLAB_ACCOUNT --repo YOUR_NAME/torchcst \
  --env benchmark-colab --body YOUR_COLAB_EMAIL
```

Google の OAuth 認証情報を GitHub の protected Environment に預ける運用となる。
認証が失効した場合は自分の端末でログインを更新し、Secret を差し替える。
Neon の接続情報や所有者の Colab 認証は参加者に渡さない。

## 日常の測定

1. Fork の main を upstream と同期する。
2. Actions → **Benchmark measurement** → Run workflow。
3. default branch を選び、`request_file` に [既存 JSON のパス](../benchmarks/requests/README.md)を指定する。
   最初の短い L4 測定は `benchmarks/requests/colab-linear.json`。
4. 完了した Actions の URL を提出する。中央の管理者が取り込む。

`benchmark-request` / `benchmark-results` は14日保持する中間成果物。
中央への提出は期限内に行う。取り込み後は元の benchmark JSON を Neon に保存する。
GPU / transport 失敗のログも保持し、別 GPU・新しい実行へ自動で再送しない。

## 中央管理者の設定と取り込み

中央は `benchmark-colab` に自身の Colab 設定、`benchmark-database` に
`NEON_DATABASE_URL`（取り込み専用 SELECT / INSERT role）を置く。両 Environment は main のみ許可する。
中央 repository 自身での測定は、検証後にそのまま Neon に保存される。

Fork の artifact は GitHub API の認証が必要。
`benchmark-database` に Secret `BENCHMARK_ARTIFACT_READ_TOKEN` を置く。
提出元の repository の Actions を読み取れる token を使う。
GitHub App / fine-grained token なら対象 repository と Actions: read に限定する。
対象を限定できる token は参加 repository が増えたらアクセス範囲を更新する。
中央自身の artifact は workflow の `GITHUB_TOKEN` で取り込める。

Actions → **Import benchmark contribution** → Run workflow で main を選び、
`repository` に `参加者名/torchcst`、`run_id` に提出された実行番号を指定する。
参加者が中央 Actions を直接起動する権限は必要ない。
中央は fork relationship、workflow、default branch、source commit、JSON・結果・GPU/runtime を確認して保存する。

これは観測の記録であり、kernel の承認や dispatcher 採用ではない。
投稿者が作った artifact の hash が一致しても、測定値の改変を暗号学的に証明して排除できるわけではない。
独立した再測定と採用判断は別工程として扱う。

## 開発時の hosted runner 検証

Main へ入れる前の短い実機確認だけ、管理者が repository variable
`BENCHMARK_VALIDATION_REF=refs/heads/確認対象ブランチ` を設定できる。
`benchmark-colab-validation` Environment の branch policy をそのブランチだけに制限し、
確認済みコードに Colab Secret / account を登録する。DB job は main 以外では実行されない。
検証後は variable と検証 Environment を削除する。通常の参加者はこの設定を必要としない。
