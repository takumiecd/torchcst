# ベンチマーク結果の提出

参加手順の入口は [CONTRIBUTING.md](../CONTRIBUTING.md)。この文書には詳細な提出規則と管理者の運用手順を記載する。

GPU での計測と、共有 Neon への提出を分ける。
自分の CUDA GPU、Colab、レンタル環境で benchmark を動かし、完成した結果 JSON を
**この repository の Issue に添付**する。提出用の fork / branch / PR、
中央の Colab 認証、Neon の接続情報は不要。

```text
手元のGPU / Colab / レンタルGPU
    → ignored output/ に結果JSON
    → torchcst の提出Issueに添付
    → 中央Actionsで形式・内部整合性・投稿者・提出上限を確認
    → Neonに原本・観測・提出者・区分・ポイントを追記
    → Issueへ受付結果を返信
```

## 計測と提出

現在受け付けるのは `benchmarks.cuda.linear.run` の完成した schema v1 JSON。
全体の status が FAIL の結果も保存する。診断用の別形式や未完成のファイルは受け付けない。
新しい benchmark 形式には明示的な adapter を追加する。
現在の測定実装は CUDA GPU 向けであり、AMD / MPS で動作するという意味ではない。

```bash
python -m pip install -e '.[cuda,benchmark-db]'
python -m benchmarks.cuda.linear.run \
  --case benchmarks/cuda/linear/cases/normalized-1024-broad.json \
  --output output/normalized-step.json

# 提出前の確認。GPU・GitHubログイン・DB接続は不要。
python -m benchmarks.submissions check output/normalized-step.json
```

`output/` は `.gitignore` 対象。ログや snapshot を Git に追加しない。
runner は execution UUID / UTC 開始時刻、Plan、Case、実際の GPU・CUDA・Torch・Triton、
各 timing sample、誤差、capture/replay の allocated / reserved peak を記録する。
小さい独立 oracle と、大きい fixture の性能測定は異なる検証範囲として残る。

1. Issues → New issue → **Submit benchmark result** を選ぶ。
2. `Benchmark JSON` 欄へ完成した JSON を１つドラッグして添付する。
3. Submit new issue を押す。ファイルをアップロードしただけでは受付は始まらない。
4. **Submit benchmark observation** Actions が起動する。
5. 保存されると、観測 ID・記録された投稿者・区分・ポイントを返信し、Issue を閉じる。

本文には添付 JSON のリンクを１つだけ置く。認証情報、Python コード、ZIP は送らない。
上限は１ファイル5 MiB。公開 Issue のファイルと計測情報は誰でも閲覧できる。
登録前の不備は本人による本文編集で再検査できる。登録済み Issue の観測は差し替えず、
別の計測は新しい Issue で送る。登録失敗時は Issue を開いたままにしておく。

## 一般・認定と信頼ポイント

正本は [公開設定](../.github/benchmark-submissions.json)。通常ユーザーの事前登録は不要。

| 区分 | 受け付ける件数 | 初期信頼ポイント |
| --- | --- | --- |
| 一般 | GitHubユーザーIDごとにUTCで1日10実行 | 1 |
| 認定 | 件数上限なし | 10（管理者が個別に変更可能） |

１つの JSON が１回の benchmark 実行。その中に複数 Plan・複数 timing sample があっても１件。
完成した FAIL / correctness-only の実行も１件として数える。
日付は投稿者が記載した測定時刻ではなく **DBが受け付けたUTC日付**を使う。
認定を取り消した場合も、その日の既存提出件数は一般の上限判定に含む。

同じ結果の再送は、別の Issue や別アカウントからでも観測を増やさず、追加枠を消費しない。
JSON の空白・キー順だけを変えた再送も同じ観測。同じ数値でも別の execution UUID の
実行は別観測として追記する。同じ UUID に異なる結果を付けると拒否する。

投稿者は GitHub API の **Issueの作者ID・ユーザー名**から取得する。
再実行した管理者や添付 JSON の自己申告を、投稿者として使わない。
ユーザー名は提出時点の表示を保存し、改名後も数字のIDで同じ人として扱う。
信頼ポイントは提出 JSON から採用せず、中央の公開設定から割り当てる。
認定・提出免除・ポイントを測定値の真実性の証明として扱わない。
自動でポイントを増やしたり、採点に掛け合わせたりするアルゴリズムは実装しない。

## 管理者による認定

一般の上限・基礎ポイント、認定の既定ポイントは公開設定の共通値で変更する。
認定するアカウントだけ `recognized` に追加する。ID は GitHub API で確認する。

```bash
gh api users/ACCOUNT_NAME --jq '{github_user_id: .id, login: .login}'
```

設定の例（実際のIDに置き換える）：

```json
{
  "github_user_id": 123456,
  "login": "approved-account",
  "reason": "管理者が確認した計測提供アカウント"
}
```

共通の10点を変える場合だけ、同じ entry に `"trust_points": 20` などを追加する。
判定に使うのは数字のIDで、`login` は公開表示用。団体か個人かで自動認定しない。
設定変更は通常のレビューを経て main に入れる。公開設定とGit履歴が現在値・変更履歴となる。
Neonにも、受け付けた時点の設定原本・hash・区分・ポイントを保存する。
変更後も過去の観測や付与記録は書き換えない。

## 中央ActionsとDBの設定

[benchmark-submit.yml](../.github/workflows/benchmark-submit.yml) を main に置く。
既存の `benchmark-database` Environment の `NEON_DATABASE_URL` を使用する。
main のみ許可し、取り込み専用 SELECT / INSERT role を使う。
`BENCHMARK_ARTIFACT_READ_TOKEN` は不要になった。

公開設定を中央コードで読み、JSON の取得・検査を終えてから、保存 step だけに DB Secret を渡す。
Issue の文字列や添付ファイルを shell / Python コードとして実行しない。
公開 GitHub 添付だけを取得し、GitHub API token を添付先へ転送しない。
URL、redirect、容量を制限し、提出したPython・workflow・source archiveを中央で実行しない。

一般ユーザーの件数検査・原本保存・投稿者との紐付けは１つのDB transaction。
同時提出もユーザー単位で直列化し、複数Issueが同時に最後の枠を使うことを防ぐ。
同じ観測が別アカウントから同時提出された場合も、１つの観測・１つの帰属にまとまる。

既存DBには **migration 2** と、新テーブルへの role 権限が必要。
新コードへの切り替え時に owner で適用する。旧コードは migration 1 を要求するので、
旧 workflow が動いている途中で先に本番 migration を適用しない。

```bash
python -m benchmarks.database migrate
```

owner で、既存の取り込み role に追加する（認証情報はSQLに書かない）：

```sql
GRANT SELECT, INSERT ON benchmark.submission_policies, benchmark.submissions
    TO torchcst_actions_ingest;
```

参照用 role にも必要に応じて新テーブルの SELECT を追加する。
既存の Plan・Case・run・metrics の原本と provenance は維持される。
以前の実行結果を新しい提出経路から受け付けても、元の出所は変更しない。

管理者は Actions → **Submit benchmark observation** → Run workflow →
`issue_number` を指定して受付を再実行できる。再実行者ではなく元の Issue 作者に帰属する。
提出済み観測の確認：

```bash
python -m benchmarks.database submissions
python -m benchmarks.database submissions --submitter-id GITHUB_NUMERIC_ID
```

## ColabとコードPR

Colab は任意の計測手段。[公開Colabツール](../tools/colab/README.md) を使える。
既存の **Benchmark measurement** Actions も、Colab 認証を設定済みのアカウントで利用できる。
この補助 workflow は結果を artifact に保存するだけで、Neonへ自動登録しない。
結果内の `artifacts/benchmark.json` を取り出して、同じ提出フォームに添付する。
Colab の認証をGitHubへ預けたくない場合は、自分の端末やColabで実行して結果だけ提出する。

新しい kernel / Algorithm のコードは通常のPRでレビューする。
計測データをコード変更としてコミットせず、提出 Issue / 観測 ID をPRから参照する。
受付時点の中央 registry にない Plan / 新しい形式はまだ取り込めないため、
コード側の追加と adapter の対応を先にレビューする。
DBへの保存はkernel承認やdispatcher採用とは独立している。

全ての提出は `consistency_checked` / `self_reported` と明記する。
GPUの実行や測定値の真正性をJSONのhashで証明できるわけではない。
再現確認・除外・leaderboard生成・dispatcher採用は、この受付とは別の処理とする。
