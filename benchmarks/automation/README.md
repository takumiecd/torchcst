# Official measurement path

測定依頼を固定し、公式の Colab 実行経路から得た観測を中央で検証して保存する。
通常の `torchcst` 実行や dispatcher の JSON 読み込みから、この tooling は呼ばない。

```text
GitHub Actions prepare → 固定 request
                           ↓
ログイン済み端末 → shared Colab queue → 正しさ / 完全 step / memory
                           ↓
GitHub Actions ingest → 依頼との照合 → PostgreSQL / Neon
```

## 境界

| ファイル | 担当 |
| --- | --- |
| `contract.py` | JSON の読み込み・request の作成・内容 hash、source inventory、job の識別。読み込みは標準ライブラリのみ |
| `colab.py` | 既存 shared pool への提出・待機・検証済み結果の回収。DB を import しない |
| `driver.py` | pool 内の専用環境を作り、固定 snapshot を既存 benchmark に渡す |
| `ingest.py` | request と結果を照合し、既存 DB API にまとめて追記。Colab を import しない |
| `__main__.py` | 上記の CLI。DB 接続は `ingest` のときだけ |

Plan / Case の設定をここへ再実装しない。Case にある全候補と baseline / dense を測る。
入口は [測定 JSON](../requests/colab-linear.json)。`catalog`、`cases`、`targets`、runtime、
回数、timeout を全てファイルに明記する。Actions / CLI に GPU や回数の上書き設定は置かない。
JSON が Case × GPU ×独立回数を指定し、各 Case の全候補 Plan を比較する。
`targets` は `T4` / `L4` / `A100` / `H100` / `G4` を指定できる。
GPU の取得可否は Colab のプラン・残量・空き状況に依存する。指定外 GPU は拒否する。
上限は6 job、各1200秒、公式 workflow は1 GPUずつ処理する。
Case の `rounds` は実行内の timing sample 数で、独立実行の回数とは別。

Request は full commit SHA、全 tracked source の SHA256、完全な Case / Plan snapshot、
対象 GPU種別 と runtime、job key / repetition を持つ。内容の hash が request ID。
作成・提出は clean な committed checkout で行う。依頼作成後の source 変更を拒否する。
Colab では指定 GPU / runtime と全 source hash を確認し、別 GPU へ fallback しない。
現在の環境は Torch 2.11.0+cu128 / CUDA 12.8 / Triton 3.6.0。
SM数やメモリ容量は機種の固定値を推測せず実機から記録し、GPU variant を分けて保存する。
job-local venv を使い、system package を変更しない。
入力・target・初期 parameter は CPU で生成して GPU にコピーする。
公式 driver は子プロセスの CPU capability を `default` に固定し、
GPU 世代やホスト CPU による乱数生成の違いを避ける。方式と実際の capability を
結果の metadata / DB protocol に記録する。生成・転送は timing の外で行う。

## ローカルでの使用

Colab CLI とアカウントは利用者が端末に設定する。
[リポジトリ内の公開 Colab ツール](../../tools/colab/README.md)に従う。
`~/.codex/skills` や所有者の固定メールアドレスには依存しない。
既存の共有 L4 クライアントも同じ host-wide queue を使用する。

```bash
# repository root。JSON を指定するだけで固定・実行・回収する。
PYTHONPATH=src python -m benchmarks.automation run \
  --request benchmarks/requests/colab-linear.json --output output/colab-run

# Actions のように依頼作成と GPU 実行を分ける場合。
PYTHONPATH=src python -m benchmarks.automation prepare \
  --request benchmarks/requests/colab-linear.json --output output/request.json

# 標準ライブラリだけの端末から、既定の host-wide queue に提出する。
# 他の supervisor が動いていれば共有し、なければ指定 GPU を1台ずつ drain / stopする。
python3 -m benchmarks.automation colab \
  --request output/request.json --output output/measurement

PYTHONPATH=src python -m benchmarks.automation validate \
  --request output/request.json --results output/measurement/results

# provenance.json は中央の実行主体が作る。GPU の投稿内容をそのまま採用しない。
PYTHONPATH=src python -m benchmarks.automation ingest \
  --request output/request.json --results output/measurement/results \
  --provenance output/provenance.json
```

`ingest` だけが環境変数 `DATABASE_URL` を読む。schema は事前に既存の
`python -m benchmarks.database migrate` で owner が初期化する。
通常の取り込みでは SELECT / INSERT の専用 role を使う。
[DB の初期化・権限](../database/README.md)を参照。
ローカル検証にも同じ CLI / PostgreSQL schema を使える。

`pending.json` に pool job ID と回収先を保存する。待機を中断しても新しい job を
自動再送せず、pool の `status` / 運用手順で状態を確認してから回収する。

```bash
python3 -m benchmarks.automation collect \
  --request output/request.json --pending output/measurement/pending/pending.json \
  --output output/recovered-results
```

wait timeout は lease を解放しない。`recover` を自動実行したり、他チャットの
supervisor / pool-owned session を直接操作したりしない。
Raw source / result archive と log は既存 pool の job directory に保存される。
Actions へ渡す bundle には元の benchmark JSON と提出・回収 hash を残す。

## GitHub Actions の初回設定

[benchmark.yml](../../.github/workflows/benchmark.yml) は手動起動のみ。
`main` の reviewed source を測り、PR のコードを自動実行しない。
測定 JSON の repository 内パスを入力すると `prepare` → `measure` → `ingest` が順に動く。

事前に以下を設定する。

1. Neon の schema を owner で migrate し、取り込み専用 role の接続文字列を用意する。
2. GitHub Environment `benchmark-database` に Secret `NEON_DATABASE_URL` を登録する。
   接続文字列はログ・チャット・tracked file に貼らない。Neon の TLS 設定を維持する。
3. Environment `benchmark-colab` と `benchmark-database` の deployment branch を
   `main` に制限する。必要なら required reviewer も設定する。
4. ログイン済みの macOS / Linux 端末に self-hosted runner を登録し、
   `colab-client` label を付ける。この CPU bridge は測定 batch のときだけ起動できる。
   Python 3.10+ と Colab CLI / アカウント設定を事前に準備する。
5. workflow が `main` に入った後、Actions → Benchmark measurement → Run workflow。

Runner の登録と label は [GitHub の手順](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow)
に従う。GPU VM を常駐させる必要はない。bridge が offline なら `measure` は実行待ちになる。
設定・ログインは自動代行しない。runner の任意 PR 実行を許可しない。

DB の Secret を参照するのは GitHub-hosted `ingest` job の最後の step だけ。
結果検証に先立って GPU 上のコードや artifact 内の Python を実行しない。
workflow metadata（repository / run ID / source SHA / actor）は中央から付与する。
取り込み job の retry attempt は測定の出所に使わず、同じ回収 bundle の再送を同じ出所として扱う。
external actions は full commit SHA に固定する。
[GitHub の security reference](https://docs.github.com/en/actions/reference/security/secure-use)を参照。

## 保存・検証の意味

全 job の bundle を検証してから、既存 `Database.import_linear` を一つの transaction で呼ぶ。
Case / Plan / source / GPU / runtime / raw hash / execution UUID と既存 adapter の計測条件を検査する。
同じ結果の再送は idempotent、独立した実行は別の観測。同じ UUID を別 repetition に使うと拒否する。
同じ原本に異なる provenance を後付けして上書きすることは拒否する。
Request と job / pool receipt、中央から付与した出所情報は run の provenance に保存する。
投稿の `certification` は信用せず、中央の記録も `not assessed` とする。

Benchmark FAIL の JSON は保存し、集計用 metric は作らない。
provisioning / transport / 環境確認で benchmark JSON 自体がない場合は観測を捏造しない。
その job は `jobs_without_observations` と bundle / pool log に残る。
初期実装では実行依頼の durable scheduling table はなく、運用失敗の永続記録は
pool と Actions artifact。Actions artifact は14日保存の中間成果物であり、Neon の代替ではない。

公式経路・hash 照合は、任意の端末所有者による測定値改変を暗号学的に証明するものではない。
再測定、独立確認、kernel の採用レビュー、leaderboard の eligibility は別工程。
ここから認証や dispatcher の既定採用を自動で行わない。

今後の provider は同じ request / result contract の adapter として追加する。
評価・leaderboard / selector 生成はこの実行経路と別の tooling にする。
複数 GPU の協調演算は別 benchmark contract / topology として version を持たせ、
単一 GPU の時間を足して代用しない。
