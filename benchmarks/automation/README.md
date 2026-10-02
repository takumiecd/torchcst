# Colab measurement helper

測定依頼を固定して Colab で実行する任意の補助ツール。
共有DBへの公開提出は [Issue受付](../submissions/README.md) に統一する。
通常の `torchcst` 実行や dispatcher の JSON 読み込みから、この tooling は呼ばない。

```text
GitHub Actions prepare → 固定 request
                           ↓
GitHub-hosted runner → Colab queue → 正しさ / 完全 step / memory
                           ↓
結果JSON → 同じrepositoryの提出Issue → 中央受付Actions → PostgreSQL / Neon
```

## 境界

| ファイル | 担当 |
| --- | --- |
| `contract.py` | JSON の読み込み・request の作成・内容 hash、source inventory、job の識別。読み込みは標準ライブラリのみ |
| `colab.py` | 既存 shared pool への提出・待機・検証済み結果の回収。DB を import しない |
| `hosted.py` | GitHub runner 上で OAuth Secret を復元し、対話なしで認証・account を確認。checkout 外に保存 |
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
`prepare` / `measure` は GitHub-hosted Ubuntu runner で動く。
測定 GPU は Colab のアカウントから一時確保する。自分の PC を起動しておく必要はない。

設定は [参加・運用ガイド](../../docs/benchmark-contributions.ja.md) を参照。
通常は既存の [測定 JSON 一覧](../requests/README.md) から repository 内パスを選ぶ。
新しい条件を増やすときだけ JSON / Case / Plan をレビューして main に追加する。

この workflow は測定・artifact 保存まで動き、Neon の接続情報を使わない。
各実行の `artifacts/benchmark.json` を取り出し、中央の提出 Issue に添付する。
Actions run ID を指定する旧 fork artifact 取り込み経路は削除した。
中央の [benchmark-submit.yml](../../.github/workflows/benchmark-submit.yml) が投稿者と公開設定を
照合し、一般はUTCで1日10実行・1点、認定は件数上限なし・既定10点として受け付ける。
ローカルGPU・Colab・レンタルGPUでも同じJSONの受付処理を使う。
DB Secret は中央の `benchmark-database` Environment の保存 step にだけ渡す。

OAuth は runner の HOME に一時復元し、token を更新して account を確認してから確保する。
対話ログインに fallback しない。Secret を driver の環境変数や source snapshot に渡さず、
終了時に削除する。CLI は `google-colab-cli==0.6.0`、外部 Actions は full SHA に固定する。

GPU の停止が確認できなかった場合は自動再測定しない。
`benchmark-pool-state` artifact に、この runner が所有した session と retrieval receipt を保存する。
強制キャンセル・runner 消失では後処理が完了しない場合があるため、Colab 上の残存 session と
artifact を確認し、[pool の復旧手順](../../tools/colab-l4-pool/references/operations.ja.md)で
所有した session だけを停止する。同じアカウントでローカルと Actions を同時に運用しない。

## 固定requestの検証と管理者向け保存API

以下の `ingest` CLI は管理者が固定 request / pool receipt を照合するためのAPI。
公開提出の入口は `benchmarks.submissions` であり、通常参加者はこのCLIやDB接続を使わない。

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

## 実機確認

[L4 / G4 と PostgreSQL の検証記録](../../docs/measurement-json-verification.ja.md)を参照。
同じ測定 JSON の入力・初期 Parameter hash の一致、独立 oracle、完全 step、
memory、回収 hash、保存・再送・元 JSON の復元を確認した。
[初回設定記録](../../docs/benchmark-actions-setup.ja.md)に Neon の実接続・専用 role・Secret・runner 接続確認を記録した。
旧 GitHub の `prepare → measure → ingest` も L4 の１ケースで実行済み。
同じ記録に実行 URL、Neon の出所照合・元 JSON 復元・重複防止、GPU / runner 終了確認と次回の起動手順を残した。

[GitHub-hosted runner の実機確認](../../docs/hosted-benchmark-verification.ja.md)で、
Mac runner を起動せず L4 の正しさ・完全 step・memory・回収・停止を確認した。
