# Colab execution tools

リポジトリから使う公開ツール。Colab の **T4 / L4 / A100 / H100 / G4** を明示的に指定し、
ソースを固定して転送、実行、hash 付きの結果回収、owned runtime の停止までを管理する。
対応する指定名は [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)と
インストール済み CLI 0.6.0 の `colab new --help` で確認済み。
利用できる GPU は Colab の契約・残量・空き状況に依存する。
取得できなければ失敗し、別の GPU で代用しない。

## 初回設定

macOS / Linux、Python 3.10+、Google の Colab CLI を使う。
CLI の認証設定は Google の手順に従い、自分のアカウントで行う。
動作確認済みの CLI は0.6.0。自動で更新・ログイン・課金しない。
CLI のバージョンを変えた場合も、使用する command / flags を確認する。

```bash
uv tool install google-colab-cli==0.6.0
colab --auth oauth2 whoami
python3 tools/colab/scripts/pool.py configure --account YOUR_ACCOUNT_EMAIL
```

`whoami` の認証フローを完了し、想定するアカウントであることを確認する。
CLI に必要な OAuth client 設定も Google の手順に従う。
このツールは serve 時に認証済みアカウントと設定したメールアドレスを照合する。
未設定・別アカウントなら GPU を割り当てない。メールや token を source に固定しない。
設定は host-wide queue の `settings.json`、OAuth の認証情報は Colab CLI の端末設定に保存される。
DB の接続情報をこの端末の測定ツールに渡す必要はない。

## 測定 JSON を実行

[測定 JSON](../../benchmarks/requests/colab-linear.json)を commit し、repository root から実行する。

```bash
PYTHONPATH=src python -m benchmarks.automation run \
  --request benchmarks/requests/colab-linear.json --output output/colab-run
```

GPU は JSON の `targets` に指定する。例えば `["A100"]`、`["L4", "G4"]`。
Case / GPU /回数を別の選択画面で上書きしない。
実行前に依頼を固定し、commit と全 source を検査する。利用者に source を手動転送させない。
正しさ確認・時間・完全 step のメモリを既存 benchmark で測定する。
取得した GPU の実名・compute capability・SM数・メモリ容量を結果に記録する。
各 GPU の kernel 適合性と数値契約は実行時に確認し、未検証の機種の性能を保証しない。

## 共有 queue と ownership

キューは全 GPU・全ローカルチャットで1つ。
既存の運用状態を保持するため、ディレクトリは `~/.local/state/colab-l4-pool` を継続使用する。
GPU や agent ごとの live `--state-root` を作らない。
同じ host の supervisor は1つだけ。別の computer から同じ VM を操作しない。

`serve --gpu L4` は L4 の job だけ、`serve --gpu G4` は G4 の job だけを処理する。
既存 L4 クライアントが、高価な別 GPU の job を意図せず割り当てることはない。
通常の measurement CLI は request に書かれた GPU ごとに1台ずつ処理し、停止を確認する。
既存 supervisor が動いていれば、その ownership を奪わず待つ。

```bash
python3 tools/colab/scripts/pool.py status
python3 tools/colab/scripts/pool.py serve --gpu A100 --workers 1 --idle-seconds 0
```

研究用の直接 submit も可能だが、公式測定の記録には request-bound automation を使う。

```bash
python3 tools/colab/scripts/pool.py submit --gpu H100 \
  --source /absolute/checkout --script /absolute/driver.py --timeout 600
```

待機の中断はキャンセルや VM 解放ではない。自動再実行・自動 recovery はしない。
実行・停止が不確かな slot は隔離し、後続の allocation を止める。
既存 [pool の運用手順](../colab-l4-pool/references/operations.ja.md)の ownership / recovery に従う。
旧 `tools/colab-l4-pool/scripts/pool.py` は同じ実装・同じ queue を呼ぶ既存クライアント用入口。
Skills のインストールは、この公開ツールの利用条件ではない。

G4 などの消費が大きい GPU は、短い測定依頼で明示的に選択する。
Colab に存在しない4090/5090などを、これらの指定名に読み替えない。
他の provider は、別 adapter として measurement contract に接続する。
