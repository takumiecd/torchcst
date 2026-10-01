# L4共有キューの運用

通常はL4を1台使い、AIによる考察・実装は並列に進める。GPUを待つ実験が増えたら、
キューを変えずに実行担当を2〜3台へ増やす。各台の測定は必ず一つずつ行う。
このプログラムは一つの実験がGPUをフルに使うことを制限しない。

## 初回起動と日常の操作

スキルは `~/.codex/skills/colab-l4-pool` に登録する。
このリポジトリの `tools/colab-l4-pool` へのリンクなので、プログラムを修正した場合も
全チャットが同じ版を参照する。Python 3.9以上で動作し、追加のPythonライブラリは不要。
Colab CLIは現在インストール済みのものを使い、勝手に更新しない。

各探索担当はGitツリーとPython driverを指定して `submit` する。
オーケストレーターは一つだけ `serve` を起動し、継続可能なtool sessionまたはterminalで保持する。
実行担当がすでにいれば、二重に起動せずそのキューへ提出する。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py status
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source /Users/ware10sai/Desktop/personal_research/torchcst \
  --script /absolute/experiment_driver.py --timeout 600 --label candidate-a
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py serve --workers 1 --idle-seconds 60
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py wait JOB_ID --timeout 3600
```

driver引数は `submit ... -- --size 8192` のように最後へ渡す。
`submit --wait` はすでに稼働中のsupervisorがある場合に使う。待機はキュー待ちも含む。
待機時間切れはGPUを停止せず、所有権も解放しない。再度 `wait` / `status` で確認する。

空キューのまま起動してもGPUは確保しない。キューから実験を取得した時に必要な台だけ確保し、
連続する実験ではVMを再利用する。idle時間経過後に各VMを停止する。
一括測定なら `serve --idle-seconds 0` でキューを処理し終えて即停止できる。
serve終了後に提出された仕事は次のserveまでqueuedのまま待機する。
台数の変更はserveが終了してから行う。稼働中の台数変更・自動増設は行わない。
他のslotで長い実験が続いている間にidle workerが退出しても、後続jobが来たら
停止確認済みのslotへworkerを再起動する。上限は最初の`--workers`の値で変わらない。
待機中のGPUは60秒で停止し続けるため、workerが再び仕事を受けられることは
GPUをidle確保し続けることを意味しない。全worker終了・待ちjob無しならserveも終了する。

## macOSのVPNと通信経路

2026-09-30の実機確認では、通常の経路でColabの結果取得・停止要求が接続エラーになり、
`curl --interface en0` のWi-Fi経由では接続できた。VPN全体を変更せず、Colab CLIと
そのkeep-aliveプロセスにだけIPv4のinterface bindingを指定する機能を追加した。
現在のこのホストでは `en0` を設定している。実験が動くColab VM側の通信には影響しない。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py configure --interface en0
# システム既定へ戻す場合
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py configure --interface default
```

設定は共通state rootの `settings.json` に保存される。稼働中のsupervisorがいる場合は
変更を拒否する。指定interfaceがなくなった場合は通信を失敗させ、別の経路へ黙って
fallbackしない。CLIだけに `PYTHONPATH` を追加して専用 `sitecustomize.py` を読み込ませ、
CLI内のソケット接続へmacOSの `IP_BOUND_IF` を指定し、認証用urllib、HTTP転送、
Jupyter用WebSocketの全経路を対象にする。
CLI内の名前解決をIPv4へ限定する。システムのPython設定には追加しない。
これはこのホストの実際の経路障害への対応で、macOS以外には設定しない。

## 実験と結果の契約

提出時にGitのtrackedファイルとignoreされていないuntrackedファイルを固定する。
Gitにまだcommitされていない変更も含む。`.git`、ignoreされたoutput、venvは含めない。
資格情報を自動判定して除外する仕組みはないので、提出前に対象を確認する。
snapshotの途中で同じソースを編集しない。同じ候補を測り直すときは新しいjobとして提出する。

Colab側はSHA256を照合し、実験固有のsourceへ展開する。driverは別Pythonプロセスで実行し、
Notebook kernelへtorch/CUDAをimportしない。通常終了・例外・driverの時間切れのいずれでも、
driverのprocess groupに残った子プロセスを停止してから結果を固める。
driverで新しいprocess groupを作ったりdaemonを残したりしない。
一つのVMのシステム環境は共有されるため、system-wideなpip installを避け、必要な場合は
source内の専用venvを使う。シェル処理はdriverから `subprocess.run([...], check=True)` で呼べる。

保存先は `CST_JOB_OUTPUT`。例えば：

```python
import json
import os
from pathlib import Path

output = Path(os.environ['CST_JOB_OUTPUT'])
# setup, independent correctness checks, warmups and complete-step measurements
(output / 'benchmark.json').write_text(json.dumps(measurements, indent=2))
```

driverの制限時間はsetupも含む。転送・GPU確認・結果回収は別の上限がある。
性能測定はdriver内でCUDA同期を適切に行う。poolのwall timeは性能の指標として使わない。
複数台を使う場合は各VMで共通baselineを測り、runtime versionsと実GPUを照合する。

`~/.local/state/colab-l4-pool/jobs/JOB_ID/` に以下を保存する。

- `spec.json`：引数、source SHA256、全sourceファイルのSHA256、提出元
- `source.tar.gz`：固定されたソースとdriver
- `transport.log`：Colab CLIの転送・実行ログ
- `results/`：driverのstdout/stderr、GPUと環境、終了コード、成果物、manifest
- `receipt.json`：結果archiveのSHA256、担当slot、session、回収時刻

ファイルは自動削除しない。記録を必要な場所へ保存した後、不要な古いjob directoryを削除してよい。
キューのSQLite DBとslot記録を勝手に編集・削除してはいけない。
結果回収後のremote cleanupに失敗しても検証済み結果は保持するが、supervisorは次の仕事を止める。
`succeeded` はdriver成功と回収・検証を表す。VMの停止完了はslot状態とlifecycleログで確認する。

## 停止・異常時

通常の停止はserveにSIGINT/SIGTERMを送り、現在の仕事の終了とVM停止を待つ。
長い実験はdriverのtimeoutまで待つ場合がある。kill -9やtool sessionの強制破棄を避ける。
supervisorが途中で死んでもOSのlockが解除されるだけで、DBとslot記録が残る。
次のserveはそのVMを再利用せず、`recover` を要求する。

接続切れ、取得失敗、hash不一致は自動再試行しない。
ランタイムを停止してserverのassignmentから消えたことを確認する。
停止確認ができなければquarantinedのままにして、別実験を同じVMへ流さない。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py recover
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py status
```

recoverはpool所有のsessionだけを停止する。作成応答が失われてendpointが分からず、
serverにassignmentが残っている場合は自動解決しない。
`slots/SLOT/lifecycle.log` とColabのsession一覧を調べ、実際の所有関係を確認する。
他の研究で使用中のVMを停止しない。

排他は同一ホストでこのpoolを使う実験同士について保証する。直接CLI、Notebook UI、
別ホストからの操作を技術的に禁止するものではない。
poolの `cst-pool-*` VMへの実行操作はこの窓口へ統一する。
