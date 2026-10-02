# 検証記録：2026-09-30

実装は `scripts/pool.py`、Colab内の実行担当は `scripts/remote_runner.py`。
追加Python依存なしのホスト側キューとして作成し、このホストのPython 3.9.6で実機運用した。
スキルは `~/.codex/skills/colab-l4-pool` へsymlink登録し、既存のColab・GPUリソーススキルと
リポジトリの `AGENTS.md` から参照させた。

## ローカル検証

`tests/test_colab_l4_pool.py` の23件が通過。Ruff lint/format、diff check、
skill-creatorのquick_validateにも合格した。

- 並列提出、4つの独立プロセスによるclaimで各jobが一度だけ取得されること
- 別プロセスから二重supervisorを起動できないこと
- 1・2・3 workerで、同じslotの実験が重ならないこと（Colabは模擬）
- 待機時間切れ・実行中のcancel拒否が所有権を解放しないこと
- 接続失敗で後続dispatchを止め、recover前の再起動を拒否すること
- stop失敗でquarantineを維持すること、別アカウントでは確保しないこと
- ソースの固定、symlink拒否、危険なresult archiveの拒否
- 実際の子プロセスによる成功・例外・timeout、残った子プロセスの終了
- L4以外のhardwareをdriver実行前に拒否すること
- network指定が子プロセスにだけ適用され、不正interfaceはfail closedになること

## L4実機検証

最終確認では、一つのL4ランタイムを再利用し、以下を順番に実行した。

| 実験 | 結果 |
|---|---|
| 0.25秒で時間切れになるdriver | SIGKILL、exit=-9、timeout=true、成果物回収 |
| 時間切れ後のGPU forward/backward | 正常終了、CPU参照の行列積と照合 |
| もう一度GPU forward/backward | 正常終了 |
| 意図的なPython例外 | exit=1、stderrと成果物回収 |

GPUはNVIDIA L4、23,034 MiB、driver 580.82.07。
Colab Python 3.13.15、PyTorch 2.11.0+cu128、CUDA 12.8、Triton 3.6.0。
4実験のdriver実行区間が重ならず、同じsessionを使ったことを生記録で確認した。
全ダウンロードのmanifest SHA256が一致し、全実験の後にowned VMを停止、
server上にactive sessionがないことを確認した。
GPUを同時に2・3台確保する確認は行っていない。複数workerの排他と並列処理は模擬で検証した。

## 実機で見つかった接続障害

最初はColab内の処理が完了したが、結果downloadとstopが接続エラーになった。
後続jobのdispatchを止め、slotをquarantineした。Wi-Fi `en0` を指定すると接続できたため、
pool側でのみ通信経路を指定する機能を追加し、recoverで停止を確認した。
HTTPだけを指定した中間版ではJupyter WebSocketが停滞し、host timeoutで再度dispatchを止めた。
最終版はsocketの接続と名前解決を対象とし、認証・転送・WebSocket・keep-aliveすべてに適用した。
既存VPNやシステム全体のネットワーク設定は変更していない。
最終の4実験はこの設定で完了した。

この検証は共有実行基盤のsmoke testであり、CSTの速度や正確性の測定ではない。
排他制御は同一ホストでpoolを使う実験同士の協調規約で、直接CLIや別ホストからの操作を禁止する
OS上のセキュリティ境界ではない。

[生結果・summary・source hashes](../evidence/20260930)を保存した。
完全なsource/result archivesと転送ログはホストの `~/.local/state/colab-l4-pool/jobs/` に残す。

```bash
.venv/bin/python -m pytest -q tests/test_colab_l4_pool.py
.venv/bin/ruff check tools/colab-l4-pool/scripts tests/test_colab_l4_pool.py
```
