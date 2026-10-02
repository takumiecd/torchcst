# PostgreSQL 保存基盤の検証（2026-10-02）

[検証記録](postgres-20261002.json)は PostgreSQL 14.19 / Python 3.11 / Torch 2.13.0 によるローカル検証。
Neon に接続せず、新規の GPU 計測も行っていない。

CPU suite は **584 passed / 117 skipped**。18件の既存 TorchScript deprecation warning。
内訳として adapter の35件、実 PostgreSQL integration の16件も実行した。
新規・変更 Python の Ruff と `git diff --check` は通過。
再送・並行 import・rollback・追記専用制約・最小権限・read snapshot・pagination・
migration history・原本復元・異なる adapter revision による追加指標を確認した。
既存 GPU テストの skip は CPU 上のためで、現 source の GPU 検証を示すものではない。

保存済み L4 job `l4job-549684aca01640e58055796b09c13c97` の broad / sharp artifact
（記録された source は短縮 SHA `8f2ec40`）を実 DB に import した。
２ Plan、２ Case、２ run、10 worker、62 metric を保持。
L4 の Plan 性能観測は４件。dense は別の参照として保存する。
両 artifact の export は元のバイト列・SHA256 と一致し、再送でも増えなかった。
これらは既存測定の保存検証であり、現在の kernel 性能を再計測した結果ではない。

検証途中で、macOS の Unix socket path 長制限によって専用 server の初回起動に失敗。
短い一時 socket directory に切り替えて TCP を無効化したまま起動・検証し、完了後に停止した。
過去の producer が持つ Plan JSON のキー順にも対応し、元の snapshot byte hash を検査する。
再送時は保存済みの原本から読み直すので、JSON のキー順を変更しても観測は増えない。

raw logs・検証用 DB・原本との比較記録は ignored
`benchmarks/database/evidence/postgres-20261002/` に保持している。
テストの実行コマンドは [database README](../README.md#テスト)を参照。
本番では別途 Neon の接続設定と owner / ingest / reader の権限設定が必要。
