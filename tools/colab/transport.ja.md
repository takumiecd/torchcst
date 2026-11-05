# 結果転送とproxy認証の期限

[English](transport.md) | **日本語**

`serve --fresh-proxy-transport`を指定した場合、ホストのpoolは設定されたColab CLIと
同じ環境にインストール済みのSDKを使い、完了した結果archiveの回収と検証済み成果物の
後処理を行う。既定では既存CLI転送を維持する。CLIは実行可能な絶対パスの
Python shebangと`from colab_cli.cli import main`を持つPython launcherである必要があり、
それ以外は拒否する。CLIのインストールや更新は行わない。uploadと学習実行は既存の
CLI経路を維持する。実験source、数値計算、checkpoint、科学的identityは変更しない。

downloadとcleanupのたびに、独立したホストprocessが既存sessionとOAuth資格情報を
読み、Google userinfoで設定済みaccountを確認してassignment一覧を取得する。
supervisorが記録したendpointとacceleratorに一致するassignmentが正確に一つ必要である。
更新されたproxyはcached HTTPS hostとpathを維持しなければならない。endpointの変更、
不在、曖昧さは拒否する。proxy資格情報とOAuth更新はメモリ内だけに保持し、adapterは
StateStoreを開かず、資格情報の保存、session同期、VM確保、slot/DB変更、実験再実行を
行わない。資格情報を含み得るSDK出力と例外は抑止し、転送失敗は固定の診断文だけを
返す。通常の状態変更とlifecycleの責任は既存supervisorに残る。

downloadはprocess全体で900秒を上限とし、HTTP requestにも上限を設け、転送完了後に
結果archiveへ確定する。既存のarchive、source、file manifest検証を終えてから、通常の
receiptとjob状態を記録する。cleanupのネットワーク接続前に、helper自身がreceiptの
slot・session・archive hash、保持済みresult manifestの全inventoryとhash、
result/source identityを独立に確認する。
このjobのremote treeとarchiveを削除するpool生成codeだけを、既存kernel/session IDを
使って実行する。永続化callbackは使わない。完了確認が得られない、またはcleanupが
不確かな場合はdispatchを停止し、検証済みlocal成果物を保持する。不確かなcodeを
再実行しない。

次のuploadや学習実行へ古いcached proxyを持ち越さないため、supervisorはVMごとの
job数を明示的に制限できる。

```bash
python3 tools/colab/scripts/pool.py serve --workers 1 --idle-seconds 0 \
  --max-jobs-per-session 1 --fresh-proxy-transport
```

既定値`0`はsession再利用を維持する。正の値では、上限に達した所有VMを停止し、
server assignmentからendpointが消えたことを確認してから、次のqueued jobのVMを
確保する。停止確認に失敗した場合はslotを隔離し、後続jobはqueuedのままにする。
worker数と共有queueは変更しない。この転送変更だけでは学習中のWebSocketが任意の
長時間にわたり維持されることを証明しないため、長時間の実機gateは別途必要である。

## 大きなsource archive

source uploadは既存の所有session用CLIを使い、全体の期限を900秒とする。
32 MiBを超えるarchiveは、最大32 MiBのchunkへ順序通り分割して送る。
全upload requestのprocess上限は180秒と残りの全体期限の短い方とし、保持するlocal一時fileは
一つのchunkだけとする。`source-upload.json`に転送方式とchunkのsize・path・hashを
記録し、提出済みjob specと元archiveのhashは変更しない。

最初のupload前に、hostは有効なjob IDと正確な`/content/<job-id>.tar.gz`だけを許可する。
chunked source展開とdriver起動の前に、remote runnerも同じ所有targetを独立に確認し、
そのjobが所有する正確な順序付き
chunk pathから、排他的な新規作成でarchiveを復元する。各chunkのsizeとSHA256、
復元archiveの元SHA256を確認した後、そのchunkだけを削除する。
既存の出力先archive、symlink、想定外のpath、不正なdescriptorは拒否する。
失敗時は既存の隔離・停止経路に渡すためremote証拠を保持し、uploadも実験も
再試行しない。CLI requestごとのsizeを制限するものであり、認証方式、実験の学習期限、
数値計算、checkpoint選択は変更しない。
