# カーネル追加と改善の手順

2026-10-05時点。小さいCST線形変換の研究は既存のLinear runner、PostgreSQL保存、
ディスパッチ生成を使う。main統合はGitHub PRで行う。

## ソースと比較条件

1. mainから名前付き`codex/`研究ブランチとworktreeを作る。
2. 実装は`src/torchcst/_backends/cuda/algorithms/`へ置く。数学的な仕様と実行設定を分ける。
   local productは`local_product/`、研究Planの登録は`benchmarks/cuda/linear/local_product.py`。
   既存Planの設定と意味を保ち、新しい候補には明示的なrouteを与える。
3. `benchmarks/cuda/linear/plans-*.json`へ候補を追加し、`cases/`で比較対象、baseline、
   dense参照を明示する。固定σを導入せず、初期幅と各stepの幅更新を区別する。
4. 既存runnerの`--validate-only`で宣言を確認する。同じ初期Parameter・入力・dtype・
   optimizer契約で、時間とメモリを比較する。

新方式の実装をbenchmarkフォルダへ埋め込まず、別runnerを追加しない。
既存の`research_local_product`は研究用registryであり、公開CSTLinearの既定ではない。

## 正しさとGPU測定

独立したFP64 scalar oracleでY、dX、全atom勾配を確認する。正規化、支持境界、
単一サイト、空支持、切り出したdomain、retained backwardを含める。
更新を変える場合は公開CSTOptimizerとParameter、AdamW moments、stepを比較する。
可変σ、支持・配置の移動、Graph replayを確認する。

L4は[共有プール](../tools/colab-l4-pool/SKILL.md)へ提出する。提出前に`git status`
を確認し、raw evidenceがsnapshotへ入らないことを確かめる。独立のdriverはignored
`evidence/`に置き、既存runnerと検証を呼び出す。driverは新しい測定runnerではない。

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-contraction.json \
  --case benchmarks/cuda/linear/cases/local-contraction-64-middle.json \
  --polar-update fused --phase-diagnostics --kernel-diagnostics \
  --output benchmarks/cuda/linear/evidence/comparison.json
```

未計装の完全学習stepを採用判断の時間とする。診断は別Graphで計測し、足し合わせたり
主測定から差し引いたりしない。メモリはcapture/replay込みのpeak allocatedとreservedを
区別し、GPU process usageを測っていなければ未測定とする。21時間サンプルを21独立runと
数えない。改善が小さい場合は独立jobで再測定する。負の結果も残す。

## DB保存とディスパッチ候補

完成runner JSONを既存[DB入口](../benchmarks/database/README.md)へ追記する。
元のバイト列のexport照合と同じprovenanceでの再送を確認する。owned poolのjob ID、
source snapshot SHA256、結果archive SHA256を付け、資格情報はGPUに転送しない。
研究local productのadapter revisionは3、normalized Stripは2。

[生成依頼JSON](../benchmarks/dispatch/README.md)でsource、GPU/runtime、case、時間目的、
メモリ方針、最小独立run数を指定する。dataset・leaderboard・dispatch JSONを生成し、
未知の条件でfallbackすることを確認する。DB保存、候補生成、公開の既定採用は別の操作。

現在のlocal productの実行条件keyには初期rho分布が含まれない。同じshapeで幅分布だけ
異なるcaseを混ぜた自動選択は完成していないため、候補はcaseごとに生成する。
`baseline-peak`は比較方針であり、ユーザーが指定したメモリ上限ではない。

## 記録と統合

`docs/research-history/local-product/`へ採否、誤差、完全step時間とピーク、job ID、
source/result hash、再現コマンドを記録する。raw logs・tensors・source copiesはignored
`evidence/`へ保存し、bulk-addしない。検証済みのまとまりをcommitする。

named branchをpushし、GitHub PRを作成して必要な検証後にGitHubでmergeする。
mainへ直接pushしない。ローカルmainのmergeをリモート統合の代わりにしない。
GitHubでmergeされた後だけ、origin/mainをfetchしてローカルmainをfast-forwardする。
worktree整理前に必要なignored evidenceを保全し、復旧先を研究ノートへ残す。
