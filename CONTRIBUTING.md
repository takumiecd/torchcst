# torchcstへの参加

コードの改善と、GPUでのベンチマーク結果の提供を受け付けています。
コードはPull Request、計測結果はこのリポジトリのIssueから提出します。
計測結果の提出には、fork・提出用ブランチ・PR・Neonの接続情報は不要です。

## コード・kernelを追加する

開発の入口はこの文書です。CUDA kernelの具体的な実装・登録・測定は
[カーネル開発ガイド](docs/kernel-development.ja.md)を参照してください。

```text
変更の数学的な契約を決める
    → codex/ ブランチで実装・Algorithm/Recipeを登録
    → Plan/Caseを用意し、宣言・独立oracle・回帰テストを確認
    → 同じ条件のbaseline・candidate・denseを実GPUで測定
    → 正しさ・完全step時間・ピークメモリ・証拠を記録
    → PRでレビュー・必要な検証を通してGitHubでmerge
```

### 1. 開発環境を用意する

repoルートで実行します。外部参加者はforkからPRを作成できます。

```bash
python -m pip install -e '.[dev]'
git switch -c codex/my-kernel
python -m tools.kernel_dev test --suite cpu
```

GPUホストでは `python -m pip install -e '.[dev,cuda]'` を使います。
CUDAが見える環境でCPU検証を行うときは `CUDA_VISIBLE_DEVICES=''` を付けます。
GPU・実DBが必要なテストのskipは、GPU・DB検証を完了したことにはなりません。

### 2. 実装と比較条件を登録する

数学的なKernelSpecとCUDAのAlgorithm/Recipeは区別します。同じ演算の計算方式を
変える場合は `_backends/cuda/algorithms/` に実装を置きます。
既存のrecipe調整、新Algorithm、新しい数学的な演算で必要な変更は
[追加する種類と変更箇所](docs/kernel-development.ja.md#追加する種類と変更箇所)に整理しています。
本体の `src/torchcst/` から `benchmarks/` や `tests/` に依存しないでください。

公開APIの契約は [README.md](README.md)、現在のpackage境界は
[backendの配置規約](src/torchcst/_backends/README.md)を正本とします。
演算の定義はatomごとのkernelの和であり、行列への因数分解は計算方式の一つです。
atom数・Parameterのshapeは学習中固定し、振幅・幅・中心などの学習可能な値はatomの
Parameterに持たせます。Moduleの状態所有、微分計算、optimizerの更新を分離してください。
旧store・birth/death/merge・slot remapping・Pullback Adamなどの互換surfaceを復活させません。

比較用のPlan catalogとCaseは既存Linear runnerで読み込める形式にします。
登録済みの候補で手順を試す例は、GPUなしで実行できます。

```bash
python -m tools.kernel_dev prepare \
  --plans benchmarks/cuda/linear/plans-local-contraction.json \
  --case benchmarks/cuda/linear/cases/local-contraction-64-sigma3.json \
  --candidate persistent-supportprep-band \
  --output output/kernel-comparison
python -m tools.kernel_dev check \
  --plans output/kernel-comparison/plans.json \
  --case output/kernel-comparison/case.json
```

`prepare`はbaseline・指定candidate・denseを含む比較用JSONと、元ファイルのhashを
新しいディレクトリへ保存します。候補は `--candidate` を繰り返して指定できます。
既存の出力先は上書きしません。新しい実装を自動生成・登録するコマンドではありません。
新候補を登録した後は、そのcatalogとCaseを入力にして同じ手順を使います。

### 3. 正しさ・性能・メモリを確認する

独立oracleでY・dX・全atom勾配、境界、支持、正規化を検証します。
optimizerを変える場合はParameter・moments・stepも比較します。
GPUでGraph replayと可変幅を確認し、未計装の完全学習stepを測定します。
時間とcapture/replay込みのallocated / reservedピークを同じ条件で比較してください。

[実行できる最小例](docs/kernel-development.ja.md#登録済みalgorithmで手順を試す)に
CPU確認からGPU correctness、完全step測定、提出前検査までのコマンドを載せています。
所有者の共有GPUを使う場合は、下の[研究運用規則](#所有者の研究環境と運用規則)も適用します。
自分のGPUで測定する外部参加者に、所有者のGPUアカウントやDBの資格情報は必要ありません。

### 4. 記録してPRを作成する

コード・テスト・簡潔な研究ノートをcommitし、raw logs・trace・tensor・source copiesは
ignored `output/` または `benchmarks/**/evidence/` に保存します。
計測結果はこの文書のIssue経路から提出し、PRから提出Issueや観測IDを参照できます。
新しいPlanや結果形式は中央registry / adapterが対応してから受け付けられるため、
コード側の対応を先にレビューします。

PRテンプレートに、変更の目的・契約、実行した検証、未実施の検証、hardware/runtime、
完全step時間・ピークメモリ、独立run数、source/result hashと再現コマンドを記載します。
PRの **CPU validation** Actionsは宣言検査・CPUテスト・wheel/sdist buildを実行します。
GPUの正しさ・性能や実DBは別途確認します。公開dispatcherへの採用も別の判断です。

mainへ直接pushせず、名前付きブランチをpushしてGitHub PRから統合します。
GitHubでmergeされた後に、ローカルmainを `origin/main` へfast-forwardします。

## ベンチマーク結果を提供する

```text
自分のGPU / Colab / レンタルGPUで計測
    → output/ に完成した結果JSONを出力
    → 提出前にローカルで形式を確認
    → torchcstのIssueにJSONを添付して投稿
    → 中央GitHub Actionsが形式・内部整合性・投稿者を確認
    → Neonで重複と提出上限を確認し、原本と観測を追記
    → Issueに観測ID・提出者・区分・ポイントを返信して閉じる
```

### 1. 用意されたCaseで計測する

現在受け付けるのは、CUDA GPU向けの `benchmarks.cuda.linear.run` が生成する
完成したschema v1 JSONです。GPUごとのAlgorithmの対応条件を満たす環境で実行してください。
Caseは入力・モデル・計測条件、Planは実行するAlgorithmとその設定を表します。
CaseやPlanの選び方は [CUDAベンチマークガイド](benchmarks/cuda/linear/README.md)を参照してください。

リポジトリのルートで実行する例：

```bash
python -m pip install -e '.[cuda,benchmark-db]'
python -m benchmarks.cuda.linear.run \
  --case benchmarks/cuda/linear/cases/normalized-1024-broad.json \
  --output output/normalized-step.json
```

runnerは実行UUID・UTC開始時刻、Plan・Case、GPU・CUDA・Torch・Tritonの情報、
時間サンプル、誤差、CUDA Graph capture/replayを含むallocated / reserved peakなどを記録します。
allocatedとreservedは別の指標です。この形式ではGPUプロセス全体のメモリ使用量を測定していません。
正しさの検査に使う小さい独立oracleと、大きい入力での性能測定は異なる検証範囲です。

`output/` はGit管理対象外です。結果JSON・ログ・trace・tensor・source snapshotを
コミットせず、手元に保管してください。完成したFAIL結果も診断用の観測として提出できます。

### 2. 提出前にJSONを確認する

```bash
python -m benchmarks.submissions check output/normalized-step.json
```

この検査にGPU・GitHubログイン・DB接続は不要です。
未完成の結果や対応していない形式は提出できません。

### 3. Issueに添付して提出する

1. このリポジトリの **Issues → New issue → Submit benchmark result** を選びます。
2. `Benchmark JSON` 欄へ完成したJSONを1つドラッグして添付します。
3. **Submit new issue** を押します。添付しただけでは受付は開始されません。
4. **Submit benchmark observation** Actionsの処理と、Issueへの返信を確認します。

一般・認定とも同じフォームを使います。本文には添付JSONのリンクを1つだけ置いてください。
1 Issueにつき1実行分のJSON、最大5 MiBです。複数Planや時間サンプルを含んでも1件です。
添付ファイルと計測情報は公開されるため、認証情報を含めないでください。

### 4. 受付結果を確認する

中央ActionsはGitHub APIからIssue作者の数値ID・ユーザー名を取得します。
JSONに記載したユーザー名・区分・ポイントは、権限の判定に使いません。
形式、選択したPlan・Caseとの整合性、snapshot hash、PASS結果のworker構成、
時間サンプルと中央値、メモリ値の整合性などをCPUで確認します。

保存するstepだけにNeonの接続Secretを渡します。参加者のコードは中央で実行しません。
DBでは提出上限の確認と、原本・観測・提出者の保存を1つのtransactionで行います。
同時提出でも上限を超えて保存されないようにしています。

| 結果 | 次にすること |
| --- | --- |
| 保存成功 | 観測ID・提出者・区分・ポイントが返信され、Issueが閉じます。 |
| 同じ結果の再送 | 既存の観測が案内され、追加登録・追加の件数消費はありません。 |
| 形式などの不備 | Issueは開いたままです。登録前なら本人が本文を修正すると再検査されます。 |
| 一般の提出上限に到達 | UTCの日付が変わってから、本文編集または管理者による再実行で再度受け付けます。 |

別の計測は新しいIssueで提出してください。登録済みIssueの観測は差し替えません。
管理者がActionsを再実行しても、提出者は元のIssue作者です。

## 提出上限と認定

| 区分 | 新しい実行の提出上限 | 信頼ポイント |
| --- | --- | --- |
| 一般 | GitHubユーザーIDごとにUTCで1日10件 | 1 |
| 認定 | 件数上限なし | 既定10、管理者が個別変更可能 |

一般ユーザーは事前登録不要です。認定は管理者が
[公開設定](.github/benchmark-submissions.json)に数値GitHub ID・表示名・理由を登録します。
表示名を変更しても数値IDで判定します。認定でも形式・容量の検査は共通です。
認定の追加・変更方法は [管理者向けの説明](docs/benchmark-contributions.ja.md#管理者による認定)を参照してください。

件数は測定日時ではなくDBの受付UTC日付で数えます。完成したFAIL結果も1件です。
同じJSONの再送は別Issue・別アカウントからでも重複を増やさず、最初の提出者の帰属を維持します。
JSONの空白・キー順だけの変更も同じ観測です。独立した再計測は別の実行UUIDで追記し、
同じUUIDに異なる結果を付けると拒否します。

信頼ポイントは受付時点の設定から各提出に記録する値です。
提出回数に応じた自動加点や、ランキングへの重み付けは現在行いません。
認定や設定を変更しても、過去の観測・区分・付与記録は書き換えません。

## Colabと計測用Actions

自分のColabで測定する場合は [公開Colabツール](tools/colab/README.md)を利用できます。
Colab認証を設定済みの環境では、**Benchmark measurement** Actionsも利用できます。
こちらはコミット済みの計測依頼JSONでCase・GPU・回数などを指定し、結果を
`benchmark-results` Artifactに保存する計測補助workflowです。

Artifact内の各実行の `artifacts/benchmark.json` を取り出し、同じIssueフォームから提出します。
このworkflowはNeonへ自動登録しません。計測依頼の形式は
[requestsガイド](benchmarks/requests/README.md)を参照してください。
自分のGPUやColabで直接計測する参加者は、GitHubへのColab認証設定なしで結果だけ提出できます。

## 観測の保存と採用

DBには結果原本、Plan・Case、観測・指標、提出者ID・ユーザー名、Issue情報、受付日時、
区分・ポイント、受付時点の設定を保存します。独立した計測は追記して蓄積します。
受付結果は `consistency_checked` / `self_reported` と記録します。
この受付はGPUでの再計測を行わないため、測定値の真正性を保証するものではありません。

DBへの保存、kernel / Algorithmの承認、leaderboardの生成、dispatcherへの採用は別の処理です。
蓄積した観測を使う集計・ランキング・dispatch生成は
[別の生成ツール・Actions](benchmarks/dispatch/README.md)で行います。

中央Actions・Neonの設定、DB移行、管理者の再実行と観測照会については
[詳しい参加・運用ガイド](docs/benchmark-contributions.ja.md)を参照してください。

## 所有者の研究環境と運用規則

以下は、このrepoの所有者のGPU・研究worktreeを使う人とエージェントに適用する規則です。
外部参加者の自前GPUには、所有者の共有プール・GPU割当を要求しません。
通常の開発手順は上の「コード・kernelを追加する」を参照してください。

### Main integration must use pull requests

- The user explicitly prohibits direct pushes to `main` (2026-10-05).
- Publish changes on a named `codex/` branch, create a GitHub pull request, and
  merge through the pull request after the required validation passes.
- Do not substitute a local merge into `main` for remote PR integration. After
  GitHub merges the PR, fetch and fast-forward the local `main` to `origin/main`.
- Preserve commits, uncommitted files and needed ignored evidence before
  archiving or removing research worktrees. Keep recovery paths in research notes.

### Shared Colab L4 experiments

For Colab L4 research experiments, read and use
[the shared pool skill](tools/colab-l4-pool/SKILL.md). It is also installed at
`~/.codex/skills/colab-l4-pool` on the user's local host.

- All local chats/worktrees submit to the default host-wide queue. Do not
  create a per-agent live queue with `--state-root`.
- One orchestrator owns `serve`; other agents submit jobs and wait for results.
  Default to one L4. The orchestrator may select two or three when warranted
  within the user's authorized allocation scope.
- Do not directly upload, execute, restart or stop the pool's `cst-pool-*`
  sessions through the CLI or notebook UI. The pool serializes complete
  experiments and manages isolated subprocesses, result retrieval and cleanup.
- A wait timeout does not release a runtime. Use `status` and the documented
  recovery path for interrupted work; do not edit the database or slot records.

Other GPUs and sessions outside this pool retain their existing workflows.
This queue coordinates one host; another computer must not operate its VMs.

### Cross-generation experiment scope

- The user authorized Blackwell/G4 and other GPU generations on 2026-10-01.
  Keep the original L4 performance objective and record other devices separately.
- G4 use is cost constrained. The orchestrator alone selects and allocates short
  comparison runs for promising, validated candidates. Agents must not provision
  G4 independently or run broad G4 parameter sweeps.
- Measure a complete dense training step on each comparison GPU under the same
  model, batch, dtype, precision settings and optimizer contract. Preserve kernel
  normalization, sharp support/one-hot behavior, dX and all atom gradients.
- Record actual hardware and runtime versions, retrieve verified results and stop
  owned runtimes after the selected batch. A faster GPU is not evidence of an
  algorithmic improvement on L4.

### Complete-step memory objective

- The user added low memory consumption as a requirement on 2026-10-01.
  Evaluate time and memory together; retain normalization, support and gradient
  correctness. There is no user-specified numerical memory ceiling.
- Report peak allocated memory for the complete step including CUDA Graph
  capture. Distinguish allocated bytes, allocator reserved bytes and total GPU
  process usage; do not present a tensor budget as a measured GPU peak.
- Prefer bounded weight/weight-gradient scratch and buffer reuse. Keep new
  implementations on research branches until independent correctness checks
  and actual GPU time/peak measurements pass. Label sharp-only fixtures separately
  from the ordinary sigma-three performance objective.

### Research Git checkpoints

- Keep experiment worktrees on named `codex/` branches. Commit coherent source,
  tests and research notes at validated checkpoints; record negative results too.
- Keep generated logs, traces, tensors and frozen source copies in ignored
  `evidence/` directories. Preserve these files on disk; do not bulk-add them.
  Track concise results, job IDs, source hashes and reproduction commands in notes.
- Keep experimental alternatives on their research branches until validated for
  the main tree's mathematical and numerical contract.
- Before Colab submission, inspect `git status` and ensure raw evidence copies
  will not enter the source snapshot. Existing intentionally tracked evidence
  remains tracked; new raw output is ignored.
