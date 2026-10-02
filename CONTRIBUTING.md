# torchcstへの参加

コードの改善と、GPUでのベンチマーク結果の提供を受け付けています。
コードはPull Request、計測結果はこのリポジトリのIssueから提出します。
計測結果の提出には、fork・提出用ブランチ・PR・Neonの接続情報は不要です。

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

## コードを改善する

1. ブランチを作成し、必要なコード・テスト・説明を変更します。外部参加者はforkからPRを作成できます。
2. 変更に対応するテストを実行します。GPU実装では正しさと実GPUでの速度・メモリを確認します。
3. 計測結果は上記のIssue経路で提出し、PRから提出Issueや観測IDを参照します。
4. PRには変更の目的、Algorithmの考え方、検証方法と結果を記載します。

開発用依存関係とCPUでのテスト実行：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

GPUやDBが必要なテストは環境に依存します。実行できなかった検証をPRに明記してください。
新しいPlanや結果形式は中央registry / adapterが対応してから受け付けられるため、
コード側の追加と対応を先にレビューします。計測結果をコードPRへ大量にコミットする必要はありません。

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
