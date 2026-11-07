# 意味的フィールド別学習率の検証

[English](optimizer-field-lrs-20261107.md) | 日本語

`CSTAdamW`と`CSTOptimizer`はparameterizationが宣言する絶対初期値`field_lrs`を受け付けます。
base optimizerの正規化後、座標更新前に提案をスケールし、Atom表ごとに1つのParameter・状態・
カウンタを保ちます。group LRは未指定フィールドの値とPolar外部時間に使います。
scheduler基準値、checkpoint復元、raw decay制限、振幅・activityの連成は
[API契約](../cst-optimizer.ja.md#field-learning-rates)を参照してください。
この機能は最適・標準の学習率を決めるものではありません。

## 検証

検証したoperatorはPR150統合後の`a6c5f12a31442a03afb79cf47ebdb88db13afbb7`です。
このPRの以後の変更は結果の文書化だけで、runtime・testsは検証ソースとバイト一致します。
実GPUはRTX4090、compute capability8.9・24GB級、Torch2.11.0+cu126、Python3.12.9、
NumPy2.0.2です。LMタスクではTF32を無効にしました。テスト用pytest9.0.2の依存は数値環境と
分離しました。

- remote CPU：意味的フィールドの新規25テストがskipなしで成功しました。独立Adam式・
  宣言されたretraction、モーメント・カウンタ、scheduler・LR0、フィールドlayout、native/wrapper、
  SGD・AMSGrad・foreach・maximize、論理decay、group設定、checkpoint検証・復元を含みます。
- native CUDA：恒久8テストがskipなしで成功しました。native/wrapper × 振幅ピーク
  0.0024/0.0048/0.0096/0.0192で、入出力中心に異なるLRを設定し、0を含む5つのschedule係数、
  CPU FP64とのParameter・モーメント・カウンタ比較、シリアライズ後の次更新を確認しました。
  radius flat-torus fused Planを明示選択しました。
- 実LM：使い捨て8設定が成功しました。native/wrapper × 振幅8/16/32/64、中心・外部時間16、
  通常ピーク0.0003です。4ブロックN256・FFN4N・c8の全8 FFN投影をCSTにしています。
  各設定で公開APIと保全した研究用の正規化後提案アダプタを比較し、通常warmupの20実タスク更新に
  続き、次の順序付きbatchでピークLRのstress更新を5回実行しました。
  batch2/context128/accumulation4、AdamW(.9,.95)、epsilon1e-8、decay0、global clip1です。
  25回すべての損失・Parameterが完全一致し（観測最大パラメータ差0）、勾配・モーメントも
  記録したFP32許容差を通り、カウンタは一致しました。
- 全8設定でmodel・optimizerをシリアライズし、フィールド指定なしで構築した公開optimizerへ
  保存済み設定を復元しました。次のピークLR実タスク更新もパラメータ差0でした。
  学習後の全64投影でY/dX/全Atom VJPがCPU FP64の分割reference検証を通りました。

ピークstress部分は正しさのfixtureで、長期学習軌跡・精度・性能ベンチマークではありません。
すべて新規の使い捨てモデルを用い、元の学習・checkpointの再開や上書きはしていません。
証拠スクリプトの診断用スカラー変換warningは結果を変えず、学習軌跡を保持するものでもありません。
再試行は不要でした。

ローカルCPUのTorch2.13.0・Python3.11.15では、実装に対して2121テストが成功し、GPU・DB用の
2474件はskipです。フィールドの個別25テストも成功しました。wheel/sdistに同一バイトのhelperが
入ることを照合し、検証headのCPU CIも成功しました（run42320197633）。最終文書headのCIは
統合前に別途確認します。skipをGPU・DB検証済みとは数えません。
stock Graph benchmarkはcold qualification時にfield指定を拒否してeagerへ戻ります。
Graph対応や速度・メモリの優位は主張しません。

## 証拠と再現

- Operator archive SHA256: `aeac783c7053f97374337f1fb0a918ae5d69e90337b05d4a0df3ed9cc8c6af7a`。
- Research consumer: `351617dc81f529ce10e964e5a762f336b7c24b39`。
- Result archive SHA256: `64ae9bac51b1565d35cae76b875f45096a9145de663a2cb0f65ffd98367dfc06`。
- Result manifest SHA256: `c37a46db68037dbdc5b3b36a1d484df8047251e222f4b86719e2a399137471f2`。
- Full-LM result SHA256: `49125fd92525bde1fba4ee1220fbe92a5fdcb9534852b626ccc5d1453b880800`。
- Data manifest SHA256: `5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`。

manifestを含む回収8ファイルのハッシュを照合しました。研究の外部証拠保存先には固定ソース、
request、controllerとプロセス識別子、JUnit・ログ、完全LM検証器と結果を保持しています。
consumerはruntime packageではなく[cst-research](https://github.com/takumiecd/cst-research)に置きます。
FFN研究と同じタスク・データ契約を使いましたが、過去の品質結果の再実行・移行ではありません。

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_optimizer_field_lrs.py -q
python -m pytest tests/test_optimizer_field_lrs_cuda.py -q
```

実タスクは固定consumer・データと保全済み
`qualify-public-field-api-r16-02.py --data TOKEN_DIR --output NEW_DIR`を記録された環境で実行します。
ハッシュとraw証拠は外部に保持し、この文書は簡潔な結果記録です。
他kernel・hardware・Graphの検証は別途行います。
