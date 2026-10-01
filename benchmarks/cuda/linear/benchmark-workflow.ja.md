# CUDA benchmark の入口と旧 `prototypes/` の整理

2026-09-29。[配置方針](../../../experiments/cuda/linear/repository-layout.ja.md)の補足。新しい方式の試作には
自由な探索を許すが、他の貢献者が数値を変えて比較するときは、共通の
`benchmarks/` 入口を使う。探索用スクリプトを増やし続けない。

## 提供する操作

```text
python -m benchmarks.cuda.linear.run --case benchmarks/cuda/linear/cases/mapped-1024.yaml
python -m benchmarks.cuda.linear.run --case benchmarks/cuda/linear/cases/mapped-1024.yaml --set window_rows=512
```

これは**実装予定の case 方式の CLI**。現時点で `--case` / `--set` は未実装。
2026-10-01 の研究 branch では、normalized full/window に限定した共通入口
`run.py --algorithm ... --size ... --output ...` を追加した。
[初期 registry と測定契約](../../../src/torchcst/nn/_backends/notes/cuda-registry-v1.ja.md)を参照。case には
演算の意味、形状 `N/K/M`、atom 数、dtype、seed、device、比較対象の recipe ID、
測定モード、warmup と反復数を書く。人が変更する値は `window_rows`、
`cache_windows`、tile、warp、candidate list 方式など、そのアルゴリズムが
明示している項目に限定する。未知の項目、非対応の組合せ、許容範囲外の値は
測定前に理由付きで拒否する。複数候補は case に明示的に並べ、意図せぬ巨大な
直積探索を起こさない。

bench runner は共通の setup と測定手順を持ち、アルゴリズム別 adapter が
モデルの構築、oracle、対応条件、1 完全学習ステップを提供する。forward のみ
の候補を学習ステップの結果と比較しない。各候補は同じ初期状態・入力・seed で
比較し、実行順を交互にして差を測る。GPU の同期、warmup、計時、ピーク割当、
エラー収集は共通 runner が担当する。CUDA Graph は eager と別のモードとし、
capture の準備と replay の計時を区別する。各アルゴリズムの `README.md` は
使える case、設定項目、正確性の閾値、既知の制限へリンクする。
PyTorch の [benchmark utility](https://docs.pytorch.org/docs/stable/benchmark_utils.html) は
非同期 accelerator の同期を考慮する。Graph の warmup、静的な入出力 buffer、
replay の条件は [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
に合わせて実装する。

出力 JSON は候補ごとの時間分布、基準との差、ピーク割当、forward・入力勾配・
atom 勾配の誤差、失格理由、選択 recipe と上書き値、GPU / CUDA / PyTorch /
Triton 版、source commit、測定条件を含む。近似方式は Exact の誤差基準に
合格したように表示せず、近似品質を別に記録する。既定の出力先は `output/`。
採用判断に必要な小さな結果だけを `benchmarks/cuda/linear/evidence/` に保存し、
[evidence 台帳](dispatch-evidence.ja.md)に採否を書く。benchmark 結果から
起動時に自動学習・自動採用はしない。レビュー済みの recipe と選択木への反映は
別の変更として扱う。

## `prototypes/` の整理結果と残る作業

整理前の `prototypes/` は 141 ファイルで、9 つのテストと多数の文書が
参照していた。開発中の実装と測定に必要な 44 Python モジュールを
`experiments/cuda/linear/` と `benchmarks/cuda/linear/` へ移し、残る 97 ファイルを
削除した。旧コードは[Git 履歴](../../../experiments/cuda/linear/legacy-prototypes.ja.md)で参照できる。
現行の `src/`、テスト、README、bench から `prototypes` への import と
実行コマンドはなくした。開発中の CUDA kernel は移動だけで、中身を変更していない。

残る作業は、開発者が方式を確定したときに実装と oracle を `src/` へ昇格し、
選別済みの bench を共通 case runner へ揃えること。旧 CLI の永続互換は
要求しない。既存の個別 CLI は数値を変えて実測できる入口として当面残す。

移行の最小単位は一つのアルゴリズムとその oracle・bench。測定候補として
残す方式と歴史的な反例を区別し、実装を増やす前にどちらかを決める。
