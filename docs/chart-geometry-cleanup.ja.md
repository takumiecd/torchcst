# Chart / Geometry の整理と検証（2026-10-02）

変更前は `56e74de`。Kernel と同じく、Geometry／Pattern／Chart を不変な宣言、共通の
Tensor State、backend の計算に分けた。旧 ABC、種類ごとの Module、宣言 adapter、
旧名の alias と旧 checkpoint の読み替えは削除した。

公開の geometry_presets が純粋な宣言を作る。CSTLinear／CSTConv2d は ChartSpec を
受け取って ChartState を所有し、既存 ChartState の明示共有も受け付ける。
NormalizedStripLinear も Strip の宣言を受け付ける。Operator は live State を参照する。

Sphere／Torus の ambient chord、center の ambient／intrinsic 表現、Strip の tile 配置、
端の部分 tile、support 検証、学習可能な明示点を保持した。Product／Strip は全点表を
保持しない。計算中の declaration snapshot は作らない。checkpoint は primitive metadata と
Tensor を保存し、異なる配置・保存表現を拒否する。ロード後の buffer 値も検証する。

## ローカル検証

- source／installed wheel 全 CPU suite: **502 passed / 197 skipped**。CUDA 等の未使用経路は skip。
- Ruff check／format、git diff --check を確認。
- 変更前後の 11 ケース・112 Tensor は **bitwise identical**。
  Product／Strip × Euclidean／Sphere／Torus、ambient／intrinsic、trainable 明示点を含む。
  座標、復号中心、距離、offset、勾配、retraction、transport、atom 初期値、forward、dX、dPを比較。
- wheel の Python 139 module が source と一致。旧 geometry／chart／pattern／adapter は含まれない。
- 最終 wheel SHA256: `6ed56e8d74fca17c45dfd2d341d4b56613deec2ba6013ff66b25c1dc7720bc7f`。
  最初の wheel は `a878c423884797b91fef702f39272f23fa27946ba72cc7b77e121d80910d5e3d`。

最初の移行テストでは旧クラス・旧メソッドを前提にした6テストが失敗した。
backend 関数を使う観測／monkeypatch、新しい宣言の validation に修正し、数値の期待値は維持した。
追加の trainable snapshot テストでは float32 の丸め値と Python float の厳密一致が失敗したため、
Tensor の同精度比較に修正した。
最終レビューで、explicit Chart の Pattern は個別 PatternState を作らないため nested
revision／独自型の検証が抜けることを見つけ、ChartState の構築境界で共通検証を追加した。
4 ケースの regression test を追加し、source／wheel の全 suite を再実行した。
CUDA registry が直接受け取る OperatorSpec にも nested Pattern revision の検証を加え、
Line／Grid の将来版を拒否する2ケースを追加した。この検証は GPU を起動しない control-plane の処理。
数値演算本体の変更は行っていない。

raw logs、変更前 source、比較 script／Tensor、wheel は ignored な
`benchmarks/cuda/linear/evidence/chart-geometry-cleanup/` に保存している。

```bash
/Users/ware10sai/Desktop/personal_research/torchcst/.venv/bin/python -m pytest -q
/Users/ware10sai/Desktop/personal_research/torchcst/.venv/bin/ruff check src tests benchmarks experiments
```

## L4 の検証と runtime の切り分け

最初の全 suite は Torch 2.11.0+cu130／CUDA 13.0／Triton 3.6.0 で
**750 passed / 7 failed / 2 skipped**。失敗はすべて routing の境界点で Torch oracle と
Triton の owner ID が厳密一致しないケースだった。

`l4job-2cb8cbbcfbbe48f3a29bf95cd4a4ac4b` で変更前 `56e74de` と変更後 `f178cd6` を
同じ CUDA 13.0 環境で比較し、両方が **13 passed / 同じ7ケース failed** と確認した。
今回の整理による regression ではない。CUDA 13.0 の既存の厳密 routing 比較の問題は
この変更で修正していない。比較後の venv 作成は ensurepip 不在で失敗したため、次の
ジョブは `--without-pip --system-site-packages` とジョブ内への pip install を使った。

一度の管理プロセス中断は `pool recover` で所有 VM の停止を確認し、未検証の結果として
記録した。最後のジョブ `l4job-e9d1ca4645ee45a881267a036b8f7810` は最終ソース `fcc621b` を
ジョブ内の Torch 2.11.0+cu128／CUDA 12.8／Triton 3.6.0 で検証した。

- installed wheel の Operator／dispatch／Profile suite: **120 passed**。
- Chart／Geometry／Optimizer／Graph／routing suite: **211 passed**。上記7ケースも通過。
- 合計 **331 個の異なるテストが成功**。最終ソースで CUDA 12.8 の全 suite を
  改めて実行したという意味ではない。
- 独立した小 shape の FP64 oracle による y／dX／全 dP／weight、support と Graph replay を確認。
- GPU は NVIDIA L4、driver 580.82.07、23034 MiB。pool の環境 probe は既定の CUDA 13.0 を
  記録するが、実際の検証・測定は job 内の CUDA 12.8。システムの package は変更していない。
- 結果 archive の SHA256 を照合。wheel／installed package／最終 source の Python 139 file が一致。
- 全 pool slot の停止を確認。失敗・中断の raw evidence も保存した。

## complete-step 時間とメモリ

N=1024、batch=128、atom=52,429、float32、TF32=false、fused/capturable AdamW
（lr=1e-4、weight_decay=0.01）。独立 process で forward／backward／optimizer の
完全 step を測定し、CUDA Graph capture／replay を含む peak を記録した。
通常条件は sigma=3、sharp 条件は別の fixture。full／window の初期 p hash は条件内で一致する。

| 条件 | 実装 | Graph median ms | peak allocated MiB | peak reserved MiB |
| --- | --- | ---: | ---: | ---: |
| sigma-three | normalized full | 0.622 | 51.85 | 106.00 |
| sigma-three | normalized window | 1.532 | 46.16 | 86.00 |
| sigma-three | dense Linear | 0.130 | 50.50 | 106.00 |
| sharp | normalized full | 0.543 | 51.85 | 106.00 |
| sharp | normalized window | 0.309 | 46.16 | 86.00 |
| sharp | dense Linear | 0.128 | 50.50 | 106.00 |

total GPU process usage は未測定であり、allocated／reserved の値と区別する。
大きな shape の全 atom 勾配を独立 oracle で全照合した測定ではない。小 shape の独立検証と
full／window の既存照合を併用した。性能による新規採用や既定 dispatch の変更はしていない。

job ID、source／result hash、失敗ケース、実測値は
[検証結果 JSON](../benchmarks/cuda/linear/results/chart-geometry-cleanup-20261002.json)に記録した。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source /Users/ware10sai/.codex/worktrees/cuda-dispatch-registry-v1/torchcst \
  --script /Users/ware10sai/.codex/worktrees/cuda-dispatch-registry-v1/torchcst/benchmarks/cuda/linear/validate_dispatch_registry.py \
  --timeout 900 --label chart-geometry-cleanup -- SOURCE_COMMIT --full-suite
```

このコマンドは VM の既定 runtime の全 suite を検証する。成功した CUDA 12.8 検証の
driver と frozen source は ignored evidence 内の上記最終 job archive に保存してある。
再現時は driver と同様に job 内の venv に `torch==2.11.0+cu128` を入れ、
`validate_dispatch_registry.py --operator-suite` と Chart／Geometry／routing の対象 suite を実行する。

API の使い方は [geometry ガイド](../src/torchcst/geometry/README.md)と
[数学契約](chart-geometry.ja.md)を参照。
