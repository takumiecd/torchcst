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

- source 全 CPU suite: **496 passed / 197 skipped**。CUDA 等の未使用経路は skip。
- Ruff check／format、git diff --check を確認。
- 変更前後の 11 ケース・112 Tensor は **bitwise identical**。
  Product／Strip × Euclidean／Sphere／Torus、ambient／intrinsic、trainable 明示点を含む。
  座標、復号中心、距離、offset、勾配、retraction、transport、atom 初期値、forward、dX、dPを比較。
- wheel の Python 139 module が source と一致。旧 geometry／chart／pattern／adapter は含まれない。
- wheel SHA256: `a878c423884797b91fef702f39272f23fa27946ba72cc7b77e121d80910d5e3d`。

最初の移行テストでは旧クラス・旧メソッドを前提にした6テストが失敗した。
backend 関数を使う観測／monkeypatch、新しい宣言の validation に修正し、数値の期待値は維持した。
追加の trainable snapshot テストでは float32 の丸め値と Python float の厳密一致が失敗したため、
Tensor の同精度比較に修正した。数値演算本体の変更は行っていない。

raw logs、変更前 source、比較 script／Tensor、wheel は ignored な
`benchmarks/cuda/linear/evidence/chart-geometry-cleanup/` に保存している。

```bash
/Users/ware10sai/Desktop/personal_research/torchcst/.venv/bin/python -m pytest -q
/Users/ware10sai/Desktop/personal_research/torchcst/.venv/bin/ruff check src tests benchmarks experiments
```

L4 での全 suite、CUDA Graph、既存の normalized Strip complete-step 時間／メモリ測定も
同じ共有 pool driver で検証し、取得した実機結果を追記する。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source /Users/ware10sai/.codex/worktrees/cuda-dispatch-registry-v1/torchcst \
  --script /Users/ware10sai/.codex/worktrees/cuda-dispatch-registry-v1/torchcst/benchmarks/cuda/linear/validate_dispatch_registry.py \
  --timeout 900 --label chart-geometry-cleanup -- SOURCE_COMMIT --full-suite
```

API の使い方は [geometry ガイド](../src/torchcst/geometry/README.md)と
[数学契約](chart-geometry.ja.md)を参照。
