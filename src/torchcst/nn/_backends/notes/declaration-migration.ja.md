# Kernel Geometry Chart の宣言分離

2026-10-01。`codex/cuda-dispatch-registry-v1` の段階的な研究変更。
公開クラスが所有する設定・Tensor 状態・checkpoint と、Torch の計算本体を分けた。
数学的な意味は frozen dataclass で参照でき、実行時の値・勾配・状態は従来の
Module が所有する。宣言だけから実行する Operator / factory は次の段階である。

## ソースの区切り

- `e27e3a4`: Kernel / Profile / parameterization / normalization の宣言、
  組み込み評価・微分・初期化・更新の Torch backend への分離。
- `0455e786edff630011a315ed8ef58127f7a5f00d`: Geometry / Pattern / Chart の宣言、
  座標生成・距離・center 変換・射影・retraction・輸送・Strip support の分離。
  検証 driver に配布 wheel の全テスト実行を追加し、bench の source hash を
  package 全体へ広げた。

現在の配置と移行の境界は[配置規約](../../../_backends/README.md)、
[Kernel](../../../kernels/README.md)、[Geometry と Chart](../../../geometry/README.md)を参照。

## CPU と移動前後の比較

ローカル Python 3.11.15 / Torch 2.13.0。移動前は 498 passed / 189 skipped、
宣言の境界テスト 36 件を追加した後は 534 passed / 189 skipped。
既存の Torch JIT deprecation warning 18 件は変わらない。

移動前 `18ec3fb` に対する Kernel 65 ケースで、値、一次・二次微分、座標更新、
状態輸送、checkpoint の keys / metadata を比較した。Geometry / Chart 15 ケースで、
Euclidean、Sphere / Torus の ambient / intrinsic 表現、Product / Strip / Explicit、
点の位置、距離、勾配、bounds、端の tile、retraction / transport と checkpoint を
比較した。有限値は `rtol=0, atol=0` で一致し、非有限値の位置も一致する。

正規化 WendlandC2 の coincident-center fixture には移動前から二次微分の非有限値が
4 ケースに計 6 要素ある。これを有限性の成功として数えず、元と同じ位置であることを
確認した。この変更ではその数学的・数値的な問題の修正は行っていない。

移した 192 メソッドの計算部分の AST も一致する。比較では docstring を除き、
移動による import の変更と、free function での明示的な `super(Class, self)` の
MRO anchor を正規化した。公開型名と checkpoint tag は既存のまま保つ。

ローカル wheel を `uv build --wheel` で作り、宣言・backend の 63 Python module が
ソースと同じ bytes で収録されていることを確認した。原始ログ・比較スクリプト・
移動前 Tensor は ignored `benchmarks/cuda/linear/evidence/kernel-declarations/` に保存。

```bash
python -m pytest -q
python benchmarks/cuda/linear/evidence/kernel-declarations/compare_kernel.py \
  --compare benchmarks/cuda/linear/evidence/kernel-declarations/before.pt
python benchmarks/cuda/linear/evidence/kernel-declarations/compare_geometry.py \
  --compare benchmarks/cuda/linear/evidence/kernel-declarations/geometry-before.pt
python benchmarks/cuda/linear/evidence/kernel-declarations/audit_bodies.py
uv build --wheel
```

## L4 配布 wheel 検証

測定 source は `0455e786edff630011a315ed8ef58127f7a5f00d`。
job ID は `l4job-94c7dc9c37a6478f8dd4ea8517f7407f`。
source archive SHA256 は
`ebd9bfd745ca046f0d89ebd111d373e36ed1f2454f4327058d6ef6c4167e6a6c`。
新しい原始 evidence が snapshot に含まれないことを確認して提出した。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_dispatch_registry.py" \
  --label declaration-migration-wheel --timeout 1200 \
  -- 0455e786edff630011a315ed8ef58127f7a5f00d --full-suite
```

全テストを配布 wheel から実行し、独立 FP64 oracle と Graph / optimizer の照合後、
N=1024 / M=128 の normalized full / window と dense の完全な AdamW 学習ステップを
測る。sigma-three と sharp は別 fixture とする。allocated peak は Graph capture
を含み、reserved と分ける。process 全体の使用量を allocated と混同しない。
この変更は方式の性能昇格や承認ポイントの導入ではない。
