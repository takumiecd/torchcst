# Chart の型と座標パッケージの分離（2026-10-02）

一つの ChartSpec に kind と全種類の nullable field を詰める方式を廃止した。
ChartSpec / ChartState は ABC とし、宣言と Tensor owner を Explicit / Product / Strip の
具体型へ分けた。Geometry・Pattern・Chart は同じ階層の別パッケージに置く。

```text
torchcst/
  geometry/   GeometrySpec、GeometryState、空間の presets
  patterns/   Line / Grid / Points の Spec、PatternState、点列の presets
  charts/     共通 ABC、Explicit / Product / Strip の Spec と State、compile_chart、presets
  kernels/    Profile・parameterization・normalization・KernelSpec / KernelState
  operators/  Chart layout と Kernel の組、live Operator
  _backends/  geometry / patterns / charts / kernels の実際の演算と dispatch
```

ChartSpec の field は geometry / shape / revision のみ。kind は各具体型の固定 ID で、
constructor に渡さない。Explicit は coordinates / trainable / optional spacing を持つ。
Product は axes を持ち、Strip は axes / tile_shape / axis / tile_pitch を持つ。
Explicit の座標を PointsPatternSpec で包む方式も廃止した。

State も同じ分担にする。Product に tile_pitch=None / spacing=None を置かず、
Explicit に axes を置かない。tile_grid / tile_count は StripChartState にのみある。
共通基底の抽象メソッドは reference / declaration などの所有・snapshot の契約であり、
座標生成や距離の演算は戻さない。backend は exact な具体 State / Spec の組と版を照合する。

旧 ChartState(spec) の代わりに compile_chart(spec) または具体的な State を使う。
Layer は Spec のまま受け取り、必要な State を構築する。入力・出力 Chart の組と単一 Chart、
trainable な明示点、bounded な直積展開、Strip の部分端 tile と物理的な pitch を維持する。
explicit の shape は座標表の点数と一致する論理 shape とし、明示した行列配置も表せる。

geometry_presets は euclidean / sphere / torus のみ。
pattern_presets は line / grid / points、chart_presets は points / grid / linspace /
product / strip。旧 mixed presets や geometry からの Chart / Pattern export は残さない。
root の具体型 export と geometry_presets / pattern_presets / chart_presets が公開入口である。

共通の finite scalar / shape / point-table validation と Tensor snapshot helper は内部の
_validation.py / _coordinate_state.py に置く。これらに距離・GPU・演算の実装は置かない。
Geometry の半径、representation、ambient chord metric と更新規則は変更していない。

Chart checkpoint は format_version=2 と具体型・shape・その型の構造を照合する。
旧 tagged ChartState の loader / kind 変換 / 互換 alias は用意しない。
Pattern / Geometry の元の checkpoint 契約は維持する。

## 検証

source とローカル配布 wheel の CPU 全テストは 524 passed / 201 skipped。
lint / format と使用例も確認した。変更前 96c3f86 との比較では、Product / Strip の
Euclidean、Sphere、Torus と ambient / intrinsic、trainable explicit を含む11ケース、
112 Tensor の positions・距離・offset・更新・輸送・初期化・y・dX・atom / 座標勾配が
bitwise 一致した。

型ごとの field / Tensor の不存在、ABC の直接構築拒否、concrete compiler、未登録型・版の
拒否、checkpoint の種類・旧形式の拒否、snapshot と演算の境界を追加検証した。

L4 の配布 wheel、既存 fused / normalized dispatch、Graph replay、時間・allocated peak は
共有 pool の validate_linear_unification.py --chart-suite で測定する。結果は完了後に追記する。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_linear_unification.py" --label chart-types --timeout 1800 -- SOURCE_COMMIT --chart-suite
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py serve --workers 1 --idle-seconds 0
```

[Chart の field と使用例](../src/torchcst/charts/README.md)、
[Geometry](../src/torchcst/geometry/README.md)、[Pattern](../src/torchcst/patterns/README.md) を参照。
