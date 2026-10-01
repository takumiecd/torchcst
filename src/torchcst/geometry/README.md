# Geometry と Chart の宣言

Geometry は空間・距離・center の座標表現を、Chart / SitePattern は有限な観測点の
配置を宣言する。第一段階では公開クラスの状態所有、constructor、checkpoint と
custom subclass の契約を保ち、組み込みの計算本体を Torch backend へ移した。

| 配置 | 責務 |
| --- | --- |
| `spec.py` | 不変の Geometry / Pattern / Chart 宣言。Tensor 演算と device の選択を持たない。 |
| `_declarations.py` | 公開クラスの設定から宣言を作る adapter。 |
| `geometry.py` | Geometry 設定・buffer・checkpoint と既存メソッドの互換入口。 |
| `chart.py`、`lazy_chart.py`、`pattern.py` | 点配置・状態所有・checkpoint と計算への遅延接続。 |
| `../_backends/torch/geometry/` | 埋め込み、距離、center 変換、射影、retraction、輸送、初期化。 |
| `../_backends/torch/charts/` | 点生成、選択、距離への接続、Strip の境界・支持域検証。 |
| `../_backends/torch/patterns/` | 規則格子の展開、明示点の選択と bounds。 |

Sphere / Torus の距離は現在の契約通り ambient chord 距離であり、geodesic 距離へ
変更しない。`representation` は center の保存形式を決め、embedding 次元と
center parameter 次元は分けて宣言する。

Strip の `tile_shape`、`axis`、`tile_pitch` は点の位置を決める。これらは CUDA の
block / window / warp 設定ではなく、宣言に残す。非連続な station 間隔、端の
部分 tile、Torus の circle axis と一周未満の範囲を保持する。

```python
from torchcst import ProductChart, LinePattern

chart = ProductChart(
    shape=(3, 4),
    axes=(LinePattern(3, spacing=0.2), LinePattern(4, spacing=0.4)),
)
spec = chart.declaration()
assert spec.kind == "product"
assert spec.geometry.id == "euclidean"
assert spec.features == 12
```

`declaration()` は設定用の snapshot。scalar buffer の参照、明示点表の CPU への
転送・tuple 化を行うため、forward / backward / Graph capture 内で呼ばない。
明示点表が大きい場合は snapshot 自体も大きい。学習可能な点の live Tensor と
勾配は元の Module が引き続き所有し、snapshot へ置き換えない。状態更新や
checkpoint load 後は必要に応じて宣言を作り直す。

Spec 単体から Module を生成する factory は次の移行段階。
[Operator](../operators/README.md) の状態 binding は既存 Chart Module を参照し、
Torch 計算は現在の Tensor / buffer を使う。
custom Geometry / Chart / Pattern を宣言として利用する場合は `declaration()` を
明示実装する。既存の計算メソッドによる拡張は維持する。
