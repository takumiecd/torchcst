# 中心移動に基づくPolar活動（研究用の明示指定）

[English (canonical)](center-polar-activity.md) | 日本語

z=(s,t)、S=||z||²、a=amplitude_max*s/sqrt(S)、p=(S-1)/3、1<=S<=4、
r=L(|a|)^(1-p)*U(|a|)^pを保ちます。既存の診断がpをq/alphaと呼ぶ場合も、Atom密度とは別です。
活動源だけを交換し、Atom配置・Linear・初期化・振幅の角度方向の提案・中心のretractionは維持します。
物理半径はタスク勾配から切り離します。半径固定でW=aKなら、dLoss/da=<G,K>、
dLoss/dmu_j=a<G,dK/dmu_j>です。同じ現在値ではY・dX・全Atom VJPが一致します。
以後はsupportとPolarノルムが変わるため、学習軌跡は変わり得ます。

## 更新

受理後のFlatTorus最短移動を使い、半周期の同点は負方向とし、更新前の物理半径で割ります。

$$E_\mu=\tfrac12[(\Delta\mu_{out}/r_{out,old})^2+
(\Delta\mu_{in}/r_{in,old})^2].$$

元の振幅の接線更新uについてd=(z+u)/||z+u||とし、活動量をE_muへ交換します。
任意の弱Atom拡張項と厳密減衰は維持します。

$$S_* = \operatorname{clip}(S+\gamma E_\mu+3\eta\delta\omega(a_*),1,4),$$
$$S_{new}=\left[1-\frac{S_*-1}{S_*}e^{-4\lambda\eta}\right]^{-1},\qquad
z_{new}=d\sqrt{S_{new}}.$$

gammaはactivity_gain、lambdaはradial_regularization、etaはoptimizer外部時間です。
delta/omegaは既存の弱Atom拡張項で、中心活動だけを見る場合はrateをゼロにします。
拡大縮小後の振幅はamplitude_max*d[0]です。隠れたEMA、sigmoid/clip目標値の直接代入、
新しい座標embeddingは導入しません。小さな活動が続いても、必ずpが1になるとは限りません。

規格化した有限stepの活動量は制御上の選択で、タスク微分から一意に決まるものではありません。
optimizer時間では割らず、学習率・更新頻度・support・gainによって尺度が変わります。
最短移動なのでtorusの周回数は数えません。式だけで学習改善を示したとは扱いません。

## 研究用の明示的実行

RadiusBoundsのkernels.polarにactivity_mode="center_chord"を指定します。
帯域幅座標では拒否し、既存finite_chord/time_energyの演算・設定は保ちます。
fused CUDA・Graphは新modeを拒否し、checkpointはmode/gainを区別します。
新しいTorch eager AlgorithmはCPU/CUDAのfloat32/float64、radius Atom、
2つの1次元FlatTorus中心を対象とし、デフォルト登録しません。
活動源の差として比較する対照・候補では、同じ更新実装を使います。

```python
from torchcst import CSTAdamW, FixedSelector, make_algorithm_registry
from torchcst._backends.schema import DefaultRecipe, ExecutionPlan
from torchcst._backends.torch.algorithms.polar_update.radius_eager import (
    RadiusEagerPolarUpdateAlgorithm,
)

registry = make_algorithm_registry()
registry.register(RadiusEagerPolarUpdateAlgorithm())
plan = ExecutionPlan("torch_radius_eager_polar_update", "v1", DefaultRecipe())
optimizer = CSTAdamW(
    model.parameters(), model=model, lr=3e-4,
    update_selector=FixedSelector(plan, registry=registry),
)
```

privateな実験用backend依存であり、安定した公開aliasではありません。
独立scalar oracle、振幅のみ・静止・周期境界、全Atom VJP、metadata/support、
checkpoint、optimizer/resumeを検査します。CPU検査だけでCUDA・完全taskのadmission、
モデル品質、Graph動作、デフォルト採用を完了したとは扱わず、別途記録します。
