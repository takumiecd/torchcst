# Optimizer選定と推奨設定

更新日: 2026-09-20

## 結論

現在の推奨構成は `Triweight + PolarAmpWidth + CSTQuadraticAdam` である。
optimizer LR、kernel temperatureともにscheduleを使わない。公開面として残す中心は
次の3経路とする。

| 経路 | 位置づけ |
| --- | --- |
| `CSTQuadraticAdam` | 主経路。有限回の局所二次更新を累積する |
| `CSTNormalizedAdam` | 同じN/D momentsを使う比較・代替経路 |
| `CSTParameterAdam` | 通常のparameter-coordinate Adamによる軽量baseline |

通常parameterを含むmixed modelでは、この3経路から所有される
`AdamWConfig` も維持する。`CSTParameterAdam` の既定cosine scheduleは今回の推奨設定では
使わず、`decay_steps=None` とする。

## A100で選択した設定

full MNIST 60,000/10,000、batch 128、5 epoch、FP32、TF32 offで比較した。
モデルは2,560 atomsの `CSTLinear(784, 64)`、bias、ReLU、dense 64-to-10 head。

```python
from torchcst import (
    AdamWConfig,
    CSTQuadraticAdam,
    PolarAmpWidth,
    Triweight,
)
from torchcst.optim import QuadraticGradientSolver

kernel = PolarAmpWidth(
    amplitude_max=1.0,
    sigma_min=0.1,
    sigma_max=10.0,
    w_c=0.0025,
    kappa=30.0,
    activity_gain=27.0,
    activity_mode="time_energy",
    radial_regularization=0.5,
    profile=Triweight(0.1),
)

optimizer = CSTQuadraticAdam(
    model,
    lr=0.00025,
    betas=(0.9, 0.999),
    eps=1e-8,
    trust_radius=2.0,
    solver=QuadraticGradientSolver(
        max_iter=8,
        tolerance=1e-7,
        damping=1.0,
    ),
    initial_zero_step=True,
    factored=True,
    kernel_step_size=0.002,
    dense=AdamWConfig(
        lr=0.005,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ),
)
```

`lr` は1回のinner step、`max_iter=8` はその反復数であり、公称horizonは
`0.00025 * 8 = 0.002`。`kernel_step_size=0.002` はPolarのtime-energy activityと
radial regularizationへ外側horizonを渡す。これは厳密な連続時間積分を意味しない。

結果はseed 17/29/43で97.07/96.48/96.88%、平均96.81%、sample SD 0.301pp。
同じPolar上の `CSTParameterAdam` 平均96.29%より0.52pp高かった。従来の
kernel-temperature scheduleを使うnon-Polar Quadratic平均96.89%との差は0.08ppで、
scheduleを外せる簡潔さを優先して本構成を採用する。

## Best practices

- 最初は `iter=1` を通常Adam相当の基準として扱い、inner LRを既知のAdam LRから
  移す。反復数を増やすときは、まず `inner_lr = horizon / iterations` として総量を
  固定する。
- 今回のincumbentはhorizon 0.002、8 iterations。確認した0.0015/0.0020/0.0025では
  0.0020が最良だった。
- Polarは `Triweight(0.1)` とsigma `[0.1, 10]` を組み合わせる。Gaussian Polarより
  seed17で95.96%から96.70%へ改善した。
- `w_c=0.0025` を使う。Quadratic側で0.0015/0.0020/0.0025/0.0035を比較しても
  96.91/96.90/97.07/97.02%で中央が最良だった。
- time-energy activity 27、radial regularization 0.5を使う。`kernel_step_size` は
  inner LRではなくouter horizonへ合わせる。
- optimizer LR scheduleとkernel temperature scheduleは加えない。比較時もdense headを
  含め全LRを固定する。
- `last_step.site_results` のfinite、residual、converged、boundaryを保存する。今回の
  strict convergenceは各run 1/2345 stepsのみなので、結果を「厳密に収束した局所二次解」
  と呼ばない。
- 単一seedの97%だけで判断せず、少なくとも3 seedsの平均と分散を併記する。

## 削除候補

まだこの文書追加と同時には削除しない。PRを分け、公開import、checkpoint、tests、docsを
まとめて落とす。

### 第1候補: 薄いwrapperと旧alias

- `CSTNormalizedSGD`, `CSTNormalizedMomentum`, `CSTNormalizedRMSProp`
- `CSTQuadraticSGD`, `CSTQuadraticMomentum`, `CSTQuadraticRMSProp`
- aliasの `CSTSGD`, `CSTMomentum`, `CSTRMSProp`, `CSTImplicitAdam`

これらはAdamと同じcoordinator/solverのmoment選択違いで、今回の採用面には不要。
内部の再利用可能なN/D componentsとsolverは残す。

### 第2候補: 独立した実験optimizer

- `CSTAdamR` / `AdamRConfig`
- `CSTLocalAdam` / `LocalAdamConfig`
- `CSTLocalVisibleAdam` / `LocalVisibleAdamConfig`
- `CSTDenseVisibleAdam` / `DenseVisibleAdamConfig`
- 旧tangent経路の `CSTAdam` / `FirstOrderAdamConfig`
- full-quartic経路の `CSTSecondOrderAdam` / `SecondOrderAdamConfig`

これらは実装・状態形式・数式文書が独立しているため、第1候補とは別PRで削除する。
削除前に、Quadratic/Normalizedが利用するderivative、moment、dense AdamW、solverまで
誤って消さないようimport graphを確認する。

## 残すもの

- `CSTQuadraticAdam`, `QuadraticOptimizerConfig`, `QuadraticGradientSolver`
- `CSTNormalizedAdam`, `NormalizedOptimizerConfig`, `NormalizedFixedPointSolver`
- `CSTParameterAdam`, `ParameterAdamConfig`
- mixed model用 `AdamWConfig` とfunctional dense AdamW
- Polarのtime-energy rule、radial regularization、Triweight profileと関連tests
- N/D coordinator、numerator/denominator moments、ball/box trust constraint

この選定はMNISTだけで全用途の優劣を証明するものではない。削除PRでは履歴をGitに残し、
必要なら旧実装をrelease tagから参照できる状態にする。
