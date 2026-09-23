# Optimizer選定と推奨設定

更新日: 2026-09-22

## 結論

CSTのoptimizerは目的で選ぶ。CST表現そのものは同じでも、保持するoptimizer状態と
更新計算量が異なる。

| 目的 | 推奨経路 | 位置づけ |
| --- | --- | --- |
| optimizer状態と計算量を小さくする | `CSTParameterAdam` | PyTorch `AdamW` を使うparameter-coordinate経路 |
| 追加コストを許容して精度を追う | `CSTQuadraticAdam` | 局所二次モデルの反復更新 |

現在のMNIST比較では、同じPolar構成で `CSTQuadraticAdam` が平均96.81%、
`CSTParameterAdam` が平均96.29%だった。これはこの実験条件での結果であり、
すべてのタスクでの優位性を示すものではない。optimizer LR、kernel temperatureともに
scheduleを使わない設定を今回のQuadraticレシピとして採用した。公開面として残す中心は
次のfamilyと拡張とする。

| 経路 | 位置づけ |
| --- | --- |
| `CSTQuadratic*` family | 精度重視。有限回の局所二次更新を累積する |
| `CSTNormalized*` family | 同じN/D momentsを使う比較・代替経路 |
| `CSTAdamR` | Normalized/Quadraticを切替可能なN/D Adamへ斥力を加える拡張 |
| `CSTParameterAdam` | 省メモリ・低計算量を重視する正式な経路 |

通常parameterを含むmixed modelでは、これらの経路から所有される
`AdamWConfig` も維持する。`CSTParameterAdam` の既定cosine scheduleは今回の推奨設定では
使わず、`decay_steps=None` とする。

### ParameterAdamの推奨kernel

振幅・帯域可変kernelを `CSTParameterAdam` で学習する場合は、現在
`DirectAmpWidth` を第一候補とする。Adam momentを回転するPolar `(s,t)` ではなく
物理振幅 `w` に保持し、`q` はaccepted `delta_w` と
`R(q)=lambda/2*(q-1)^2` だけで更新する。同一初期operator・同一minibatch順の
full MNIST 3-seed比較ではDirect 95.9467%、Polar 94.8367%で、全seed正、平均
+1.11 ppだった。詳細は
[direct-amplitude-bandwidth.ja.md](direct-amplitude-bandwidth.ja.md)を参照。

これはfirst-order parameter-coordinate経路内のkernel比較であり、下記の
Polar `CSTQuadraticAdam` 選定を置き換える比較ではない。

`CSTAdamR` は独立したmoment familyではない。実装は `_NDModelOptimizer`、EMA
numerator、EMA denominatorを共有し、`update_rule="normalized"` または
`update_rule="quadratic"` でtask solverの更新則を選ぶ。task solve後の変位へ
`-lr * repulsion * grad(R)` を加え、合成後に同じtrust geometryへ再projectする。
既定は後方互換のNormalized。斥力ゼロなら選択した通常Adamと同じ更新になる。
このため今後の斥力実験用にN/D familyと一緒に保持する。

## A100で選択した設定

full MNIST 60,000/10,000、batch 128、5 epoch、FP32、TF32 offで比較した。
モデルは2,560 atomsの `CSTLinear(784, 64)`、bias、ReLU、dense 64-to-10 head。
このoptimizer比較はraw Triweightを使うため、下の例では
`normalize_columns=False` を明示する。

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
    profile=Triweight(0.1, normalize_columns=False),
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

### 曲率ブロックのablation

`CurvatureBlockMask` は、原子ごとの曲率 `H:[K,P,P]` を先頭 `split` 座標と残りの
座標に分割し、N/D momentへ渡す前にブロックを選択する。したがってnumeratorの
`C` だけでなく、denominatorの `y=gH` と `Z=H⊗H` にも同じマスクが反映される。
Polarでは先頭2座標が `(s,t)` なので、振幅・幅と中心座標のmixed block `M` を
調べるablationは次のように指定できる。

```python
from torchcst import CurvatureBlockMask

optimizer = CSTQuadraticAdam(
    model,
    # ...上記と同じ設定...
    curvature_mask=CurvatureBlockMask(split=2, mode="no_m"),
)
```

`mode` は、全曲率の `"full"`、対角2ブロックだけを残す `"no_m"`、mixed block
だけを残す `"m_only"`、全曲率を0にする `"none"` の4種類。これは機構分離用の
実験面であり、`split` はkernelの座標順序に合わせて明示する。

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
- Polarは `Triweight(0.1, normalize_columns=False)` とsigma `[0.1, 10]` を組み合わせる。Gaussian Polarより
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

## 公開optimizer surface

### 保持するN/D family surface

- `CSTNormalizedSGD`, `CSTNormalizedMomentum`, `CSTNormalizedRMSProp`
- `CSTQuadraticSGD`, `CSTQuadraticMomentum`, `CSTQuadraticRMSProp`
- aliasの `CSTSGD`, `CSTMomentum`, `CSTRMSProp`, `CSTImplicitAdam`
- `CSTNormalizedAdam`, `CSTQuadraticAdam`
- `CSTAdamR` / `AdamRConfig`

これらは同じcoordinator/solver上でmoment構成や更新則を比較するfamilyなので削除しない。
内部の再利用可能なN/D componentsとsolverも残す。

### 削除済み: 独立した旧実験optimizer

- `CSTLocalAdam` / `LocalAdamConfig`
- `CSTLocalVisibleAdam` / `LocalVisibleAdamConfig`
- `CSTDenseVisibleAdam` / `DenseVisibleAdamConfig`
- 旧tangent経路の `CSTAdam` / `FirstOrderAdamConfig`
- full-quartic経路の `CSTSecondOrderAdam` / `SecondOrderAdamConfig`

これらは実装・状態形式・数式文書を含めて公開面から削除した。Git履歴からは参照できるが、
import互換性とcheckpoint互換性は提供しない。Quadratic/Normalizedが利用するderivative、
N/D moments、dense AdamW、solverは維持する。

## 残すもの

- `CSTQuadraticAdam`, `QuadraticOptimizerConfig`, `QuadraticGradientSolver`
- `CSTNormalizedAdam`, `NormalizedOptimizerConfig`, `NormalizedFixedPointSolver`
- Quadratic/NormalizedのSGD、Momentum、RMSProp wrapperと既存alias
- `CSTAdamR`, `AdamRConfig` とatom-operator repulsion実装
- `CSTParameterAdam`, `ParameterAdamConfig`
- mixed model用 `AdamWConfig` とfunctional dense AdamW
- Polarのtime-energy rule、radial regularization、Triweight profileと関連tests
- N/D coordinator、numerator/denominator moments、ball/box trust constraint

この選定はMNISTだけで全用途の優劣を証明するものではない。旧実装が必要な場合は
削除前のGit履歴を参照する。
