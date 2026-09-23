# DirectAmpWidth: 物理振幅座標でAdam状態を保持するkernel

更新日: 2026-09-22

## 結論

`CSTParameterAdam` で振幅・帯域可変kernelを使う場合、現在の第一候補は
`DirectAmpWidth` とする。各atomの先頭2座標をPolarの `(s, t)` ではなく
物理量 `(w, q)` として保存し、Adamの一次・二次momentを振幅 `w` に直接保持する。
`q` はtask-lossの最適化変数ではなく、実際に受理された振幅移動から更新する
帯域activity stateである。
以下のpaired MNIST比較は `normalize_columns=False` のraw Triweightを使用した。
`Triweight` の既定は旧 `main` と同じL2列正規化である。

full MNISTの同一初期operator・同一minibatch順による3-seed paired比較では、
`DirectAmpWidth` が既存 `PolarAmpWidth` を全seedで上回った。

| Seed | Polar | Direct | Direct - Polar |
| ---: | ---: | ---: | ---: |
| 17 | 96.17% | 96.28% | +0.11 pp |
| 29 | 95.46% | 96.46% | +1.00 pp |
| 43 | 92.88% | 95.10% | +2.22 pp |
| **Mean** | **94.8367%** | **95.9467%** | **+1.11 pp** |

これは5 epoch MNISTにおけるfirst-order kernel比較であり、他dataset、長期学習、
`CSTQuadraticAdam`、すべてのchart/profileに対する一般的優位性の証明ではない。

## 解決する問題

`PolarAmpWidth` は振幅と帯域状態を

```text
p_polar = (s, t)
q = s² + t²
w = W s / sqrt(q)
```

として表す。時刻ごとのtask gradientは現在のPolar方向に対して接線方向になるが、
座標frameは更新に伴って回転する。通常のAdamは過去の `exp_avg` と
`exp_avg_sq` を座標成分ごとに蓄積するため、過去の `(s, t)` frameで得たmomentを
現在のframeへ完全には読み替えられない。特に二次momentの対角軸は回転で閉じない。

`DirectAmpWidth` はこの問題を座標変換で解く。

```text
p_direct = (w, q, input_center..., output_center...)
w in [-W, W]
q in [1, 4]
alpha = (q - 1) / 3
```

task-lossの振幅gradientは常に同じ物理座標 `w` に作用する。`q` のtask gradientは
kernelが明示的に0へprojectし、Adamの `q` 用 `exp_avg` と `exp_avg_sq` も厳密に
0のまま維持される。追加のmodel parameterや別のoptimizer stateは必要ない。

## 更新則

時刻 `n` の状態を `(w_n, q_n)`、Adamが提案した振幅変位を `d_w`、振幅上限を
`W` とする。まず実際に受理される振幅を求める。

```text
w_accepted = clamp(w_n + d_w, -W, W)
delta_w = w_accepted - w_n
```

`q` の増加は、提案値ではなくclamp後の実移動量から計算する。

```text
delta_squared = q_n * (delta_w / W)²
q_geom = clamp(q_n + delta_squared, 1, 4)
```

ここで平方根が現れないのは、実装上の `q` が半径そのものではなく半径の二乗だからで
ある。`r_n = sqrt(q_n)`、正規化振幅変位を `d = delta_w / W` と置くと、直角三角形の
接線辺は `delta = r_n * d` であり、半径としては

```text
r_geom = sqrt(r_n² + delta²)
```

となる。保存している座標へ戻せば

```text
q_geom = r_geom²
       = q_n + q_n * d²
```

である。つまり平方根の三角形更新を省略しているのではなく、二乗半径 `q` 上で同じ
更新を直接計算している。

この更新は符号に依存せず、振幅が動かなければ `q` を増やさない。Polar角、
`tan(delta_theta)`、activity gain、time-energyのstep-size除算、dormant expansionは
使わない。

最後に

```text
R(q) = lambda / 2 * (q - 1)²
```

の通常の勾配ステップを1回適用する。

```text
q_new = clamp(q_geom - step_size * lambda * (q_geom - 1), 1, 4)
```

ここで `step_size` は `CSTParameterAdam` がその更新で使ったCST learning rate、
`lambda` は `radial_regularization` である。これは指数decayやPolar半径上の
gradient flowではなく、明示的な `q` 座標に対するEuler stepである。最終clampは
極端なstep sizeでもstate contractを維持する。

## PyTorch optimizer state

`CSTParameterAdam` はatom tableと同じshapeのPyTorch AdamW stateを確保する。
`DirectAmpWidth` の先頭2列に対する意味は次の通り。

| 列 | parameter | task gradient | Adam `m`, `v` | kernel後処理 |
| ---: | --- | --- | --- | --- |
| 0 | `w` | 有効 | 物理振幅gradientを蓄積 | `[-W, W]`へclamp |
| 1 | `q` | 常に0 | 常に0 | accepted `delta_w`と正則化で更新 |

`q` 用のstate列はPyTorchのparameter-shape contractのためtensor上には存在するが、
値は0である。`w` を別parameterとして追加保存するわけではなく、旧 `(s, t)` の2列を
`(w, q)` の2列へ置き換えている。

## 利用方法

```python
from torchcst import (
    AdamWConfig,
    CSTParameterAdam,
    DirectAmpWidth,
    ParameterAdamConfig,
    Triweight,
)

kernel = DirectAmpWidth(
    amplitude_max=1.0,
    sigma_min=0.1,
    sigma_max=10.0,
    w_c=0.0025,
    kappa=30.0,
    radial_regularization=0.5,
    profile=Triweight(0.1, normalize_columns=False),
)

optimizer = CSTParameterAdam(
    model,
    cst=ParameterAdamConfig(
        lr=0.005,
        betas=(0.9, 0.999),
        eps=1e-8,
        decay_steps=None,
    ),
    dense=AdamWConfig(
        lr=0.005,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ),
)
```

`DirectAmpWidth` は `PolarAmpWidth` を継承せず、独立したkernelとして振幅とactivityを
処理する。両者の帯域上限・下限の数式と設定項目は似ているが、atom rowの座標、更新則、
checkpoint contractは別である。`activity_gain`、`activity_mode`、
`dormant_expansion_rate` はPolar専用であり、Directへ指定するとエラーになる。

## Polarとの互換性

`PolarAmpWidth` と `DirectAmpWidth` は同じoperatorを表現できるが、parameter rowと
checkpointの意味は異なる。Polar checkpointをDirect modelへそのままloadしてはならない。
新しく保存するcheckpointにはkernel座標系、profile型、列正規化設定を記録し、
異なる設定への `load_state_dict` は拒否する。`main` 時代の単一帯域Polar checkpointは
旧 `sigma_max` と `profile.sigma` から識別できるため、現在の入出力別bufferへ自動移行する。
一方、このfeature branchで作った識別情報なしの新帯域checkpointは、Polar/Directが
同じbuffer名とatom shapeを持つため自動判別できず、明示的に識別するまで拒否する。
atomごとの表現値だけを変換する場合は、Polarの先頭2列から

```text
q = s² + t²
w = W s / sqrt(q)
```

を計算し、center列をそのままコピーする。Adamのmomentはこの変換では正しく移せないため、
optimizer stateは新規初期化する。

## 検証結果と観測

検証条件はfull MNIST 60,000/10,000、batch 128、5 epoch、seed 17/29/43、
`CSTLinear(784, 64)`、2,560 atoms、ReLU、dense 64-to-10 head、FP32、TF32 off、
NVIDIA A100 80GB PCIe MIG 3g.40gb。CST/dense LRは0.005、betasは
`(0.9, 0.999)`、scheduleなし。

- 初期実体化operatorはpaired arm間で最大絶対差 `1.12e-8` 以下。
- minibatch順のhashはpaired armで一致。
- Directの全runで `q` のAdam一次・二次momentは厳密に0。
- mean final test lossはDirect 0.1333、Polar 0.1696。
- accuracyのseed間sample SDはDirect 0.739 pp、Polar 1.731 pp。
- warm stepはDirect 10.71 ms、Polar 11.84 msで約9.5%短い。
- Directのfinal mean `q` は1.000189、mean `alpha` は `6.31e-5`。

最後の値は、旧Polarの強いactivity膨張がこの条件の精度に必要なかったことを示す。
一方、Directの振幅上限1%以内にあるatomは平均1.50%で、Polarの0.69%より多い。
長期学習では振幅clamp率を継続監視する。

## 旧Direct実験との違い

最初のDirect試作は `w` momentだけをDirect化した一方、`q` に
`activity_gain * delta² / step_size` と旧Polar radial flowを残していた。その版は
3-seed平均でPolar比 -0.0367 ppだった。`q` 則だけを本書のphysical-displacement更新へ
置き換えると +1.11 ppへ反転し、全seedが正になった。したがって旧試作の結果を
Direct `w` 座標の否定として扱わない。

## 現在の制限

- evidenceは5 epoch MNISTの3 seedsに限られる。
- `CSTConv2d`、深いnetwork、長期run、他datasetでは未確認。
- `CSTQuadraticAdam` はparameter-coordinate Adamとは異なるstateとsolverを使うため、
  本結果からDirectがQuadraticでも優位とは結論しない。
- `q` がほぼ1に留まるため、帯域探索を必要とするtaskでは別の挙動になり得る。
- checkpoint変換utilityは未提供。表現変換は明示的に行い、optimizer stateは再初期化する。

実験runner、事前登録、raw JSON、図を含む完全な記録はcompanion `cst` repositoryの
`docs/experiments/direct_amp_width_physical_q_20260922.md` にある。
