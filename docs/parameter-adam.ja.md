# CSTParameterAdam: パラメータ数に比例する状態のoptimizer

`CSTParameterAdam(model)` は、PyTorchの `AdamW` を再利用するCST-aware wrapperである。
Adamのmomentとparameter-coordinate updateをCST側で再実装していない。CSTとdenseの
parameter partition、CSTのownership検証、任意のCST learning-rate schedule、および
kernel固有のcoordinate updateをまとめた公開APIである。重み空間のAdamの近似や新しい
moment transport方式ではない。

## 更新と既定値

勾配 `g_t = ∂loss/∂p` に対して、

```text
m_t = beta1 m_(t-1) + (1-beta1) g_t
v_t = beta2 v_(t-1) + (1-beta2) g_t²
p_t = p_(t-1) - eta_t (m_t / (1-beta1^t)) / (sqrt(v_t / (1-beta2^t)) + eps)
eta_t = lr [r + (1-r)(1+cos(pi min((t-1)/(T-1), 1)))/2]
```

既定値は `lr=.03, betas=(.5,.99), eps=1e-8, T=128, r=.1`。
1回目の更新はlr=.03、128回目はlr=.003。以降もlr=.003を維持する。
`decay_steps=None` で減衰を無効にできる。CSTパラメータにweight decayはかけない。

保存状態は、各パラメータと同じshapeの `exp_avg`, `exp_avg_sq` とstep scalarのみ。
K原子・q座標ならmomentは `2 K q` scalars。K=64、q=4、FP32なら2048 bytes、
PyTorch Adamのstep tensorを含めて2052 bytes。学習率scheduleの整数counterと設定は
parameter groupに保存し、checkpointから再開できる。

全CST siteでfrozen chartsおよび `backend="factored"` を要求する。
optimizer wrapperは `W`、visible-space m/v、Jacobian、Hessian、Gram、小行列block、過去frameを
生成・保存しない。forward/backwardは通常のautogradを使い、既存の因子表
`Phi_in[I,K]`, `Phi_out[O,K]` と
batch activationを一時的に使う。この作業領域やデータセットのメモリはmoment容量とは別であり、
学習全体のメモリが2 KiBという意味ではない。

## 利用方法

```python
from torchcst import CSTParameterAdam, ParameterAdamConfig

# model中のCSTLinearはbackend="factored"で構築する。
optimizer = CSTParameterAdam(model, cst=ParameterAdamConfig(decay_steps=128))
for images, labels in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(images), labels)
    loss.backward()
    optimizer.step()
```

通常のtrainable parameterを含む場合は `dense=AdamWConfig(...)` を明示する。
そのgroupは指定した通常のAdamWを使い、CST向けcosine減衰は適用しない。
`param_groups[i]['lr']` は減衰前の基準lrを表す。別schedulerを重ねるとそのlrに
内部cosine係数を乗じるため、外部schedulerを使うなら通常は `decay_steps=None` にする。

`zero_grad()`後に勾配がないgroupはmomentもscheduleも進めない。group中の一部だけに
勾配がある場合、scheduleはそのgroupについて進み、各parameterのAdam bias correctionは
そのparameterが勾配を受けた回数で進む。非有限gradientは更新前に検出して例外を出す。
この検査はGPU同期を伴う。チェックポイント読み込みはpartition、名前、shapeを照合する。

## 検証

テストでは通常のPyTorch Adam＋同じscheduleとの更新一致、減衰終了後の動作、
途中checkpointからの再開、通常AdamWとのmixed-model比較、closureの1回実行を確認する。
原子数と入出力サイズを変え、保存状態が `2Kq+1` scalarsであること、および
materializationとCST derivative APIを禁止しても学習できることを検査する。

学習設定はtrain prefix8192の次の2048例を使ったvalidation平均で選んだ。
モデル・kernel・初期化・128回の更新は既存protocolと同じ。
A100での精度・速度・メモリおよび全探索履歴はcompanion cst repoの
`docs/experiments/torchcst_compact128_20260912.md` を参照。
