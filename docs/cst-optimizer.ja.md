# CSTのoptimizer

## CSTAdamW: 単一のPyTorch lifecycle

`CSTAdamW(params, model=model, ...)`（PyTorch 2.6+）はAdamWを継承し、
`param_groups`・`state`・hook・scheduler接続を1つのoptimizerが所有する。
step内では勾配射影と旧座標snapshot、PyTorch functional AdamW、既存Kernelの
更新・vector transportの順に実行する。wrapped parentの`step()`は呼ばず、
標準step pre/post-hookが1回ずつ動き、post-hookはCST更新後の値を見る。
既存wrapperのbase post-hookが見るAdam proposalとは境界が違うため、移行は明示的に行う。

managed atom groupはraw `weight_decay=0`が既定で、ampだけの減衰は
`amplitude_weight_decay`で指定する。通常parameter groupはAdamW既定のraw decay=.01を
保つ。明示的にatomへ非ゼロraw decayを与えると拒否する。Kernelの数式は下記wrapperと共通。
`grad=None`では更新せず、`lr=0`でもAdam clock/momentsの通常の更新は維持する。
`check_finite=False`が既定で、追加の厳密検査を使う場合はTrueにする。
lr/decay・active atomのmetadataとcallback変更の検証は残す。model全体の所有・構造の走査は
構築、add_param_group、checkpoint load時に行い、stepごとには繰り返さない。

`zero_grad`、scheduler、closure、state-dict hookはPyTorchの通常の仕組みを使う。
GradScalerは通常のunscale/skip経路で全stepをskipする。fused Adamの内部overflow判定だけで
Polarが実行される状態にはしない。managed atomのcapturable/differentiable modeは未対応。
checkpoint post-hookはcst manifestと再接続済みのatom-owned stateを観測できる。
標準`load_state_dict`はnative checkpoint用。現在のexact AdamW wrapperからの移行は、
新しいmodelへmodel checkpointを読み込んでnative optimizerを作り、
`load_legacy_state_dict`を明示的に呼ぶ。すでに別optimizerがbindしたmodelは共用しない。

## CSTOptimizer: 任意optimizerの互換wrapper

`CSTOptimizer(base_optimizer, model=model, state_adapter=None)` は既存の
PyTorch optimizer を包む。`check_finite=True`が既定。旧 optimizer、config、N/D solver、moment system、
AtomGrad は削除した。旧実験の再現には変更前の Git revision `33504c6` を使う。

## 所有と更新

| 項目 | 所有者 / 処理 |
| --- | --- |
| Parameter | model |
| learning rate、通常のweight decay、optimizer state | base optimizer |
| amp専用weight decay | groupのamplitude_weight_decayをKernel更新時に適用 |
| gradient | backend の通常の autograd |
| 勾配射影・更新規則・vector transport | KernelSpec が定める Parameterization / Geometry。計算本体は backend |
| 更新順序・CST site 発見 | CSTOptimizer |
| scheduler | PyTorch scheduler を wrapper に接続 |

closure があれば現在点で一度評価する。検査有効時は全 group の勾配の有限性を確認する。
atom 勾配を射影し、更新する atom table の旧座標を保存してから base が更新する。
実際の差分 `proposal - old` とその group の現在の `lr` を Kernel に渡し、
更新結果へ vector state を transport して Parameter に書き戻す。

`grad is None` の atom は更新規則も進めない。ゼロ勾配は有効な step として
活動状態の減衰等を進める。`lr=0` では CST の更新規則を止めるが、base optimizer
自身の clock は通常の仕様どおり進む。wrapper 独自の schedule は持たない。

通常の Parameter は base の更新をそのまま使う。各 base が所有する atom だけを
更新するため、model のパラメータを複数 optimizer に分担できる。同じ atom に
複数の CST site が更新規則を割り当てることは拒否する。

wrapper は atom の列を解釈しない。Polar の活動状態の時間単位は group の lr。
optimizer を替えても同じ規則を使うが、学習軌道が同じになるという意味ではない。
raw atom coordinates への weight decay は base の proposal に含まれ、その後
Kernel の規則で射影・調整される。Kernel の正則化とは別なので、必要に応じて
CST group を `weight_decay=0` にする。

### Polarのampだけにweight decayを掛ける

AdamWのatom groupに`weight_decay=0`と`amplitude_weight_decay=λ`を指定する。
通常のDense parameterは別groupの通常`weight_decay`を使う。groupを自動分割したり、
baseのgroup設定をstep中に書き換えたりしない。

```python
base = torch.optim.AdamW([
    {"params": cst_layer.parameters(), "weight_decay": 0.0,
     "amplitude_weight_decay": 0.01},
    {"params": dense_layer.parameters(), "weight_decay": 0.01},
], lr=1e-3)
optimizer = CSTOptimizer(base, model=model)
```

λは有限・非負のPython scalar、`0 <= lr*λ <= 1`とする。このoptionはexact
`torch.optim.AdamW`用であり、subclassや別optimizerのclosure意味を推定しない。
raw decayとの併用は、AdamWがParameter/stateを書き換える前に拒否する。
optionを指定しないgroupは互換性のため従来どおりで、raw decayが自動的にamp用へ変わるわけではない。

Polarで`q=s²+t²`、`amp=Amax*s/sqrt(q)`とする。まずraw decayなしのAdamW proposalへ
従来のPolar task/activity/radial更新を適用して`(sf,tf)`を得る。
`rf=sqrt(sf²+tf²)`、`u_old=amp_old/Amax`として、

\[
u'=\operatorname{clamp}\left(\frac{s_f}{r_f}-\mathrm{lr}\lambda u_{old},-1,1\right),
\quad s'=r_f u',\quad
t'=\operatorname{copysign}\left(r_f\sqrt{(1-|u'|)(1+|u'|)},t_f\right).
\]

これは`amp_target = amp_after_Polar - lr*λ*amp_old`というold-value項の減衰で、
taskによる増分をさらに減衰させない。上限を超えた場合は振幅領域へ射影する。
taskがなければ`amp_new=(1-lr*λ)*amp_old`。λ=0では再構成自体を省く。
`t=±0`でも符号側を保ち、中心座標・Adam moments・stepにはこの減衰を適用しない。
実数上`q_new=q_after_Polar`であり、既存の活動状態の更新則は変えない。
profileのradius/bandwidthの上限がampに依存する契約では、その通常のdecodeにより
profile radiusが変わる可能性がある。`q`の保存とprofile radiusの固定は別である。

baseの既存pre-hook後に一時的な最後のpre-hookで検証・勾配射影・snapshot・Plan準備を行い、
実際にAdamWが使うlr/λ/gradを捕捉する。post-hookが設定を変えても現在stepは捕捉値を使う。
closureはwrapper入口で一度だけ評価し、baseへの再提示では再評価しない。
pre-hookによるclosureの差し替えはこのmodeでは拒否する。一時hookは例外時にも除去する。
定常stepでGPUの値をCPUへ読み出す操作は追加しない。

amp decay対応の汎用Torch/Polar Torch/Euclidean・FlatTorus・radius融合Planはv2。
旧Sphere融合・intrinsic Torus専用v1 Planは正のamp decayを拒否するため、対応する
Torch参照Planを使う。λはstepごとのInputsとして渡し、係数の変更でFourier係数や
atomのコピーを常駐させない。CUDA融合版では既存Polar kernelの中で再構成する。

## State adapter

| base optimizer | vector として運ぶ state |
| --- | --- |
| SGD | momentum_buffer |
| Adam / AdamW / Adamax | exp_avg |
| RMSprop | momentum_buffer、centered の grad_avg |
| Rprop | prev（前回の勾配） |
| Adagrad / Adadelta | なし |

二乗の蓄積と step clock は元の保存座標系で保持する。これは座標ベースの
optimizer に vector transport を組み合わせた規則であり、内在的な Riemannian
Adam や covariance transport を実装したものではない。

custom optimizer / subclass は state の意味を明示する。

```python
from torchcst import CSTOptimizer, OptimizerStateAdapter

optimizer = CSTOptimizer(
    custom_base,
    model=model,
    state_adapter=OptimizerStateAdapter(vector_keys=("velocity",)),
)
```

指定した parameter-shaped state を 共通 KernelState を使う backend で運ぶ。それ以外の state を
保持することが正しいかは custom optimizer 側の契約である。subclass を名前
だけで既知の state semantics と推定しない。ASGD の平均パラメータや LBFGS の
複数 proposal / closure 再評価は、既定経路では扱わない。

## PyTorch との接続

wrapper は `torch.optim.Optimizer` であり、`param_groups` / `state` は base と
共有する。scheduler、GradScaler、step hook は wrapper を使う。closure は一度
評価し、base へはその結果を返す closure を渡す。`add_param_group()` も base に
委譲する。base を直接 step / load して wrapper と併用しない。

checkpoint は標準の state / param_groups に `cst` manifest を追加する。
optimizer type、vector keys、parameter 名・shape、KernelSpec と CST policy 所有を検証して
復元する。Kernel / Geometry の設定は model checkpoint で復元する。load 後も
base と wrapper の state / groups は共有する。旧 optimizer や raw base の
checkpoint は受け入れない。model を目的の device へ移してから wrapper を作り、
atom Parameter を置き換えた場合は wrapper を作り直す。

## 通常PyTorchと同じく独自finite検査を追加しない経路

`CSTOptimizer(..., check_finite=False)`はoptimizer内の値の有限性検査を追加しない。
NaN/Infの自動拒否やstep skipを行わない通常AdamWの流れに合わせる選択肢である。
shape/dtype/device、group/Plan、固定設定の検証は常に保つ。既定Trueは従来の検査を維持する。
modeは呼出しのscope内だけに適用し、closureや呼出し外のgeometry検証は抑制しない。
checkpointの数学的契約やAdam内部状態は変えず、modeは呼出し側で選ぶ。

次節のFlatTorus用Planを選ぶ構成なら、末尾に`check_finite=False`を加える。
固定設定のFP32/fused AdamW/FlatTorusで、定常optimizer stepの同期API0・DtoH0を実測した。
設定の初回準備・変更後の再検証、他のPlan/optimizer、closure、AMP/GradScaler、学習全体の
同期ゼロまで保証するものではない。CUDA Graphは引き続き未対応。

## 現在の範囲

CUDA FP32のPolarと1次元FlatTorus chartの組では、更新専用Planを明示して
Polar更新と中心の周期折り返しをTritonで融合できる。

```python
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.polar_update.plans import FLAT_TORUS_FUSED

optimizer = CSTOptimizer(
    base_optimizer,
    model=model,
    update_selector=FixedSelector(FLAT_TORUS_FUSED, registry=make_algorithm_registry()),
)
```

これは`atom_update`の実装選択であり、LinearのPlanとは独立する。2座標の中心を
持つ連続FP32 `[atom_count, 4]`が対象で、ほかのgeometryやdtypeにはこのPlanを
指定しない。既定は引き続きTorch参照経路。Adamのproposal・moments・step、
勾配射影とstate adapter、異常値検査は共通wrapperが担当する。
周期とPolar scalarはlive bufferから読み、設定変更時の宣言検証も維持する。
既存のEuclidean用`FUSED`とは別Planで、周期版もeager実行を要求する。

Linear / Conv2d の forward backend に依存しない。Euclidean trainable Chart は
処理できる。trainable 非 Euclidean Chart は atom center と異なる更新契約が
必要なので、専用 policy を追加するまで拒否する。

CUDA実数の有限性検査はdevice/dtypeごとにまとめ、小さな判定vectorだけをhostへ返す。
raw勾配は全Parameter分をAdam前に確認する。Euclidean/FlatTorusのtransportも旧点・新点・
vectorをまとめる。更新則やmomentsの演算は変えず、不正値をその場で拒否する同期は保つ。
complex/CPU/非対応layoutは従来の検査を使い、CUDA Graphの対応範囲は広げない。

snapshot と更新作業領域は atom parameter 数に比例する。dense weight、
weight-gradient、Hessian は構築しない。有限性検査は device を同期する。
CST policy を含む CUDA Graph capture、Tensor learning rate、differentiable
optimizer step は未対応。`capturable=True` は拒否する。通常の eager CUDA と
foreach / fused base optimizer は別の実行設定である。

`check_finite=True`ではraw gradientの非有限値をbase更新前に拒否する。base 更新後の custom policy
failure まで全 optimizer state を rollback するトランザクションではない。
custom policy は shape と有限性の契約を守る必要がある。
