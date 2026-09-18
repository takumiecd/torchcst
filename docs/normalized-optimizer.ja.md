# torchcst 正規化 optimizer の設計

この文書は、CSTNormalizedSGD、CSTNormalizedMomentum、CSTNormalizedRMSProp、
CSTNormalizedAdam からなる正規化 optimizer family の設計を説明する。これは
従来の CSTAdam を置き換える互換層ではなく、分子 N、分母 D、更新 solver を
独立に差し替えられる新しい optimizer 経路である。CSTSGD、CSTMomentum、
CSTRMSProp は対応する CSTNormalized* wrapper の別名である。

N と D の評価、trust 制約（全体球 / 成分 box）、反復ループは Quadratic
family と共有する。Normalized は `d ← Π(-η N/D)`、Quadratic は
`d ← Π(d - η N/D)` であり、解いている方程式だけが違う。

実装の中心は次の三つである。

1. LinearJGHAtomGrad が各atomの一次勾配と損失Hessianブロックを集める。
2. 分子momentと分母momentが、それぞれ固定されたatom座標で独立に状態を持つ。
3. candidate-dependentな正規化更新を、注入された NormalizedSolver が解く。

## 1. 何を更新するoptimizerか

CSTLinear のatom parameter tableを

    p: [K, P]

とする。K はatom数、P はkernelが解釈する1 atomあたりのopaque parameter幅である。
optimizerが探す局所変位も

    d: [K, P]

であり、denseな表現weight空間の変位ではない。

現在の点のまわりで、atom座標上の損失勾配を一次近似すると

$$
r(d)=g+Hd
$$

となる。ここで

$$
g\in\mathbb R^{K\times P},
\qquad
H\in\mathbb R^{K\times P\times P}
$$

である。H はatom間を混ぜた巨大な全体Hessianではなく、各atomの局所的な
損失Hessianブロックである。

正規化optimizerの更新は、candidate d に依存する

$$
\boxed{
d=-\eta\frac{N(d)}{D(d)}
}
$$

を満たす変位を探すことに相当する。割り算は [K, P] の各成分に対して行う。
制約付きsolverでは、実際にはこの写像をfeasible setへ射影した

$$
T(d)=\Pi_{\mathcal C}\left(-\eta\frac{N(d)}{D(d)}\right)
$$

の固定点として反復する。

## 2. AtomGrad: jg と gh

### 2.1 収集する量

正規化optimizer familyは LinearJGHAtomGrad を使う。名前の通り、主な観測値は

    jg: [K, P]
    gh: [K, P, P]

である。

jg はatom parameterに対する損失勾配で、表現weightの勾配をatom座標へ
pullbackしたものに対応する。

gh は、各atomについて損失をatom parameterのスカラー関数として見たときの
Hessianブロックである。実装では、各atomの寄与とbackwardで得た出力勾配から
スカラー損失contractionを作り、そのatom parameterに対して
torch.func.grad / torch.func.hessian を適用する。したがって、モデル全体の
dense Hessianを作るのではなく、必要な [K, P, P] だけを作る。

### 2.2 反復呼び出しの扱い

同じ optimizer step中に CSTLinear が複数回使われた場合、LinearJGHAtomGrad は
各callbackの jg と gh を加算する。row/column squareを使う経路では、backward
中の項を保持して最後にaggregateしてから二乗するため、呼び出し間のcross termも
失われない。

factored=True を指定すると、factorizationを提供するkernelでは、同じ観測値を
factorized contraction経路から得られる。これは観測値の意味を変えるものではなく、
AmpWidth のようなfactor-capable kernelでの実行方法を変える
だけである。

## 3. 分子 N(d)

分子momentは NumeratorMoment である。状態は

    m: [K, P]
    C: [K, P, P]

を持ち、step-localな分子を

$$
N(d)=m+Cd
$$

として評価する。

### SGD

CSTNormalizedSGD はmomentを持たず、現在の観測値をそのまま使う。

$$
N_{\mathrm{SGD}}(d)=g+Hd,
\qquad
D_{\mathrm{SGD}}(d)=1.
$$

### Momentum

CSTNormalizedMomentum は g と H の両方を同じ一次moment clockでEMAする。

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
C_t&=\beta_1C_{t-1}+(1-\beta_1)H_t.
\end{aligned}
$$

bias correction後の値で

$$
N_{\mathrm{Momentum}}(d)=\hat m_t+\hat C_td,
\qquad
D_{\mathrm{Momentum}}(d)=1
$$

を評価する。

## 4. 分母 D(d)

D(d) は通常のRMSProp/Adamと同じく、勾配の二乗を座標ごとに正規化する。ただし
勾配が g+Hd というcandidate-dependentな量なので、その二乗を多項式として保持する。

atom k とparameter座標 i を固定すると、

$$
r_{k,i}(d)=g_{k,i}+H_{k,i:}d_k
$$

である。この二乗は

$$
r_{k,i}(d)^2
=g_{k,i}^2
+2g_{k,i}H_{k,i:}d_k
+d_k^\top H_{k,i:}^\top H_{k,i:}d_k
$$

と展開できる。そこで分母momentは

    x: [K, P]
    y: [K, P, P]
    Z: [K, P, P, P]

を持つ。DenominatorMoment はこれらを二次moment clockでEMAし、bias correction後に

$$
q(d)=x+2y[d]+d^\top Z[d,d]
$$

を評価する。成分ごとの分母は

$$
\boxed{
D(d)=\sqrt{\max(q(d),0)}+\varepsilon
}
$$

である。結果のshapeは

$$
D(d)\in\mathbb R^{K\times P},
$$

であり、d と同じshapeになる。

eps は分母に直接加える。理想的には q(d) は二乗平均なので非負だが、EMAされた
係数の評価では浮動小数点誤差があり得るため、実装は平方根の前で0にclampする。

### RMSProp

CSTNormalizedRMSProp は分子を現在のaffine gradient、分母を二乗EMAにする。

$$
N_{\mathrm{RMSProp}}(d)=g+Hd,
\qquad
D_{\mathrm{RMSProp}}(d)
=\sqrt{\hat x+2\hat y[d]+d^\top\hat Z[d,d]}+\varepsilon.
$$

### Adam

CSTNormalizedAdam は分子と分母の両方をEMAする。

$$
N_{\mathrm{Adam}}(d)=\hat m+\hat C d,
$$

$$
D_{\mathrm{Adam}}(d)
=\sqrt{\hat x+2\hat y[d]+d^\top\hat Z[d,d]}+\varepsilon.
$$

CSTImplicitAdam は CSTNormalizedAdam の別名である。

## 5. expand と compress

NとDの状態は別々に保持する。NormalizedMomentSystem の1 stepは次の順序である。

    AtomGrad observation
            ↓
    numerator.expand / denominator.expand
            ↓
    step-local N(d), D(d) と pending state
            ↓
    solver.solve
            ↓
    numerator.compress / denominator.compress
            ↓
    次のpersistent state

expand は前stepのstateと現在のimmutable observationから、今回のEMA値と
bias-correctedなsolver viewを作る。ここではpersistent stateを変更しない。

compress はsolverが返したaccepted displacementを受け取るが、現在のN/D実装では
それを使ってstateをrecenterしたり、別のframeへ移送したりしない。expand が準備した
pending stateをそのままcommitするだけである。したがって、最後にNとDを混ぜたり、
accepted displacementに合わせてmomentを圧縮し直したりする処理は行わない。

## 6. 差し替え可能なsolver

optimizer wrapperはmomentの組み合わせだけを選び、更新を解く方法は
NormalizedSolverとして注入する。

### global L2 ball

NormalizedFixedPointSolver は、全 [K,P] を連結したEuclidean normに対して

$$
\lVert d\rVert_2\le r
$$

というglobal ballを使う。

### elementwise box

NormalizedBoxFixedPointSolver は各成分に独立なboundを課す。

$$
\mathcal C_{\mathrm{box}}
=\left\{d\mid -r\le d_{k,i}\le r\right\}.
$$

ここで trust_radius は球の半径ではなく、boxのhalf-widthとして解釈される。
例えば trust_radius=1.0 なら、各要素が [-1,1] に入る。原子数やparameter幅が
増えても、1要素あたりのboundは変わらない。

solverはdamped fixed-point iterationを行う。

$$
\begin{aligned}
u^{(j)}&=\Pi_{\mathcal C}\left(-\eta\frac{N(d^{(j)})}{D(d^{(j)})}\right),\\
d^{(j+1)}&=\Pi_{\mathcal C}\left((1-\lambda)d^{(j)}+\lambda u^{(j)}\right).
\end{aligned}
$$

box境界に張り付くことは、box制約の正常な結果であり、それ自体を失敗とは扱わない。
optimizerがhard errorにするのは、変位が非有限、shape/device/dtype不一致、または
solverのfeasible set外に出た場合である。

converged=False は、指定された max_iter 後も残差が tolerance 以下にならなかった
ことを表す。これは必ずしも学習値がNaNになったという意味ではない。実験では有限値の
approximate fixed-point updateとして継続することもできる。

## 7. 1 step目の d=0

NormalizedOptimizerConfig(initial_zero_step=True) を指定すると、最初のoptimizer
stepだけは

$$
d^{(1)}=\Pi_{\mathcal C}\left(-\eta\frac{N(0)}{D(0)}\right)
$$

を使う。つまり、solverの反復を最初から回すのではなく、candidate displacementを
d=0 としてN/Dを一度評価する。その後のstepでは通常のcandidate-dependent fixed-point
solverへ戻る。

これはoptimizer stateの初期化と、最初のimplicit feedbackの影響を分離して調べるための
オプションであり、常に有効とは限らない。

## 8. 公開optimizerの対応表

| optimizer | 分子 N(d) | 分母 D(d) |
|---|---|---|
| CSTNormalizedSGD | 現在の g + H d | 1 |
| CSTNormalizedMomentum | g と H のEMA | 1 |
| CSTNormalizedRMSProp | 現在の g + H d | (g + H d)^2 のcomponent-wise EMA |
| CSTNormalizedAdam | g と H のEMA | (g + H d)^2 のcomponent-wise EMA |
| CSTAdamR | CSTNormalizedAdam と同じ | 同じ。CST変位に decoupled な `-lr λ ∇L` を後から足す |

これらは同じmodel-level coordinatorを使う薄いwrapperであり、optimizerごとに別の
CST engineを複製しない。moment componentとsolverを交換することで、組み合わせを
増やせる。

## 9. 使用例

### Adam + elementwise box

    from torchcst import CSTNormalizedAdam
    from torchcst.optim import NormalizedBoxFixedPointSolver

    solver = NormalizedBoxFixedPointSolver(
        max_iter=32,
        tolerance=1e-7,
        damping=1.0,
    )

    optimizer = CSTNormalizedAdam(
        model,
        lr=0.05,
        betas=(0.9, 0.99),
        eps=1e-8,
        trust_radius=0.1,
        solver=solver,
        initial_zero_step=True,
        factored=True,
    )

    for inputs, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()

factored=True は、kernelがfactorizationを提供するときだけ使用できる。
AmpWidth はこの経路を提供する。

### 明示的なconfig

    from torchcst import CSTNormalizedAdam, NormalizedOptimizerConfig
    from torchcst.optim import NormalizedBoxFixedPointSolver

    optimizer = CSTNormalizedAdam(
        model,
        cst=NormalizedOptimizerConfig(
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            trust_radius=1.0,
            solver=NormalizedBoxFixedPointSolver(max_iter=64),
            initial_zero_step=False,
        ),
    )

### dense parameterとの混在

modelにCST siteと通常のdense parameterが混在する場合、normalized optimizerは
CST siteをこの経路で更新し、dense parameterを別途 AdamWConfig で所有できる。

    from torchcst import AdamWConfig, CSTNormalizedAdam

    optimizer = CSTNormalizedAdam(
        model,
        dense=AdamWConfig(lr=1e-3),
        lr=1e-3,
    )

同じoptimizer stepの中で、CST proposalとdense AdamW proposalを検証してから、model全体
へcommitする。dense parameterが無い場合は dense を渡す必要はない。

## 10. stateとメモリ

正規化Adamのreference実装は、各atomについて次を保持する。

    numerator:   m [K,P]       + C [K,P,P]
    denominator: x [K,P]       + y [K,P,P] + Z [K,P,P,P]

特に Z は H の二次積を完全に保持するため、分母側は O(KP^3) のreference
stateになる。これはdense represented-weightの m,v を保存する方式とは異なり、
atom coordinateに閉じたstateである一方、P が大きい場合には無視できない。

現在は正しさを優先した完全表現であり、Z のfactorized/compressed representationは
今後の最適化候補である。N/Dのcomponent contractを維持すれば、低メモリ実装へ差し
替えてもoptimizer wrapperとsolverの設計は変えずに済む。

## 11. 既存optimizerとの関係と現在の境界

- CSTNormalizedAdam は新しいcomposable normalized familyのAdamである。
- CSTNormalizedSGD、CSTNormalizedMomentum、CSTNormalizedRMSProp は同じ family の
  残りの wrapper である。CSTSGD、CSTMomentum、CSTRMSProp はその別名である。
- CSTAdamR はそのAdam wrapperに、タスクmomentsの外でatom演算子斥力を足したものである。
- 歴史的な CSTAdam は別のtangent-moment behaviorを持ち、同じ名前の実装ではない。
- CSTImplicitAdam は CSTNormalizedAdam のaliasである。
- 現在のoptimizerは固定shape・frozen chartを前提とする。
- LinearJGHAtomGrad は現在のLinear経路を実装している。Convなど他のmodule familyへ
  広げる場合も、optimizer側は同じ observation contractを使う設計にする。
- solverの非収束は、solverの反復予算・damping・momentのスケール・learning rateの
  影響を受ける。したがって、converged の有無と学習lossの発散を同一視しない。

## 12. 関連する実験

AmpWidth kernel上でのMNIST比較では、elementwise box [-1,1]、
1 step目の d=0 評価、K=64、3 seedの条件で、現時点では
CSTNormalizedAdam が最も安定した結果を示している。この比較はoptimizerの設計を
確定する証明ではなく、次のhyperparameter調整（learning rate、moment係数、solver
反復数、分母の数値スケール）を選ぶための初期診断である。

実験側の詳細な結果とraw JSONは、companion cst repositoryの
torchcst-normalized-optimizer-a100 laneに保存している。
