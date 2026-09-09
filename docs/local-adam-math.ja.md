# CSTLocalAdam の数式

2026-09-09。公開実装 `CSTLocalAdam` / `LocalAdamConfig` の定式化。
第1〜8節は既定の `whitening="eigen"`。Cholesky比較版の差分は第9節。
atom内で一次モーメントαと二次モーメントCを輸送し、小さい正則化行列を
直接解く。trust regionは使わない。ここでいう「二次モーメント」は勾配の
外積EMAであり、重み表現の二次Taylor近似や損失Hessianではない。

## 1. 記号と時刻

1つのCST層について、dense重みを並べたベクトルを

$$
w(\theta)=\sum_{a=1}^{K}w_a(\theta_a)\in\mathbb R^N,
\qquad \theta_a\in\mathbb R^q
$$

とする。$q$ はkernelが定める1 atomの座標数で、4に固定しない。
$t=1,2,\ldots$ はoptimizer step、$\theta_{t,a}$ はそのstepの**更新前**の値。

$$
J_{t,a}=\left.\frac{\partial w_a}{\partial\theta_a}\right|_{\theta_{t,a}}
\in\mathbb R^{N\times q},\qquad
g_{t,a}=J_{t,a}^{\top}g_{w,t}\in\mathbb R^q.
$$

$g_{w,t}$ はそのstepに累積されたdense重み勾配を表す説明用の量。
実装はAtomGradから $g_{t,a}$ を受け取り、この式のためにdense勾配や
Jacobian全体を保存する必要はない。以降はatom添字 $a$ を省略する。

$$
R_t=J_t^{\top}J_t,\qquad S_t=J_t^{\top}J_{t-1}
\quad\in\mathbb R^{q\times q}.
$$

$R_t$ は同時刻の局所Gram、$S_t$ は同じatomの異なる時刻間の内積。
異なるatom間の $J_{t,a}^{\top}J_{s,b}$（$a\ne b$）は計算しない。
以下の式はこのatom単位の近似に対する式であり、atom間の直交性を保証するものではない。

## 2. 一次モーメントαの輸送と再圧縮

前stepの履歴を、当時の接空間の係数 $\alpha_{t-1}$ として持つ。

$$
\begin{aligned}
b_t&=\beta_1 S_t\alpha_{t-1}+(1-\beta_1)g_t,\\
(R_t+\lambda I)\alpha_t&=b_t,\\
\widehat b_t&=\frac{b_t}{1-\beta_1^t}.
\end{aligned}
$$

$\lambda>0$ は `first_moment_damping`。$t=1$ の履歴項はゼロとする。
$\alpha_t$ は補正前の $b_t$ を圧縮して保存し、実際の更新には
バイアス補正した $\widehat b_t$ を使う。

この再圧縮は、説明用に
$m_t=\beta_1J_{t-1}\alpha_{t-1}+(1-\beta_1)g_{w,t}$ と置いたときの

$$
\alpha_t=\mathop{\mathrm{argmin}}_x
\left\{\frac12\|J_tx-m_t\|_2^2+\frac\lambda2\|x\|_2^2\right\}
$$

に対応する。単なる座標ごとのスケール調整ではなく、同じatom内の
方向の相関も使う。$\lambda>0$ なので厳密な直交射影ではなく、
表現できる方向にも縮小が入る。これにより $R_t$ が特異でも解を定められる。

## 3. 白色化基底Bと可視方向Q

数値上は対称化した局所Gramを固有値分解する。

$$
R_t=V_t\operatorname{diag}(\rho_{t,i})V_t^{\top},\qquad
\rho_{\max}=\max_i\max(\rho_{t,i},0).
$$

$\tau=$ `tangent_rtol` として

$$
s_{t,i}=\begin{cases}
\rho_{t,i}^{-1/2}&\rho_{t,i}>\tau\rho_{\max},\\
0&\text{それ以外},
\end{cases}
\qquad B_t=V_t\operatorname{diag}(s_{t,i}),\qquad Q_t=J_tB_t.
$$

$B_t\in\mathbb R^{q\times q}$。$Q_t\in\mathbb R^{N\times q}$ は説明用で、
実装では作らない。採用した列だけが正規直交し、不採用の列はゼロなので、
$Q_t^{\top}Q_t=\operatorname{diag}(\text{active}_i)$。
全方向が不採用なら $B_t=Q_t=0$。

ここは現在も固有値の**閾値除外**であり、$(R_t+\lambda I)^{-1/2}$ に
置き換えたものではない。αの正則化と、Bの方向選択は別の処理である。

## 4. 二次モーメントCの輸送とEMA

前のQ座標から現在のQ座標へ移す小行列は

$$
T_t=B_t^{\top}S_tB_{t-1}=Q_t^{\top}Q_{t-1}
\in\mathbb R^{q\times q}.
$$

現在の観測とEMAを

$$
\begin{aligned}
h_t&=B_t^{\top}g_t=Q_t^{\top}g_{w,t},\\
C_t&=\beta_2T_tC_{t-1}T_t^{\top}+(1-\beta_2)h_th_t^{\top},\\
\widehat C_t&=\frac{C_t}{1-\beta_2^t}
\end{aligned}
$$

で定義する。$t=1$ の輸送履歴はゼロ。実装は丸め誤差対策としてCを対称化する。
バイアス補正はAdam型の係数補正であり、射影で捨てた履歴を復活させるものではない。

CはQ座標で保存するため、輸送には $S_tC_{t-1}S_t^{\top}$ ではなく
$T_tC_{t-1}T_t^{\top}$ を用いる。前の方向が現在の同じatomの可視空間に
直交すれば、その履歴は消える。一部重なる場合は重なる成分だけを残す。
別atomがその方向を表現できても、そこへの引き継ぎは行わない。

$h_th_t^{\top}$ は**累積勾配の外積**で、勾配を観測するたびに外積を
作って加算する処理とは異なる。また、$g_w^{\odot2}$ のdense対角EMAを
引き戻したものでも、$g_\theta^{\odot2}$ の座標別EMAでもない。

## 5. 更新に使う計量と直接解

$$
\begin{aligned}
A_t&=B_t^{\top}R_t=Q_t^{\top}J_t,\\
D_t^{(Q)}&=\sqrt{\widehat C_t}_{\mathrm{PSD}}+\varepsilon I,\\
M_t&=A_t^{\top}D_t^{(Q)}A_t,\\
(M_t+\mu I)\Delta\theta_t&=-\eta\widehat b_t,\\
\theta_{t+1}&=\theta_t+\Delta\theta_t.
\end{aligned}
$$

PSD平方根はCの固有値をゼロ以上に切り上げて平方根を取り、再構成する。
$\eta=$ `lr`、$\varepsilon=$ `eps`、$\mu>0=$ `update_damping`。
$\varepsilon I$ はQ座標で平方根の**外側**に加え、$\mu I$ はparameter座標で
Mに加える。$\sqrt{\widehat C_t+\varepsilon I}$ とは異なる。

同じ更新は

$$
\Delta\theta_t=\mathop{\mathrm{argmin}}_d
\left\{\widehat b_t^{\top}d+
\frac{1}{2\eta}d^{\top}(M_t+\mu I)d\right\}
$$

とも書ける。半径制約やnorm clippingはない。$\mu$ は学習率で割る前の
Mに加える規約であり、$M_t/\eta+\mu I$ を使う規約とは異なる。

ここでのMは今回定義した局所計量である。元々検討していたdenseの
対角EMA $D_t$ に対する $J_t^{\top}D_tJ_t$ を厳密に高速計算したものではない。
計量と履歴の表現を変更して、小さい直接解へ落としている。

## 6. 時刻・保存状態・計算量

1 stepで $R_t,S_t,B_t,C_t,b_t$ を更新前の $\theta_t$ から計算し、
更新を適用するときに、次回用の $(\theta_t,\alpha_t,B_t,C_t)$ を保存する。
保存する基底は $\theta_{t+1}$ の基底ではない。次のstepで新旧の接空間を輸送する。
実装では一次・二次モーメントがそれぞれ旧point等を持つため、これは論理上の状態一覧。

| 量 | 全atom分の形 | 保存精度・役割 |
| --- | --- | --- |
| α、旧point | 各 $K\times q$ | parameter dtype、一次履歴と旧Jacobianの再評価 |
| B | $K\times q\times q$ | FP64、旧白色化基底 |
| C | $K\times q\times q$ | parameter dtype、Q座標の二次履歴 |
| R、S、T、M | 各 $K\times q\times q$ | step内の作業量、全体Gramではない |

小行列の固有値分解・CholeskyはFP64で行う。局所Gram自体はparameter dtypeで
構成してからFP64へ変換するので、その構成時の丸めまでFP64になるわけではない。
checkpoint復元でもBのFP64を保持する。

モーメント状態のオーダーは $O(Kq^2+Kq)$、小行列分解は $O(Kq^3)$。
これとは別に勾配・kernel微分・局所Gram構成の時間と作業メモリが必要。
分離可能な行列kernelで微分が定数個の外積に分かれる場合、行数m・列数nに対して
局所内積は $O(Kq^2(m+n))$ で構成できる。これはkernelの構造に依存し、
一般のkernelへの一律の計算量保証ではない。

再圧縮と更新はbatched Choleskyの直接解で、逆行列を明示的に作らず、
反復線形solverも使わない。ただし小さい固有値分解は残る。
factorized経路はdense Jacobianを避けるが、reference経路は検証用に実体化し得る。
`device_execution=True` も、ライブラリ内部を含む同期ゼロの保証ではない。

## 7. 近似と今後の検討点

- atom間の履歴・共分散の結合を省略する。将来の斥力正則化は未実装であり、
  それによるatom間独立性は現在の式の保証に含めない。
- Bの閾値を跨ぐ方向は不連続に出入りする。αと更新の正則化だけで
  この不連続性が消えるわけではない。
- $\lambda I$ と $\mu I$ はparameter座標の単位に依存する。
  trust regionを外しても、位置・幅・振幅の単位依存性すべてを解決したわけではない。
- 二次モーメントはこの局所表現に対するEMAである。dense Adamとの同値性や、
  多様な学習問題での優位性は主張しない。

## 8. API・実装との対応

`CSTLocalAdam(model, cst=LocalAdamConfig(...), dense=AdamWConfig(...))` として
model全体を渡す。CSTLinearは上記の式、通常のLinear等はdense AdamWで扱う。
更新提案を検証してから同時に適用する。既存のCSTAdamとCSTSecondOrderAdamは別方式。

| 記号 | config | 既定値 |
| --- | --- | --- |
| $\eta$ | `lr` | `1e-3` |
| $(\beta_1,\beta_2)$ | `betas` | `(0.9, 0.999)` |
| $\lambda$ | `first_moment_damping` | `1e-2` |
| $\mu$ | `update_damping` | `1e-2` |
| $\varepsilon$ | `eps` | `1e-8` |
| $\tau$ | `tangent_rtol` | `1e-6` |

`solve_rtol=1e-5` は直接解の相対残差の検査用で、反復回数の制御ではない。

- [αのEMA](../src/torchcst/optim/moments/tangent.py)
- [atom内輸送・再圧縮](../src/torchcst/_derivatives/local_tangent.py)
- [B・T・C・Mの構成](../src/torchcst/optim/moments/transported_rms.py)
- [直接更新](../src/torchcst/optim/_regularized_update.py)
- [公開optimizer](../src/torchcst/optim/local.py)
- [API移植時の検証記録](experiments/local-adam-api.ja.md)
- [trust regionなしの学習実験](experiments/no-trust.ja.md)

## 9. 正則化Cholesky比較版

`LocalAdamConfig(whitening="cholesky")` では、第3節の固有値分解・閾値除外を
次に置き換える。この節の $\lambda$ は白色化用の正則化を表す。
`whitening_damping` を指定すると一次の `first_moment_damping` と独立に設定できる。
既定の `None` は従来どおり一次と同じ値を使う。指定値は有限の正数。

$$
R_t+\lambda I=L_tL_t^\top,\qquad B_t=L_t^{-\top}.
$$

ここでの $L_t$ はCholesky因子。第5節の平方根計量 $D_t^{(Q)}$ とは別。実装は
`solve_triangular(L_t.T, I)` で小さいBを得る。全体Jacobianの逆行列は作らない。
既定では一次再圧縮と正則化値が共通だが、分解の計算結果自体は再利用しない。
`whitening_damping` を別の値にすれば、分解対象の行列も異なる。
第4・5節のT、h、C、A、M、更新式は同じ形を用いる。したがって

$$
h_t=L_t^{-1}g_t,\qquad
T_t=L_t^{-1}S_tL_{t-1}^{-\top}
$$

となる。$\lambda=0$ かつRが正定値ならQは正規直交し、閾値で方向を除外しない
eigen版との違いは直交基底の選択になる。この条件で更新計量の一致をテストしている。
公開configでは特異なRも扱うため $\lambda>0$ を必要とする。

正則化すると

$$
Q_t^\top Q_t=B_t^\top R_tB_t
=I-\lambda B_t^\top B_t\preceq I
$$

であり、厳密な白色化ではなく収縮する座標になる。Qの特異値の二乗は
$\rho_i/(\rho_i+\lambda)$。弱い方向を閾値で切る代わりに連続的に弱める。
前後でJが変わらなくてもTは一般にIではなく、Cの輸送で追加の減衰が入る。
従って現在版の単なる等価な高速実装とは扱わない。

Rに対する固有値判定はなくなるが、CのPSD平方根では固有値分解と負固有値の
ゼロへの切り上げが残る。数値的失敗の検査も残る。これは全処理からの分岐除去ではない。
`tangent_rtol` はこの方式のBに影響しない。方式の異なるcheckpointの読み込みは拒否する。

### 9.1. Lの安定化は分解する前のRに加える

現在のCholesky版は、無正則化のRからLを作っているわけではない。
**分解前のRの対角に、白色化用の正則化を加えている。**
既定では一次モーメントと同じ `first_moment_damping` を使い、
`whitening_damping` を指定した場合はその値を使う。
この値を白色化用のepsilonと呼ぶなら、$\varepsilon_R=\lambda$ に相当する。

$$
\begin{aligned}
\widetilde R_t&=\tfrac12(R_t+R_t^\top)+\varepsilon_R I,\\
L_t&=\operatorname{chol}(\widetilde R_t),\\
(R_t+\lambda_\alpha I)\alpha_t&=b_t,\\
L_t^\top B_t&=I.
\end{aligned}
$$

一次再圧縮は上から3行目、白色化基底は4行目のsolveに対応する。
一次側の $\lambda_\alpha$ は `first_moment_damping`。
$\varepsilon_R=\lambda_\alpha$ の場合だけ同じ行列になり、現実装はその場合も別々に分解する。
公開configは $\varepsilon_R>0$ を要求する。既定値と最初の比較実験値は0.01。

厳密演算でRが半正定値なら、$\widetilde R_t$ の最小固有値は
$\varepsilon_R$ 以上なので、特異なRにもCholeskyを適用できる。また、

$$
\|L_t^{-1}\|_2=\|B_t\|_2\leq\frac1{\sqrt{\varepsilon_R}}.
$$

例えば $R=0$、$\varepsilon_R=0.01$ なら $L=0.1I$、$B=10I$ と有限になる。
このとき $J=0$ なら $Q=JB=0$ であり、見えない方向が復活するわけではない。
浮動小数点の丸め誤差まで無条件に保証する式ではなく、実装は分解の成功と有限値を検査する。

Lを作った**後**に $L+\epsilon I$ とする操作とは異なる。
後者では

$$
(L+\epsilon I)(L+\epsilon I)^\top
=LL^\top+\epsilon(L+L^\top)+\epsilon^2I
$$

となり、元のGramへの単純な対角正則化ではなくなる。さらに、無正則化Rの
分解自体が失敗する場合を防げない。現在は分解前に加える方式で対応している。

| 安定化する場所 | config | 既定値 | 式 |
| --- | --- | --- | --- |
| 一次再圧縮のR | `first_moment_damping` | `0.01` | $R+\lambda_\alpha I$ |
| Cholesky白色化のR | `whitening_damping` | `None`（一次と同値） | $R+\varepsilon_R I$ |
| Q座標のCの平方根の外側 | `eps` | `1e-8` | $\sqrt{\widehat C}+\varepsilon I$ |
| 最終更新のparameter計量 | `update_damping` | `0.01` | $M+\mu I$ |

したがって、APIの `eps` を変えてもLの安定化強度は変わらない。
`whitening_damping=None` の場合だけ、`first_moment_damping` を変えると両方に作用する。
独立指定した場合、一次再圧縮の正則化は変わらない。
Rへの正則化自体は最初のCholesky比較から存在し、後から強度を独立設定できるようにした。
checkpointには有効な白色化強度を反映し、異なる強度の履歴は黙って読み替えない。
