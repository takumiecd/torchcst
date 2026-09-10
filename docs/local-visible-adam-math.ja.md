# CSTLocalVisibleAdam の数式

`CSTLocalVisibleAdam` は、dense Adam の対角二次モーメント作用素を
atomごとの現在の接空間へ射影する。`CSTLocalAdam` の白色化接空間における
勾配外積EMAとは異なるoptimizerである。

## 記号

atom $a$ のJacobian、同時刻Gram、異時刻Gramを

$$
J_{t,a}\in\mathbb R^{N\times p},\qquad
R_{t,a}=J_{t,a}^\top J_{t,a},\qquad
S_{t,a}=J_{t,a}^\top J_{t-1,a}
$$

とする。以下ではatom添字を省略する。異なるatom間のblockは構成しない。

## 一次モーメント

一次側は `CSTLocalAdam` と共通である。

$$
\begin{aligned}
b_t&=\beta_1S_t\alpha_{t-1}+(1-\beta_1)J_t^\top g_t^W,\\
(R_t+\lambda_1I)\alpha_t&=b_t,\\
\widehat b_t&=b_t/(1-\beta_1^t).
\end{aligned}
$$

## 対角二次モーメントのvisible operator

dense Adam のraw second momentを $v_t^W$、対角作用素を
$V_t=\operatorname{Diag}(v_t^W)$ とする。現在のatomから観測できるblockは

$$
U_t=J_t^\top V_tJ_t\in\mathbb R^{p\times p}
$$

である。現在勾配による新しい観測は

$$
E_t=J_t^\top\operatorname{Diag}((g_t^W)^{\odot2})J_t.
$$

実装は `AtomGrad.atom_square` によって $E_t$ を `[K,p,p]` の縮約として
取得する。勾配は同一stepの全寄与を合算してから二乗する。full Jacobian、
対角行列、atom間blockは作らない。

前stepのvisible operator代表を

$$
\widehat V_{t-1}=J_{t-1}\Gamma_{t-1}J_{t-1}^\top
$$

と置く。現在frameからの観測は

$$
J_t^\top\widehat V_{t-1}J_t
=S_t\Gamma_{t-1}S_t^\top
$$

なので、raw EMA numeratorを

$$
U_t=\beta_2S_t\Gamma_{t-1}S_t^\top+(1-\beta_2)E_t
$$

とする。保存係数はridge再圧縮

$$
(R_t+\lambda_2I)\Gamma_t(R_t+\lambda_2I)=U_t
$$

で定義する。実装は逆行列を作らず、Cholesky因子を用いた左右solveで計算する。

## 更新計量

正定値かつ無減衰の場合、$V_t$ を接空間に支持された代表

$$
\widehat V_t=J_tR_t^{-1}\widehat U_tR_t^{-1}J_t^\top,
\qquad
\widehat U_t=U_t/(1-\beta_2^t)
$$

へ置き換える。この代表に対して

$$
J_t^\top\sqrt{\widehat V_t}J_t
=R_t\mathbin{\#}\widehat U_t
$$

が厳密に成り立つ。$\#$ は正定値行列の幾何平均である。したがってdense
対角Adamの計量を

$$
J_t^\top\operatorname{Diag}(\sqrt{v_t^W})J_t
\approx R_t\mathbin{\#}\widehat U_t
$$

と近似する。

実装では $\bar R_t=R_t+\lambda_2I=L_tL_t^\top$ として

$$
\begin{aligned}
C_t&=L_t^{-1}\widehat U_tL_t^{-\top},\\
M_t&=L_t(\sqrt{C_t}_{\rm PSD}+\varepsilon I)L_t^\top
\end{aligned}
$$

を計算する。$\sqrt{C_t}_{\rm PSD}$ は小さいbatched固有値分解で得る。
Cholesky因子はstep内だけで使い、白色化基底 $B$ は保存しない。

最終更新は

$$
(M_t+\mu I)\Delta\theta_t=-\eta\widehat b_t
$$

をatomごとに直接解く。

## 近似境界とメモリ

- 異なるatom間の $J_{t,a}^\top V_tJ_{t,b}$ は省略する。supportが重なる場合は近似である。
- dense対角作用素 $V_t$ を、前stepの接空間に支持された
  $J_{t-1}\Gamma_{t-1}J_{t-1}^\top$ で置き換える。
- 正則化 $\lambda_1,\lambda_2,\mu$ は特異系を扱うためのridge近似である。
- 永続二次状態は $\Gamma\in\mathbb R^{K\times p\times p}$。$B$、dense $v$、
  `[K,K,p,p]` は保存しない。

主要な実装対応は次の通り。

- `ProjectedVisibleSecondMoment`: $E,U,\Gamma,M$ の構成
- `AtomGradientObservation.atom_square`: exactなatom-local $E$ 観測
- `CSTLocalVisibleAdam`: model全体の所有、一次状態との合成、直接更新
