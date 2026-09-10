# CSTDenseVisibleAdam の数式

`CSTDenseVisibleAdam` は、精度を優先して Adam の一次・二次モーメントを
visible weight と同じ空間に保持し、更新だけを CST の atom-local 接空間へ落とす。
`CSTLocalAdam` と `CSTLocalVisibleAdam` のメモリ節約版とは別optimizerである。

## 状態

visible weight を $W\in\mathbb{R}^{O\times I}$、step $t$ の勾配を
$G_t=\partial\mathcal{L}/\partial W$ とする。永続状態は

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)G_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)(G_t\odot G_t)
\]

であり、$m_t,v_t\in\mathbb{R}^{O\times I}$ である。bias correction は

\[
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}
\]

とする。`AtomGrad` は同じ `CSTLinear` が1 backward中に複数回使われても、
各寄与を先に加算して正確な $G_t$ を作り、その後に二乗する。

## atom-local 更新

更新前の atom $a$ の可視 Jacobian を
$J_{t,a}\in\mathbb{R}^{OI\times p}$ とする。一次側は

\[
b_{t,a}=J_{t,a}^{\mathsf T}\operatorname{vec}(\widehat m_t)
\]

である。二次側は Adam の分母

\[
d_t=\sqrt{\widehat v_t}+\epsilon
\]

を先に visible 空間で厳密に作り、

\[
M_{t,a}
=J_{t,a}^{\mathsf T}\operatorname{Diag}(\operatorname{vec}(d_t))J_{t,a}
\in\mathbb{R}^{p\times p}
\]

を計算する。更新は $K$ 個の独立な系

\[
\boxed{
(M_{t,a}+\lambda I)\Delta p_{t,a}=-\eta b_{t,a}
}
\]

を直接解く。逆行列も $[K,K,p,p]$ 行列も作らない。

## 厳密な部分と近似する部分

次は dense Adam と同じである。

- visible 勾配 $G_t$ の集約
- visible $m_t,v_t$ の EMA
- $\sqrt{\widehat v_t}+\epsilon$ の要素ごとの構成
- 各 $J_{t,a}^{\mathsf T}\operatorname{Diag}(d_t)J_{t,a}$

相違は、dense $W$ を直接更新せず、現在の各 atom 接空間へ更新を制限すること、
および $a\ne b$ の metric block
$J_{t,a}^{\mathsf T}\operatorname{Diag}(d_t)J_{t,b}$ を捨てることである。
atom間の重なりを斥力で小さくする設計とは整合するが、有限の重なりがある間は
これは明示的な atom-local 近似である。

`CSTLocalVisibleAdam` と違い、移動する接空間での再帰的な射影輸送も、
matrix geometric meanによる平方根の再構成も行わない。

## メモリ

永続momentは

\[
2OI\quad\text{scalars}
\]

であり、dense Adam の $m,v$ と同じ大きさである。CSTによるmoment-stateの
メモリ削減はない。一方、Jacobianは永続化も全体 materialize もせず、行とatomを
tileして $K$ 個の $p\times p$ blockへ縮約する。一時メモリはtile幅で制御する。

このoptimizerは、compact版の精度低下がmoment履歴の射影によるものかを調べる
accuracy baselineであり、メモリ優位を主目的とするoptimizerではない。
