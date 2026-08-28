# CST Initialization as a Multi-Objective / Constrained Optimization Problem

## 1. 目的

CST (Continuous Synaptic Tensor) の初期化では、パラメータ

$$
\theta
=
\left(
w_1,\ldots,w_M,
s_1^\top,\ldots,s_M^\top,
t_1^\top,\ldots,t_M^\top
\right)^\top
\in\mathbb{R}^{p}
$$

を単にランダムに与えるのではなく、初期状態で以下の性質を同時に満たすように決定する。

1. 出力の平均を所望の値にする。
2. 出力の分散を所望の値にする。
3. CST パラメータ \(\theta\) に対する勾配の大きさを所望の値にする。
4. 必要であれば、各パラメータ方向の勾配感度を均一化し、パラメータ間の干渉を抑える。

したがって、CST の初期化を

$$
\theta^\star
=
\operatorname*{arg\,min}_{\theta}
\mathcal{L}_{\mathrm{init}}(\theta)
$$

または

$$
F(\theta)=0
$$

を満たす陰関数として定義する。

---

# 2. CST による重み行列

CST で生成される重み行列を

$$
W(\theta)\in\mathbb{R}^{n_{\mathrm{out}}\times n_{\mathrm{in}}}
$$

とする。

各成分は

$$
W_{ji}(\theta)
=
\sum_{a=1}^{M}
w_a\,
\kappa_{\mathrm{out}}
\left(
\mu_j^{\mathrm{out}},t_a
\right)
\,
\kappa_{\mathrm{in}}
\left(
\mu_i^{\mathrm{in}},s_a
\right)
$$

で与える。

略記として

$$
K^o_{ja}
=
\kappa_{\mathrm{out}}
\left(
\mu_j^{\mathrm{out}},t_a
\right),
\qquad
K^i_{ia}
=
\kappa_{\mathrm{in}}
\left(
\mu_i^{\mathrm{in}},s_a
\right)
$$

とおけば、

$$
W_{ji}
=
\sum_{a=1}^{M}
w_a K^o_{ja}K^i_{ia}.
$$

---

# 3. 入力・出力分布

線形変換

$$
y=W(\theta)x
$$

を考える。

入力分布を

$$
x
\sim
\mathcal{N}
\left(
\mu_x,\Sigma_x
\right)
$$

とすると、\(\theta\) を固定した条件下で

$$
y
\sim
\mathcal{N}
\left(
W(\theta)\mu_x,
\;
W(\theta)\Sigma_xW(\theta)^\top
\right).
$$

したがって、

$$
\mu_y(\theta)
=
W(\theta)\mu_x
$$

および

$$
\Sigma_y(\theta)
=
W(\theta)\Sigma_xW(\theta)^\top
$$

が出力の平均・共分散を決定する。

---

# 4. 平均条件

目標出力平均を

$$
\mu_y^\star
$$

とする。

平均保存条件は

$$
W(\theta)\mu_x
=
\mu_y^\star
$$

である。

したがって残差を

$$
F_\mu(\theta)
=
W(\theta)\mu_x-\mu_y^\star
$$

と定義し、

$$
\boxed{
F_\mu(\theta)=0
}
$$

を初期化条件とする。

各出力ニューロン \(j\) について書けば、

$$
F_{\mu,j}(\theta)
=
\sum_{i=1}^{n_{\mathrm{in}}}
W_{ji}(\theta)\mu_{x,i}
-
\mu_{y,j}^\star.
$$

CST を展開すると

$$
F_{\mu,j}(\theta)
=
\sum_{a=1}^{M}
w_a K^o_{ja}
\left(
\sum_{i=1}^{n_{\mathrm{in}}}
K^i_{ia}\mu_{x,i}
\right)
-
\mu_{y,j}^\star.
$$

特に

$$
\mu_x=0
$$

かつ

$$
\mu_y^\star=0
$$

なら、この条件は自動的に満たされる。

---

# 5. 分散条件

出力共分散は

$$
\Sigma_y(\theta)
=
W(\theta)\Sigma_xW(\theta)^\top.
$$

まず、各ニューロンの分散のみを保存する場合、

$$
v_y(\theta)
=
\operatorname{diag}
\left(
W(\theta)\Sigma_xW(\theta)^\top
\right)
$$

とする。

目標分散ベクトルを

$$
v_y^\star
$$

とすれば、

$$
F_\sigma(\theta)
=
\operatorname{diag}
\left(
W(\theta)\Sigma_xW(\theta)^\top
\right)
-
v_y^\star.
$$

したがって、

$$
\boxed{
F_\sigma(\theta)=0
}
$$

を要求する。

---

## 5.1 IID 入力の場合

入力成分が独立で

$$
\Sigma_x
=
\sigma_x^2 I
$$

なら、

$$
\operatorname{Var}[y_j]
=
\sigma_x^2
\sum_{i=1}^{n_{\mathrm{in}}}
W_{ji}^2.
$$

したがって

$$
F_{\sigma,j}(\theta)
=
\sigma_x^2
\sum_i
W_{ji}(\theta)^2
-
\sigma_{y,j}^{\star 2}.
$$

CST を代入すると、

$$
\boxed{
F_{\sigma,j}(\theta)
=
\sigma_x^2
\sum_i
\left(
\sum_a
w_a K^o_{ja}K^i_{ia}
\right)^2
-
\sigma_{y,j}^{\star 2}
}
$$

となる。

---

## 5.2 atom 間相互作用

二乗項を展開すると、

$$
\sum_i W_{ji}^2
=
\sum_i
\sum_{a,b}
w_aw_b
K^o_{ja}K^o_{jb}
K^i_{ia}K^i_{ib}.
$$

したがって、

$$
\sum_i W_{ji}^2
=
\sum_{a,b}
w_aw_b
K^o_{ja}K^o_{jb}
\left\langle
K^i_a,K^i_b
\right\rangle.
$$

よって

$$
\boxed{
\operatorname{Var}[y_j]
=
\sigma_x^2
\sum_{a,b}
w_aw_b
K^o_{ja}K^o_{jb}
\left\langle
K^i_a,K^i_b
\right\rangle
}
$$

となる。

この式から、CST の分散は単純な

$$
\operatorname{Var}(w_a)
$$

だけでは決まらず、

$$
\left\langle K_a^i,K_b^i \right\rangle
$$

という atom 間の重なりにも依存することが分かる。

---

# 6. 勾配条件

損失を

$$
L(W(\theta))
$$

とする。

dense weight space における勾配を

$$
g_W
=
\operatorname{vec}
\left(
\frac{\partial L}{\partial W}
\right)
$$

とする。

また CST 写像の Jacobian を

$$
J(\theta)
=
\frac{\partial\operatorname{vec}W(\theta)}
{\partial\theta}
$$

とする。

chain rule より、

$$
\boxed{
g_\theta
=
\nabla_\theta L
=
J(\theta)^\top g_W
}
$$

である。

## 6.1 CST Jacobian の具体形

kernel 列ベクトルを

$$
k_a^o
=
\left(K^o_{1a},\ldots,K^o_{n_{\mathrm{out}}a}\right)^\top,
\qquad
k_a^i
=
\left(K^i_{1a},\ldots,K^i_{n_{\mathrm{in}}a}\right)^\top
$$

と定義すると、atom \(a\) が生成する重み行列は

$$
W^{(a)}
=
w_a k_a^o(k_a^i)^\top,
\qquad
W
=
\sum_{a=1}^{M}W^{(a)}
$$

である。したがって、振幅に関する微分は

$$
\boxed{
\frac{\partial W}{\partial w_a}
=
k_a^o(k_a^i)^\top
}
$$

となる。\(s_a\in\mathbb{R}^{d_i}\)、\(t_a\in\mathbb{R}^{d_o}\) とし、
\(s_{a,r}\)、\(t_{a,q}\) をそれぞれの座標成分とすると、

$$
\boxed{
\frac{\partial W_{ji}}{\partial s_{a,r}}
=
w_a K^o_{ja}
\frac{\partial K^i_{ia}}{\partial s_{a,r}},
\qquad
\frac{\partial W_{ji}}{\partial t_{a,q}}
=
w_a
\frac{\partial K^o_{ja}}{\partial t_{a,q}}
K^i_{ia}
}
$$

である。すなわち、CST Jacobian の列は

$$
\boxed{
\begin{aligned}
J_{w_a}
&=
\operatorname{vec}\!\left(k_a^o(k_a^i)^\top\right),
\\[1mm]
J_{s_{a,r}}
&=
w_a\operatorname{vec}\!\left(
k_a^o
\left(\partial_{s_{a,r}}k_a^i\right)^\top
\right),
\\[1mm]
J_{t_{a,q}}
&=
w_a\operatorname{vec}\!\left(
\left(\partial_{t_{a,q}}k_a^o\right)
(k_a^i)^\top
\right).
\end{aligned}
}
$$

よって Frobenius norm は CST の kernel 列を用いて

$$
\boxed{
\begin{aligned}
\|J(\theta)\|_F^2
=
\sum_{a=1}^{M}
\Bigg[
&\|k_a^o\|_2^2\|k_a^i\|_2^2
\\
&+
w_a^2\|k_a^o\|_2^2
\sum_{r=1}^{d_i}
\left\|
\partial_{s_{a,r}}k_a^i
\right\|_2^2
\\
&+
w_a^2\|k_a^i\|_2^2
\sum_{q=1}^{d_o}
\left\|
\partial_{t_{a,q}}k_a^o
\right\|_2^2
\Bigg]
\end{aligned}
}
$$

と展開できる。この式では、\(J\) の列同士の内積ではなく各列の二乗
norm を加算しているため、異なる atom 間の交差項は現れない。一方、
\(J^\top J\) には異なる atom 間の kernel overlap と kernel 微分の overlap
が現れる。

---

# 7. 勾配ノルム保存

初期化時に CST パラメータ空間で所望の勾配ノルム

$$
G^\star
$$

を持たせたいとする。

その場合、

$$
\left\|
J(\theta)^\top g_W
\right\|_2^2
=
(G^\star)^2
$$

を要求できる。

ただし \(g_W\) は入力データや上流勾配に依存するため、初期化条件としては期待値を取って

$$
\boxed{
\mathbb{E}
\left[
\|
\nabla_\theta L
\|_2^2
\right]
=
(G^\star)^2
}
$$

とするのが自然である。

残差は

$$
F_g(\theta)
=
\mathbb{E}
\left[
\|
J(\theta)^\top g_W
\|_2^2
\right]
-
(G^\star)^2.
$$

---

## 7.1 dense gradient が等方的な場合

初期化時に

$$
\mathbb{E}
\left[
g_Wg_W^\top
\right]
=
cI
$$

と近似すると、

$$
\begin{aligned}
\mathbb{E}
\left[
\|
g_\theta
\|_2^2
\right]
&=
\mathbb{E}
\left[
g_W^\top
JJ^\top
g_W
\right]
\\
&=
\operatorname{tr}
\left(
JJ^\top
\mathbb{E}[g_Wg_W^\top]
\right)
\\
&=
c\operatorname{tr}(JJ^\top)
\\
&=
c\operatorname{tr}(J^\top J)
\\
&=
c\|J\|_F^2.
\end{aligned}
$$

したがって、

$$
\boxed{
c\|J(\theta)\|_F^2
=
(G^\star)^2
}
$$

を満たせばよい。

つまり

$$
\boxed{
F_g(\theta)
=
\|J(\theta)\|_F^2
-
\frac{(G^\star)^2}{c}
}
$$

とできる。

---

# 8. 勾配の等方性

勾配ノルムだけを保存すると、一部のパラメータにのみ大きな勾配が入り、他のパラメータでは勾配が消える初期化も許容される。

そこで、より強い条件として

$$
\boxed{
J(\theta)^\top J(\theta)
\approx
\gamma I
}
$$

を考える。

dense gradient が

$$
\mathbb{E}[g_Wg_W^\top]
=
cI
$$

なら、

$$
\mathbb{E}
[g_\theta g_\theta^\top]
=
cJ^\top J.
$$

したがって

$$
J^\top J
=
\gamma I
$$

であれば、

$$
\boxed{
\mathbb{E}
[g_\theta g_\theta^\top]
=
c\gamma I
}
$$

となり、CST パラメータ空間における初期勾配が等方的になる。

---

## 8.1 対角成分

各パラメータの感度をそろえる条件として、

$$
\boxed{
\operatorname{diag}
(J^\top J)
=
\gamma\mathbf{1}
}
$$

を課す。

---

## 8.2 非対角成分

パラメータ間干渉を抑える条件として、

$$
\boxed{
\operatorname{offdiag}
(J^\top J)
\approx 0
}
$$

を課すことができる。

なお、\(p=\dim\theta\)、\(d=n_{\mathrm{out}}n_{\mathrm{in}}\) とすると
\(J\in\mathbb{R}^{d\times p}\) である。\(J^\top J=\gamma I_p\) を
\(\gamma>0\) で厳密に満たすには \(p\le d\) が必要であり、\(p>d\) の場合は
この条件を soft objective として用いるか、非零特異値に対する等方性
\(JJ^\top\approx\gamma I_d\) を用いる。

---

# 9. 陰関数としての初期化

平均・分散・勾配条件をまとめて

$$
F(\theta)=0
$$

と定義する。

例えば、

$$
\boxed{
F(\theta)
=
\begin{bmatrix}
F_\mu(\theta)
\\
F_\sigma(\theta)
\\
F_g(\theta)
\end{bmatrix}
}
$$

とする。

すなわち、

$$
\boxed{
\begin{cases}
W(\theta)\mu_x
=
\mu_y^\star
\\[2mm]
\operatorname{diag}
\left(
W(\theta)\Sigma_xW(\theta)^\top
\right)
=
v_y^\star
\\[2mm]
\mathbb{E}
\left[
\|
J(\theta)^\top g_W
\|_2^2
\right]
=
(G^\star)^2
\end{cases}
}
$$

を同時に満たす \(\theta^\star\) を初期値とする。

したがって CST initialization を

$$
\boxed{
\theta^\star:
F(\theta^\star)=0
}
$$

という陰関数として定義できる。

---

# 10. 多目的最適化として解く場合

一般には、すべての条件を完全に

$$
F(\theta)=0
$$

とする解が存在するとは限らない。

その場合は各条件の残差を目的関数に変換する。

$$
\mathcal{L}_{\mu}(\theta)
=
\frac{1}{2}
\|
F_\mu(\theta)
\|_2^2,
$$

$$
\mathcal{L}_{\sigma}(\theta)
=
\frac{1}{2}
\|
F_\sigma(\theta)
\|_2^2,
$$

$$
\mathcal{L}_{g}(\theta)
=
\frac{1}{2}
F_g(\theta)^2.
$$

さらに勾配等方性を入れるなら、

$$
\mathcal{L}_{\mathrm{iso}}(\theta)
=
\frac{1}{2}
\left\|
J^\top J-\gamma I
\right\|_F^2.
$$

これらを統合して

$$
\boxed{
\mathcal{L}_{\mathrm{init}}(\theta)
=
\lambda_\mu
\mathcal{L}_{\mu}
+
\lambda_\sigma
\mathcal{L}_{\sigma}
+
\lambda_g
\mathcal{L}_{g}
+
\lambda_{\mathrm{iso}}
\mathcal{L}_{\mathrm{iso}}
}
$$

とする。

そして

$$
\boxed{
\theta^\star
=
\operatorname*{arg\,min}_{\theta}
\mathcal{L}_{\mathrm{init}}(\theta)
}
$$

を CST の初期パラメータとする。

---

# 11. 制約付き最適化として解く場合

分散保存などを絶対条件として扱い、勾配特性をその制約下で最適化する方法も考えられる。

例えば、

$$
\boxed{
\begin{aligned}
\min_{\theta}
\quad&
\left\|
J(\theta)^\top J(\theta)
-
\gamma I
\right\|_F^2
\\
\text{subject to}
\quad&
W(\theta)\mu_x
=
\mu_y^\star
\\
&
\operatorname{diag}
\left(
W(\theta)\Sigma_xW(\theta)^\top
\right)
=
v_y^\star
\end{aligned}
}
$$

とする。

この定式化では、

- 平均保存
- 分散保存

を hard constraint とし、

- 勾配等方性
- 勾配伝播の良さ

を optimization objective として扱う。

CST の初期化では、この形は特に自然である。

---

# 12. ラグランジュ形式

制約付き問題は Lagrange multiplier を用いて

$$
\mathcal{J}
(\theta,\lambda_\mu,\lambda_\sigma)
=
\mathcal{L}_g(\theta)
+
\lambda_\mu^\top F_\mu(\theta)
+
\lambda_\sigma^\top F_\sigma(\theta)
$$

と書ける。

必要条件は

$$
\nabla_\theta
\mathcal{J}
=
0,
$$

$$
F_\mu(\theta)=0,
$$

$$
F_\sigma(\theta)=0.
$$

したがって KKT 系として

$$
\boxed{
\begin{bmatrix}
\nabla_\theta
\mathcal{J}
\\
F_\mu(\theta)
\\
F_\sigma(\theta)
\end{bmatrix}
=
0
}
$$

を数値的に解くこともできる。

---

# 13. 各条件の Jacobian

陰関数や Newton 法を使う場合、

$$
\frac{\partial F}{\partial\theta}
$$

が必要になる。

---

## 13.1 平均条件

$$
F_\mu
=
W(\theta)\mu_x-\mu_y^\star
$$

なので、

$$
\boxed{
\frac{\partial F_\mu}
{\partial\theta_r}
=
\frac{\partial W}
{\partial\theta_r}
\mu_x
}
$$

である。

---

## 13.2 分散条件

$$
\Sigma_y
=
W\Sigma_xW^\top
$$

なので、

$$
d\Sigma_y
=
dW\Sigma_xW^\top
+
W\Sigma_xdW^\top.
$$

したがって

$$
\boxed{
\frac{\partial\Sigma_y}
{\partial\theta_r}
=
\frac{\partial W}{\partial\theta_r}
\Sigma_xW^\top
+
W\Sigma_x
\left(
\frac{\partial W}{\partial\theta_r}
\right)^\top
}
$$

である。

分散成分だけなら、

$$
\boxed{
\frac{\partial F_\sigma}
{\partial\theta_r}
=
\operatorname{diag}
\left[
\frac{\partial W}{\partial\theta_r}
\Sigma_xW^\top
+
W\Sigma_x
\left(
\frac{\partial W}{\partial\theta_r}
\right)^\top
\right]
}
$$

となる。

IID 入力なら、

$$
\Sigma_x
=
\sigma_x^2I
$$

なので、

$$
\boxed{
\frac{\partial
\operatorname{Var}[y_j]}
{\partial\theta_r}
=
2\sigma_x^2
\sum_i
W_{ji}
\frac{\partial W_{ji}}
{\partial\theta_r}
}
$$

まで簡約できる。

---

## 13.3 勾配条件

$$
F_g(\theta)
=
\|J(\theta)\|_F^2-C,
\qquad
C
=
\frac{(G^\star)^2}{c}
$$

とする場合、

$$
\|J\|_F^2
=
\sum_{m,r}
J_{mr}^2.
$$

したがって、

$$
\frac{\partial F_g}{\partial\theta_q}
=
2
\sum_{m,r}
J_{mr}
\frac{\partial J_{mr}}
{\partial\theta_q}.
$$

ここで

$$
J_{mr}
=
\frac{\partial W_m}{\partial\theta_r}
$$

なので、

$$
\frac{\partial J_{mr}}{\partial\theta_q}
=
\frac{\partial^2 W_m}
{\partial\theta_q\partial\theta_r}.
$$

したがって、

$$
\boxed{
\frac{\partial F_g}{\partial\theta_q}
=
2
\sum_{m,r}
\frac{\partial W_m}{\partial\theta_r}
\frac{\partial^2W_m}
{\partial\theta_q\partial\theta_r}
}
$$

となる。

つまり、勾配条件そのものを最適化する場合、

$$
\boxed{
\frac{\partial^2W}{\partial\theta^2}
}
$$

が自然に必要になる。

---

# 14. 推奨する最終定式化

CST initialization では、まず平均・分散を hard constraint とし、勾配特性を soft objective とする形が扱いやすい。

$$
\boxed{
\begin{aligned}
\theta^\star
=
\operatorname*{arg\,min}_{\theta}
\quad&
\alpha
\left(
\|J(\theta)\|_F^2-C
\right)^2
\\
&+
\beta
\left\|
J(\theta)^\top J(\theta)
-
\gamma I
\right\|_F^2
\\[2mm]
\text{subject to}
\quad&
W(\theta)\mu_x
=
\mu_y^\star
\\
&
\operatorname{diag}
\left(
W(\theta)\Sigma_xW(\theta)^\top
\right)
=
v_y^\star.
\end{aligned}
}
$$

入力が

$$
x
\sim
\mathcal{N}
(0,\sigma_x^2I)
$$

であり、出力平均も 0 を要求するなら平均条件は消え、

$$
\boxed{
\begin{aligned}
\theta^\star
=
\operatorname*{arg\,min}_{\theta}
\quad&
\alpha
\left(
\|J(\theta)\|_F^2-C
\right)^2
+
\beta
\left\|
J^\top J-\gamma I
\right\|_F^2
\\
\text{subject to}
\quad&
\sigma_x^2
\operatorname{diag}
\left(
W(\theta)W(\theta)^\top
\right)
=
v_y^\star.
\end{aligned}
}
$$

となる。

---

# 15. 陰関数としての最終表現

外部条件を

$$
\xi
=
\left(
n_{\mathrm{in}},
n_{\mathrm{out}},
M,
\mu_x,
\Sigma_x,
\mu_y^\star,
v_y^\star,
G^\star,
\gamma,
\text{kernel parameters}
\right)
$$

とまとめる。

すると CST initialization は

$$
\boxed{
F(\theta^\star;\xi)=0
}
$$

を満たす解として定義できる。

したがって陽関数

$$
\theta^\star=f(\xi)
$$

を解析的に求める必要はなく、

$$
\boxed{
\theta^\star
\text{ is implicitly defined by }
F(\theta^\star;\xi)=0
}
$$

とすればよい。

さらに、\(F:\mathbb{R}^{p}\times\mathbb{R}^{q}\to\mathbb{R}^{p}\) で
\(F\) の式数と \(\theta\) の自由度が一致し、条件

$$
\det
\left(
\frac{\partial F}
{\partial\theta}
\right)
\neq0
$$

が局所的に成立する場合、陰関数定理により \(\theta^\star\) は \(\xi\) の局所的な関数として存在する。式数と自由度が一致しない場合は、この逆行列をそのまま用いることはできず、制約の独立性を確認した上で KKT 系または擬似逆による感度解析を用いる。

その微分は

$$
\boxed{
\frac{d\theta^\star}{d\xi}
=
-
\left(
\frac{\partial F}
{\partial\theta}
\right)^{-1}
\frac{\partial F}
{\partial\xi}
}
$$

で与えられる。

これは将来的に、

- kernel width
- atom 数
- layer width
- 入力分散
- target gradient scale

などに対して初期値を自動適応させる場合にも利用できる。

---

# 16. まとめ

CST の初期化は、単一の weight variance を決める問題ではなく、

$$
\boxed{
\text{forward statistics}
+
\text{backward statistics}
+
\text{CST geometry}
}
$$

を同時に整える問題として考えられる。

具体的には、

$$
\boxed{
\begin{aligned}
&\text{Mean:}
&
W(\theta)\mu_x
&=
\mu_y^\star
\\[1mm]
&\text{Variance:}
&
\operatorname{diag}
\left(
W(\theta)\Sigma_xW(\theta)^\top
\right)
&=
v_y^\star
\\[1mm]
&\text{Gradient norm:}
&
\mathbb{E}
\|
J^\top g_W
\|^2
&=
(G^\star)^2
\\[1mm]
&\text{Gradient geometry:}
&
J^\top J
&\approx
\gamma I.
\end{aligned}
}
$$

これらを

$$
F(\theta)=0
$$

として陰的に解くか、

$$
\min_\theta
\mathcal{L}_{\mathrm{init}}(\theta)
$$

として多目的最適化する。

CST の場合、分散条件には atom 間の kernel overlap が入り、勾配条件には

$$
\frac{\partial W}{\partial\theta}
$$

および、その条件自体を最適化する際には

$$
\frac{\partial^2W}{\partial\theta^2}
$$

まで現れる。

したがって、CST initialization は

$$
\boxed{
\text{distribution preservation}
+
\text{Jacobian conditioning}
}
$$

を同時に満たす制約付き幾何学的初期化問題として定式化できる。
