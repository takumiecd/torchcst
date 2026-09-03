# CST implicit projected moment optimizer

Status: historical research derivation, unimplemented.

この文書の accepted-frame $\alpha$ transport と quartic stationarity の導出は
現在の設計でも使用する。一方、§3.2、§8以降の force-level $\gamma$ を第二状態として
EMAする案は、operator $D$ と過去の $\Delta W$ を早すぎる段階で結合するため、現在の
最終候補ではない。実験結果、棄却理由、diagonal product-space transportを含む現在の
判断は
[`implicit-projected-adam-decisions.ja.md`](implicit-projected-adam-decisions.ja.md)
を参照すること。

English version: [implicit-projected-moment-transport.md](implicit-projected-moment-transport.md)

この文書では、CST が実際に観測できる部分だけに optimizer state を圧縮し、
dense な represented weight、勾配、first moment、second state を生成・保存せずに
更新量を決める方法を、multi-parameter の行列形式で整理する。

scalar の $v,q,\alpha,\gamma$ はすべて $p=1$ の特殊例である。一般には、

$$
d\in\mathbb R^p,
\qquad
V(d)\in\mathbb R^{N\times p},
\qquad
Q(d)=V(d)^\top V(d)\in\mathbb R^{p\times p},
$$

$$
\alpha(d),\gamma(d)\in\mathbb R^p
$$

として扱う。

## 1. CST map の二次展開

represented weight を vectorize して $W(\theta)\in\mathbb R^N$ とする。現在の
CST parameter に対する候補 step を

$$
d:=\delta\theta\in\mathbb R^p
$$

とし、CST map の差分を二次まで展開する。

$$
\boxed{
\Delta W_t(d)
=J_td+\frac12H_t[d,d]
}
$$

ここで

$$
J_t\in\mathbb R^{N\times p},
\qquad
H_t\in\mathbb R^{N\times p\times p}
$$

であり、$H_t[d,d]\in\mathbb R^N$ は parameter index 二つを $d$ と縮約したもの
である。$H_t$ は parameter index について対称とする。

候補 $d$ における微分は、一つの vector ではなく行列

$$
\boxed{
V_t(d)
:=\frac{\partial\Delta W_t(d)}{\partial d}
=J_t+H_t[d,\cdot]
\in\mathbb R^{N\times p}
}
$$

である。$V_t(d)$ の column space が、その候補 step において CST が観測できる
ambient 空間の部分空間になる。

## 2. 導出上の ambient objective

導出上、Adam 型の局所 objective を

$$
\phi_t(d)
=m_t^\top\Delta W_t(d)
+\frac1{2\eta}
\Delta W_t(d)^\top D_t\Delta W_t(d)
$$

と書く。$m_t\in\mathbb R^N$ と $D_t\in\mathbb R^{N\times N}$ は概念上の
ambient 量であり、runtime に dense tensor として作ることを意味しない。

内側で候補 $d$ を解いている間は $m_t,D_t,J_t,H_t$ を固定する。
$D_t=D_t^\top$ なら、chain rule より

$$
\boxed{
\nabla_d\phi_t(d)
=V_t(d)^\top m_t
+\frac1\eta V_t(d)^\top D_t\Delta W_t(d)
\in\mathbb R^p
}.
$$

したがって optimizer が必要とするのは ambient state 全体ではなく、
$V_t(d)^\top$ を通して見える二つの $p$ 次元量だけである。

## 3. 行列版の可視代表

以下では一時的に time index を省略する。現在の visible Gram matrix を

$$
\boxed{
Q(d):=V(d)^\top V(d)\in\mathbb R^{p\times p}
}
$$

とする。

### 3.1 First-moment side

$m$ の可視代表を $\operatorname{col}(V(d))$ 内で

$$
\widehat m(d)=V(d)\alpha(d)
$$

と書く。ここで $\alpha(d)$ は scalar ではなく

$$
\alpha(d)\in\mathbb R^p
$$

である。元の force と同じ pullback を保つ条件は

$$
V(d)^\top m=V(d)^\top\widehat m(d).
$$

$\widehat m(d)=V(d)\alpha(d)$ を代入すると、

$$
V(d)^\top m
=V(d)^\top V(d)\alpha(d)
=Q(d)\alpha(d).
$$

したがって minimum-norm coordinate は

$$
\boxed{
\alpha(d)=Q(d)^\dagger V(d)^\top m
}.
$$

ambient 空間の可視代表そのものは

$$
\widehat m(d)
=V(d)Q(d)^\dagger V(d)^\top m
$$

であり、これは $m$ の $\operatorname{col}(V(d))$ への直交射影である。

### 3.2 D-side

D-side の ambient force を

$$
u_D(d):=D\Delta W(d)
$$

とし、その可視代表を

$$
\widehat u_D(d)=V(d)\gamma(d),
\qquad
\gamma(d)\in\mathbb R^p
$$

とする。同じ pullback を保つ条件から、

$$
\boxed{
\gamma(d)
=Q(d)^\dagger V(d)^\top D\Delta W(d)
}.
$$

## 4. なぜ $Q(d)$ が現れ、逆行列が消えるのか

元の gradient は

$$
\nabla_d\phi(d)
=V(d)^\top m
+\frac1\eta V(d)^\top D\Delta W(d)
$$

である。可視代表は同じ pullback を保つため、

$$
V(d)^\top m=V(d)^\top\widehat m(d),
$$

$$
V(d)^\top D\Delta W(d)=V(d)^\top\widehat u_D(d).
$$

一つずつ展開すると、

$$
\begin{aligned}
V(d)^\top\widehat m(d)
&=V(d)^\top\bigl(V(d)\alpha(d)\bigr)\\
&=V(d)^\top V(d)\alpha(d)\\
&=Q(d)\alpha(d),
\end{aligned}
$$

$$
\begin{aligned}
V(d)^\top\widehat u_D(d)
&=V(d)^\top\bigl(V(d)\gamma(d)\bigr)\\
&=Q(d)\gamma(d).
\end{aligned}
$$

よって

$$
\boxed{
\nabla_d\phi(d)
=Q(d)
\left(
\alpha(d)+\frac1\eta\gamma(d)
\right)
}.
$$

ここで

$$
A(d):=V(d)^\top m,
\qquad
G(d):=V(d)^\top D\Delta W(d)
$$

とおけば、

$$
\alpha(d)=Q(d)^\dagger A(d),
\qquad
\gamma(d)=Q(d)^\dagger G(d).
$$

$A(d),G(d)$ はそれぞれ $V(d)^\top$ の range に入るため、Moore--Penrose
pseudoinverse の恒等式により

$$
Q(d)Q(d)^\dagger A(d)=A(d),
$$

$$
Q(d)Q(d)^\dagger G(d)=G(d).
$$

したがって求解に使う gradient は

$$
\boxed{
\nabla_d\phi(d)
=A(d)+\frac1\eta G(d)
}.
$$

scalar case での「分母 $q(d)$ が消える」は、一般形では
「Gram pseudoinverse $Q(d)^\dagger$ を求解式に入れる必要がない」に対応する。

$Q(d)$ が正則なら stationarity は

$$
\alpha(d)+\frac1\eta\gamma(d)=0
$$

と同値である。rank deficient の場合、正しい条件は

$$
Q(d)\left(\alpha(d)+\frac1\eta\gamma(d)\right)=0
$$

または直接

$$
A(d)+\frac1\eta G(d)=0
$$

であり、$Q(d)$ を単純に消してはいけない。

## 5. $d$ に依存しない係数への展開

$V(d)=J+H[d,\cdot]$ は $d$ に対して affine である。
任意の固定 ambient vector $z\in\mathbb R^N$ に対して

$$
V(d)^\top z=J^\top z+C_zd,
$$

ここで

$$
\boxed{
(C_z)_{ab}:=H_{:ab}^\top z
},
\qquad
C_z\in\mathbb R^{p\times p}.
$$

したがって first-side numerator $A(d)$ は $d$ の affine vector になる。

Gram matrix は $d$ の二次 matrix polynomial である。

$$
\boxed{
\begin{aligned}
Q(d)
={}&J^\top J
+J^\top H[d,\cdot]
+H[d,\cdot]^\top J\\
&+H[d,\cdot]^\top H[d,\cdot].
\end{aligned}
}
$$

D-side numerator

$$
G(d)=V(d)^\top D\Delta W(d)
$$

は $p$ 次元の三次 vector polynomial である。tensor notation では

$$
\boxed{
G(d)
=G^{(1)}[d]
+G^{(2)}[d,d]
+G^{(3)}[d,d,d]
}
$$

と書ける。実装は巨大な coefficient tensor を必ずしも明示保存せず、candidate
$d$ に対する contraction oracle として評価してよい。

## 6. 時間をまたぐ圧縮 state

前 step の圧縮 state は

$$
\widehat m_{t-1}=V_{t-1}^\star\alpha_{t-1},
$$

$$
\widehat u_{t-1}=V_{t-1}^\star\gamma_{t-1},
$$

$$
V_{t-1}^\star
:=V_{t-1}(d_{t-1})
=J_{t-1}+H_{t-1}[d_{t-1},\cdot].
$$

ここで

$$
\alpha_{t-1},\gamma_{t-1}\in\mathbb R^p.
$$

古い coordinate を現在の $V_t$ の coordinate としてそのまま解釈してはいけない。
必ず $V_{t-1}^\star$ へ一度展開し、現在の $V_t(d)^\top$ で再び押し潰す。

$V_{t-1}^\star$ は ambient dimension を含むため保存しない。加法更新

$$
\theta_t=\theta_{t-1}+d_{t-1}
$$

なら、$d_{t-1}$ または $\theta_{t-1}$ のどちらかから旧点と旧stepを再構成できる。
旧点での JVP、VJP、HVP、CST factor contraction により
$V_t(d)^\top V_{t-1}^\star$ の作用だけを求める。

論理的な永続 state は

$$
\boxed{
(\alpha_{t-1},\gamma_{t-1},d_{t-1})
}
$$

または加法更新のもとで同値な

$$
\boxed{
(\alpha_{t-1},\gamma_{t-1},\theta_{t-1})
}
$$

である。

## 7. First moment の行列版輸送

概念上の first-moment EMA を

$$
\widetilde m_t
=\beta_1V_{t-1}^\star\alpha_{t-1}
+(1-\beta_1)g_t
$$

とする。dense な $\widetilde m_t,g_t$ は作らない。現在の observable numerator
は

$$
\boxed{
A_t(d)
:=V_t(d)^\top\widetilde m_t
}
$$

$$
=\beta_1V_t(d)^\top V_{t-1}^\star\alpha_{t-1}
+(1-\beta_1)V_t(d)^\top g_t.
$$

$V_t(d)$ は affine なので

$$
\boxed{
A_t(d)=A_t^{(0)}+A_t^{(1)}d
}
$$

と書ける。係数は

$$
\boxed{
A_t^{(0)}
=\beta_1J_t^\top V_{t-1}^\star\alpha_{t-1}
+(1-\beta_1)J_t^\top g_t
}
$$

および

$$
\boxed{
A_t^{(1)}
=\beta_1C_{V_{t-1}^\star\alpha_{t-1}}
+(1-\beta_1)C_{g_t}
}.
$$

$A_t^{(0)}\in\mathbb R^p$、$A_t^{(1)}\in\mathbb R^{p\times p}$ である。
これらは毎 step 再生成する現在の係数であり、永続 EMA buffer ではない。

## 8. D-side の行列版輸送

この方式では $\gamma$ を CST 固有の compressed second-side EMA state として
直接定義する。dense Adam の raw elementwise second moment と厳密に同値である
とは主張しない。

現在の $g_t^{\odot2}$ などから定義する新しい D-side action を、導出上

$$
D_t^{\mathrm{new}}
$$

と書く。current evidence は

$$
B_t(d)
:=V_t(d)^\top D_t^{\mathrm{new}}\Delta W_t(d)
\in\mathbb R^p
$$

であり、$d$ の三次 vector polynomial である。

古い $\gamma_{t-1}$ と current evidence を force level で混ぜた numerator を

$$
\boxed{
\begin{aligned}
G_t(d)
:={}&\beta_2V_t(d)^\top
V_{t-1}^\star\gamma_{t-1}\\
&+(1-\beta_2)B_t(d)
\end{aligned}
}
$$

とする。これは

$$
\boxed{
G_t(d)
=G_t^{(0)}
+G_t^{(1)}[d]
+G_t^{(2)}[d,d]
+G_t^{(3)}[d,d,d]
}
$$

という三次 vector polynomial になる。古い state の寄与は affine、current
D-side evidence の寄与は最大三次である。

完全な coefficient tensor を保存するか、candidate ごとに contraction で
$G_t(d)$ を評価するかは実装上の選択である。どちらの場合も ambient
$D_t^{\mathrm{new}}$ や $V_{t-1}^\star\gamma_{t-1}$ は実体化しない。

## 9. 現在の $d$ を解く

求解中に current $\alpha_t(d),\gamma_t(d)$ を作る必要はない。それぞれの
observable numerator を直接使う。

$$
\boxed{
R_t(d)
:=A_t(d)+\frac1\eta G_t(d)=0,
\qquad
R_t(d)\in\mathbb R^p
}.
$$

$A_t(d)$ は affine、$G_t(d)$ は最大三次なので、$R_t(d)$ は $p$ 変数の三次
vector polynomial system である。scalar case の一つの三次方程式とは異なり、
一般には多変数非線形方程式または quartic surrogate の最小化問題になる。

対応する有効 surrogate は、概念上

$$
\begin{aligned}
\Phi_t(d)
={}&\left[
\beta_1V_{t-1}^\star\alpha_{t-1}
+(1-\beta_1)g_t
\right]^\top\Delta W_t(d)\\
&+\frac{\beta_2}{\eta}
\left(V_{t-1}^\star\gamma_{t-1}\right)^\top\Delta W_t(d)\\
&+\frac{1-\beta_2}{2\eta}
\Delta W_t(d)^\top D_t^{\mathrm{new}}\Delta W_t(d)
\end{aligned}
$$

であり、

$$
\nabla_d\Phi_t(d)=R_t(d).
$$

これは最大四次である。candidate の評価中に model parameter と永続 state を
変更せず、trust region 内で選んだ $d_t^\star$ を一度だけ適用する。

## 10. 確定後の再圧縮

current step が確定したら

$$
V_t^\star:=V_t(d_t^\star),
\qquad
Q_t^\star:=(V_t^\star)^\top V_t^\star
$$

を暗黙的に評価し、

$$
\boxed{
\alpha_t
=(Q_t^\star)^\dagger A_t(d_t^\star)
}
$$

$$
\boxed{
\gamma_t
=(Q_t^\star)^\dagger G_t(d_t^\star)
}
$$

として初めて current state を圧縮する。

したがって、前 step の $\alpha_{t-1},\gamma_{t-1}$ は現在の polynomial を作る
入力であり、current $\alpha_t,\gamma_t$ は $d_t^\star$ 確定後に得られる出力である。

$Q_t^\star$ が rank deficient なら、$\alpha_t,\gamma_t$ は null space 方向まで
一意には定まらない。minimum-norm pseudoinverse、damping、または state reset
policy を明示する必要がある。

## 11. 一 step のアルゴリズム

1. 保存済みの $(\alpha_{t-1},\gamma_{t-1})$ と旧点情報から、
   $V_{t-1}^\star$ を含む cross-time operator action を暗黙計算する。
2. current loss から $V_t(d)^\top g_t$ の affine coefficients を、dense $g_t$
   なしで計算する。
3. current $g_t^{\odot2}$ 由来の $B_t(d)$ を、D-side contraction oracle
   または compact coefficient tensors として構成する。
4. $A_t(d)$ と $G_t(d)$ を組み立てる。
5. trust region 内で $R_t(d)=0$ または $\Phi_t(d)$ を解き、$d_t^\star$ を選ぶ。
6. 確定点で $Q_t^\star,A_t(d_t^\star),G_t(d_t^\star)$ を評価する。
7. $(\alpha_t,\gamma_t)$ へ再圧縮する。
8. $d_t^\star$ を一度だけ適用し、新しい compact state と旧点情報を保存する。

## 12. Dense-free implementation boundary

理論式に現れる $V_t(d)\in\mathbb R^{N\times p}$ は巨大でも、行列そのものは
必要ない。必要なのは $V_t(d)$ と $V_t(d)^\top$ の作用だけである。

任意の $x\in\mathbb R^p$ に対する pushforward を

$$
\boxed{
\operatorname{push}_t(d,x)
:=V_t(d)x
=J_tx+H_t[d,x]
}
$$

とする。これは $\Delta W_t$ の $d$ における $x$ 方向 JVP である。

$$
\operatorname{push}_t(d,x)
=\left.
\frac{\partial}{\partial\epsilon}
\Delta W_t(d+\epsilon x)
\right|_{\epsilon=0}.
$$

任意の ambient cotangent $y$ に対する pullback を

$$
\boxed{
\operatorname{pull}_t(d,y)
:=V_t(d)^\top y
}
$$

とする。これは $\Delta W_t$ に対する VJP である。$H_t[d,x]$ は JVP の JVP、
forward-over-reverse、または CST factor の二次 directional contraction として
計算できる。

### 12.1 Gram matrix を作らない

$Q_t(d)=V_t(d)^\top V_t(d)$ 自体を構築する代わりに、matvec を

$$
\boxed{
Q_t(d)x
=\operatorname{pull}_t
\left(d,\operatorname{push}_t(d,x)\right)
}
$$

として計算する。したがって accepted point での

$$
Q_t^\star\alpha_t=A_t(d_t^\star),
\qquad
Q_t^\star\gamma_t=G_t(d_t^\star)
$$

は、$Q_t^\star$ を形成せずに CG、MINRES、LSMR などの反復法で解ける。
実際には rank deficiency に備えて

$$
(Q_t^\star+\lambda I)x=b
$$

という damped system を解くのが自然である。$p$ が十分小さい場合だけ、作用を
basis vector に適用して $p\times p$ の $Q_t^\star$ を明示構築してもよい。

### 12.2 Cross-time Gram も作らない

輸送に必要な

$$
V_t(d)^\top V_{t-1}^\star x
$$

も、概念上は

$$
\boxed{
V_t(d)^\top V_{t-1}^\star x
=\operatorname{pull}_t
\left(
d,
\operatorname{push}_{t-1}(d_{t-1},x)
\right)
}
$$

という old pushforward と current pullback の合成である。この式により
$N\times p$ の current/old Jacobian を保存する必要はない。

### 12.3 Stationarity residual も作用として評価する

求解に必要な二つの numerator は

$$
A_t(d)=V_t(d)^\top\widetilde m_t,
$$

$$
G_t(d)=V_t(d)^\top\widetilde u_t(d)
$$

であるため、どちらも current pullback として評価できる。したがって

$$
R_t(d)=A_t(d)+\frac1\eta G_t(d)
$$

を返す matrix-free residual oracle を作れる。solver が必要とする
$\partial R_t(d)x/\partial d$ も residual に対する JVP で求められるため、完全な
三次 coefficient tensor や Jacobian を作らずに nonlinear trust-region、
Newton--Krylov、fixed-point 系の inner solverを実装できる。ここでの
Newton--Krylov は小さな inner polynomial system の数値解法であり、dense loss
Hessian を使う Newton optimizer だという意味ではない。

### 12.4 Matrix-free と ambient-free の違い

通常の autograd JVP/VJP だけでも $N\times p$ の $V$ は作らずに済む。しかし、
framework の実装によっては $Vx\in\mathbb R^N$ や $D\Delta W\in\mathbb R^N$ という
中間 vector を materialize する。この場合は **matrix-free ではあるが、まだ
ambient-free ではない**。

CST の本来の memory advantage を守るには、次の合成作用を factor level で fused
実装する必要がある。

$$
\boxed{
x\longmapsto V_t(d)^\top V_t(d)x
}
$$

$$
\boxed{
x\longmapsto V_t(d)^\top V_{t-1}^\star x
}
$$

$$
\boxed{
d\longmapsto V_t(d)^\top
D_t^{\mathrm{new}}\Delta W_t(d)
}.
$$

つまり、JVP/VJP は正しい数学的 interface であり、generic autograd は最初の
correctness implementation に使える。最終的な dense-free implementation では、
同じ interface を CST 固有の structured contraction kernel で実現する。

実装が公開すべき最小 operator API は、例えば次である。

- `current_gram_mv(d, x)`：$V_t(d)^\top V_t(d)x$
- `cross_gram_mv(d, x, old_state)`：$V_t(d)^\top V_{t-1}^\star x$
- `first_numerator(d)`：$A_t(d)$
- `second_numerator(d)`：$G_t(d)$
- `residual(d)`：$A_t(d)+G_t(d)/\eta$

これらの operator を JVP、VJP、HVP、CST factor contraction で実装し、巨大な
行列や ambient vector を optimizer 側へ公開しない。

まとめると、実装では次のような作用だけを計算する。

$$
V_t(d)^\top g_t,
\qquad
V_t(d)^\top V_{t-1}^\star x,
\qquad
V_t(d)^\top V_t(d)x,
$$

$$
V_t(d)^\top D_t^{\mathrm{new}}\Delta W_t(d).
$$

実装の contract は次の通りである。

- dense $W,g_W,m_W,D_W,V$ を生成・保存しない。
- dense 記号は導出と小規模 correctness oracle にだけ使う。
- old visible operator は保存せず、旧点から必要な作用だけを再計算する。
- candidate solve 中に parameter と永続 state を変更しない。
- $\alpha_t,\gamma_t$ は採用した $d_t^\star$ でのみ確定する。

## 13. Scalar case は $p=1$ の特殊例

$p=1$ のときだけ

$$
V(d)=v(d)\in\mathbb R^N,
\qquad
Q(d)=v(d)^\top v(d)=q(d)\in\mathbb R
$$

となり、$\alpha,\gamma$ も scalar になる。

$$
v(d)^\top\bigl(\alpha(d)v(d)\bigr)
=q(d)\alpha(d).
$$

そして

$$
\alpha(d)=\frac{v(d)^\top m}{q(d)},
\qquad
\gamma(d)=\frac{v(d)^\top D\Delta W(d)}{q(d)}.
$$

この scalar 表記は説明と unit test には有用だが、$p>1$ の optimizer state を
scalar に潰してよいことを意味しない。$V(d)$ 全体を一つの vector として flatten
して scalar 投影すると、CST が観測できる $p$ 個の gradient component の一部を
失う。

## 14. 主張しないことと検証項目

この提案は次を主張しない。

- true dense-weight loss Hessian $J^\top H_WLJ$ を計算できること。
- Newton 法であること。
- dense Adam の elementwise second-moment recurrence と厳密に同値であること。
- dense state を一度作ってから圧縮する実装が必要であること。

小規模 dense oracle では最低限、次を検証する。

1. $V(d)^\top m=A(d)$ が成立する。
2. $V(d)^\top D\Delta W(d)$ が compact cubic evaluation と一致する。
3. 明示的な expand-add-recompress と compact transport recurrence が一致する。
4. $Q(d)\alpha(d)=A(d)$、$Q(d)\gamma(d)=G(d)$ が observable range 上で
   成立する。
5. $Q(d)(\alpha(d)+\gamma(d)/\eta)=A(d)+G(d)/\eta$ が成立する。
6. implicit operator implementation が explicit dense oracle と数値精度内で一致する。
7. 保存する $\alpha_t,\gamma_t$ が実際に採用した $d_t^\star$ での値である。

これらを確認した後に、solver、trust policy、damping、bias correction、atom block、
block diagonal、cross-atom truncation を独立した実験軸として追加する。
