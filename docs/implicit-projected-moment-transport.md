# CST implicit projected moment optimizer

Status: historical research derivation, unimplemented.

The accepted-frame $\alpha$ transport and quartic stationarity derivation are
still used by the selected design. The force-level $\gamma$ EMA in Sections
3.2 and 8 onward is not the selected second-state model: it binds the operator
$D$ to a historical $\Delta W$ before transport. See the current Japanese
decision record,
[`implicit-projected-adam-decisions.ja.md`](implicit-projected-adam-decisions.ja.md),
for the experimental evidence, diagonal product-space transport, and remaining
open problems.

Japanese version: [implicit-projected-moment-transport.ja.md](implicit-projected-moment-transport.ja.md)

This note derives a CST-native optimizer in multi-parameter matrix form. It
compresses optimizer state to the subspace visible through the CST map without
materializing or retaining dense represented weights, gradients, first
moments, or second states.

The scalar symbols $v,q,\alpha,\gamma$ are only the $p=1$ specialization. In
the general theory,

$$
d\in\mathbb R^p,
\quad
V(d)\in\mathbb R^{N\times p},
\quad
Q(d)=V(d)^\top V(d)\in\mathbb R^{p\times p},
$$

and

$$
\alpha(d),\gamma(d)\in\mathbb R^p.
$$

The second-order structure belongs to the CST map. This is not a Newton method
and does not assume access to the dense-weight loss Hessian.

## 1. Second-order CST map

Vectorize the represented weight as $W(\theta)\in\mathbb R^N$ and let

$$
d:=\delta\theta\in\mathbb R^p.
$$

Approximate its displacement at the current parameter by

$$
\boxed{
\Delta W_t(d)=J_td+\frac12H_t[d,d]
},
$$

where

$$
J_t\in\mathbb R^{N\times p},
\qquad
H_t\in\mathbb R^{N\times p\times p},
$$

and $H_t$ is symmetric in its two parameter indices. The derivative at
candidate $d$ is the matrix

$$
\boxed{
V_t(d)
:=\frac{\partial\Delta W_t(d)}{\partial d}
=J_t+H_t[d,\cdot]
\in\mathbb R^{N\times p}
}.
$$

Its column space is the ambient subspace visible to CST at candidate $d$.

## 2. Conceptual ambient objective

For the derivation only, write

$$
\phi_t(d)
=m_t^\top\Delta W_t(d)
+\frac1{2\eta}\Delta W_t(d)^\top D_t\Delta W_t(d).
$$

The ambient $m_t\in\mathbb R^N$ and $D_t\in\mathbb R^{N\times N}$ are
conceptual, not runtime dense tensors. Hold $m_t,D_t,J_t,H_t$ fixed during the
inner solve. For symmetric $D_t$,

$$
\boxed{
\nabla_d\phi_t(d)
=V_t(d)^\top m_t
+\frac1\eta V_t(d)^\top D_t\Delta W_t(d)
\in\mathbb R^p
}.
$$

Only these two pullbacks are required, not the full ambient states.

## 3. Matrix-valued visible projection

Suppress the time index and define the visible Gram matrix

$$
\boxed{
Q(d):=V(d)^\top V(d)\in\mathbb R^{p\times p}
}.
$$

### 3.1 First-moment side

Write the visible representative as

$$
\widehat m(d)=V(d)\alpha(d),
\qquad
\alpha(d)\in\mathbb R^p.
$$

Preserving the original pullback requires

$$
V(d)^\top m
=V(d)^\top\widehat m(d)
=Q(d)\alpha(d).
$$

The minimum-norm coordinate is therefore

$$
\boxed{
\alpha(d)=Q(d)^\dagger V(d)^\top m
}.
$$

The ambient representative

$$
\widehat m(d)=V(d)Q(d)^\dagger V(d)^\top m
$$

is the orthogonal projection of $m$ onto $\operatorname{col}(V(d))$.

### 3.2 D-side

For the ambient force $u_D(d):=D\Delta W(d)$, write

$$
\widehat u_D(d)=V(d)\gamma(d),
\qquad
\gamma(d)\in\mathbb R^p.
$$

The same pullback-preservation condition gives

$$
\boxed{
\gamma(d)=Q(d)^\dagger V(d)^\top D\Delta W(d)
}.
$$

## 4. Why the Gram appears and its pseudoinverse disappears

Starting from the original gradient and replacing each force by its visible
representative,

$$
\begin{aligned}
V(d)^\top\widehat m(d)
&=V(d)^\top\bigl(V(d)\alpha(d)\bigr)\\
&=Q(d)\alpha(d),
\end{aligned}
$$

and

$$
V(d)^\top\widehat u_D(d)=Q(d)\gamma(d).
$$

Therefore

$$
\boxed{
\nabla_d\phi(d)
=Q(d)\left(\alpha(d)+\frac1\eta\gamma(d)\right)
}.
$$

Define the observable numerators

$$
A(d):=V(d)^\top m,
\qquad
G(d):=V(d)^\top D\Delta W(d).
$$

Then

$$
\alpha(d)=Q(d)^\dagger A(d),
\qquad
\gamma(d)=Q(d)^\dagger G(d).
$$

Both numerators lie in the range of $Q(d)$, hence

$$
Q(d)Q(d)^\dagger A(d)=A(d),
\qquad
Q(d)Q(d)^\dagger G(d)=G(d).
$$

The solve therefore uses

$$
\boxed{
\nabla_d\phi(d)=A(d)+\frac1\eta G(d)
}.
$$

The scalar denominator cancellation is, in the general form, the fact that no
$Q(d)^\dagger$ is needed in the stationarity residual. If $Q(d)$ is full rank,
stationarity is equivalent to

$$
\alpha(d)+\frac1\eta\gamma(d)=0.
$$

If it is rank deficient, do not cancel $Q(d)$; use

$$
A(d)+\frac1\eta G(d)=0
$$

or the equivalent Gram-weighted equation.

## 5. Candidate-independent coefficient expansion

Because $V(d)=J+H[d,\cdot]$ is affine in $d$, for any fixed ambient vector
$z\in\mathbb R^N$,

$$
V(d)^\top z=J^\top z+C_zd,
$$

where

$$
\boxed{
(C_z)_{ab}:=H_{:ab}^\top z
},
\qquad
C_z\in\mathbb R^{p\times p}.
$$

The first numerator is therefore affine. The Gram is a quadratic matrix
polynomial:

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

The D-side numerator is a cubic vector polynomial:

$$
\boxed{
G(d)
=G^{(1)}[d]+G^{(2)}[d,d]+G^{(3)}[d,d,d]
}.
$$

An implementation may evaluate these terms through contraction oracles rather
than storing the complete coefficient tensors.

## 6. Persistent state and old-frame reconstruction

The accepted compressed state from the previous iteration represents

$$
\widehat m_{t-1}=V_{t-1}^\star\alpha_{t-1},
\qquad
\widehat u_{t-1}=V_{t-1}^\star\gamma_{t-1},
$$

where

$$
V_{t-1}^\star
:=V_{t-1}(d_{t-1})
=J_{t-1}+H_{t-1}[d_{t-1},\cdot].
$$

The old coordinates must first be expanded through their old visible operator
before being measured in the current frame. They cannot be interpreted as
coefficients of $V_t$ directly.

Do not store the ambient $V_{t-1}^\star$. Under the additive update

$$
\theta_t=\theta_{t-1}+d_{t-1},
$$

store either $d_{t-1}$ or $\theta_{t-1}$ and reconstruct the other. Recompute
only the old-point operator actions required for cross-time contractions using
JVPs, VJPs, HVPs, or CST factor contractions.

The logical persistent state is

$$
\boxed{(\alpha_{t-1},\gamma_{t-1},d_{t-1})}
$$

or, equivalently under additive updates,

$$
\boxed{(\alpha_{t-1},\gamma_{t-1},\theta_{t-1})}.
$$

## 7. Matrix first-moment transport

The conceptual first-moment EMA is

$$
\widetilde m_t
=\beta_1V_{t-1}^\star\alpha_{t-1}
+(1-\beta_1)g_t.
$$

Neither ambient vector is materialized. Its current observable numerator is

$$
\boxed{
\begin{aligned}
A_t(d)
:={}&V_t(d)^\top\widetilde m_t\\
={}&\beta_1V_t(d)^\top V_{t-1}^\star\alpha_{t-1}
+(1-\beta_1)V_t(d)^\top g_t.
\end{aligned}
}
$$

It is affine:

$$
\boxed{A_t(d)=A_t^{(0)}+A_t^{(1)}d},
$$

with

$$
\boxed{
A_t^{(0)}
=\beta_1J_t^\top V_{t-1}^\star\alpha_{t-1}
+(1-\beta_1)J_t^\top g_t
},
$$

$$
\boxed{
A_t^{(1)}
=\beta_1C_{V_{t-1}^\star\alpha_{t-1}}
+(1-\beta_1)C_{g_t}
}.
$$

Here $A_t^{(0)}\in\mathbb R^p$ and
$A_t^{(1)}\in\mathbb R^{p\times p}$. They are rebuilt each iteration, not
persistent EMA buffers.

## 8. Matrix D-side transport

This method defines $\gamma$ as a CST-native compressed second-side EMA state;
it does not claim exact equivalence to dense Adam's raw elementwise second
moment.

Let $D_t^{\mathrm{new}}$ conceptually denote the current D-side action derived
from $g_t^{\odot2}$ or another chosen signal. Define

$$
B_t(d)
:=V_t(d)^\top D_t^{\mathrm{new}}\Delta W_t(d)
\in\mathbb R^p.
$$

It is a cubic vector polynomial. Mix old and current evidence at the force
level:

$$
\boxed{
\begin{aligned}
G_t(d)
:={}&\beta_2V_t(d)^\top V_{t-1}^\star\gamma_{t-1}\\
&+(1-\beta_2)B_t(d).
\end{aligned}
}
$$

Consequently,

$$
\boxed{
G_t(d)
=G_t^{(0)}+G_t^{(1)}[d]
+G_t^{(2)}[d,d]+G_t^{(3)}[d,d,d]
}.
$$

The old-state term is affine and the current D-side term is at most cubic.
Complete coefficient tensors may be stored or replaced by candidate-wise
contraction oracles. Neither option materializes the ambient forces.

## 9. Solve the current step

Current $\alpha_t(d)$ and $\gamma_t(d)$ are unnecessary during the solve. Use
their observable numerators directly:

$$
\boxed{
R_t(d)
:=A_t(d)+\frac1\eta G_t(d)=0,
\qquad
R_t(d)\in\mathbb R^p
}.
$$

This is a cubic vector-polynomial system in $p$ variables, or equivalently the
stationarity condition of a quartic surrogate. Conceptually that surrogate is

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
\Delta W_t(d)^\top D_t^{\mathrm{new}}\Delta W_t(d),
\end{aligned}
$$

and $\nabla_d\Phi_t(d)=R_t(d)$. Do not mutate model parameters or persistent
state while evaluating candidates. Apply only the accepted $d_t^\star$.

## 10. Recompress only after acceptance

After fixing the current step, define

$$
V_t^\star:=V_t(d_t^\star),
\qquad
Q_t^\star:=(V_t^\star)^\top V_t^\star.
$$

Only now compute

$$
\boxed{
\alpha_t=(Q_t^\star)^\dagger A_t(d_t^\star)
},
$$

$$
\boxed{
\gamma_t=(Q_t^\star)^\dagger G_t(d_t^\star)
}.
$$

The previous $\alpha_{t-1},\gamma_{t-1}$ are inputs to the current coefficient
construction; the current $\alpha_t,\gamma_t$ are outputs created only after
$d_t^\star$ is known.

If $Q_t^\star$ is rank deficient, the coordinates are not unique in its null
space. Specify a minimum-norm pseudoinverse, damping, or state-reset policy.

## 11. One-step algorithm

1. Use saved $(\alpha_{t-1},\gamma_{t-1})$ and old-point metadata to evaluate
   cross-time actions involving $V_{t-1}^\star$ implicitly.
2. Obtain the affine coefficients of $V_t(d)^\top g_t$ without materializing
   dense $g_t$.
3. Build the current D-side polynomial $B_t(d)$ from squared-gradient evidence
   through compact tensors or a contraction oracle.
4. Assemble $A_t(d)$ and $G_t(d)$.
5. Solve $R_t(d)=0$ or minimize $\Phi_t(d)$ inside a trust region.
6. Evaluate $Q_t^\star,A_t(d_t^\star),G_t(d_t^\star)$ at the accepted step.
7. Recompress into $(\alpha_t,\gamma_t)$.
8. Apply $d_t^\star$ once and persist the new compact state and old-point
   metadata.

## 12. Dense-free implementation boundary

The potentially huge matrix $V_t(d)\in\mathbb R^{N\times p}$ never needs to be
formed. Only its forward and transpose actions are required. Define

$$
\boxed{
\operatorname{push}_t(d,x)
:=V_t(d)x
=J_tx+H_t[d,x]
}
$$

for $x\in\mathbb R^p$. This is a JVP of $\Delta W_t$ at $d$ in direction $x$:

$$
\operatorname{push}_t(d,x)
=\left.
\frac{\partial}{\partial\epsilon}
\Delta W_t(d+\epsilon x)
\right|_{\epsilon=0}.
$$

For an ambient cotangent $y$, define

$$
\boxed{
\operatorname{pull}_t(d,y):=V_t(d)^\top y
}.
$$

This is the corresponding VJP. The term $H_t[d,x]$ can be evaluated by a JVP
of a JVP, forward-over-reverse AD, or a CST-specific second directional
contraction.

### 12.1 Do not form the Gram matrix

Instead of constructing $Q_t(d)=V_t(d)^\top V_t(d)$, evaluate its matvec as

$$
\boxed{
Q_t(d)x
=\operatorname{pull}_t
\left(d,\operatorname{push}_t(d,x)\right)
}.
$$

The accepted-point systems

$$
Q_t^\star\alpha_t=A_t(d_t^\star),
\qquad
Q_t^\star\gamma_t=G_t(d_t^\star)
$$

can therefore be solved by CG, MINRES, or LSMR without forming $Q_t^\star$.
In rank-deficient or poorly conditioned cases, solve the damped system

$$
(Q_t^\star+\lambda I)x=b.
$$

If $p$ is small, explicitly constructing the $p\times p$ Gram by applying the
operator to basis vectors remains a valid implementation choice.

### 12.2 Do not form the cross-time Gram

The transport action is likewise the composition

$$
\boxed{
V_t(d)^\top V_{t-1}^\star x
=\operatorname{pull}_t
\left(
d,
\operatorname{push}_{t-1}(d_{t-1},x)
\right)
}.
$$

Neither the current nor old $N\times p$ Jacobian is stored.

### 12.3 Evaluate stationarity as an operator

Both numerators are pullbacks,

$$
A_t(d)=V_t(d)^\top\widetilde m_t,
\qquad
G_t(d)=V_t(d)^\top\widetilde u_t(d),
$$

so a matrix-free oracle can return

$$
R_t(d)=A_t(d)+\frac1\eta G_t(d).
$$

If an inner solver needs $\partial R_t(d)x/\partial d$, obtain it by a JVP of
the residual. This permits nonlinear trust-region, Newton--Krylov, or
fixed-point solvers without forming the full cubic coefficient tensor or
residual Jacobian. Newton--Krylov here is only a numerical method for the small
inner polynomial system; it does not turn the optimizer into a dense-loss
Newton method.

### 12.4 Matrix-free is not automatically ambient-free

Generic autograd JVPs and VJPs avoid the $N\times p$ matrix, but a framework may
still materialize intermediate ambient vectors such as
$V_t(d)x\in\mathbb R^N$ or $D\Delta W\in\mathbb R^N$. Such code is
**matrix-free but not yet ambient-free**.

To preserve CST's memory advantage, fuse the composed actions at the factor
level:

$$
x\longmapsto V_t(d)^\top V_t(d)x,
$$

$$
x\longmapsto V_t(d)^\top V_{t-1}^\star x,
$$

$$
d\longmapsto V_t(d)^\top
D_t^{\mathrm{new}}\Delta W_t(d).
$$

JVP/VJP is therefore the correct mathematical interface, and generic autograd
is suitable for the first correctness implementation. The final dense-free
implementation should realize the same interface with CST-specific structured
contraction kernels.

A minimal operator API could expose:

- `current_gram_mv(d, x)` for $V_t(d)^\top V_t(d)x$;
- `cross_gram_mv(d, x, old_state)` for
  $V_t(d)^\top V_{t-1}^\star x$;
- `first_numerator(d)` for $A_t(d)$;
- `second_numerator(d)` for $G_t(d)$;
- `residual(d)` for $A_t(d)+G_t(d)/\eta$.

These operators should be implemented with JVPs, VJPs, HVPs, and CST factor
contractions without exposing huge matrices or ambient vectors to the
optimizer.

In summary, the implementation needs actions such as

$$
V_t(d)^\top g_t,
\qquad
V_t(d)^\top V_{t-1}^\star x,
\qquad
V_t(d)^\top V_t(d)x,
$$

and

$$
V_t(d)^\top D_t^{\mathrm{new}}\Delta W_t(d).
$$

The implementation contract is:

- never materialize or retain dense $W,g_W,m_W,D_W,V$;
- use dense notation only for derivation and small correctness oracles;
- reconstruct old visible-operator actions from compact old-point metadata;
- do not mutate parameters or persistent state during the candidate solve;
- finalize $\alpha_t,\gamma_t$ only at the accepted $d_t^\star$.

## 13. Scalar specialization

Only when $p=1$ do

$$
V(d)=v(d)\in\mathbb R^N,
\qquad
Q(d)=v(d)^\top v(d)=q(d)\in\mathbb R
$$

and scalar $\alpha,\gamma$ apply. Then

$$
v(d)^\top\bigl(\alpha(d)v(d)\bigr)=q(d)\alpha(d),
$$

$$
\alpha(d)=\frac{v(d)^\top m}{q(d)},
\qquad
\gamma(d)=\frac{v(d)^\top D\Delta W(d)}{q(d)}.
$$

This form is useful for exposition and unit tests. Flattening the full
$V(d)$ into one vector and using a scalar projection for $p>1$ would discard
some of the $p$ observable gradient components.

## 14. Non-claims and validation

This proposal does not claim to compute $J^\top H_WLJ$, implement Newton's
method, reproduce dense Adam exactly, or require dense state before
compression.

A small dense oracle should verify:

1. $V(d)^\top m=A(d)$;
2. compact cubic evaluation equals $V(d)^\top D\Delta W(d)$;
3. explicit expand-add-recompress equals compact transport;
4. $Q(d)\alpha(d)=A(d)$ and $Q(d)\gamma(d)=G(d)$ on the observable range;
5. $Q(d)(\alpha(d)+\gamma(d)/\eta)=A(d)+G(d)/\eta$;
6. implicit operators match explicit dense directions numerically;
7. stored $\alpha_t,\gamma_t$ are evaluated at the accepted $d_t^\star$.

Solver choice, trust policy, damping, bias correction, atom blocks,
block-diagonal approximations, and cross-atom truncation should be evaluated
as separate experimental axes only after these identities pass.
