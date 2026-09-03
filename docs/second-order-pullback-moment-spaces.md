# Second-order pullback Adam: five moment-space models

This document is a research derivation, not an implemented optimizer contract.
It separates five candidates for extending Pullback Adam through the
second-order CST map:

1. parameter moments;
2. tangent moments with a frozen frame;
3. tangent moments following the pullback-frame flow;
4. tangent moments using a pullback-metric geodesic retraction.
5. tangent moments using a CST-curvature least-squares retraction.

All displayed mathematics uses `$...$` or `$$...$$` delimiters because the
target Markdown renderer does not reliably render alternative display-math
delimiters.

## 1. Common second-order construction

Let $W(\theta)$ be the represented weight at the current parameter value. Write

$$
J=\frac{\partial W}{\partial\theta},
\qquad
H=\frac{\partial^2W}{\partial\theta^2}.
$$

Choose a local moment coordinate $z$ and a second-order map from that coordinate
to a parameter displacement:

$$
d_\chi(z)=A_\chi z+\frac12B_\chi[z,z].
$$

The label $\chi$ names one of the five variants. Composing this retraction with
the second-order CST map gives

$$
\begin{aligned}
\delta W_\chi(z)
&=Jd_\chi(z)+\frac12H[d_\chi(z),d_\chi(z)]+O(\lVert z\rVert^3)\\
&=Q_\chi z+\frac12K_\chi[z,z]+O(\lVert z\rVert^3),
\end{aligned}
$$

where

$$
Q_\chi=JA_\chi
$$

and

$$
K_\chi[z,z]
=H[A_\chi z,A_\chi z]+JB_\chi[z,z].
$$

The two pieces of $K_\chi$ have different meanings. The first is the curvature
of the CST map $W(\theta)$. The second is the curvature of the selected local
map from moment coordinates back to parameters.

Let $\xi_{\chi,t}$ be the stochastic signal whose EMA lives in the chosen local
coordinate. Define five raw-moment EMA objects:

$$
M_{1,\chi}=\operatorname{EMA}_{\beta_1}(\xi_\chi),
\qquad
M_{2,\chi}=\operatorname{EMA}_{\beta_1}(\xi_\chi^{\otimes2}),
$$

and

$$
V_{2,\chi}=\operatorname{EMA}_{\beta_2}(\xi_\chi^{\otimes2}),
\qquad
V_{3,\chi}=\operatorname{EMA}_{\beta_2}(\xi_\chi^{\otimes3}),
\qquad
V_{4,\chi}=\operatorname{EMA}_{\beta_2}(\xi_\chi^{\otimes4}).
$$

Distinct clocks $\beta_1\ne\beta_2$ require five objects because the second raw
moment appears once on each clock. Tying the clocks permits those two objects
to be shared.

The represented-weight first moment is

$$
m_\chi^W
=Q_\chi M_{1,\chi}+\frac12K_\chi:M_{2,\chi},
$$

where $(K_\chi:M_{2,\chi})_a=(K_\chi)_{aij}(M_{2,\chi})_{ij}$. The elementwise
second raw moment is

$$
(v_\chi^W)_a
=(Q_\chi)_{ai}(Q_\chi)_{aj}(V_{2,\chi})_{ij}
+(Q_\chi)_{ai}(K_\chi)_{ajk}(V_{3,\chi})_{ijk}
+\frac14(K_\chi)_{aij}(K_\chi)_{akl}(V_{4,\chi})_{ijkl}.
$$

Set

$$
D_\chi=\operatorname{Diag}(\sqrt{v_\chi^W}).
$$

The Adam variational objective in the chosen local coordinate is

$$
\Phi_\chi(z)
=(m_\chi^W)^\top\delta W_\chi(z)
+\frac1{2\eta}\delta W_\chi(z)^\top D_\chi\delta W_\chi(z).
$$

Using the second-order map gives the common fourth-degree form

$$
\begin{aligned}
\Phi_\chi(z)
={}&(m_\chi^W)^\top Q_\chi z\\
&+\frac12(m_\chi^W)^\top K_\chi[z,z]\\
&+\frac1{2\eta}(Q_\chi z)^\top D_\chi(Q_\chi z)\\
&+\frac1{2\eta}(Q_\chi z)^\top D_\chi K_\chi[z,z]\\
&+\frac1{8\eta}K_\chi[z,z]^\top D_\chi K_\chi[z,z].
\end{aligned}
$$

The five candidates differ only in the EMA signal $\xi_\chi$, the first-order
map $A_\chi$, and the second-order retraction $B_\chi$. Those choices change
$Q_\chi$, $K_\chi$, the reconstructed $m_\chi^W,v_\chi^W$, and therefore every
coefficient of the final fourth-degree objective.

## 2. Candidate P: parameter-moment model

The EMA signal is the parameter-space gradient:

$$
\xi_{P,t}=g_t^\theta.
$$

The local coordinate is the parameter displacement itself:

$$
A_P=I,
\qquad
B_P=0,
\qquad
d_P(z)=z.
$$

Consequently,

$$
Q_P=J,
\qquad
K_P[z,z]=H[z,z],
$$

and

$$
m_P^W=JM_{1,P}+\frac12H:M_{2,P}.
$$

The second raw moment is

$$
(v_P^W)_a
=J_{ai}J_{aj}(V_{2,P})_{ij}
+J_{ai}H_{ajk}(V_{3,P})_{ijk}
+\frac14H_{aij}H_{akl}(V_{4,P})_{ijkl}.
$$

With $D_P=\operatorname{Diag}(\sqrt{v_P^W})$, the solved objective is

$$
\begin{aligned}
\Phi_P(z)
={}&(m_P^W)^\top Jz
+\frac12(m_P^W)^\top H[z,z]\\
&+\frac1{2\eta}(Jz)^\top D_P(Jz)\\
&+\frac1{2\eta}(Jz)^\top D_PH[z,z]\\
&+\frac1{8\eta}H[z,z]^\top D_PH[z,z].
\end{aligned}
$$

The emitted parameter displacement is $d=z$.

## 3. Common tangent frame

Let $P=P(G)$ be the current pullback map. For the inverse full metric this is
approximately $G^{-1/2}$; diagonal, block, bounded, and damped implementations
use their corresponding current map. The tangent-coordinate gradient is

$$
r_t=P^\top g_t^\theta.
$$

For the symmetric pullback maps currently used by Pullback Adam,
$r_t=Pg_t^\theta$. All four tangent candidates accumulate their five EMA
objects from

$$
\xi_{\chi,t}=r_t.
$$

They share

$$
A_\chi=P,
\qquad
Q_\chi=JP.
$$

They differ in the second-order retraction $B_\chi$ and hence in
$K_\chi$. Moments from old steps remain expressed in old moving frames unless
they are explicitly transported. To isolate the five models above, the first
experiment should use the current Pullback Adam convention: no transport and
an EMA reset after structural mutation. Moment transport is a separate
experimental axis.

## 4. Candidate TF: frozen-frame tangent model

The parameter displacement freezes the current pullback map:

$$
B_{TF}=0,
\qquad
d_{TF}(z)=Pz.
$$

Therefore,

$$
Q_{TF}=JP,
$$

and

$$
K_{TF}[z,z]=H[Pz,Pz].
$$

The represented moments are

$$
m_{TF}^W
=Q_{TF}M_{1,TF}+\frac12K_{TF}:M_{2,TF},
$$

and

$$
\begin{aligned}
(v_{TF}^W)_a
={}&(Q_{TF})_{ai}(Q_{TF})_{aj}(V_{2,TF})_{ij}\\
&+(Q_{TF})_{ai}(K_{TF})_{ajk}(V_{3,TF})_{ijk}\\
&+\frac14(K_{TF})_{aij}(K_{TF})_{akl}(V_{4,TF})_{ijkl}.
\end{aligned}
$$

With $D_{TF}=\operatorname{Diag}(\sqrt{v_{TF}^W})$, the solved objective is

$$
\begin{aligned}
\Phi_{TF}(z)
={}&(m_{TF}^W)^\top Q_{TF}z
+\frac12(m_{TF}^W)^\top K_{TF}[z,z]\\
&+\frac1{2\eta}(Q_{TF}z)^\top D_{TF}(Q_{TF}z)\\
&+\frac1{2\eta}(Q_{TF}z)^\top D_{TF}K_{TF}[z,z]\\
&+\frac1{8\eta}K_{TF}[z,z]^\top D_{TF}K_{TF}[z,z].
\end{aligned}
$$

The emitted parameter displacement is $d=Pz$.

## 5. Candidate TM: moving-frame-flow tangent model

Hold a tangent coefficient $z$ fixed while following the vector field

$$
\frac{d\theta(\tau)}{d\tau}=P(\theta(\tau))z.
$$

At $\tau=0$ the second derivative is

$$
\theta''(0)=DP[Pz]z.
$$

Thus

$$
B_{TM}[z,z]=DP[Pz]z
$$

and

$$
d_{TM}(z)=Pz+\frac12DP[Pz]z.
$$

The first- and second-order represented maps are

$$
Q_{TM}=JP
$$

and

$$
K_{TM}[z,z]
=H[Pz,Pz]+JDP[Pz]z.
$$

The represented moments are

$$
m_{TM}^W
=Q_{TM}M_{1,TM}+\frac12K_{TM}:M_{2,TM},
$$

and

$$
\begin{aligned}
(v_{TM}^W)_a
={}&(Q_{TM})_{ai}(Q_{TM})_{aj}(V_{2,TM})_{ij}\\
&+(Q_{TM})_{ai}(K_{TM})_{ajk}(V_{3,TM})_{ijk}\\
&+\frac14(K_{TM})_{aij}(K_{TM})_{akl}(V_{4,TM})_{ijkl}.
\end{aligned}
$$

With $D_{TM}=\operatorname{Diag}(\sqrt{v_{TM}^W})$, the solved objective is

$$
\begin{aligned}
\Phi_{TM}(z)
={}&(m_{TM}^W)^\top Q_{TM}z
+\frac12(m_{TM}^W)^\top K_{TM}[z,z]\\
&+\frac1{2\eta}(Q_{TM}z)^\top D_{TM}(Q_{TM}z)\\
&+\frac1{2\eta}(Q_{TM}z)^\top D_{TM}K_{TM}[z,z]\\
&+\frac1{8\eta}K_{TM}[z,z]^\top D_{TM}K_{TM}[z,z].
\end{aligned}
$$

The emitted parameter displacement includes the moving-frame correction:

$$
d=Pz+\frac12DP[Pz]z.
$$

For a bilinear implementation, use the symmetric extension

$$
B_{TM}[u,z]
=\frac12\left(DP[Pu]z+DP[Pz]u\right).
$$

## 6. Candidate TG: pullback-geodesic tangent model

Let $\Gamma$ denote the Levi-Civita connection of the selected pullback metric
$G$. A second-order geodesic retraction has

$$
B_{TG}[z,z]=-\Gamma[Pz,Pz]
$$

and

$$
d_{TG}(z)=Pz-\frac12\Gamma[Pz,Pz].
$$

The first- and second-order represented maps are

$$
Q_{TG}=JP
$$

and

$$
K_{TG}[z,z]
=H[Pz,Pz]-J\Gamma[Pz,Pz].
$$

The represented moments are

$$
m_{TG}^W
=Q_{TG}M_{1,TG}+\frac12K_{TG}:M_{2,TG},
$$

and

$$
\begin{aligned}
(v_{TG}^W)_a
={}&(Q_{TG})_{ai}(Q_{TG})_{aj}(V_{2,TG})_{ij}\\
&+(Q_{TG})_{ai}(K_{TG})_{ajk}(V_{3,TG})_{ijk}\\
&+\frac14(K_{TG})_{aij}(K_{TG})_{akl}(V_{4,TG})_{ijkl}.
\end{aligned}
$$

With $D_{TG}=\operatorname{Diag}(\sqrt{v_{TG}^W})$, the solved objective is

$$
\begin{aligned}
\Phi_{TG}(z)
={}&(m_{TG}^W)^\top Q_{TG}z
+\frac12(m_{TG}^W)^\top K_{TG}[z,z]\\
&+\frac1{2\eta}(Q_{TG}z)^\top D_{TG}(Q_{TG}z)\\
&+\frac1{2\eta}(Q_{TG}z)^\top D_{TG}K_{TG}[z,z]\\
&+\frac1{8\eta}K_{TG}[z,z]^\top D_{TG}K_{TG}[z,z].
\end{aligned}
$$

The emitted parameter displacement is

$$
d=Pz-\frac12\Gamma[Pz,Pz].
$$

## 7. Candidate TCLS: CST-curvature least-squares model

The moving-frame and geodesic terms put frame or metric curvature into $B$.
A more direct alternative puts the CST map curvature itself into the parameter
acceleration. For each local direction $z$, define

$$
B_{TCLS}[z,z]
=
\mathop{\operatorname{argmin}}_b
\left\|Jb+H[Pz,Pz]\right\|_2^2
+\lambda\left\|b\right\|_2^2.
$$

Its closed form is

$$
B_{TCLS}[z,z]
=
-(J^\top J+\lambda I)^{-1}J^\top H[Pz,Pz].
$$

Thus

$$
K_{TCLS}[z,z]
=H[Pz,Pz]+JB_{TCLS}[z,z]
$$

is the residual CST curvature that cannot be removed by a regularized
tangent-space correction. In the undamped limit, $JB_{TCLS}$ removes the
component of $H[Pz,Pz]$ lying in the image of $J$; the normal component remains.
This gives $B$ a least-squares and geometric meaning rather than inserting an
arbitrary curvature tensor.

The EMA signal remains $P^\top g^\theta$, and its five moments are reconstructed
with the same common formulas above. The emitted parameter displacement is

$$
d=Pz+\frac12B_{TCLS}[z,z].
$$

## 8. Comparison and experiment boundary

The five candidates are summarized below.

| candidate | EMA signal | first-order parameter map | second-order parameter map |
| --- | --- | --- | --- |
| P | $g^\theta$ | $A_P=I$ | $B_P=0$ |
| TF | $P^\top g^\theta$ | $A_{TF}=P$ | $B_{TF}=0$ |
| TM | $P^\top g^\theta$ | $A_{TM}=P$ | $B_{TM}[z,z]=DP[Pz]z$ |
| TG | $P^\top g^\theta$ | $A_{TG}=P$ | $B_{TG}[z,z]=-\Gamma[Pz,Pz]$ |
| TCLS | $P^\top g^\theta$ | $A_{TCLS}=P$ | $B_{TCLS}[z,z]=-(J^\top J+\lambda I)^{-1}J^\top H[Pz,Pz]$ |

All five solve a fourth-degree objective, but they do not share its
coefficients. P accumulates moments in parameter coordinates. TF, TM, TG, and
TCLS accumulate moments in the current tangent coefficients. TM, TG, and TCLS
additionally change both the quadratic pushforward moment and the
cubic/fourth-degree terms through their distinct $K_\chi$.

The first comparison should hold the following fixed across arms:

- identical minibatches, initial parameters, Adam clocks, trust radius, and
  fourth-degree solver budget;
- the same pullback form $P(G)$ for TF, TM, TG, and TCLS;
- no moment transport in any tangent arm;
- reset all five moment objects after structural mutation;
- report both exact represented-weight loss and the fourth-degree model value
  before accepting an update.

This isolates the choice of moment space and second-order retraction. Moment
transport, diagonal or atom-block approximations to the five raw moments, and
faster fourth-degree solvers remain separate experimental axes.
