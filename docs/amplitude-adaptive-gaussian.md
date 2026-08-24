# Amplitude-adaptive Gaussian atoms

## Question

The ordinary one-sided CST Gaussian is

$$
\phi(x;s)=\exp\!\left(-\frac{\lVert x-s\rVert^2}{2\sigma^2}\right),
\qquad
u(s)=\frac{\phi(s)}{\lVert\phi(s)\rVert_2}.
$$

Its two-sided, L2-gauged atom is

$$
A(w,s,t)=w\,q(s,t),\qquad q(s,t)=v(t)u(s)^\top,
\qquad \lVert q\rVert_F=1.
$$

The proposed continuation mechanism replaces distance by an
amplitude-dependent scaled distance. For a dimensionless inverse-width
multiplier $a(w)>0$,

$$
\phi_a(x;s,w)=
\exp\!\left(
  -\frac{a(w)^2\lVert x-s\rVert^2}{2\sigma^2}
\right),
\qquad
\sigma_{\mathrm{eff}}(w)=\frac{\sigma}{a(w)}.
$$

Small atoms can therefore see a broad score field and mature atoms can
recover the ordinary local Gaussian.

## What the raw rule proves

The literal rule $d\mapsto wd$ needs a reference amplitude $w_\star$ for
units:

$$
a_{\mathrm{raw}}(w)^2=\frac{w^2}{w_\star^2}.
$$

The square makes the kernel smooth and independent of the sign of $w$; using
$|w|$ directly would introduce an unnecessary cusp at zero.

For a finite neuron population, let $u_w(s)$ be the normalised column. Since
the unnormalised exponent is $O(w^2)$,

$$
u_w(s)=u_0+w^2u_2(s)+O(w^4),
$$

where $u_0$ is the normalised constant column and is independent of $s$.
Consequently the represented atom satisfies

$$
A(w,s,t)=wq_0+w^3q_2(s,t)+O(w^5).
$$

This gives three exact asymptotic facts:

$$
\frac{\partial A}{\partial w}=q_0+O(w^2),
\qquad
\frac{\partial A}{\partial s}=O(w^3),
\qquad
G_{ss}=\left\langle A_s,A_s\right\rangle=O(w^6).
$$

Thus the raw rule has an infinite-width degeneracy. At $w=0$, the amplitude
birth score is the same at every coordinate, and near zero its spatial
variation is only $O(w^2)$. It sees the whole chart but loses the direction
in which it should travel. A pullback inverse can cancel the vanishing scale,
but it must then amplify an increasingly flat, noise-sensitive field.

## Finite exploration width

A bounded smooth multiplier avoids both infinite width and unit ambiguity:

$$
a(w)^2
=a_{\min}^2
+(a_{\max}^2-a_{\min}^2)
\frac{w^2}{w^2+w_\star^2},
\qquad
0<a_{\min}<a_{\max}.
$$

The recommended first experiment sets $a_{\max}=1$. Then

$$
\sigma_{\mathrm{eff}}(0)=\frac{\sigma}{a_{\min}},
\qquad
\lim_{|w|\to\infty}\sigma_{\mathrm{eff}}(w)=\sigma.
$$

For example, $a_{\min}=1/4$ gives a reserve atom a $4\sigma$ exploration
width while a mature atom returns continuously to the existing Gaussian.

Because $a(0)>0$, the zero-amplitude limit keeps its coordinate-dependent
column:

$$
\left.\frac{\partial A}{\partial w}\right|_{w=0}
=q_{a_{\min}}(s,t).
$$

The birth-score field therefore remains spatially informative. Moreover,

$$
\frac{\partial A}{\partial s}=w\,\partial_s q_{a(w)}=O(w),
\qquad G_{ss}=O(w^2),
$$

which has the same non-degenerate order as the ordinary fixed-width CST
atom under pullback normalisation.

## Gauge and orthogonality

L2 normalisation preserves the interpretation of $w$ as functional
amplitude for every width:

$$
\lVert A(w,s,t)\rVert_F=|w|.
$$

It also gives $\langle q,q_s\rangle=\langle q,q_t\rangle=0$. However, hard
width-amplitude coupling changes the amplitude tangent:

$$
A_w=q+wq_w.
$$

Hence

$$
\langle A_w,A_s\rangle
=w^2\langle q_w,q_s\rangle,
$$

and similarly for $t$. Scale and translation are orthogonal in the ideal
continuous symmetric Gaussian by parity, but not necessarily on a finite,
irregular or boundary-truncated neuron chart. A correct pullback metric for
the hard-coupled family must therefore retain these terms rather than reuse
the fixed-width closed form.

## Expressivity distinction

Hard coupling $a=a(w)$ removes width as an independent degree of freedom. It
cannot in general represent an arbitrarily small, narrow correction with one
atom, so it is not a strict superset of the current Gaussian family.

A mathematically cleaner extension gives each atom an independent maturity
$m$:

$$
A(w,s,t,m)=wq_{a(m)}(s,t).
$$

The old CST is recovered exactly at $a(m)=1$, while broad reserve atoms are
available at $a(m)<1$. Since the partial derivative with respect to $w$ holds
$m$ fixed, amplitude remains exactly orthogonal to translation. A continuous
regulariser or optimiser dynamics may encourage $m$ to follow amplitude
without making representability depend on that relation.

The hard-coupled form is still a useful first causal experiment: it tests the
specific hypothesis that low-amplitude atoms benefit from broad sensing. The
independent-maturity form is the stronger representation if that experiment
succeeds.

## Implemented maturity chart

The experimental family uses an unconstrained logit $m$:

$$
a(m)^2=a_{min}^2+(1-a_{min}^2)\operatorname{sigmoid}(m).
$$

Thus every finite parameter value is valid, $m\to-\infty$ approaches the
finite exploration width, and $m\to+\infty$ approaches the ordinary Gaussian.
The ordinary family lies in the closure rather than at a finite logit.

Initialising every atom with the same negative logit is not the intended
low-amplitude mechanism: it broadens useful atoms too and can collapse the
initial effective rank. A scale-selective initialisation follows directly
from the bounded amplitude law. Write
$r=|w|/\operatorname{median}|w|$ and choose a threshold ratio $c>0$. Equating

$$
\operatorname{sigmoid}(m)=\frac{r^2}{r^2+c^2}
$$

gives the exact logit

$$
m=2\log\frac{r}{c}.
$$

This relation is used only to initialise the independent coordinate $m$;
afterwards, task gradients may change it freely. The representation therefore
keeps a pure amplitude tangent and does not acquire the $q+wq_w$ term of hard
coupling.

## Falsifiable checks before training

For both the raw and bounded forms, measure across amplitudes:

1. spatial contrast of the birth score $\partial L/\partial w$;
2. coordinate-gradient norm and minibatch signal-to-noise ratio;
3. pullback eigenvalues and amplitude-coordinate cross energy;
4. displacement toward a planted narrow residual;
5. recovery of the ordinary Gaussian as $a\to1$.

The decisive toy problem plants a narrow target away from a reserve atom.
The proposal predicts that an intermediate finite exploration width reaches
the target from farther away than the fixed Gaussian, while the raw
$a(0)=0$ rule becomes spatially flat and fails at sufficiently small $|w|$.

In a 401-point one-dimensional chart with $\sigma=0.08$, target centre
$0.78$, reserve centre $0.15$, and $w_\star=0.05$, the first numerical check
gave the following at $w=10^{-4}$:

| form | birth-score contrast over the chart | coordinate gradient toward target |
|---|---:|---:|
| raw $a=|w|/w_\star$ | $2.8\times10^{-4}$ | $1.0\times10^{-8}$ |
| bounded $a_{\min}=0.25$ | $0.722$ | $6.2\times10^{-5}$ |

The raw zero-amplitude score was exactly spatially constant, as the expansion
predicts. The toy also exposed a separate hard-coupling effect: far from the
target, increasing $w$ simultaneously narrows the atom and can reduce target
overlap, so $\partial\langle A,q_\star\rangle/\partial w$ can become negative
even though both columns are positive. The ordinary interpretation of
$dL/dw$ as a fixed-dictionary birth score therefore no longer applies without
including the width transition in that score.

## First language-model screen

A fixed-atom screen used 298 atoms per site, tangent PullbackAdam with
$\beta_1=0$, and 1000 Shakespeare steps on an RTX 4090. The ordinary Gaussian
reached validation loss 1.7839. Initialising every maturity to $m=-2$ reduced
the initial effective rank from about 26 to 4--5 and reached only 2.0054.
Amplitude-aware initialisation with $c=0.5$ recovered much of the rank but
reached 1.8405. Restricting broad sensing to the smallest few percent with
$c=0.05$ preserved the initial rank, improved the 250-step loss from 2.3599
to 2.3219, and finished essentially tied but slightly worse at 1.7860.

This falsifies the indiscriminate broad-kernel version, while leaving a
narrower claim: broad reserve atoms can improve early discovery, but the
current free maturity dynamics do not convert that early advantage into a
better final solution.
