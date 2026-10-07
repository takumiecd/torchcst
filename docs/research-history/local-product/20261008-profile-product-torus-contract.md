# Proposed Torus profile-product contract

This is a CPU FP64 proposal awaiting the owner's distance interpretation.
It is not a public declaration or an adopted CUDA route. Existing radial Torus
and Euclidean profile-product semantics remain the controls.

The existing intrinsic Torus is S1 x S2 embedded in R4, with embedding

\[
 E(\theta,q)=((R+r q_0)\cos\theta,(R+r q_0)\sin\theta,r q_1,r q_2).
\]

Let an atom's canonical centre be (arc,s1,s2), theta_a=arc/R,
q_a=(cos(t),sinc(t/pi)s/r), t=norm(s)/r. The output Strip samples the circle;
the input sites sample the S2 section. For a centre-fibre interpretation,
hold the other geometry axis at the atom centre for each profile query:

\[
 d_c(i;a)^2=4(R+r q_{a,0})^2\sin^2((\theta_i-\theta_a)/2),
 \qquad d_s(j;a)^2=r^2\|q_j-q_a\|^2.
\]

Then u_a(i)=triweight(d_c(i;a)^2/sigma_a^2),
v_a(j)=triweight(d_s(j;a)^2/sigma_a^2), and

\[
 W_{ij}=\sum_a A_a\frac{u_a(i)v_a(j)}
 {\max(\|u_a\|_2\|v_a\|_2,\epsilon)}.
\]

The complete Cartesian chart norm factorizes, with one global floor. Two
geometry-axis profiles correspond to three intrinsic centre coordinates;
this requires an explicit declaration contract rather than reusing the current
one-profile-per-Euclidean-coordinate validation.

Site separability does not imply parameter separability. The circular radius
R+r*q_a0 depends on the two section-centre coordinates. Their VJP must include
both u and v derivatives, and both norm derivatives. Detaching that dependency
in the prototype produces section-gradient error0.0323801093. Euclidean's
independent centre-factor VJP cannot simply be reused for this proposal.

The physical chord to a jointly displaced Torus site instead obeys

\[
 d_{joint}(i,j;a)^2=r^2\|q_j-q_a\|^2
 +4(R+r q_{j,0})(R+r q_{a,0})\sin^2((\theta_i-\theta_a)/2).
\]

The site-dependent q_j0 prevents the same circular profile factorization.
Centre-fibre profile product changes composition; it is not the old joint
radial kernel. Which distance interpretation to adopt remains undecided.

## CPU proof

The ignored probe uses the current geometry's physical embeddings to construct
independent axis queries and a full-matrix L2 reference, then compares the
analytic product and all five canonical-parameter gradients. FP64, Strip65x12,
output tile16/pitch4.1,3x4 S2 cross grid,R3.2626763334/r0.4. Six atoms cover
asymmetry, wrapped negative arc, zero section centre, empty support and tiny
profiles. Floors1e-6 and0.5 both pass: maximum value error2.776e-17,
dX1.111e-16,all-five-parameter gradient4.309e-13. Joint chord analytic formula
matches embeddings within2.132e-14. Widths are fixed current widths under the
existing task-VJP law; optimizer, retraction, CUDA and Graph updates are not
validated by this study.

Raw source and report remain in ignored
benchmarks/cuda/linear/evidence/profile-product-torus-contract-20261008/.

Source/report SHA256:

- probe.py: `0043990d9b870de00b3dbe0ffe549f25a6fbc97cb9140a587fe9e7756382c6d2`
- report.json: `f0e5df886e3928748212993500b6fef4d6b45aeddbdaa72e524442df159af2ed`
