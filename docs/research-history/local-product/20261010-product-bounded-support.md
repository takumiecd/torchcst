# Regular Product: bounded support metadata

Research candidate only. No public dispatcher adoption or performance result
is declared by this initial checkpoint.

Keep the existing single Euclidean Product, two positive equally spaced Line
axes, normalized Triweight profile product and Polar[A,4]. The whole-product
denominator is `max(norm(u)*norm(v), floor)`. Live activity widths, all positive
supports, full norm derivatives, canonical gradients and the public Polar
optimizer contract remain unchanged. Existing matrix controls retain their code.

The candidate uses the same conservative coordinate-to-index span, grouped norm
preparation, grouped W assembly, IEEE GEMM and atom VJP kernels. Only executor
storage changes: prepare at most C atoms, assemble their W contributions, then
reuse the support scratch. Backward reconstructs the same metadata from the
forward's immutable parameters, all nine decoder scalars, grid origin/spacing,
normalization floor and recipe. Scratch is not an autograd saved tensor.

Metadata scratch is at most `13*4*min(K,C)` bytes per preparation, rather than
retaining `13*4*K` bytes until backward. C262144 reserves at most13 MiB; at
K3355443 the earlier support table alone is about166.4 MiB. These are tensor
budgets, not measured CUDA peaks. W/dW remain full-size; removing them is a
separate execution candidate. Recomputing norms may make this route slower.

The first frozen comparison uses B32, seed41, 5% atoms, initial rho3, live bounds
1..16 and the existing capturable AdamW/production Polar update. Compare native
G8P8 and Torch G8P8 with same-engine bounded counterparts (C262144), plus dense,
at N2048 and then8192 after the small/2048 correctness gates. Include complete
Graph step and capture/replay allocated/reserved peaks. Phase diagnostics are
separate instrumented graphs and do not replace the primary step measurements.

Predeclare no tolerance changes or retries of an unchanged failed source. Run
the independent full-site FP64 oracle for every atom and Y/dX/all four atom
gradients. Supplemental validation after24 captured learning updates requires
both max-absolute and relative-L2<=4e-4. Boundary tests cover chunk tails,
empty/singleton/tiny/floor/wide/precision supports, strides, scalar/lattice
mutation after old forward, zero atoms, required gradients and Graph moments.

Assess each engine against its own control. A memory candidate qualifies at
>=5% allocated-peak reduction with <=3% complete-step time regression; a speed
candidate requires >=3% step improvement with no allocated-peak increase. Small
improvements need independent confirmation. An initial losing measurement is
retained, not replaced by a more favorable run. The 2048 study can be negative
while the independently declared8192 memory objective remains meaningful.

Raw drivers, logs, source and pool receipts are retained under ignored output;
record measured source/result hashes and disposition here after retrieval.
