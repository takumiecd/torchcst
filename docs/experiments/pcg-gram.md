# Matrix-free separated PCG experiment

`ImplicitAdamConfig(gram_solver="pcg", factored_geometry=True,
first_moment_damping=1e-4)` routes first-moment compression through the blocked
Gram action, bypassing GramSystem construction and Cholesky. The default
solver remains unchanged. This is an experimental reference implementation,
not a recommended speed improvement.

## Numerical contract

The solver uses diagonal-preconditioned conjugate gradients, zero initial
solution, and a fixed host loop over `gram_iterations` (default 64). Tensor
predicates mask converged/broken iterations without a host scalar read. All
iteration slots still execute. `gram_block_size` (default 32) limits column
pair tiles. `gram_rtol` (default 1e-5) is a true relative residual acceptance
threshold, not a pseudoinverse rank cutoff. Damping must be explicitly finite
and positive. The solve runs without autograd recording.

Factor derivatives and displacement are promoted to FP64 before contraction.
The diagonal is evaluated from all sixteen separated term pairs without
constructing the Gram. The final returned, input-dtype solution is reapplied
to the actual operator; validity requires finite output, positive finite
preconditioner, no active-iteration breakdown, and
`norm(A*x-b) <= gram_rtol*norm(b)`. Zero rhs returns zero. Failed checks use the
existing optimizer latch to freeze parameter and moment updates under deferred
device execution. No direct-solve or CPU fallback is attempted.

This targets the same real-valued damped equations as Cholesky, but the
existing Cholesky path assembles and damps the Gram in input precision before
promoting it. Thus factor-contraction rounding and finite-iteration error are
separate differences. Benchmarks report error against both an FP64 factor
Gram oracle and the existing input-precision assembled Cholesky system.
Residual acceptance alone does not bound forward error in ill-conditioned
systems or establish learning accuracy.

## Memory and execution limits

No full D-by-D Gram is retained, where D=KP. Tiles are at most b-by-b (and can
cover the entire system when D <= b). Existing factor derivative caches and
their FP64 conversions remain O(K*(I+O)*(1+P+P^2)). PCG adds O(D) vectors and
the blocked action workspace described in [blocked Gram](blocked-gram.md).
K proportional to width therefore still entails quadratic factor storage.
This does not solve the complete multi-layer checkpoint memory problem.

The eager blocked implementation recomputes many factor blocks and launches
many small kernels each iteration. It is deliberately not captured as a
single unrolled CUDA graph; no graph-memory scaling improvement is assumed.
CUDA synchronization-error tests do not replace a complete profiler audit.

## Validation

Tests cover FP32/FP64 reference solves, both kernel variants, diagonal identity,
zero rhs/zero frame, nonfinite rhs, insufficient iterations, invalid config,
three consecutive updates against Cholesky with full Gram construction
forbidden in the PCG branch, and parameter/moment freeze plus permanent error
latching. The CUDA test checks the solve under synchronization-error mode.

## Reproduction

Run each GPU case in a separate process. `pcg_gram_benchmark` extracts the first
real proposed update frame/rhs, stopping before compression. Warm the selected
method, then measure three isolated compression calls. Peak counters are read
before constructing reference matrices. Baseline includes retained proposal
compilation/graph buffers; incremental peak is not total training memory.

```bash
PYTHONPATH=src python -m experiments.pcg_gram_benchmark \
  --method pcg --iterations 256 --block-size 64 --output output/pcg.json
PYTHONPATH=src python -m experiments.pcg_gram_benchmark \
  --method cholesky --output output/direct.json
PYTHONPATH=src python -m experiments.scaling_benchmark \
  --method cst_ray1 --inputs 128 --outputs 128 --atoms 64 \
  --factored-geometry --gram-solver pcg --damping .0001 \
  --gram-iterations 256 --gram-block-size 64 --steps 2 --repeats 3 \
  --output output/pcg_training.json
```

A failed convergence check is a failed case, not a valid timing/accuracy result.

## A100 measurements (2026-09-08)

A100 80GB PCIe MIG3g.40gb, PyTorch2.6.0+cu126; width128x128, K64/P4,
batch128, float32 model, highest matmul precision, damping1e-4, block64.
One first-update compression system, same seed17/model/data for each method:

| Method / fixed budget | Median compression ms | Relative residual | Relative solution error vs input-precision direct oracle | Accepted |
| --- | ---: | ---: | ---: | --- |
| Captured Gram + Cholesky | 0.715 | 4.96e-7 | 3.52e-7 | Yes |
| PCG16 | 331.82 | 3.37e-2 | 1.93e-1 | No |
| PCG64 | 1252.92 | 1.32e-3 | 6.80e-3 | No |
| PCG128 | 2503.69 | 7.72e-6 | 3.73e-5 | Yes |
| PCG256 | 4958.61 | 7.72e-6 | 3.73e-5 | Yes |

PCG becomes inactive after115 iterations on this rhs. Fixed budgets continue
executing masked slots, so256 does not improve this solution over128. FP64
factor-oracle solution error is3.81e-5 for the accepted PCG cases. The direct
oracle uses eager input-precision Gram assembly; production compilation and
output conversion can introduce additional rounding differences.

Isolated compression peak allocated memory (including retained preparation
buffers) is55.72MiB for Cholesky and41.66MiB for all PCG budgets. Baseline is
54.72 versus36.66MiB; incremental peak is1.00 versus5.00MiB. The PCG reduction
in this case is largely from avoiding retained dense Gram/solve graph buffers,
not from eliminating all quadratic storage. FP64 factor conversions increase
PCG's transient footprint. Equal peaks across budgets confirm no iteration
history is retained in this eager no-grad measurement.

### Complete training updates

Fresh processes, same synthetic eight resident batches and model seed as the
scaling benchmark. Eight warm updates followed by three blocks of two updates
(14 total); the short blocks bound the cost of this deliberately slow reference
implementation. Warmup/cold compilation excluded from the following peaks:

| Method | Median ms/step | Peak allocated MiB | Peak reserved MiB |
| --- | ---: | ---: | ---: |
| Cholesky | 11.39 | 76.78 | 206 |
| PCG256 | 5356.95 | 62.40 | 156 |

Both complete runs pass the final error latch check and the full timed-loop
synchronization-error guard. Final synthetic loss is4.8478536606 for both;
this is neither an MNIST accuracy result nor a long-trajectory equivalence
claim. The previous longer timing protocol gave19.99ms for Cholesky at this
width; do not treat the short-run host-sensitive11.39ms as a new speedup.

PCG saves18.7% of warmed training peak allocated memory but is about470x slower
in this comparison. It remains above the historical Dense+Adam17.13MiB peak
at this width. The matrix-free action is correct, but diagonal preconditioning
plus this eager implementation is not a practical replacement. At D256 and
block64 each action evaluates16 block pairs times16 separated term pairs;
repeating these small matrix operations over hundreds of iteration slots
causes substantial submission/computation overhead. No speed lower bound for
a fused matrix-free implementation follows from this result.

Next work should improve the preconditioner and fuse/reuse bounded GPU tile
operations before attempting large K or long learning runs. A multi-layer,
checkpointed peak-memory advantage has not been established. Defaults stay
unchanged, and insufficient-iteration cases remain rejected.

Final validation:247 local tests passed,25 skipped;15 targeted A100 tests
passed. Ruff and diff checks passed. All experiment subprocesses completed;
raw JSON/logs are in `output/pcg_a100/results.tgz` and extracted alongside it.
Runtime checkpoint:`0f58605`.
