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
