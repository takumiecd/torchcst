# Bounded complete-support benchmark reports

The whole-chart FP64 oracle remains unchanged. Existing Product CPU support
reports evaluate every axis site again before and after each measurement worker,
and before the dense worker. That diagnostic work grows as A*(NI+NO), although
the compact Triweight support is small. This change reduces only CPU report
work; it does not reduce the independent Y/dX/all-atom oracle, the measured step,
its samples or capture/replay memory scope.

Compute conservative support bounds against the actual rounded sorted site
arrays using FP64 search bounds enlarged by8*eps*(radius+abs(centre)+maxAbsSite).
Evaluate every site inside those bounds with the original dtype and arithmetic;
omitted sites are necessarily zero. Bound width at128 sites; wide, unsorted,
nonfinite or non-FP32/64 cases use the original dense axis computation. The
Product and input-Strip wrappers retain their original report fields/meanings.

Norm reduction order changes within compact rows. With at most128 positive
terms, the relative FP32 summation error is bounded far below1e-3; zero terms do
not add rounding error. Recompute both original dense axes when the resulting
product norm lies within1e-9 of the existing1e-6 floor threshold. The report
uses all atoms/sites in support and no sampling. The earlier full-site mechanism
is retained as the numerical/structural fallback.

Thirty-one independent dense-oracle CPU scenarios pass, including both dtypes,
1/33/1024/8192 site axes, Strip gaps, partial boundaries, tiny/floor-crossing
profiles, nextafter support endpoints, wide/zero/negative/nonfinite precisions,
large or tiny coordinate scales, unsorted axes, chunk tails, empty atoms and
both actual8192 fixture wrappers. Complete CPU gate1144/1898 skips passed with
29 of these scenarios; the additional two nextafter cases passed in the31-case
targeted suite. New benchmark code has no CUDA runtime/package modifications.

Local and eventual Colab CPU diagnostic times are separate from CUDA training
step results. Raw tests and diagnostic driver/reports are retained in ignored
benchmarks/cuda/linear/evidence/profile-product-support-report-20261008/.

A local single-thread diagnostic at8192 axes/4096 atoms/rho3 produced identical
integer reports: complete-axis median0.2962745s versus bounded-support0.00110617s
(three samples each). This is CPU diagnostic time only, not a CUDA step gain.
A preliminary run overlapped the complete CPU gate and is retained separately
as local-diagnostic-contended.json; the subsequent report ran after that gate
completed. Source driver and every sample remain in the ignored evidence root.

The original full8192 Product family job l4job-1db8c943dbc04f89b01e052025af36fc
terminated when its runner child exhausted the predeclared1800s budget. The pool
driver itself did not time out (exit1,timeoutFalse). Four full-scale correctness
workers and only the prepared measurement worker completed; no paired performance
comparison is qualified. Preserve the original failed/incomplete artifact and
all workers. The new support-report source change will be validated on the same
Colab CPU before any new same-budget complete-step experiment. Neither timeout
budget nor oracle scope is relaxed; no unchanged-source retry is submitted.
