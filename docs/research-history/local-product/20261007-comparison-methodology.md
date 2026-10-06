# Small Linear comparison and correctness scope

CST candidate and CST baseline are the same declared mathematical contract: local_polar_product.normalized_triweight.shared_width.v1 on the eligible one-dimensional regular input/output domains and sliced domains. Tests cover all source gradients,continuous positions,current normalization/norm floor,singleton/empty support and20 reference production optimizer updates. They do not establish arbitrary-geometry support beyond this declaration.

For atom a,let g_a(t)=max(1-((t-c_a)/sigma_a)^2,0)^3. Each direction has its own center and full-domain discrete L2 normalizer,with floor epsilon=1e-6. Norms are computed over the full unsliced domain before selecting the sliced input/output sites. The shared current width is decoded from the canonical Polar source. With normalized input factor v_a[i],output factor u_a[j],current amplitude w_a:

D_a^in = max(sqrt(sum over full input sites of g_a^in(t)^2), 1e-6),
D_a^out = max(sqrt(sum over full output sites of g_a^out(t)^2), 1e-6),
v_a[i] = g_a^in(t_i) / D_a^in,
u_a[j] = g_a^out(t_j) / D_a^out,
Y[b,j] = sum_a w_a u_a[j] H[b,a],
H[b,a] = sum over selected input sites i of X[b,i] v_a[i].

The measured routes recompute H within owner/parameter blocks; no global H tensor is allocated. Exact gradients include factor derivatives,normalizer derivatives and the fixed-source Polar pullback. A singleton is based on actual support and normalization semantics,not rho alone. Initial performance rho=sigma/spacing is strictly>1; correctness tests also exercise permitted sigma<spacing and normalization-floor cases.

Within one condition and across the independent repeats,the CST recipes use byte-identical initial canonical Parameter tensors and input/target tensors. Each CST plan independently matches a scalar FP64 mathematical oracle for Y,dX and every canonical source gradient; center gradients must be nonzero where the broad support oracle requires them. Captured20-step checks compare the same source optimizer/Polar update to its independent reference,including moments and steps. They do not compare training trajectories to dense.

The dense timing worker is nn.Linear(N,N,bias=False),initialized independently with a seeded uniform matrix. It is a separate performance reference,not a materialization of the initial CST matrix. It uses the same dimensions,batch,dtype,input/target,linear loss,AdamW settings andcapture/replay protocol. Its parameters are a full matrix; CST's source parameters and Polar update are different. See benchmarks/cuda/linear/run.py _measure/step. Initial Parameter hash equality is asserted only among CST recipes. Input/target hash equality includes dense. We do not claim equal initial dense/CST matrices or equal optimizer trajectories.

Complete-step timing includes forward,backward,optimizer and source Polar updates for CST,including partial reductions and topology/current-value refresh. It is not a forward-GEMM measurement. Each independent execution has21 graph samples; reported2-run values are medians of execution medians. Reverse case/plan ordering on repeat2 reduces an order confound but is not a statistical-significance test. Descriptive best-route selection across measured recipes is post-measurement; a runtime dispatcher is not certified by that selection.

Peak allocated includes all live benchmark/capture/replay buffers and workspace. Dense allocated peak around34MB is not its matrix parameter size alone. Reserved/process memory is separate. Compiler shared/register/spill reports do not prove physical DRAM traffic or L2 residency. Cache repair counters cover52 primary refreshes per measured direction; separate diagnostic timings cannot be summed to replace complete-step timing. Counter-free variants explicitly disable telemetry and cannot supply execution counter evidence.

Frozen source/result archives,SHA256 receipts,committed/submitted/worker runtime hashes,FP64 errors,input/source hashes and raw21-sample timing arrays are retained. Host CPU success with GPU skips is not GPU proof. A source-only/queued/failed/interrupted/cancelled job is not a performance result. New research routes remain opt-in draft PR experiments until integration/dispatch validation.

Equation verified directly against tests/test_local_product_research.py scalar_oracle and CUDA kernels.py prepare: normalization is discrete L2,not L1. This source check corrects the brief initial draft wording; runtime and all frozen measurements have always used L2.
