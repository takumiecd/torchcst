# M8 Atom-budget task pilot

English (canonical) | [日本語](m8-atom-budget-20261110.ja.md)

This note records the initial preparation and subsequent measured validation.
The owner closed unmerged M16 implementation PRs159–162
and authorized an M8 Atom-count search, moving the reserved GPU to RTX4090.
The closed branches and all results remain preserved; closure is not evidence
that M16 caused the task gap. Prior equal-count M8/M16 losses were similar.
M8 PR163 and its evidence remain open/preserved; this research branch is stacked
on it, so M16 history remains in ancestry without being selected for new runs.
Integration must extract the M8 changes from the abandoned stack before adoption.

A separate physical/envelope budget Algorithm admits512↔2048,FP32,1..256 rows,
unconnected M8/base1/merge2/origin0/spacing1,A16384/24576/32768/49152/65536/98304/131072.
Only a conditional host guard and private Plan/extension declarations are added;
remove the RADIUS_LM_BUDGET block to recover the existing header byte-for-byte.
All numerical kernels,old c8/c16 guards,initialization and optimizer are unchanged.
The public/default Registry is unchanged. This is not central benchmark fixture
certification or a Graph route.

Before task learning require metadata/rejection/roundtrip and native retained,
multiple-forward and noncontiguous-input tests,plus per-stage full-A FP64
Y,dX,each raw field (maxabs≤3e-4 AND relativeL2≤1e-4 separately),exact serialized
next-update/observer equality,and checkpoint/moment/clock/count audits.
The consumer freezes source/data/protocol and records actual4090/CC8.9 identity.
Full-step time and allocated/reserved peaks are distinct. No merged/default adoption.

CPU validation before freezing:2288 passed,2677 optional skips; Ruff and diff checks pass.


## Completed validation and task pilot

Frozen library0bed48655b1d988e0c59b4442005aaec8c463a31 and consumer
9b9e2eb1e4579fe035ca2442b36bad99f1df029d completed17 fresh arms on actualRTX4090
(CC8.9,Torch2.9.1+cu128,CUDA12.8). Native123 tests,17 exact serialized-next-update/
observer qualifications,34 checkpoint and32 explicit Atom/mask-one count audits
passed. Initial/final full-A FP64 checks cover96 tensor/raw-field pairs:
maxabs1.9908421e-5,relativeL2 max5.9737507e-7,zero older component violations;
maxabs<=3e-4 AND relativeL2<=1e-4 remained unchanged. Finite FP32 moments,
clocks976 CST/973 DST and final validation reload passed. Task-qualified counts
arec16/32/64/128;unused midpoints24/48/96 were metadata-tested,not task-trained.

DEV is validation:8192 story-disjoint targets,model.eval()/no_grad(),not logged
minibatch TRAIN CE. Validation is also used for hyperparameter selection;32768-target
confirmation has no performance evaluation. All8 FFN maps are replaced,not the
attention/embedding/tied-vocabulary maps. Final validation CE:

| A/map | M8 validation CE | DEV-selected dense-mask DST CE |
| --- | ---: | ---: |
|16384|3.350850|3.373828|
|32768|3.344251|3.358979|
|65536|3.331671|3.323396|
|131072|3.330237|3.315642|

Fresh Dense is3.308981. No CST count meets the fixed+.01 CE and-.5pp Top1 pilot
margins,so no midpoint was triggered and selected_coefficient remains null.
65536 is only a descriptive follow-up compromise:Top137.280% vs Dense37.451%,
CE gap.022690,step459.236ms,allocated1622.90MiB,reserved2114MiB. Increasing to131072
yields only.001435 lower CE and observedTop136.780%. No convergence,formal
equivalence,optimum or unique-cause claim. Same initialization policy changes
radius distributions with A;this does not isolate pure representation capacity.
The small LM shows no actual speed/allocated-memory advantage over Dense
(18.194ms,948.74MiB allocated). DST remains dense weights+Boolean mask;no COO/CSR.

[Consumer PR40](https://github.com/takumiecd/cst-research/pull/40) contains bilingual
validation/TRAIN plots,all twelve DST LR trials,geometry,actual costs and derived
provenance. Source archive e77a8b6ed48d24a8dc44600f3f6459d44774391194b44b858665c27d5abd5a07;
final compact60f83fa3f78f9135c5c8d9612cca02d6479e1021860d02f80c2327936e952ffa.
All195 payloads/2237 source hashes were retrieved/verified. All33 owned commands
finished successfully;GPU idle,raw checkpoints retained remote,Hub remains running.
Temporary API403 monitoring loss recovered after owner token renewal;no duplicate
job was launched. No merge/default adoption. M16 PRs159–162 remain closed with
code/evidence preserved;extract M8 from the abandoned ancestry before integration.
