# Semantic field learning-rate qualification

English (canonical) | [日本語](optimizer-field-lrs-20261107.ja.md)

`CSTAdamW` and `CSTOptimizer` accept absolute initial `field_lrs` declared by the
parameterization. Proposal scaling happens after base-optimizer normalization
and before the coordinate update, with one Parameter/state/counter per Atom table.
Group LR remains the unspecified-field fallback and Polar outer clock. See the
[API contract](../cst-optimizer.md#field-learning-rates) for scheduler anchors,
checkpoint restoration, raw-decay restrictions and amplitude/activity coupling.
This feature does not select an optimal/default learning rate.

## Validation

The tested operator is `a6c5f12a31442a03afb79cf47ebdb88db13afbb7`, based on merged
PR150. Later changes in this PR only document this result; runtime/tests remain
byte-identical to the tested source. Actual GPU: RTX4090, compute capability8.9,
24GB-class; Torch2.11.0+cu126, Python3.12.9, NumPy2.0.2. TF32 was disabled for the
LM task. Test-only pytest9.0.2 dependencies were isolated from the numeric runtime.

- Remote CPU:25 new semantic-field tests passed, no skips. They include independent
  Adam equations, declared retraction, moments/counters, schedulers/zero LR,
  field layouts, native/wrapper behavior, SGD/AMSGrad/foreach/maximize, logical
  decay, group settings and checkpoint validation/restoration.
- Native CUDA:8 permanent tests passed, no skips: native/wrapper × amplitude peaks
  0.0024/0.0048/0.0096/0.0192, with distinct input/output center rates, five schedule
  factors including0, CPU FP64 comparison, Parameters/moments/counters and a
  serialized next update. The explicit radius flat-torus fused Plan was selected.
- Actual LM:8 disposable configurations passed: native/wrapper × amplitude
  8/16/32/64 with centers/outer16, ordinary peak0.0003. The four-block N256,
  FFN4N, c8 model has all eight FFN projections CST. Each configuration compared
  the public API against the retained research normalized-proposal adapter on
  20 ordinary warmup task updates followed by5 full-peak stress updates on the
  next ordered batches. Batch2/context128/accumulation4, AdamW(.9,.95), epsilon1e-8,
  decay0 and global clip1 were used. All25 losses and Parameters matched exactly
  (maximum observed parameter difference0); gradients and moments passed their
  recorded FP32 tolerances and counters matched.
- All8 configurations serialized model/optimizer, restored saved field settings
  into a public optimizer constructed without field overrides, then reproduced
  the next full-peak task update with parameter difference0. All64 trained
  projection Y/dX/all-Atom VJPs passed CPU FP64 chunked-reference checks.

The peak stress portion is a correctness fixture, not a long learning trajectory
or an accuracy/performance benchmark. All models were fresh disposable copies;
no original learner/checkpoint was resumed or overwritten. A diagnostic scalar
conversion warning in the evidence script did not alter results or retain a
training trajectory. No retries were needed.

Local CPU validation used Torch2.13.0/Python3.11.15: the implementation passed
2121 tests with2474 GPU/database skips; focused field tests passed25 cases.
Wheel/sdist builds included the helper with verified matching bytes. CPU CI on
exact tested head passed (run42320197633); final documentation-head CI is checked
separately before integration. A skip is not counted as GPU/database validation.
The stock Graph benchmark rejects configured field rates during cold
qualification and uses eager fallback. No Graph qualification or speed/memory
advantage is claimed.

## Evidence and reproduction

- Operator archive SHA256: `aeac783c7053f97374337f1fb0a918ae5d69e90337b05d4a0df3ed9cc8c6af7a`.
- Research consumer: `351617dc81f529ce10e964e5a762f336b7c24b39`.
- Result archive SHA256: `64ae9bac51b1565d35cae76b875f45096a9145de663a2cb0f65ffd98367dfc06`.
- Result manifest SHA256: `c37a46db68037dbdc5b3b36a1d484df8047251e222f4b86719e2a399137471f2`.
- Full-LM result SHA256: `49125fd92525bde1fba4ee1220fbe92a5fdcb9534852b626ccc5d1453b880800`.
- Data manifest SHA256: `5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.

All8 retrieved archive members, including the manifest, were hash verified.
The study's external evidence locator retains frozen source, requests,
controller/process identities, JUnit/logs, the full-LM qualifier and result.
The consumer belongs in [cst-research](https://github.com/takumiecd/cst-research),
not the runtime package. The same task/data contract as its FFN study was used;
this validation does not rerun or migrate the historical quality results.

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_optimizer_field_lrs.py -q
python -m pytest tests/test_optimizer_field_lrs_cuda.py -q
```

For the actual task, use the pinned consumer/data plus the retained
`qualify-public-field-api-r16-02.py --data TOKEN_DIR --output NEW_DIR` script in
the recorded runtime. All hashes and raw evidence remain external; this note is
a concise result record. Wider kernel/hardware/Graph qualification is separate.
