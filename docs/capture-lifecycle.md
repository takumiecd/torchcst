# Backward capture lifecycle

`BackwardContext` is an update-scoped mailbox. It does not run autograd and it
does not decide structure. Compute modules attach an output tensor hook during
scheduled observation windows; that hook delivers the module input and output
gradient to the mailbox.

```text
begin_update
  prepare instruments
  attach context to observed modules

forward
  tensor hook closes over the detached module input

backward
  hook receives g_out
  deferred       -> queue x and g_out
  inline_reduced -> call instrument.measure and queue its small result

observe_microbatch
  attach the microbatch aggregation weight

finalize_backward
  normalize both capture modes to WeightedMeasurement
  call instrument.finalize_update
  release context and module hooks

optimizer.step
engine.step
  schedule, retention, selection, and structural transactions
```

## Why top-K is not a backward operation

Backward produces evidence. The schedule-issued birth budget is available at
`engine.step()`, after retention has also reported replacement counts. A scored
birth proposer therefore reads finalized scores and selects at most its
allocated budget at the structural boundary.

This separation guarantees that hooks cannot mutate structure and that the
same scores can be paired with different selectors or allocation rules.

## Memory trade-off

`deferred` preserves maximum flexibility because an instrument can inspect the
full detached boundary tensors after autograd finishes. Those tensors remain
alive until `finalize_backward()`.

`inline_reduced` computes sufficient statistics while the layer's output
gradient is available and retains only the returned tensors. It is usually the
appropriate mode for large convolutional activations, provided the reduction
itself is chunked to control temporary memory.

Capture mode can be selected per site. Policy code is independent of that
choice because both paths call the same instrument `measure` method.
