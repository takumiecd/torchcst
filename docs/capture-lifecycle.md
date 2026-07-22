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
  after_backward  -> queue x and g_out
  backward_inline -> call instrument.reduce_backward and queue its small result
  mixed site      -> do both for their respective instruments

observe_microbatch
  attach the microbatch aggregation weight

finalize_backward
  call deferred instrument.measure_after_backward
  normalize both stages to WeightedMeasurement
  call instrument.finalize_update
  release context and module hooks

optimizer.step
engine.step
  cadence, quota distribution, selection, and structural transactions
```

## Why top-K is not a backward operation

Backward produces evidence. The event's distributed structural quota is available at
`engine.step()`, after retention has also reported replacement counts. A scored
birth proposer therefore reads finalized scores and selects at most its
allocated budget at the structural boundary.

This separation guarantees that hooks cannot mutate structure and that the
same scores can be paired with different selectors or allocation rules.

## Memory trade-off

`after_backward` preserves maximum flexibility because an instrument can inspect the
full detached boundary tensors after autograd finishes. Those tensors remain
alive until `finalize_backward()`.

`backward_inline` computes sufficient statistics while the layer's output
gradient is available and retains only the returned tensors. It is usually the
appropriate mode for large convolutional activations, provided the reduction
itself is chunked to control temporary memory.

Timing is declared per observation request, so one site can mix inline gradient
reduction with deferred residual or certificate computation. `capture_mode`
remains only as a global/per-site test override; the engine rejects an override
when an instrument does not implement the required stage capability.
