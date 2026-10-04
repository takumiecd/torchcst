## Change

Describe the problem, the resulting behavior and the affected mathematical contract.

## Validation

- CPU checks / tests:
- For GPU changes: independent oracle, dX / all atom gradients, optimizer state,
  live widths / support movement and CUDA Graph replay:
- Complete training step: hardware / runtime, baseline → candidate time,
  peak allocated / reserved memory including capture, independent run count:
- Evidence: result / observation or Issue, job ID, source hash, reproduction command:
- Checks not run:

Start with [CONTRIBUTING](https://github.com/takumiecd/torchcst/blob/main/CONTRIBUTING.md)
and the [kernel development guide](https://github.com/takumiecd/torchcst/blob/main/docs/kernel-development.ja.md). Keep raw logs,
tensors and source snapshots in ignored evidence directories.
