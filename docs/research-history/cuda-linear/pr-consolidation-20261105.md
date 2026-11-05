# Qualified CUDA research consolidation

English (canonical) | [日本語](pr-consolidation-20261105.ja.md)

The owner requested PR consolidation on 2026-11-05. PR #140 targets main and
preserves the individual commits instead of squashing or rewriting the research
history. Its pre-consolidation head is 76204ef47505f1945cdaba4de581b2ce4aa377c7.

## Included implementation and evidence

- #128–#132: guarded benchmark-local whole-step Graph execution, explicit FP32
  owner-window Plans, shared geometry, int32 temporary sort keys and parallel C.
- #136–#137: the ordered weak-root topology correctness fix, shared stock/callable
  qualification and the bounded weak-leaf guard optimization.
- #139: large-width admission and corrected G4 qualification through 16384.
- #140: completed one-epoch HIGGS middle-layer density/width study and its
  source/data/result provenance. This is distinct from the later confirmation.

These are explicit opt-ins. Automatic Plan selection and ordinary-entry Graph
rejection are unchanged. The benchmark Graph cohorts are not a general public
CUDA Graph training API. FP32 bodies follow the same operator law but do not
claim bit equality with FP64 evaluation.

The recorded gates include independent Y/dX/all-atom-gradient Fourier and
quadrature references, optimizer/gradient/clock comparisons, lifecycle and memory
checks, and two-order 45000-update experiments. Read each revision's recorded
scope: #131 retains its failed original long-memory flag and a retrospective
supplement, rather than converting that supplement into a prospective pass.
#139 larger-width measurements use short 768-update runs and hashes; direct
state-byte admission is at 2048. These are not long-horizon quality evidence.
The sibling G4 confirmation in #138 is integrated separately as documentation.

## Excluded and continuing work

#133–#135 close without merging. Their complete-step results respectively
regressed by 2.1–3.2%, improved only 2.40–2.58% against a fixed 5% threshold, and
triggered the stop rule with a roughly 6.8% regression. Their branches, commits
and retained evidence remain available. Closing a PR does not delete them.

#141 remains the running 10/20-epoch, three-seed HIGGS confirmation; #143 is the
separate DST pilot pending GPU qualification. Neither preparation nor a single
development seed is a completed confirmation result. Existing runs retain their
frozen source identities regardless of main integration.

Model-level protocols and future runners are organized in
[cst-research](https://github.com/takumiecd/cst-research). Kernel correctness and
implementation validation stay here. Integration CI is rerun on the consolidated
PR against main; no new arithmetic, dispatcher promotion, branch deletion,
worktree cleanup or evidence removal is introduced by this consolidation note.
