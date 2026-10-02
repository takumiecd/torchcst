---
name: colab-l4-pool
description: Share Colab L4 GPUs across research agents and Codex chats using a host-wide experiment queue, exclusive runtime leases, isolated source snapshots, and verified result retrieval. Use for L4 experiments that may overlap with other work; other GPU types use their existing resource skills.
---

# Shared Colab L4 experiments

The implementation now lives in `tools/colab/scripts`; this existing entry
point still uses the same host-wide queue and defaults to L4 only. Public
measurement users use [the Colab tool](../colab/README.md), without installing
a skill. Configure the expected authenticated account on this host once with
`pool.py configure --account EMAIL`; the implementation contains no personal
email. Keep L4 jobs on this shared queue. Other GPU types must be explicitly
selected by the orchestrator, never by an existing L4 supervisor implicitly.

Use the installed entry point `~/.codex/skills/colab-l4-pool/scripts/pool.py`
with Python 3.9+. It needs the standard library and `~/.local/bin/colab`.
The default queue is `~/.local/state/colab-l4-pool`, shared across local chats,
projects and worktrees. Do not override `--state-root` for live experiments:
separate queues would defeat the host-wide ownership rule.

Read [the operating guide](references/operations.ja.md) when first using the
pool or handling a failure. The low-level CLI is documented by
`colab skill` and `colab <command> --help`.
The [validation record](references/validation-20260930.ja.md) documents local
tests, the real single-L4 smoke campaign, and the limits of those checks.

If macOS VPN routing prevents Colab access, the pool can persist an interface
with `configure --interface en0` (or another confirmed working interface).
This binds only its CLI/keep-alive requests to IPv4 on that interface; system
VPN settings and remote experiment networking remain unchanged. Use
`configure --interface default` to restore system routing. See the guide for
the observed connectivity issue that motivated this optional setting.

## Submit and receive results

Write a Python driver that performs setup, correctness checks and measurements.
Save files under `os.environ['CST_JOB_OUTPUT']`; exceptions or nonzero exit codes
indicate failure. The driver runs in a fresh subprocess, from a frozen snapshot
of the supplied Git working tree. `src/` and the tree root are on `PYTHONPATH`.
Review the tree's ignored/untracked files before submission: every tracked and
nonignored untracked regular file is included, including uncommitted edits.
Symlinks and submodules are rejected; do not submit credentials or datasets
unnecessarily. The driver itself can be outside the Git tree.

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source /absolute/git/tree --script /absolute/driver.py \
  --label my-experiment --timeout 600
```

Keep the emitted job ID and directory. One orchestrator starts the supervisor
in a long-running tool/terminal session; agents only submit and wait:

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py serve --workers 1
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py wait JOB_ID --timeout 3600
```

`submit --wait` combines submission with waiting when a supervisor is already
running. A wait timeout does not cancel remote work. Read the final JSON and
`results/result.json`, `stdout.log`, `stderr.log`, and `artifacts/`; do not infer
success merely from a CLI exit or a log message. Driver-specific accuracy,
benchmark timing and memory must be saved by the driver, not inferred from
the pool's wall time. Downloaded results are SHA256-verified.

## Runtime ownership

- Default to one L4. The orchestrator can choose `serve --workers 2` or `3`
  when experiment traffic warrants it and allocation is authorized. Workers
  allocate only when they claim jobs; no hardware fallback is permitted.
- Only the supervisor runs provisioning, upload, execution, download or stop
  against its `cst-pool-*` sessions. Do not bypass the queue with direct CLI
  calls or notebook execution. Use `status` for pool state.
- Each L4 runs one whole experiment at a time, including setup, compilation,
  warmup, measurements and retrieval. Multiple L4s run independent jobs.
  A single experiment may fully use its GPU through GEMM, multiple streams,
  CUDA Graphs and other internal parallelism.
- Drivers must finish all work before exit; do not detach processes, change
  process groups, leave background services, or mutate system packages.
  Install optional dependencies into a job-local environment inside the
  source directory, and invoke that environment from the driver.
- All managed sessions use separate CLI state files. Identity is checked
  against the user's authorized Colab account before allocation. Actual
  hardware must be exactly NVIDIA L4; another accelerator is rejected.
- The supervisor stops owned VMs after 60 seconds without work and exits.
  Use `--idle-seconds 0` to drain a batch and stop immediately. SIGINT/SIGTERM
  stops taking new jobs, finishes active jobs within their timeouts, and stops
  the VMs. Keep the supervisor alive until cleanup completes.

## Failures and recovery

`cancel JOB_ID` cancels only queued work. Execution/transport uncertainty
interrupts dispatch and triggers an owned-session stop with server verification.
If stop cannot be confirmed, the slot remains quarantined. No experiment is
automatically retried. A killed supervisor leaves durable ownership records;
the next supervisor refuses to start until recovery.

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py status
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py recover
```

Recovery stops only pool-owned sessions and marks interrupted jobs failed.
It refuses to overlap a live supervisor. If allocation lost its endpoint,
inspect `slots/*/lifecycle.log` and server assignments with the Colab CLI;
never stop unrelated assignments. Re-submit explicitly after inspecting logs.
This is cooperative exclusion on one host, not an OS security boundary or a
cross-host scheduler. Experiments from another computer need a shared broker
before using the same sessions.

For TorchCST preserve kernel normalization and sharply localized/one-hot
contributions. Compare at about 5% atoms and 1024²/8192² using an independent
oracle and the complete forward/backward/optimizer step. Record batch size,
kernel/chart definitions, dtype, updated support behavior, all atom gradients,
preparation cost and allocated peak memory. Do not attribute a model change
or reduced correctness requirements to implementation speedup.
