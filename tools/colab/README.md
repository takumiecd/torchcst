# Colab execution tools

**English** | [日本語](README.ja.md)

These public repository tools explicitly select a Colab **T4 / L4 / A100 / H100 /
G4**, freeze and transfer source, execute it, retrieve results with hashes, and
stop owned runtimes. Supported accelerator names were checked against the
[Google Colab CLI](https://github.com/googlecolab/google-colab-cli) and the
installed CLI 0.6.0's `colab new --help`. GPU availability depends on your Colab
subscription, remaining quota and capacity. Unavailable allocations fail; another
GPU is never substituted.

## Initial setup

Use macOS or Linux, Python 3.10+, and Google's Colab CLI. Follow Google's
authentication instructions with your own account. CLI 0.6.0 has been validated;
the tools do not automatically update, log in or incur charges. Check the
commands and flags again when changing CLI versions.

```bash
uv tool install google-colab-cli==0.6.0
colab --auth oauth2 whoami
python3 tools/colab/scripts/pool.py configure --account YOUR_ACCOUNT_EMAIL
```

Complete the `whoami` authentication flow and verify the expected account.
Follow Google's instructions for any required OAuth client configuration. At
`serve` startup, the tools compare the authenticated account with the configured
email. They allocate no GPU if the account is missing or different. Do not embed
emails or tokens in source. Settings are stored in the host-wide queue's
`settings.json`; OAuth credentials remain in the Colab CLI's local configuration.
These local measurement tools do not need database connection credentials.

## Execute measurement JSON

Commit the [measurement JSON](../../benchmarks/requests/colab-linear.json) and
run from the repository root:

```bash
PYTHONPATH=src python -m benchmarks.automation run \
  --request benchmarks/requests/colab-linear.json --output output/colab-run
```

Specify GPUs in the JSON's `targets`, for example `["A100"]` or `["L4", "G4"]`.
Do not override the Case, GPU or repetition count through a separate selection
interface. Freeze the request and inspect the commit and all source before
execution; users should not have to transfer source manually. Use the existing
benchmark to check correctness and measure time and complete-step memory. Record
the actual GPU name, compute capability, SM count and memory capacity. Check
kernel compatibility and numerical contracts on each GPU at execution time;
performance on unvalidated devices is not guaranteed.

## Shared queue and ownership

All GPUs and local chats share one queue. Preserve existing operational state
by continuing to use `~/.local/state/colab-l4-pool`. Do not create a live
`--state-root` per GPU or agent. Only one supervisor may run on a host; another
computer must not operate the same VM.

`serve --gpu L4` processes only L4 jobs; `serve --gpu G4` processes only G4 jobs.
Existing L4 clients cannot accidentally allocate jobs for a more expensive GPU.
The normal measurement CLI processes one device at a time for each requested
GPU and verifies stopping. Wait for an existing supervisor without taking its
ownership.

```bash
python3 tools/colab/scripts/pool.py status
python3 tools/colab/scripts/pool.py serve --gpu A100 --workers 1 --idle-seconds 0
```

Direct research submission is supported; use request-bound automation for
formal measurement records.

```bash
python3 tools/colab/scripts/pool.py submit --gpu H100 \
  --source /absolute/checkout --script /absolute/driver.py --timeout 600
```

Interrupting a wait does not cancel work or release a VM. Experiments and
recovery are never automatically retried. Quarantine slots with uncertain
execution or stopping and stop further allocations. Follow the ownership and
recovery rules in the [pool operating guide](../colab-l4-pool/references/operations.ja.md).
The legacy `tools/colab-l4-pool/scripts/pool.py` entry point calls the same
implementation and queue. Installing a skill is not required to use these
public tools.

Explicitly select GPUs with higher resource consumption, such as G4, through
short measurement requests. Do not reinterpret these accelerator names as GPUs
Colab does not offer, such as 4090 or 5090. Other providers connect to the
measurement contract through separate adapters.

For long-job proxy renewal and job limits per VM, see the
[result transport guide](transport.md) ([日本語](transport.ja.md)). Default CLI
transport is unchanged; SDK retrieval and cleanup require an explicit opt-in.
