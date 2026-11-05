# Result transport and proxy expiry

**English** | [日本語](transport.ja.md)

With `serve --fresh-proxy-transport`, the host pool retrieves completed result
archives and performs verified-artifact cleanup using the SDK installed with the
configured Colab CLI. The default keeps the existing CLI transport. The configured
CLI must be a Python launcher with an executable absolute Python shebang and
`from colab_cli.cli import main`; other launchers fail closed. This does not
install or update the CLI. Upload and training execution retain their existing
CLI path. Experiment source, arithmetic, checkpoints and scientific identities
are unchanged.

For each download and cleanup, an isolated host process reads the existing
session and OAuth credentials, verifies the configured account through Google
userinfo, and lists assignments. Exactly one assignment must match the
supervisor's recorded endpoint and accelerator. The fresh proxy must retain the
cached HTTPS host and path. Changed, absent or ambiguous endpoints fail closed.
Proxy credentials and OAuth refreshes remain in memory: the adapter never opens
a StateStore, saves credentials, synchronizes sessions, allocates a VM, changes
slot/database state, or retries an experiment. Credential-bearing SDK output and
exceptions are suppressed; transport failures expose only a fixed diagnostic.
The existing supervisor remains responsible for normal status/lifecycle changes.

Downloads have a 900-second process ceiling, with bounded HTTP requests, and
become the result archive only after successful transfer. Existing archive,
source and file-manifest verification still precedes the normal receipt and job
status update. Before any cleanup network access, the helper independently
checks the receipt's slot, session and archive hash, the retained result
manifest's complete inventory and hashes, and the result/source identities. It
executes only the pool-generated removal of this job's remote tree and archives,
using existing kernel/session IDs, without persistence callbacks. A missing
acknowledgment or uncertain cleanup stops dispatch and retains verified local
results. No uncertain code is replayed.

To avoid starting another upload or training execution with an old cached proxy,
the supervisor can explicitly limit jobs per VM:

```bash
python3 tools/colab/scripts/pool.py serve --workers 1 --idle-seconds 0 \
  --max-jobs-per-session 1 --fresh-proxy-transport
```

The default `0` preserves session reuse. A positive limit stops each owned VM and
verifies its endpoint has disappeared from server assignments before another VM
is allocated for the next queued job. Failure to verify stopping quarantines the
slot and leaves later jobs queued. The worker count and shared queue do not
change. This transport change does not prove that an existing training WebSocket
survives an arbitrarily long job; a live long-duration gate remains separate.
