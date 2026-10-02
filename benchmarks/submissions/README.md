# Submission intake

GPU-provider-neutral admission of completed benchmark JSON through issues in the
same repository. This is tooling, not a runtime dependency of `torchcst`.

| Module | Responsibility |
| --- | --- |
| `policy.py` | Public settings, GitHub numeric identity, general/recognized limits and manually assigned trust points |
| `github.py` | Authenticated issue metadata; bounded public attachment downloads without forwarded credentials |
| `__main__.py` | Local preflight, fetch/validation before DB access, transactional admission and issue feedback |

[`Database.import_submission`](../database/postgres.py) atomically stores immutable
observations, the submission's author/login, policy snapshot, tier and points.
It serializes per-user quota checks and global run deduplication with transaction
advisory locks. UTC admission time comes from PostgreSQL, not user data.
The runner artifact remains byte-for-byte restorable. Admission metadata is kept
outside untrusted JSON and outside immutable historical run provenance.

Default: general accounts need no registration, have 10 new runs/UTC day and 1
trust point. Recognized IDs have no daily count cap and default to 10 points.
Individual point overrides are manual. No scoring, vote, automatic promotion or
dispatcher selection algorithm is added.

One issue holds one completed run, not one Plan or one timing sample. A fresh UUID
is a fresh observation; retries, copied uploads, whitespace and key order do not
create another observation or another submitter's credit. Same UUID with changed
data is rejected. Already accepted issue content cannot replace the stored run.

Only JSON is parsed. Contributor archives/scripts/workflows are never executed.
Uploaded points, identity, certification and role are not used as authority.
`consistency_checked` is internal validation; hardware and measurement authenticity
remain `self_reported`. Raw failures are preserved without performance metrics.
New artifact kinds need an explicit adapter.

[Participation and deployment](../../docs/benchmark-contributions.ja.md) cover the
issue form, public settings, schema migration 2, reader/writer permissions and
the optional Colab measurement helper. Source/configuration is Git-managed;
benchmark output and operational data are not committed.

[Verification](../../docs/issue-submission-verification.ja.md) records the real
PostgreSQL quota/race/migration tests and the remaining production activation.
