# Saved CUDA Linear measurements

This directory contains selected raw JSON, CSV, and text outputs cited by
[research notes](../../../../experiments/cuda/linear/notes/README.md) and the
[dispatch evidence ledger](../dispatch-evidence.ja.md). Each dated subdirectory
groups a measurement campaign; the associated note records its source commit,
GPU, comparison method, and limits. These files are evidence, not runtime
dispatch configuration.

New measurements start in ignored `output/`. Preserve only results used in a
decision, with a note and a reproducible command. Keep large traces outside
the repository and record their source and hash when needed.
