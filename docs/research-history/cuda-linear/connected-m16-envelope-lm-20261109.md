# Connected M16 envelope LM opt-in

**English** | [日本語](connected-m16-envelope-lm-20261109.ja.md)

This separate Algorithm composes the existing differentiable radius-envelope map
with the qualified contiguous-row physical M16 Algorithm. Revision10 raw rows
remain `[amplitude, radius_fraction, output_center, input_center]`; the optimizer
updates those raw rows, not the decoded physical radius. Both the envelope's
amplitude-to-radius and fraction-to-radius derivatives remain live. Immutable
map constants and the revision9 physical declaration are prepared; trainable rows
are decoded every call. The existing small FP64 envelope map is unchanged; the
M16 execution and saved geometry/moments remain FP32.

Admission is metadata-only: default connected M16, radius within [.5,50],
amplitude cap<=32, and the physical Algorithm's eager FP32/B1..256/first-order
cohort. This does not broaden existing Algorithms or production dispatch.
The consumer must explicitly register `ConnectedEnvelopeAlgorithm` and select
`CONNECTED_ENVELOPE`. Connected declarations retain revision10 checkpoint guards.

Qualification requires independent rectangular quadrature with the full raw map
chain, actual FFN dimensions/Atom counts, task-model serialized resume including
Adam moments/counters, and observational hooks that preserve the next update.
Short LM quality and geometry diagnostics are a separate cst-research study;
no recommended recipe or public default is adopted here.
