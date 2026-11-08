# Wider-amplitude envelope cohort

**English** | [日本語](radius-envelope-amplitude-range-20261109.ja.md)

An initial radius_fraction of1 sets every radius to50; fraction.5 has a minimum
radius of5 when L=.5,U=50. With8192 Atoms and amplitude cap1, the existing bounded
initializer cannot reach the FFN-up target variance.0004 in the .5 or1 examples.
The cap1/fraction1 down target.00005 also fails. This is an initialization
feasibility failure, not a failed learning run. Fraction.25 remains feasible.

The user approved a common wider amplitude range while preserving the initial
Dense moment targets. Add a separate explicit
`RADIUS_ENVELOPE_WIDE_AMPLITUDE_LM` / `RadiusEnvelopeWideAmplitudeLMAlgorithm`
with amplitude_max<=32; old reference/cached Plans remain limited to1. The map,
physical CUDA arithmetic, endpoints, all four gradients, optimizer field units
and public defaults do not change. Preparation caches only immutable scalars;
Parameters and autograd saves remain per-call. This is not an automatic dispatcher
adoption or a performance improvement claim.

At seed123 in the public initializer, cap32 reaches all up/down targets at
fractions0,.25,.5,1. Up amplitude RMS is approximately.264,.395,1.156,10.029;
down.094,.140,.409,3.562. Relative moment errors stay below.005. These are
expected operator-energy targets, not exact per-example output RMS equalities;
actual model activation RMS must also be recorded. Only changing fraction while
keeping cap1 would mix initialization-scale failure with radius dynamics.

GPU qualification must cover encoded-row FP64 Y/dX/all-field references at
|a|<=32, both directions, rows7/256, radius caps4.7/50, retained backward and
serialized optimizer continuation. Keep the existing atol1e-4/rtol4e-3 gate.
Native support remains width512/expansion4,A8192,R1..256,FP32,TF32off,eager M8.
Other cohorts, Graph execution and a public default require separate evidence.
