# Initial compact-view proposal (superseded)

**Correction (2026-10-07):** the ownership premise below was wrong. Parameter VJP
reuses the full13 ordered forward snapshot; canonical preparation is transient.
GPU gradient checks rejected both-directions11. The corrected experiment keeps
forward/VJP13 and reduces only dX to11, saving8*A bytes. See
[the implementation and failure record](20261007-compact-views.md). The original
proposal is preserved below as an unvalidated historical hypothesis.

Source inspection confirms parameter VJP consumes the canonical `packed` tensor with 13 fields, while forward and dX consume separate ordered `views[0/1]` snapshots. Full metadata contains amplitude, inverse width squared, two centers, two norms, two L2 normalizer center-derivative coefficients, flags and four support bounds.

An opt-in 11-field physical view could omit only the two normalizer-derivative fields from the forward/dX copies, while preserving canonical 13-field parameter metadata and all position derivatives. The ordered field map would be canonical [0,1,2,3,4,5,8,9,10,11,12]. This requires explicit flag/bound field mapping in consumer/owner kernels, immutable snapshots, and tests of source-to-view coverage, sliced/empty/singleton/multiple support and repeated captured optimizer updates.

Two directions would reduce logical view payload by 16*A bytes: 3,264 B for A204 and 13,104 B for A819. Eleven rather than thirteen fields would reduce logical copied fields by 15.4%. Allocator rounding, runtime improvement and physical DRAM traffic are unmeasured; these numbers are a storage calculation, not a benchmark. Other routes keep their current layout. The source-width VJP contract must stay unchanged.

This is a future proposal because the remaining overnight time is reserved for repeatability checks, G4 follow-up and safe result retrieval. No new recipe, runtime change or speed claim has been made for this proposal.

## G4 follow-up: number of thread blocks (not yet tuned)

For the measured owner16/batch16/owner-split4 recipes, the forward/dX grid is ceil(B/16) * ceil(N/16) * 4: 32 CTAs at N64/B32 and64 at N128/B32. Parameter VJP is ceil(A/parameter_atom_block) * batch_partitions:26 CTAs at N64/A204/atom16/split2,52 at N128/A819/atom32/split2,104 for the N128 atom16 control. The observed G4 has188 SMs; these small grids cannot occupy every SM simultaneously in those individual kernels. These are launch-grid counts read from executor.py, not measured achieved occupancy or evidence of the dominant bottleneck.

A useful next controlled experiment is increasing the parameter batch partitions or owner partitions specifically on G4, while measuring complete-step time, partial-buffer/reduction cost and all source gradients. Increasing partitions can add padding/work and scratch memory; the existing serial-loop and fewer-owner results show why it requires a paired measurement. This has not been implemented or timed in the overnight run.
