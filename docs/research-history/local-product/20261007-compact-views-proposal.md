# Follow-up proposal: compact ordered physical views (not implemented)

Source inspection confirms parameter VJP consumes the canonical `packed` tensor with 13 fields, while forward and dX consume separate ordered `views[0/1]` snapshots. Full metadata contains amplitude, inverse width squared, two centers, two norms, two L2 normalizer center-derivative coefficients, flags and four support bounds.

An opt-in 11-field physical view could omit only the two normalizer-derivative fields from the forward/dX copies, while preserving canonical 13-field parameter metadata and all position derivatives. The ordered field map would be canonical [0,1,2,3,4,5,8,9,10,11,12]. This requires explicit flag/bound field mapping in consumer/owner kernels, immutable snapshots, and tests of source-to-view coverage, sliced/empty/singleton/multiple support and repeated captured optimizer updates.

Two directions would reduce logical view payload by 16*A bytes: 3,264 B for A204 and 13,104 B for A819. Eleven rather than thirteen fields would reduce logical copied fields by 15.4%. Allocator rounding, runtime improvement and physical DRAM traffic are unmeasured; these numbers are a storage calculation, not a benchmark. Other routes keep their current layout. The source-width VJP contract must stay unchanged.

This is a future proposal because the remaining overnight time is reserved for repeatability checks, G4 follow-up and safe result retrieval. No new recipe, runtime change or speed claim has been made for this proposal.
