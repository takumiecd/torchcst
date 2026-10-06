# Count-free ordered cache with a 16-atom parameter tile

This combines PR #51’s exact cached ordering without diagnostic counters with PR #47’s 16-atom parameter tile, two batch partitions and four parameter warps. Eight CST controls/candidates and a separately initialized dense baseline cover N64/N128 and initial rho 1.25, 3, 8 and mixed; every decoded initial rho is strictly greater than one.

The runtime is unchanged from PR #51. Ordering repairs still use exact neighbor checks and full-sort fallback. Disabling counters removes their allocation and writes, not correctness checks. Position gradients, full-domain discrete L2 normalization before slicing, dX and all source-parameter gradients remain part of the contract; source-width VJP follows the existing fixed-decoded-width contract while optimizer steps change forward widths.

Host validation: 887 passed, 877 CUDA skips in 22.65 seconds, eight kernel comparison preparations/checks, wheel and source build. GPU correctness and performance are pending. No automatic dispatch or performance claim is introduced.
