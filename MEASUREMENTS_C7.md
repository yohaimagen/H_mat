# C.7 real-subset spectral diagnostics

These are deterministic diagnostics for the largest interaction-list block at
each compressed level of the representative 192-patch subsets. They are not a
claim that every admissible real-data block is low rank. Numerical rank uses a
relative singular-value threshold of `1e-6`; the rank-4 tail is the relative
Frobenius norm after the first four singular values.

| Dataset | Regions | `(width, numerical rank, rank-4 tail)` by level |
|---|---:|---|
| BP3 | 4 | `(49, 5, 1.09e-5)`, `(25, 5, 8.00e-6)`, `(13, 5, 2.51e-6)`, `(7, 4, 9.12e-7)` |
| BP3 | 6 | `(49, 5, 2.14e-5)`, `(25, 5, 4.26e-6)`, `(13, 5, 2.39e-6)`, `(8, 5, 1.31e-6)` |
| BP7 | 4 | `(30, 30, 0.0301)`, `(34, 34, 0.2776)`, `(10, 10, 0.00204)` |
| BP7 | 6 | `(39, 39, 0.0223)`, `(22, 22, 0.0716)` |

BP3 has strong observed decay for these probes. BP7 is different: all selected
blocks are full numerical rank at `1e-6`, and one selected level has a
substantial rank-4 tail of about `0.278`. Most BP7 selected tails still fall
below `0.1`; that is evidence of some spectral decay, not evidence that a
fixed rank four accurately represents every real far-field block. C.10's
multi-seed operator and input checks are therefore the acceptance evidence;
these spectra remain a diagnostic limitation.
