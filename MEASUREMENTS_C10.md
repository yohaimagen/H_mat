# C.10 fixed-path operating-point measurements

All figures below are from the 192-patch representative subsets, every one of
three construction seeds and two independent validation starts. Storage is
retained numerical bytes; columns are forward plus transpose construction
columns. Apply figures are median warm single-RHS seconds across the six
records and are noisy at these small sizes.

| Dataset | Choice `(m,k,p)` | Worst global error | Bytes | Columns | Forward / adjoint apply | Decision |
|---|---|---:|---:|---:|---:|---|
| BP3 | primary `(8,4,2)` | `2.90e-10` | 223,104 | 285 | 0.644 / 0.640 ms | Accuracy margin dominates; apply times are tied within noise. |
| BP3 | alternative `(8,3,2)` | `7.26e-9` | 180,792 | 241 | 0.639 / 0.639 ms | Smaller storage/sample cost, weaker accuracy. |
| BP3 | alternative `(8,4,1)` | `1.53e-8` | 223,104 | 241 | 0.644 / 0.650 ms | Fewer columns only; weaker accuracy. |
| BP7 | primary `(8,6,2)` | `1.32e-5` | 1,573,496 | 1,498 | 7.842 / 7.780 ms | Best accuracy and passes the genuinely truncated selected-block gate; timing advantage is not claimed. |
| BP7 | alternative `(8,4,2)` | `5.26e-5` | 1,240,744 | 1,146 | 7.840 / 7.732 ms | Lower storage/sample cost, but fails the genuinely truncated selected-block gate. |
| BP7 | alternative `(8,3,2)` | `9.89e-5` | 1,035,024 | 970 | 7.411 / 7.427 ms | Lowest storage/sample cost; least accuracy headroom. |

Selected blocks are the largest genuinely truncated admissible block at every
compressed level, chosen with a deterministic geometric tie-break and driven
by a unit source-box vector. Their criterion is relative to that block
response (not the general near-zero response floor), so a zero selected-block
output has error exactly `1.0`. BP3's primary maximum is `1.89e-4`, frozen at
`3e-4`; BP7 rank 6's maximum is `0.294`, frozen at `0.4`. Both therefore have
measured margin while rejecting zero output. BP7 rank 4 reaches `1.35` and is
kept only as a lower-cost alternative that does not meet this gate.

These are subset results and not a full-baseline speedup claim. The frozen
criteria and the `0.1` coloring comparison slack are carried in the dataset
configuration files; full runs remain opt-in.
