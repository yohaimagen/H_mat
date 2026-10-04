# C.10 fixed-path operating-point measurements

All figures below are from the 192-patch representative subsets, every one of
three construction seeds and two independent validation starts. Storage is
retained numerical bytes; columns are forward plus transpose construction
columns. Apply figures are median warm single-RHS seconds across the six
records and are noisy at these small sizes.

| Dataset | Choice `(m,k,p)` | Worst global error | Bytes | Columns | Forward / adjoint apply | Decision |
|---|---|---:|---:|---:|---:|---|
| BP3 | primary `(8,4,2)` | `2.90e-10` | 223,104 | 285 | 0.651 / 0.660 ms | Accuracy margin dominates; apply times are tied within noise. |
| BP3 | alternative `(8,3,2)` | `7.26e-9` | 180,792 | 241 | 0.662 / 0.657 ms | Smaller storage/sample cost, weaker accuracy. |
| BP3 | alternative `(8,4,1)` | `1.53e-8` | 223,104 | 241 | 0.649 / 0.667 ms | Fewer columns only; weaker accuracy. |
| BP7 | primary `(8,4,2)` | `5.26e-5` | 1,240,744 | 1,146 | 8.527 / 8.409 ms | Best accuracy and leaves-only margin; timing advantage is not claimed. |
| BP7 | alternative `(8,4,1)` | `6.21e-5` | 1,240,744 | 970 | 8.181 / 8.186 ms | Fewer columns and slightly faster in this subset timing; weaker component margin. |
| BP7 | alternative `(8,3,2)` | `9.89e-5` | 1,035,024 | 970 | 8.038 / 7.947 ms | Lowest storage/sample cost; least accuracy headroom. |

The BP3 selected-block criterion is `1e-5`: the measured primary maximum is
`6.02e-6`, while a zero selected-block output would have relative-with-floor
error at least `8.86e-5` because every selected reference response exceeds
that value and the frozen response floor is 1.0. It therefore has measured
margin without allowing the zero far-field output to qualify.

These are subset results and not a full-baseline speedup claim. The frozen
criteria and the `0.1` coloring comparison slack are carried in the dataset
configuration files; full runs remain opt-in.
