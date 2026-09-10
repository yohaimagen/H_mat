# C.4 synthetic oversampling record

Recorded 2026-09-10 after replacing traversal-counter sketch seeds with the
stable `(base seed, level, side, dyadic cell coordinates)` stream key.

Configuration: smooth `MockGF` on a 16 by 16 unit-spaced 2D patch grid;
`m=4`, `k=4`, fixed sampling, construction seeds `0`, `1`, and `2`.
Operator-wide relative errors were estimated through the power-method
`relative_error` utility with independent validation seeds `1000`, `1001`, and
`1002`, respectively. No dense compressor matrix was assembled.

| construction seed | explicit `p=0` | default `p=10` |
| --- | ---: | ---: |
| 0 | 3.65076758e-05 | 2.03169512e-08 |
| 1 | 3.25513306e-05 | 2.33523882e-08 |
| 2 | 1.82191798e-05 | 2.31417860e-08 |

This is a reproducible diagnostic, not a per-seed monotonicity guarantee for
randomized compression. `p=0` remains an explicit supported experiment;
public compression and fixed-pattern sketch defaults now use `p=10`.
