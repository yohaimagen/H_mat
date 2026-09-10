# C.4 synthetic oversampling record

Recorded 2026-09-10 after replacing traversal-counter sketch seeds with the
stable `(base seed, level, side, dyadic cell coordinates)` stream key. Updated
after adding explicit signed arbitrary-integer cell-coordinate word boundaries.

Configuration: smooth `MockGF` on a 16 by 16 unit-spaced 2D patch grid;
`m=4`, `k=4`, fixed sampling, construction seeds `0`, `1`, and `2`.
Operator-wide relative errors were estimated through the power-method
`relative_error` utility with independent validation seeds `1000`, `1001`, and
`1002`, respectively. No dense compressor matrix was assembled.

| construction seed | explicit `p=0` | default `p=10` |
| --- | ---: | ---: |
| 0 | 4.95964014e-06 | 1.93146364e-08 |
| 1 | 1.62554053e-06 | 2.45065787e-08 |
| 2 | 2.69610256e-06 | 2.17693349e-08 |

This is a reproducible diagnostic, not a per-seed monotonicity guarantee for
randomized compression. `p=0` remains an explicit supported experiment;
public compression and fixed-pattern sketch defaults now use `p=10`.
