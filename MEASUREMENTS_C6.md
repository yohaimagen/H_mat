# C.6 synthetic accuracy record

Recorded 2026-10-01 on the fixed-pattern `MockGF` fixtures.  Construction
uses seed `0`; independent validation uses seed `1` for power estimates and
seeds `101` (2D) and `103` (3D) for normalized localized/smooth vectors.
The tables report the selected first admissible block at each compressed
level.  `rank gap` is `sigma_(k_eff+1) / sigma_1`; `block error` and `best`
are Frobenius errors divided by the block Frobenius norm.

| fixture | levels | k_eff | rank gap (coarse, fine) | block error (coarse, fine) | best rank-k_eff (coarse, fine) | global error | leaves-only baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D 32x32, m=16, k=10, p=10 | 2, 3 | 10 | 1.3679e-04, 4.0646e-05 | 6.0131e-04, 1.1887e-04 | 1.6477e-04, 4.1735e-05 | 6.3118e-10 | 7.2044e-06 |
| 3D 8x16x16, m=8, k=6, p=6 | 2, 3 | 6 | 4.2881e-02, 2.6409e-02 | 1.2354e-01, 8.1848e-02 | 5.3167e-02, 3.0334e-02 | 2.2335e-10 | 8.0511e-09 |

The global baseline is the dense near-field leaves with all far-field factors
removed.  It is deliberately retained because a small global error alone can
hide a missing far field.  The selected 3D blocks are only modestly
rank-decaying; their wider best-rank gap is why the block comparison allows a
measured factor above the optimum rather than claiming exact SVD recovery.

Exact-recovery boundary tests remain separate from these genuinely truncated
cases.  The power-method tests use small dense references solely to verify the
estimator against exact norms; compression quality itself is tested only with
the smooth `MockGF` kernel and black-box products.
