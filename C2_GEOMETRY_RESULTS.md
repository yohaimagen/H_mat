# C.2 geometry measurements

Measured on 2026-09-09 with the read-only `real_gfs` symlink, NumPy float64,
and `.venv/bin/python`.  The command loaded each coordinate CSV with
`gfcompress.realgf._load_coords_csv`, applied `pca_align`, and built a
level-synchronous tree from the retained coordinates using unit patch lengths.
Leaf columns are `m`, leaf count, minimum/median/maximum occupancy, and the
number of leaves holding at most two patches.

| Dataset | Raw → retained tree dimensions | PCA variance ratios | m | Leaves | Occupancy min / median / max | ≤2 leaves |
| --- | --- | --- | ---: | ---: | --- | ---: |
| BP3 | 2 → 1 | 1.000e+00, 4.706e-30 | 64 | 256 | 53 / 55 / 56 | 0 |
| BP3 | 2 → 1 | 1.000e+00, 4.706e-30 | 128 | 128 | 108 / 110 / 111 | 0 |
| BP3 | 2 → 1 | 1.000e+00, 4.706e-30 | 256 | 64 | 218 / 219 / 220 | 0 |
| BP3 | 2 → 1 | 1.000e+00, 4.706e-30 | 512 | 32 | 436 / 437 / 439 | 0 |
| BP7 | 3 → 2 | 5.116e-01, 4.884e-01, 3.172e-64 | 64 | 764 | 1 / 4 / 28 | 321 |
| BP7 | 3 → 2 | 5.116e-01, 4.884e-01, 3.172e-64 | 128 | 252 | 1 / 9 / 94 | 41 |
| BP7 | 3 → 2 | 5.116e-01, 4.884e-01, 3.172e-64 | 256 | 252 | 1 / 9 / 94 | 41 |
| BP7 | 3 → 2 | 5.116e-01, 4.884e-01, 3.172e-64 | 512 | 64 | 7 / 39 / 280 | 0 |

The automatic singular cutoff was 4.247e-09 for BP3 and 1.452e-11 for BP7;
their equivalent variance-ratio cutoffs were 9.664e-24 and 7.685e-25.  The
discarded projection residuals were respectively 2.988e-12 (relative
2.187e-15) and 6.050e-15 (relative 3.654e-16).  The operator layout remains
unchanged: only its tree coordinates are rotated/reduced.
