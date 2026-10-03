#!/usr/bin/env python3
"""Write the compact, reproducible MockGF C.8 benchmark report."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from gfcompress.benchmark import isolated_peak_rss, synthetic_report, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/synthetic.json"))
    parser.add_argument("--n-side", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run = partial(synthetic_report, n_side=args.n_side, seed=args.seed)
    report, peak_rss = isolated_peak_rss(run)
    storage = report["storage"]
    storage["peak_rss_bytes"] = peak_rss
    dense_bytes = storage["dense_reference_bytes"]
    storage["rss_ratio"] = peak_rss / dense_bytes if dense_bytes else None
    write_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
