#!/usr/bin/env python3
"""Write the compact, reproducible MockGF C.8 benchmark report."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from gfcompress.benchmark import (
    isolated_peak_rss,
    synthetic_dense_only,
    synthetic_report,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/synthetic.json"))
    parser.add_argument("--n-side", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run = partial(synthetic_report, n_side=args.n_side, seed=args.seed)
    report, compressed_peak = isolated_peak_rss(run)
    _, dense_peak = isolated_peak_rss(synthetic_dense_only, args.n_side)
    storage = report["storage"]
    storage["compressed_peak_rss_bytes"] = compressed_peak
    storage["dense_peak_rss_bytes"] = dense_peak
    storage["rss_ratio"] = compressed_peak / dense_peak if dense_peak else None
    write_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
