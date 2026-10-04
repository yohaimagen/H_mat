#!/usr/bin/env python3
"""Write C.9's compact fixed-path improvement measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

from gfcompress.benchmark import c9_synthetic_report, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/c9_fixed_improvements.json")
    )
    parser.add_argument("--n-side", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    write_report(args.output, c9_synthetic_report(n_side=args.n_side, seed=args.seed))
    print(args.output)


if __name__ == "__main__":
    main()
