"""CLI wrapper for the multi-seed sweep — the reported training path.

The logic lives in src/training/multiseed.py (importable and testable); this file is only
the command-line face of it.

Run:  PYTHONPATH=. ./env/bin/python tools/multiseed.py --seeds 5
      PYTHONPATH=. ./env/bin/python tools/multiseed.py --seeds 5 --families xgb crnn
"""
from __future__ import annotations

import argparse

from src.training import DEFAULT_FAMILIES, run_multiseed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds per sweep point")
    ap.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES)
    args = ap.parse_args()
    run_multiseed(list(range(args.seeds)), args.families)


if __name__ == "__main__":
    main()
