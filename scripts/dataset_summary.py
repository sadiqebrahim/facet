#!/usr/bin/env python
"""Print SCUT-FBP5500 facts, recomputed from the raw data.

The numbers in docs/DATASETS.md are claims. This regenerates them from the files so the
documentation can be checked rather than trusted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.data.scut_fbp5500 import ScutFbp5500  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(ROOT / "data/raw/SCUT-FBP5500_v2"))
    args = ap.parse_args()

    ds = ScutFbp5500(args.data_root)
    print(json.dumps(ds.summary(), indent=2))

    lab = ds.labels
    print("\nrater disagreement (per-image std of the 60 ratings):")
    print(f"  paper claims most values fall in [0.6, 0.7]")
    print(f"  measured mean   = {lab['std'].mean():.4f}")
    print(f"  measured median = {lab['std'].median():.4f}")
    print(f"  fraction in [0.6, 0.7] = {lab['std'].between(0.6, 0.7).mean():.3f}")

    print("\nmean score by subgroup:")
    for g, v in lab.groupby("subgroup")["mean"].agg(["count", "mean", "std"]).iterrows():
        print(f"  {g}: n={int(v['count']):<5} mean={v['mean']:.3f} std={v['std']:.3f}")

    _, valid = ds.all_landmarks(ds.filenames)
    bad = [n for n, v in zip(ds.filenames, valid) if not v]
    print(f"\nunusable landmark files: {len(bad)} {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
