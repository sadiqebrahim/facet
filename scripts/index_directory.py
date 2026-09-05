#!/usr/bin/env python
"""Index a directory of images: discover, detect, assess quality, align, encode, store.

The expensive half of the pipeline (docs/RESEARCH.md 15.1), and the only stage that touches
pixels. Re-running on an unchanged directory does no work: discovery compares stat data
first, so an incremental run costs a directory walk.

Usage:
    python scripts/index_directory.py PATH [PATH ...] --index facet.db --features feats/
    python scripts/index_directory.py PATH --limit 1000        # partial run, resumable
    python scripts/index_directory.py PATH --force             # re-index everything
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.pipeline.indexer import IndexConfig, Indexer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--index", default=str(ROOT / "artifacts/index/facet.db"))
    ap.add_argument("--features", default=str(ROOT / "artifacts/index/features"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = IndexConfig(batch_size=args.batch_size, clip=not args.no_clip,
                      use_gpu=not args.cpu)
    ix = Indexer(args.index, args.features, cfg)
    print(f"encoder: {ix.encoder_version}\ncrop:    {cfg.crop_version}\n"
          f"detector:{ix.detector.version}\n")
    st = ix.index_directories(args.roots, force=args.force, limit=args.limit)
    print("\n" + json.dumps(
        {k: v for k, v in st.__dict__.items() if k != "errors"}, indent=2, default=str))
    if st.errors:
        print(f"\nfirst {len(st.errors)} failures:")
        for e in st.errors[:10]:
            print(f"  {e['path']}: {e['error']}")
    print(f"\nindex: {ix.index.stats()}")
    ix.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
