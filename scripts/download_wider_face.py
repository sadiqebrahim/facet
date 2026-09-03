#!/usr/bin/env python
"""Download WIDER FACE validation split for detector evaluation (E8).

WIDER FACE is the standard face-detection benchmark. What makes it the right set for E8 is
not the leaderboard but the **per-face attribute annotations**: every box is labelled with
blur, expression, illumination, occlusion and pose. That turns "what is our recall" into
"what do we miss, and is it the small/blurred/occluded/profile faces" - which is the question
that decides whether a directory scanner silently loses results.

LICENSE: CC BY-NC-ND 4.0 - non-commercial, NO DERIVATIVES. We evaluate on it and never
redistribute or train on it. See docs/LICENSING.md.
Cite: Yang, Luo, Loy, Tang. WIDER FACE: A Face Detection Benchmark. CVPR 2016.

Usage:
    python scripts/download_wider_face.py
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "CUHK-CSE/wider_face"
FILES = ["data/WIDER_val.zip", "data/wider_face_split.zip"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(ROOT / "data/raw/WIDERFACE"))
    args = ap.parse_args()

    dest = Path(args.dest)
    if (dest / "WIDER_val" / "images").is_dir():
        print(f"[skip] already present at {dest}")
        return 0
    dest.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    for f in FILES:
        print(f"downloading {f} ...")
        p = hf_hub_download(repo_id=REPO, filename=f, repo_type="dataset")
        print(f"  extracting {Path(f).name}")
        with zipfile.ZipFile(p) as z:
            z.extractall(dest)

    n = len(list((dest / "WIDER_val" / "images").rglob("*.jpg")))
    gt = dest / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    print(f"[ok] {n} images (expected 3226); annotations present: {gt.exists()}")
    return 0 if n > 0 and gt.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
