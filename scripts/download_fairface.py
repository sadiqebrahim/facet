#!/usr/bin/env python
"""Download the FairFace validation split and materialise it for the fairness audit (E11).

FairFace is balanced by construction across seven race groups, which is exactly what a bias
audit needs - an imbalanced audit set confounds "the model is worse on group X" with "there
are fewer examples of group X".

LICENSE: CC BY 4.0 - commercial use permitted, attribution required. This is the only face
dataset in the project without a non-commercial restriction (docs/LICENSING.md), which also
makes it the anchor for any commercial track.

Cite: Karkkainen & Joo, FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age.

Note the labels are annotator-PERCEIVED race, gender and age bucket, not self-reported
identity. They are used here only to slice our own error rates, never as prediction targets
or as a user-facing facet.

Usage:
    python scripts/download_fairface.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "HuggingFaceM4/FairFace"
# padding 1.25 keeps surrounding context, so detection recall is measured on something
# closer to a real photo than a pre-cropped thumbnail.
FILE = "1.25/validation-00000-of-00001-09e3e67bb00ab4ec.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(ROOT / "data/raw/FairFace"))
    args = ap.parse_args()

    dest = Path(args.dest)
    img_dir = dest / "val_images"
    if (dest / "labels.csv").exists():
        print(f"[skip] already present at {dest}")
        return 0
    img_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from huggingface_hub import hf_hub_download
    from PIL import Image

    print(f"downloading {FILE} from {REPO} (~237 MB)...")
    path = hf_hub_download(repo_id=REPO, filename=FILE, repo_type="dataset")
    df = pd.read_parquet(path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    rows = []
    for i, r in df.iterrows():
        img = r["image"]
        raw = img["bytes"] if isinstance(img, dict) else img
        name = f"{i:06d}.jpg"
        Image.open(io.BytesIO(raw)).convert("RGB").save(img_dir / name, quality=95)
        rows.append({
            "file": name,
            "age": r.get("age"),
            "gender": r.get("gender"),
            "race": r.get("race"),
        })
        if (i + 1) % 2000 == 0:
            print(f"  wrote {i+1}/{len(df)}")

    out = pd.DataFrame(rows)
    out.to_csv(dest / "labels.csv", index=False)
    (dest / "SOURCE.json").write_text(json.dumps({
        "repo": REPO, "file": FILE, "license": "CC BY 4.0",
        "citation": "Karkkainen & Joo, FairFace (WACV 2021)",
        "note": "annotator-perceived labels; used only for slicing our own error rates",
    }, indent=2))
    print(f"[ok] {len(out)} images -> {img_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
