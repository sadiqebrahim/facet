#!/usr/bin/env python
"""Download SCUT-FBP5500 v2 from the authors' official release.

LICENSE: SCUT-FBP5500 is NON-COMMERCIAL RESEARCH USE ONLY. By running this you accept
those terms. The dataset is never redistributed by this repository - only downloaded
from the authors' own link. See docs/LICENSING.md.

Cite: Liang, Lin, Jin, Xie, Li. SCUT-FBP5500: A Diverse Benchmark Dataset for
Multi-Paradigm Facial Beauty Prediction. ICPR 2018.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Official Google Drive link from github.com/HCIILAB/SCUT-FBP5500-Database-Release
DRIVE_ID = "1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"

BANNER = """
SCUT-FBP5500 is licensed for NON-COMMERCIAL RESEARCH USE ONLY.
Any model trained on it inherits that restriction and cannot be used commercially.
Continuing means you accept those terms.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(ROOT / "data/raw"))
    ap.add_argument("--yes", action="store_true", help="accept the license non-interactively")
    args = ap.parse_args()

    print(BANNER)
    if not args.yes:
        if input("Accept and download? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("aborted")
            return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "SCUT-FBP5500_v2" / "Images").is_dir():
        print(f"[skip] already present at {dest/'SCUT-FBP5500_v2'}")
        return 0

    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown is required:  pip install gdown", file=sys.stderr)
        return 1

    archive = dest / "SCUT-FBP5500_v2.zip"
    print(f"downloading to {archive} (~180 MB)...")
    subprocess.run([sys.executable, "-m", "gdown", DRIVE_ID, "-O", str(archive)], check=True)

    print("extracting...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    inner = dest / "SCUT-FBP5500_v2" / "train_test_files.zip"
    if inner.exists():
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(inner.parent)
        inner.unlink()
    archive.unlink()

    n = len(list((dest / "SCUT-FBP5500_v2" / "Images").glob("*.jpg")))
    print(f"[ok] {n} images at {dest/'SCUT-FBP5500_v2'} (expected 5500)")
    return 0 if n == 5500 else 1


if __name__ == "__main__":
    raise SystemExit(main())
