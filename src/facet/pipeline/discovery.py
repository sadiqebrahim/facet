"""Image discovery and change detection.

Phase 8 asks for very large directories, incremental indexing, corrupt-file handling and
duplicate detection. The cheap-to-expensive ordering here is deliberate:

    stat (mtime, size)  ->  content hash  ->  decode  ->  detect/encode

`stat` alone decides the common case - re-running on an unchanged directory should touch no
pixels at all. Content hashing runs only for files that look new or changed, and catches
files that were moved or copied rather than edited. Decoding is where corruption surfaces,
and a corrupt file becomes a recorded row rather than a crash or a silent omission.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif",
}

#: Files smaller than this are almost certainly not photographs (favicons, spacers,
#: truncated downloads). Recorded as skipped rather than attempted.
MIN_BYTES = 1024


@dataclass
class Candidate:
    path: str
    size_bytes: int
    mtime: float


@dataclass
class ScanPlan:
    """What a run intends to do, computed before any expensive work starts."""

    new: list[Candidate]
    changed: list[Candidate]
    unchanged: list[str]
    too_small: list[str]
    missing: list[str]        # in the index but no longer on disk

    @property
    def to_process(self) -> list[Candidate]:
        return self.new + self.changed

    def summary(self) -> dict:
        return {
            "new": len(self.new), "changed": len(self.changed),
            "unchanged": len(self.unchanged), "too_small": len(self.too_small),
            "missing": len(self.missing),
        }


def walk_images(roots: list[str | Path], follow_symlinks: bool = False) -> Iterator[Candidate]:
    """Yield image candidates under one or more roots.

    Uses os.scandir rather than Path.rglob: it returns stat data from the directory entry,
    which avoids a second syscall per file and matters at directory sizes where this stage
    would otherwise dominate.
    """
    seen: set[str] = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if root.is_file():
            stack = [root.parent]
        elif not root.is_dir():
            continue
        else:
            stack = [root]
        while stack:
            d = stack.pop()
            try:
                entries = list(os.scandir(d))
            except (PermissionError, OSError):
                continue
            for e in entries:
                try:
                    if e.is_dir(follow_symlinks=follow_symlinks):
                        stack.append(Path(e.path))
                        continue
                    if not e.is_file(follow_symlinks=follow_symlinks):
                        continue
                    if Path(e.name).suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    rp = str(Path(e.path).resolve())
                    if rp in seen:      # overlapping roots must not double-index
                        continue
                    seen.add(rp)
                    st = e.stat(follow_symlinks=follow_symlinks)
                    yield Candidate(rp, st.st_size, st.st_mtime)
                except OSError:
                    continue


def plan_scan(roots, fingerprints: dict[str, tuple[float, int, str]],
              force: bool = False) -> ScanPlan:
    """Decide what needs work, using only stat data."""
    new, changed, unchanged, too_small = [], [], [], []
    found: set[str] = set()
    for c in walk_images(roots):
        found.add(c.path)
        if c.size_bytes < MIN_BYTES:
            too_small.append(c.path)
            continue
        prev = fingerprints.get(c.path)
        if prev is None:
            new.append(c)
        elif force or prev[0] != c.mtime or prev[1] != c.size_bytes:
            changed.append(c)
        else:
            unchanged.append(c.path)
    missing = [p for p in fingerprints if p not in found and p not in too_small]
    return ScanPlan(new, changed, unchanged, too_small, missing)


def content_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()[:32]


def load_image(path: str | Path):
    """Decode to BGR uint8, or return (None, reason).

    Corruption is common in real directories and must not stop a run: truncated JPEGs,
    mislabelled extensions, zero-byte files and images too large to be worth decoding all
    become recorded statuses.
    """
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None, "empty file"
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None, "decode failed"
        if img.ndim != 3 or img.shape[2] != 3:
            return None, f"unexpected shape {img.shape}"
        return img, None
    except Exception as e:  # noqa: BLE001 - any decode failure is a recorded status
        return None, f"{type(e).__name__}: {e}"
