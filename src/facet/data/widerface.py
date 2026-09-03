"""WIDER FACE loader - detection evaluation with per-face attributes (experiment E8).

The value of this set for us is not the leaderboard; it is that every ground-truth box
carries blur / expression / illumination / occlusion / pose flags. That converts "what is
our recall" into "WHAT do we miss", which is the question that decides whether a directory
scanner silently loses results (docs/RESEARCH.md 13.2: a missed face is an invisible failure,
because the user cannot tell "no matching faces" from "the detector never saw them").

Annotation format, one block per image:
    <relative path>
    <n faces>
    x1 y1 w h blur expression illumination invalid occlusion pose

License: CC BY-NC-ND 4.0 - non-commercial, no derivatives. Evaluation only.
Cite: Yang, Luo, Loy, Tang. WIDER FACE. CVPR 2016.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

BLUR = {0: "clear", 1: "normal", 2: "heavy"}
OCCLUSION = {0: "none", 1: "partial", 2: "heavy"}
POSE = {0: "typical", 1: "atypical"}
ILLUMINATION = {0: "normal", 1: "extreme"}

#: WIDER FACE's own difficulty tiers are defined by detectability; we approximate the
#: distinction with face height in pixels, which is the dominant factor and is what a
#: product actually cares about (a 12-pixel face is not a useful search result).
SIZE_BUCKETS = ((0, 16, "tiny <16px"), (16, 32, "small 16-32"), (32, 64, "medium 32-64"),
                (64, 128, "large 64-128"), (128, 10**9, "huge >=128"))


@dataclass
class ImageAnns:
    path: str
    boxes: np.ndarray      # (N, 4) x1 y1 x2 y2
    attrs: np.ndarray      # (N, 6) blur expression illumination invalid occlusion pose


def size_bucket(h: float) -> str:
    for lo, hi, name in SIZE_BUCKETS:
        if lo <= h < hi:
            return name
    return SIZE_BUCKETS[-1][2]


class WiderFace:
    def __init__(self, root: str | Path, split: str = "val"):
        self.root = Path(root)
        self.images_dir = self.root / f"WIDER_{split}" / "images"
        self.gt_file = self.root / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"{self.images_dir} not found - run scripts/download_wider_face.py"
            )

    @cached_property
    def annotations(self) -> list[ImageAnns]:
        lines = self.gt_file.read_text().splitlines()
        out: list[ImageAnns] = []
        i = 0
        while i < len(lines):
            path = lines[i].strip()
            if not path:
                i += 1
                continue
            n = int(lines[i + 1].strip())
            i += 2
            boxes, attrs = [], []
            # A count of 0 is still followed by one filler line in the official file.
            for k in range(max(n, 1)):
                parts = [int(float(v)) for v in lines[i + k].split()]
                if n == 0:
                    continue
                x, y, w, h = parts[:4]
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])
                attrs.append(parts[4:10])
            i += max(n, 1)
            out.append(ImageAnns(
                path=path,
                boxes=np.array(boxes, dtype=np.float64).reshape(-1, 4),
                attrs=np.array(attrs, dtype=np.int64).reshape(-1, 6),
            ))
        return out

    def image_path(self, rel: str) -> Path:
        return self.images_dir / rel

    def summary(self) -> dict:
        anns = self.annotations
        boxes = np.concatenate([a.boxes for a in anns if len(a.boxes)])
        attrs = np.concatenate([a.attrs for a in anns if len(a.attrs)])
        heights = boxes[:, 3] - boxes[:, 1]
        valid = attrs[:, 3] == 0
        buckets: dict[str, int] = {}
        for h in heights:
            buckets[size_bucket(h)] = buckets.get(size_bucket(h), 0) + 1
        return {
            "n_images": len(anns),
            "n_faces": int(len(boxes)),
            "n_valid_faces": int(valid.sum()),
            "median_face_height_px": float(np.median(heights)),
            "faces_under_32px": float((heights < 32).mean()),
            "size_buckets": buckets,
            "blur": {BLUR[k]: int((attrs[:, 0] == k).sum()) for k in BLUR},
            "occlusion": {OCCLUSION[k]: int((attrs[:, 4] == k).sum()) for k in OCCLUSION},
            "pose": {POSE[k]: int((attrs[:, 5] == k).sum()) for k in POSE},
        }


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) boxes in x1y1x2y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def match(gt: np.ndarray, det: np.ndarray, det_scores: np.ndarray, thr: float = 0.5):
    """Greedy highest-score-first matching. Returns a boolean 'was this GT box found'."""
    found = np.zeros(len(gt), dtype=bool)
    if len(det) == 0 or len(gt) == 0:
        return found
    order = np.argsort(-det_scores)
    ious = iou_matrix(det[order], gt)
    taken = set()
    for di in range(len(order)):
        row = ious[di].copy()
        for t in taken:
            row[t] = -1
        gi = int(np.argmax(row))
        if row[gi] >= thr:
            found[gi] = True
            taken.add(gi)
    return found
