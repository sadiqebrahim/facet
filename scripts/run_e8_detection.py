#!/usr/bin/env python
"""Experiment E8 - detector evaluation on real scenes.

E11 measured detection recall at 0.9998 with essentially no demographic disparity - but on
FairFace, whose images are pre-cropped and face-centred. That is close to a best case. This
runs the same detector on WIDER FACE, where the median annotated face is 20 pixels tall and
68% are under 32px, and asks the question that actually matters for a directory scanner:

    WHAT do we miss, and is it the small / blurred / occluded / profile faces?

docs/RESEARCH.md 13.2: a missed face is an INVISIBLE failure. The user cannot distinguish
"no matching faces in this directory" from "the detector never saw them", so detection recall
is a product metric, not an infrastructure metric.

It also tests a specific tension E5 created. E5 chose detector padding (pad_frac 0.25)
because SCRFD's recall on frame-filling portrait crops collapses to 46% without it. Padding
makes faces smaller relative to the frame - which is exactly the wrong direction for a scene
full of 20-pixel faces. Whether the portrait fix breaks the scene case is measured here, not
assumed.

Usage:
    python scripts/run_e8_detection.py
    python scripts/run_e8_detection.py --limit 800     # faster sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.data.widerface import (  # noqa: E402
    BLUR, ILLUMINATION, OCCLUSION, POSE, WiderFace, match, size_bucket,
)
from facet.models.insightface_backend import InsightFaceDetector  # noqa: E402
from facet.utils.run import RunManifest  # noqa: E402
from facet.utils.seed import seed_everything  # noqa: E402

#: (label, pack, det_size, pad_frac)
CONFIGS = [
    ("buffalo_l d640 pad0.00", "buffalo_l", 640, 0.00),
    ("buffalo_l d640 pad0.25", "buffalo_l", 640, 0.25),   # E5 production setting
    ("buffalo_l d1024 pad0.00", "buffalo_l", 1024, 0.00),
    ("buffalo_l d1024 pad0.25", "buffalo_l", 1024, 0.25),
    ("antelopev2 d1024 pad0.00", "antelopev2", 1024, 0.00),
]

#: Faces below this height are not useful search results even if detected, so headline
#: recall is reported on this subset as well as overall.
USEFUL_MIN_PX = 32


def evaluate(ds, anns, label, pack, det_size, pad, iou=0.5):
    det = InsightFaceDetector(pack=pack, det_size=det_size, pad_frac=pad)
    rec = {"found": [], "height": [], "attrs": [], "n_det": 0, "n_img": 0}
    t0 = time.time()
    for a in anns:
        img = cv2.imread(str(ds.image_path(a.path)))
        if img is None or len(a.boxes) == 0:
            continue
        ds_ = det.detect(img)
        boxes = np.array([d.bbox for d in ds_], dtype=np.float64).reshape(-1, 4)
        scores = np.array([d.score for d in ds_], dtype=np.float64)
        valid = a.attrs[:, 3] == 0          # drop boxes flagged invalid by the annotators
        gt = a.boxes[valid]
        at = a.attrs[valid]
        if len(gt) == 0:
            continue
        found = match(gt, boxes, scores, thr=iou)
        rec["found"].append(found)
        rec["height"].append(gt[:, 3] - gt[:, 1])
        rec["attrs"].append(at)
        rec["n_det"] += len(boxes)
        rec["n_img"] += 1
    elapsed = time.time() - t0

    found = np.concatenate(rec["found"])
    height = np.concatenate(rec["height"])
    attrs = np.concatenate(rec["attrs"])
    useful = height >= USEFUL_MIN_PX

    def by(keyfn, mapping):
        out = {}
        for k, name in mapping.items():
            m = keyfn(k)
            if m.sum() >= 50:
                out[name] = {"n": int(m.sum()), "recall": float(found[m].mean())}
        return out

    buckets = {}
    for h, f in zip(height, found):
        b = size_bucket(h)
        buckets.setdefault(b, []).append(f)
    return {
        "config": label, "pack": pack, "det_size": det_size, "pad_frac": pad,
        "n_images": rec["n_img"], "n_faces": int(len(found)),
        "recall_all": float(found.mean()),
        "recall_useful_ge32px": float(found[useful].mean()),
        "n_useful": int(useful.sum()),
        "detections_per_image": rec["n_det"] / max(rec["n_img"], 1),
        "images_per_sec": rec["n_img"] / max(elapsed, 1e-9),
        "by_size": {k: {"n": len(v), "recall": float(np.mean(v))} for k, v in buckets.items()},
        "by_blur": by(lambda k: attrs[:, 0] == k, BLUR),
        "by_illumination": by(lambda k: attrs[:, 2] == k, ILLUMINATION),
        "by_occlusion": by(lambda k: attrs[:, 4] == k, OCCLUSION),
        "by_pose": by(lambda k: attrs[:, 5] == k, POSE),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT / "data/raw/WIDERFACE"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/e8_detection"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all 3226 images")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    seed_everything(1337)
    manifest = RunManifest(
        experiment="e8_detection",
        description="Detector recall on real scenes, by face size / blur / occlusion / pose",
        config=vars(args),
        dataset="WIDER FACE validation",
        split_methodology=(
            "Official validation split, annotator-flagged invalid boxes excluded. Greedy "
            f"highest-score-first matching at IoU {args.iou}. Evaluation only - WIDER FACE "
            "is CC BY-NC-ND 4.0."
        ),
    )

    ds = WiderFace(args.root)
    anns = ds.annotations
    if args.limit:
        anns = anns[: args.limit]
    print(json.dumps(ds.summary(), indent=2)[:400] + " ...\n")
    print(f"evaluating {len(anns)} images at IoU {args.iou}\n")

    results = {}
    print(f"{'config':<26} {'recall all':>11} {'recall >=32px':>14} {'det/img':>8} {'img/s':>7}")
    print("-" * 72)
    for label, pack, dsz, pad in CONFIGS:
        r = evaluate(ds, anns, label, pack, dsz, pad, iou=args.iou)
        results[label] = r
        print(f"{label:<26} {r['recall_all']:>11.4f} {r['recall_useful_ge32px']:>14.4f} "
              f"{r['detections_per_image']:>8.1f} {r['images_per_sec']:>7.1f}")

    best = max(results, key=lambda k: results[k]["recall_useful_ge32px"])
    prod = "buffalo_l d640 pad0.25"
    print(f"\nbest on useful faces: {best}")
    print(f"E5 production setting: {prod} -> {results[prod]['recall_useful_ge32px']:.4f} "
          f"(gap {results[best]['recall_useful_ge32px'] - results[prod]['recall_useful_ge32px']:+.4f})")

    print(f"\n=== recall by face size ({best}) ===")
    for k, v in results[best]["by_size"].items():
        print(f"  {k:<16} n={v['n']:<6} recall={v['recall']:.4f}")
    for axis in ("by_blur", "by_occlusion", "by_pose", "by_illumination"):
        print(f"\n=== {axis.replace('by_','recall by ')} ({best}) ===")
        for k, v in results[best][axis].items():
            print(f"  {k:<10} n={v['n']:<6} recall={v['recall']:.4f}")

    print(f"\n=== effect of E5 padding on scenes (det_size 640 and 1024) ===")
    for dsz in (640, 1024):
        a = results.get(f"buffalo_l d{dsz} pad0.00")
        b = results.get(f"buffalo_l d{dsz} pad0.25")
        if a and b:
            print(f"  d{dsz}: pad0.00 {a['recall_useful_ge32px']:.4f} -> "
                  f"pad0.25 {b['recall_useful_ge32px']:.4f} "
                  f"({b['recall_useful_ge32px']-a['recall_useful_ge32px']:+.4f})")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    manifest.metrics = results
    manifest.finish(out)
    print(f"\n[ok] -> {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
