#!/usr/bin/env python
"""Extract and cache face features for SCUT-FBP5500.

This is the expensive half of the pipeline (docs/RESEARCH.md section 15.1). Features are
written once, keyed by (encoder version, crop protocol), and every downstream experiment
reads the cache. Re-running with the same key is a no-op unless --force is given.

Usage:
    python scripts/extract_features.py --encoder arcface_buffalo_l
    python scripts/extract_features.py --encoder clip --margin 0.25
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

from facet.data.mebeauty import MEBeauty  # noqa: E402
from facet.data.scut_fbp5500 import ScutFbp5500  # noqa: E402
from facet.models.insightface_backend import (  # noqa: E402
    ArcFaceEmbedder,
    InsightFaceDetector,
    align_to_template,
    crop_bbox,
)
from facet.utils.seed import seed_everything  # noqa: E402


def detect_all(ds, names: list[str], dataset: str, use_gpu: bool, cache_dir: Path):
    """Run detection once per dataset and cache bbox + keypoints.

    Detection dominates extraction cost (~7 ms/image vs ~1 ms to align and encode), and
    it does not depend on the crop protocol. Caching it makes a margin sweep such as
    experiment E5 cheap instead of quadratic in the number of configurations.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{dataset}__buffalo_l.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["names"]) == list(names):
            print(f"  [cache] detections from {cache.name}")
            return z["bboxes"], z["kps"], z["scores"]
        print("  [cache] stale (name list changed), re-detecting")

    detector = InsightFaceDetector(pack="buffalo_l", use_gpu=use_gpu)
    bboxes = np.zeros((len(names), 4), dtype=np.float32)
    kps = np.full((len(names), 5, 2), np.nan, dtype=np.float32)
    scores = np.zeros(len(names), dtype=np.float32)
    t0 = time.time()
    for i, name in enumerate(names):
        img = cv2.imread(str(ds.image_path(name)))
        if img is None:
            raise RuntimeError(f"unreadable image: {name}")
        dets = detector.detect(img)
        if dets and dets[0].keypoints is not None:
            bboxes[i] = dets[0].bbox
            kps[i] = dets[0].keypoints
            scores[i] = dets[0].score
        if (i + 1) % 1000 == 0:
            print(f"  detect {i+1}/{len(names)}  ({(i+1)/(time.time()-t0):.0f} img/s)", flush=True)
    np.savez_compressed(cache, names=np.array(names), bboxes=bboxes, kps=kps, scores=scores)
    print(f"  detected {int((scores > 0).sum())}/{len(names)}; cached -> {cache.name}")
    return bboxes, kps, scores


def build_crops(ds, names, dataset, margin, size, use_gpu, align="template",
                cache_dir=None):
    """Crop every image under one crop protocol. Returns (crops, metadata).

    `align="template"` applies the ArcFace 5-point similarity transform (production
    behaviour). `align="bbox"` is the E5 control: a plain square bbox crop, no rotation
    correction.
    """
    bboxes, kps, scores = detect_all(ds, names, dataset, use_gpu, cache_dir)
    crops = np.zeros((len(names), size, size, 3), dtype=np.uint8)
    meta = []
    n_fallback = 0
    for i, name in enumerate(names):
        img = cv2.imread(str(ds.image_path(name)))
        ok = scores[i] > 0 and np.isfinite(kps[i]).all()
        if ok and align == "template":
            crops[i] = align_to_template(img, kps[i], size=size, margin=margin)
        elif ok and align == "bbox":
            crops[i] = crop_bbox(img, tuple(bboxes[i]), size=size, margin=margin)
        else:
            # No usable detection: plain resize. Recorded so it is never invisible.
            n_fallback += 1
            crops[i] = cv2.resize(img, (size, size))
        meta.append({"file": name, "det_score": float(scores[i]), "fallback": not ok})
    if n_fallback:
        print(f"  detection failed on {n_fallback}/{len(names)} images (resize fallback)")
    return crops, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="scut", choices=["scut", "mebeauty"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/features"))
    ap.add_argument(
        "--encoder",
        default="arcface_buffalo_l",
        choices=["arcface_buffalo_l", "arcface_antelopev2", "clip", "geometry"],
    )
    ap.add_argument("--margin", type=float, default=0.0, help="crop margin (E5 sweeps this)")
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--align", default="template", choices=["template", "bbox"],
                    help="template = ArcFace 5-point similarity transform; bbox = plain crop")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seed_everything(1337)
    if args.dataset == "scut":
        root = args.data_root or str(ROOT / "data/raw/SCUT-FBP5500_v2")
        ds = ScutFbp5500(root)
        names = ds.filenames
    else:
        root = args.data_root or str(ROOT / "data/raw/MEBeauty")
        ds = MEBeauty(root)
        names = ds.keys
    use_gpu = not args.cpu

    prefix = "" if args.dataset == "scut" else f"{args.dataset}__"
    suffix = "" if args.align == "template" else f"__{args.align}"
    key = f"{prefix}{args.encoder}__m{args.margin:g}__s{args.size}{suffix}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / f"{key}.npy"
    meta_path = out_dir / f"{key}.json"

    if feat_path.exists() and not args.force:
        print(f"[skip] {feat_path} exists (use --force to recompute)")
        return 0

    t0 = time.time()

    if args.encoder == "geometry":
        if args.dataset != "scut":
            raise SystemExit(
                "geometry features require SCUT-FBP5500's supplied 86-point landmarks; "
                "MEBeauty does not ship them"
            )
        # Uses the dataset's own 86 hand-annotated landmarks - no detection needed.
        from facet.models.geometry import geometry_features

        lms, valid = ds.all_landmarks(names)
        n_bad = int((~valid).sum())
        if n_bad:
            # Impute the mean shape so the feature matrix stays aligned with `names`.
            # Recorded in metadata AND in a mask file so downstream code can exclude them.
            print(f"  {n_bad} image(s) have unusable landmarks; imputing mean shape")
            lms[~valid] = np.nanmean(lms[valid], axis=0)
        feats, col_names = geometry_features(lms)
        np.save(out_dir / f"{key}__valid.npy", valid)
        extra = {
            "columns": col_names[-6:],
            "n_landmarks": int(lms.shape[1]),
            "n_invalid_landmarks": n_bad,
            "invalid_files": [n for n, v in zip(names, valid) if not v],
        }
        version = "procrustes86:v1"
        crop_meta = []
    else:
        crops, crop_meta = build_crops(
            ds, names, args.dataset, args.margin, args.size, use_gpu,
            align=args.align, cache_dir=Path(args.out_dir).parent / "detections",
        )
        if args.encoder.startswith("arcface"):
            pack = args.encoder.replace("arcface_", "")
            enc = ArcFaceEmbedder(pack=pack, use_gpu=use_gpu)
            feats, norms = enc.encode_with_norm(crops)
            # Embedding norm is the free MagFace-style quality proxy (RESEARCH.md 2.4).
            np.save(out_dir / f"{key}__embnorm.npy", norms)
            extra = {"embedding_norm_saved": True}
        else:
            from facet.models.clip_backend import ClipEmbedder

            enc = ClipEmbedder(use_gpu=use_gpu)
            feats = enc.encode(crops)
            extra = {}
        version = enc.version

    elapsed = time.time() - t0
    np.save(feat_path, feats)
    meta = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "version": version,
        "crop_margin": args.margin,
        "crop_size": args.size,
        "align": args.align,
        "n": len(names),
        "dim": int(feats.shape[1]),
        "filenames": names,
        "elapsed_sec": round(elapsed, 2),
        "per_image_ms": round(1000 * elapsed / len(names), 2),
        "device": "cpu" if args.cpu else "cuda",
        "detection": crop_meta,
        **extra,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[ok] {feat_path.name}  shape={feats.shape}  "
        f"{elapsed:.1f}s  ({1000*elapsed/len(names):.1f} ms/img)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
