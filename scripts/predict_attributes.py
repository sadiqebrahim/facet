#!/usr/bin/env python
"""Run prediction heads over an existing index.

The cheap, replaceable half of docs/RESEARCH.md 15.1. Separated from indexing on purpose:
heads get swapped and retrained often, and none of that should re-detect or re-encode
anything. Re-running this after a head change touches no pixels.

Passes:
    beauty   over cached features - milliseconds per face, run for everything
    age      MiVOLO, re-cropping from the original image - E4 measured it at ~190x the
             retired baseline's cost, so it is LAZY: only faces above a quality floor, and
             optionally only the top-N by attractiveness
    dupes    exact (content hash) and near-duplicate (embedding) grouping

Each pass writes `model_version` and `config_hash` per row, so a version bump re-queues
exactly the affected faces and nothing else.

Usage:
    python scripts/predict_attributes.py --index INDEX --features DIR --pass beauty
    python scripts/predict_attributes.py --index INDEX --features DIR --pass age --min-quality 0.4
    python scripts/predict_attributes.py --index INDEX --features DIR --pass dupes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.models.beauty_head import BeautyHead  # noqa: E402
from facet.pipeline.db import Index  # noqa: E402
from facet.pipeline.duplicates import near_duplicate_groups  # noqa: E402
from facet.pipeline.store import FeatureStore  # noqa: E402


def open_store(features_dir: Path, index: Index):
    """Locate the shard the indexed faces actually reference."""
    row = index.conn.execute(
        "SELECT encoder_version, crop_version, COUNT(*) n FROM faces "
        "WHERE feature_row IS NOT NULL GROUP BY 1,2 ORDER BY n DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise SystemExit("no faces with features in this index - run the indexer first")
    meta = json.loads(
        (features_dir / f"{_slug(row['encoder_version'])}__{_slug(row['crop_version'])}.json")
        .read_text()
    )
    return FeatureStore(features_dir, row["encoder_version"], row["crop_version"],
                        dim=meta["dim"]), row["encoder_version"], row["crop_version"]


def _slug(s):
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def run_beauty(index: Index, store, head_path: Path, batch: int = 4096) -> dict:
    head = BeautyHead.load(head_path)
    faces = index.faces_missing_prediction("beauty", head.version)
    print(f"beauty: {len(faces)} faces need prediction (version {head.version})")
    if not faces:
        return {"n": 0}
    t0 = time.time()
    n_ood = 0
    for s in range(0, len(faces), batch):
        chunk = faces[s : s + batch]
        rows = [f["feature_row"] for f in chunk]
        X = store.take(rows).astype(np.float32)
        preds = head.predict(X)
        payload = []
        for f, p in zip(chunk, preds):
            n_ood += int(p.ood)
            payload.append({
                "face_id": f["id"], "model": "beauty",
                "model_version": head.version, "config_hash": head.config_hash,
                "value": p.mean, "confidence": p.confidence,
                "std": float(np.sqrt(p.aleatoric ** 2 + p.epistemic ** 2)),
                "interval_lo": p.interval[0] if p.interval else None,
                "interval_hi": p.interval[1] if p.interval else None,
                "distribution": [float(x) for x in p.distribution],
                "extra": {"p_ge4": p.p_ge4, "aleatoric": p.aleatoric,
                          "epistemic": p.epistemic, "ood_score": p.ood_score,
                          "ood": p.ood, "warnings": p.warnings, "source": p.source},
            })
        with index.tx():
            index.upsert_predictions(payload)
        print(f"  {min(s+batch,len(faces))}/{len(faces)}", flush=True)
    el = time.time() - t0
    print(f"beauty: {len(faces)} faces in {el:.1f}s ({len(faces)/el:.0f} faces/s); "
          f"{n_ood} flagged out-of-distribution ({100*n_ood/len(faces):.1f}%)")
    return {"n": len(faces), "seconds": el, "ood": n_ood}


def run_age(index: Index, min_quality: float, limit: int, batch: int = 32) -> dict:
    """Lazy MiVOLO pass. E4: wide crop, and only on faces worth the compute."""
    import cv2

    from facet.models.insightface_backend import crop_bbox
    from facet.models.mivolo_backend import MiVOLOPredictor

    m = MiVOLOPredictor()
    version = m.version
    faces = index.faces_missing_prediction("age", version)
    faces = [f for f in faces if (f["quality"] or 0) >= min_quality]
    if limit:
        faces = sorted(faces, key=lambda f: -(f["quality"] or 0))[:limit]
    print(f"age/gender: {len(faces)} faces (quality >= {min_quality}"
          + (f", top {limit} by quality" if limit else "") + ")")
    if not faces:
        return {"n": 0}

    paths = {r["id"]: r["path"] for r in index.conn.execute("SELECT id, path FROM images")}
    t0 = time.time()
    for s in range(0, len(faces), batch):
        chunk = faces[s : s + batch]
        crops, keep = [], []
        for f in chunk:
            img = cv2.imread(paths[f["image_id"]])
            if img is None:
                continue
            # E4: MiVOLO wants a WIDE crop - it is a face+body model and uses the context.
            crops.append(crop_bbox(img, (f["x1"], f["y1"], f["x2"], f["y2"]),
                                   size=384, margin=1.0))
            keep.append(f)
        if not crops:
            continue
        pf, age = m.predict(np.stack(crops), batch_size=batch)
        with index.tx():
            index.upsert_predictions([
                {"face_id": f["id"], "model": "age", "model_version": version,
                 "config_hash": None, "value": float(a), "confidence": None,
                 "std": None, "interval_lo": None, "interval_hi": None,
                 "distribution": None, "extra": None}
                for f, a in zip(keep, age)
            ] + [
                {"face_id": f["id"], "model": "gender", "model_version": version,
                 "config_hash": None, "value": float(p), "confidence": float(max(p, 1 - p)),
                 "std": None, "interval_lo": None, "interval_hi": None,
                 "distribution": None,
                 "extra": {"label": "female" if p >= 0.5 else "male",
                           "note": "predicted perceived presentation, not identity"}}
                for f, p in zip(keep, pf)
            ])
        print(f"  {min(s+batch,len(faces))}/{len(faces)}", flush=True)
    el = time.time() - t0
    print(f"age/gender: {len(faces)} faces in {el:.1f}s ({len(faces)/el:.1f} faces/s)")
    return {"n": len(faces), "seconds": el}


def run_dupes(index: Index, store, threshold: float) -> dict:
    rows = list(index.conn.execute(
        "SELECT id, feature_row FROM faces WHERE feature_row IS NOT NULL ORDER BY id"))
    exact = list(index.conn.execute(
        "SELECT content_hash, GROUP_CONCAT(id) ids, COUNT(*) n FROM images "
        "WHERE content_hash IS NOT NULL GROUP BY content_hash HAVING n > 1"))
    with index.tx():
        index.conn.execute("DELETE FROM duplicates")
        for gid, r in enumerate(exact):
            for image_id in r["ids"].split(","):
                for f in index.conn.execute("SELECT id FROM faces WHERE image_id=?",
                                            (int(image_id),)):
                    index.conn.execute(
                        "INSERT OR REPLACE INTO duplicates(face_id,group_id,kind) "
                        "VALUES(?,?,'exact')", (f["id"], gid))
        if rows:
            X = store.take([r["feature_row"] for r in rows])
            labels = near_duplicate_groups(X, threshold=threshold)
            _, counts = np.unique(labels, return_counts=True)
            multi = {int(l) for l, c in zip(*np.unique(labels, return_counts=True)) if c > 1}
            for r, lab in zip(rows, labels):
                if int(lab) in multi:
                    index.conn.execute(
                        "INSERT OR REPLACE INTO duplicates(face_id,group_id,kind) "
                        "VALUES(?,?,'near')", (r["id"], int(lab)))
    n_near = index.conn.execute(
        "SELECT COUNT(DISTINCT group_id) FROM duplicates WHERE kind='near'").fetchone()[0]
    print(f"duplicates: {len(exact)} exact-file groups, {n_near} near-duplicate face groups "
          f"(cosine > {threshold})")
    return {"exact_groups": len(exact), "near_groups": int(n_near)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--pass", dest="which", default="beauty",
                    choices=["beauty", "age", "dupes", "all"])
    ap.add_argument("--head", default=str(ROOT / "artifacts/models/beauty_head.npz"))
    ap.add_argument("--min-quality", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dupe-threshold", type=float, default=0.92)
    args = ap.parse_args()

    index = Index(args.index)
    store, enc, crop = open_store(Path(args.features), index)
    print(f"index: {index.stats()}\nfeatures: {len(store)} rows, dim {store.dim}\n")

    out = {}
    if args.which in ("beauty", "all"):
        out["beauty"] = run_beauty(index, store, Path(args.head))
    if args.which in ("age", "all"):
        out["age"] = run_age(index, args.min_quality, args.limit)
    if args.which in ("dupes", "all"):
        out["dupes"] = run_dupes(index, store, args.dupe_threshold)

    print(f"\nfinal: {index.stats()}")
    index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
