#!/usr/bin/env python
"""Experiment E1 - identity-leakage audit of SCUT-FBP5500's official splits.

The highest-priority experiment in docs/RESEARCH.md section 14, and it gates every other
number this project produces.

SCUT-FBP5500 ships official 5-fold and 60/40 splits, but they are random IMAGE splits.
The dataset is assembled from three third-party sources, at least one of which (the 10k
US Adults Faces Database) is widely redistributed. If the same person appears in both
train and test, a model can partly memorise identities rather than learn attractiveness,
and every published number on this benchmark - including ours - is optimistically biased.

Method: use the cached ArcFace embeddings, find all pairs above a face-verification
cosine threshold, build connected components (identity clusters), and count clusters that
straddle the train/test boundary.

Usage:
    python scripts/run_e1_identity_audit.py --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.data.scut_fbp5500 import ScutFbp5500  # noqa: E402
from facet.utils.run import RunManifest  # noqa: E402


def connected_components(n: int, edges: np.ndarray) -> np.ndarray:
    """Union-find over the pair list. Returns a component label per item."""
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(ROOT / "data/raw/SCUT-FBP5500_v2"))
    ap.add_argument("--features", default="arcface_antelopev2__m0__s112")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="cosine similarity above which two crops are treated as the same identity; "
        "0.5 is a conventional ArcFace verification operating point",
    )
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/e1_identity_audit"))
    args = ap.parse_args()

    manifest = RunManifest(
        experiment="e1_identity_audit",
        description="Are SCUT-FBP5500's official splits subject-disjoint?",
        config=vars(args),
        dataset="SCUT-FBP5500_v2",
        split_methodology="official splits, audited for identity leakage",
    )

    ds = ScutFbp5500(args.data_root)
    feat_dir = ROOT / "artifacts/features"
    emb = np.load(feat_dir / f"{args.features}.npy").astype(np.float32)
    names = json.loads((feat_dir / f"{args.features}.json").read_text())["filenames"]
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    idx = {n: i for i, n in enumerate(names)}

    # Full 5500x5500 cosine matrix is only ~120 MB - compute it directly.
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    iu = np.triu_indices(len(emb), k=1)
    pair_sims = sim[iu]

    results: dict = {
        "n_images": len(names),
        "features": args.features,
        "threshold": args.threshold,
        "similarity_percentiles": {
            str(p): float(np.percentile(pair_sims, p)) for p in (50, 90, 99, 99.9, 99.99)
        },
        "max_similarity": float(pair_sims.max()),
    }

    hits = np.argwhere(sim > args.threshold)
    hits = hits[hits[:, 0] < hits[:, 1]]
    labels = connected_components(len(names), hits)
    _, counts = np.unique(labels, return_counts=True)

    results["n_pairs_above_threshold"] = int(len(hits))
    results["n_clusters"] = int(len(counts))
    results["n_multi_image_clusters"] = int((counts > 1).sum())
    results["largest_cluster"] = int(counts.max())
    results["n_images_in_multi_clusters"] = int(counts[counts > 1].sum())

    print(f"images                      : {len(names)}")
    print(f"pairs above {args.threshold:.2f}            : {len(hits)}")
    print(f"identity clusters           : {len(counts)}")
    print(f"clusters with >1 image      : {(counts > 1).sum()}")
    print(f"images in multi-image cluster: {counts[counts > 1].sum()}")
    print(f"max pairwise similarity     : {pair_sims.max():.4f}")
    print()

    per_split = {}
    for split_name, split in ds.splits.items():
        tr = {idx[n] for n in split.train}
        te = {idx[n] for n in split.test}
        straddling, leaked_test = 0, set()
        for lab in np.unique(labels):
            members = np.where(labels == lab)[0]
            if len(members) < 2:
                continue
            m = set(members.tolist())
            if m & tr and m & te:
                straddling += 1
                leaked_test |= m & te
        per_split[split_name] = {
            "straddling_clusters": straddling,
            "leaked_test_images": len(leaked_test),
            "leaked_test_fraction": round(len(leaked_test) / len(te), 5),
        }
        print(
            f"{split_name:<10} straddling clusters={straddling:<4} "
            f"leaked test images={len(leaked_test):<4} "
            f"({100*len(leaked_test)/len(te):.2f}% of test)"
        )

    results["per_split"] = per_split
    worst = max(v["leaked_test_fraction"] for v in per_split.values())
    results["verdict"] = (
        "CLEAN - no identity cluster straddles any official split boundary"
        if worst == 0
        else f"LEAKAGE - up to {worst*100:.2f}% of test images share an identity with training"
    )
    print(f"\nVERDICT: {results['verdict']}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    manifest.metrics = results
    manifest.finish(out)
    print(f"[ok] -> {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
