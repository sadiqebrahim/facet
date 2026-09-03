#!/usr/bin/env python
"""Experiment E9 - is the free quality composite actually a quality signal?

docs/RESEARCH.md section 2.4 proposed NOT training a FIQA model, and instead building a
composite from signals we already compute (blur, exposure, contrast, face size, detector
confidence, embedding norm), then validating it. `src/facet/models/quality.py` is that
composite, and its reference constants are currently documented guesses. This validates it.

The obvious evaluation - correlate against CR-FIQA - would only measure agreement with
another estimator. The field's actual ground truth is functional: **a face quality metric is
good if rejecting low-quality faces reduces face-recognition error.** That is the
Error-versus-Reject Curve (ERC), and it needs identity labels, so this runs on LFW's official
6,000-pair verification protocol.

The question is not "does our composite correlate with something" but:

    1. Does rejecting by our composite lower verification error faster than rejecting at random?
    2. Does the COMPOSITE beat its individual components, or is one signal doing all the work?
    3. Do the guessed constants matter - would fitting them help?

License note: LFW is a research benchmark; used here for evaluation only, never redistributed.

Usage:
    python scripts/run_e9_quality.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.models.insightface_backend import (  # noqa: E402
    ArcFaceEmbedder, InsightFaceDetector, align_to_template,
)
from facet.models.quality import composite, composite_v2, per_crop_signals  # noqa: E402
from facet.utils.run import RunManifest  # noqa: E402
from facet.utils.seed import seed_everything  # noqa: E402

LFW_HOME = Path.home() / "scikit_learn_data" / "lfw_home"
REJECT_RATES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def load_pairs(pairs_file: Path, img_root: Path):
    """Official LFW protocol: 10 folds x (300 genuine + 300 impostor)."""
    lines = pairs_file.read_text().splitlines()
    pairs = []
    for ln in lines[1:]:
        p = ln.split()
        if len(p) == 3:
            a = img_root / p[0] / f"{p[0]}_{int(p[1]):04d}.jpg"
            b = img_root / p[0] / f"{p[0]}_{int(p[2]):04d}.jpg"
            pairs.append((str(a), str(b), 1))
        elif len(p) == 4:
            a = img_root / p[0] / f"{p[0]}_{int(p[1]):04d}.jpg"
            b = img_root / p[2] / f"{p[2]}_{int(p[3]):04d}.jpg"
            pairs.append((str(a), str(b), 0))
    return pairs


def fnmr_at_fmr(gen: np.ndarray, imp: np.ndarray, fmr: float = 0.01) -> tuple[float, float]:
    """Threshold set on impostor scores at the target FMR; returns (threshold, FNMR)."""
    if len(imp) == 0 or len(gen) == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(imp, 1.0 - fmr))
    return thr, float((gen < thr).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lfw-home", default=str(LFW_HOME))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/e9_quality"))
    ap.add_argument("--fmr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    seed_everything(args.seed)
    manifest = RunManifest(
        experiment="e9_quality",
        description="Does the free quality composite reduce recognition error when used to reject?",
        config=vars(args), seed=args.seed,
        dataset="LFW (official 6000-pair protocol)",
        split_methodology=(
            "Error-vs-Reject: threshold fixed on ALL impostor pairs at the target FMR, then "
            "pairs are progressively rejected by the quality of their WORSE face and FNMR is "
            "recomputed on what remains. Evaluation only."
        ),
    )

    home = Path(args.lfw_home)
    pairs = load_pairs(home / "pairs.txt", home / "lfw_funneled")
    files = sorted({p for a, b, _ in pairs for p in (a, b)})
    print(f"LFW: {len(pairs)} pairs ({sum(l for _,_,l in pairs)} genuine), "
          f"{len(files)} unique images\n")

    det = InsightFaceDetector(pack="buffalo_l")   # adaptive det_size + padding (E8)
    emb = ArcFaceEmbedder(pack="buffalo_l")
    idx = {f: i for i, f in enumerate(files)}
    crops = np.zeros((len(files), 112, 112, 3), dtype=np.uint8)
    sig = {k: np.zeros(len(files)) for k in
           ("blur", "luminance", "contrast", "clipped_dark", "clipped_bright")}
    det_score = np.zeros(len(files))
    face_px = np.zeros(len(files))

    cache = Path(args.out_dir) / "lfw_signals.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["files"]) == files:
            print("  [cache] reusing LFW detections/embeddings/signals")
            E = z["E"]; norms = z["norms"]; det_score = z["det_score"]
            face_px = z["face_px"]
            sig = {k: z[f"sig_{k}"] for k in
                   ("blur", "luminance", "contrast", "clipped_dark", "clipped_bright")}
            sig["det_score"] = det_score; sig["face_pixels"] = face_px
            return_early = True
        else:
            return_early = False
    else:
        return_early = False

    t0 = time.time()
    for i, f in enumerate(files) if not return_early else []:
        img = cv2.imread(f)
        ds = det.detect(img)
        if ds and ds[0].keypoints is not None:
            d = ds[0]
            crops[i] = align_to_template(img, d.keypoints, size=112, margin=0.25)
            det_score[i] = d.score
            face_px[i] = max(d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1])
        else:
            crops[i] = cv2.resize(img, (112, 112))
        for k, v in per_crop_signals(crops[i]).items():
            sig[k][i] = v
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(files)}  ({(i+1)/(time.time()-t0):.0f} img/s)")

    if not return_early:
        sig["det_score"] = det_score
        sig["face_pixels"] = face_px
        E, norms = emb.encode_with_norm(crops)
        print(f"  embedded {len(E)} faces in {time.time()-t0:.0f}s\n")
        np.savez_compressed(
            cache, files=np.array(files), E=E, norms=norms, det_score=det_score,
            face_px=face_px, **{f"sig_{k}": v for k, v in sig.items()
                                if k not in ("det_score", "face_pixels")},
        )

    sig_full = dict(sig)
    sig_full["embedding_norm"] = norms.astype(np.float64)
    signals = {
        "composite_v1": composite(sig),
        "composite_v2": composite_v2(sig_full),
        "blur": sig["blur"],
        "contrast": sig["contrast"],
        "face_pixels": face_px,
        "det_score": det_score,
        "embedding_norm": norms.astype(np.float64),
        "random": np.random.default_rng(args.seed).random(len(files)),
    }

    ia = np.array([idx[a] for a, _, _ in pairs])
    ib = np.array([idx[b] for _, b, _ in pairs])
    label = np.array([l for _, _, l in pairs])
    score = np.einsum("ij,ij->i", E[ia], E[ib]).astype(np.float64)  # cosine, E is L2-normed

    thr, base_fnmr = fnmr_at_fmr(score[label == 1], score[label == 0], args.fmr)
    auc = float(stats.mannwhitneyu(score[label == 1], score[label == 0]).statistic /
                ((label == 1).sum() * (label == 0).sum()))
    print(f"baseline verification: AUC={auc:.4f}  FNMR@FMR={args.fmr} = {base_fnmr:.4f}  "
          f"(threshold {thr:.4f})\n")

    results = {"n_pairs": len(pairs), "n_images": len(files), "auc": auc,
               "baseline_fnmr": base_fnmr, "fmr": args.fmr, "erc": {}}

    print("=== Error-vs-Reject: FNMR after rejecting the lowest-quality pairs ===")
    print(f"{'signal':<16}" + "".join(f"{int(r*100):>8}%" for r in REJECT_RATES) + f"{'AUERC':>9}")
    print("-" * (16 + 8 * len(REJECT_RATES) + 10))
    for name, q in signals.items():
        # A pair is only as good as its worse face.
        pq = np.minimum(q[ia], q[ib])
        row, curve = [], {}
        for r in REJECT_RATES:
            keep = pq >= np.quantile(pq, r) if r > 0 else np.ones(len(pq), bool)
            g, i_ = score[keep & (label == 1)], score[keep & (label == 0)]
            fnmr = float((g < thr).mean()) if len(g) else float("nan")
            row.append(fnmr)
            curve[str(r)] = fnmr
        auerc = float(np.trapezoid(row, REJECT_RATES) / (REJECT_RATES[-1] - REJECT_RATES[0]))
        results["erc"][name] = {"curve": curve, "auerc": auerc,
                                "reduction_at_20pct": float(row[0] - row[3])}
        print(f"{name:<16}" + "".join(f"{v:>9.4f}" for v in row) + f"{auerc:>9.4f}")

    rnd = results["erc"]["random"]["auerc"]
    print(f"\n(lower AUERC is better; random rejection = {rnd:.4f})")
    ranked = sorted(((v["auerc"], k) for k, v in results["erc"].items()))
    print("\nranking (best first):")
    for a, k in ranked:
        print(f"  {k:<16} AUERC={a:.4f}   vs random {100*(rnd-a)/rnd:+.1f}%")
    results["best_signal"] = ranked[0][1]
    for v in ("composite_v1", "composite_v2"):
        parts = min(a for k, a in ((k, x["auerc"]) for k, x in results["erc"].items())
                    if not k.startswith("composite") and k != "random")
        results[f"{v}_beats_random"] = bool(results["erc"][v]["auerc"] < rnd)
        results[f"{v}_beats_all_parts"] = bool(results["erc"][v]["auerc"] <= parts)

    print("\n=== do the guessed constants matter? (correlations among signals) ===")
    for v in ("composite_v1", "composite_v2"):
        print(f"  {v}:")
        for a in ("embedding_norm", "blur", "face_pixels", "det_score"):
            r = stats.spearmanr(signals[v], signals[a]).statistic
            print(f"    corr({v}, {a:<15}) = {r:+.3f}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    manifest.metrics = results
    manifest.finish(out)
    print(f"\n[ok] -> {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
