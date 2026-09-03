#!/usr/bin/env python
"""Experiment 001 - frozen representation + linear head on SCUT-FBP5500.

This is experiment E2 in docs/RESEARCH.md section 14, and it is the pivotal Phase-4
baseline. It asks two questions:

  Q1  Which frozen representation best predicts attractiveness ratings?
      (ArcFace-R50 / ArcFace-R100 / CLIP / 86-point geometry / fusions)

  Q2  How close does frozen-features-plus-a-linear-head get to a fine-tuned CNN?
      Published reference points on the exact same official splits:
          AlexNet     PC 0.8634 | ResNet-18 PC 0.8900 | ResNeXt-50 PC 0.8997  (5-fold)
          AlexNet     PC 0.8298 | ResNet-18 PC 0.8513 | ResNeXt-50 PC 0.8777  (60/40)
      and the human inter-rater ceiling on this data is PC ~0.77.

If frozen features come close, the whole system architecture changes: encoding becomes a
one-time cached cost, heads become swappable and personalisable, and the index does not
need rebuilding when the beauty model changes.

It also compares two HEADS on identical features:
  * ridge on the mean score            - the conventional baseline
  * ridge on the 5-bin rating histogram - label-distribution learning, which additionally
                                          yields per-face rater-disagreement estimates

No test data touches any fitting decision: ridge alpha is chosen by cross-validation
inside the training split only.

Usage:
    python scripts/run_exp001.py
    python scripts/run_exp001.py --splits cv1 cv2 --reps arcface_buffalo_l
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

from facet.data.scut_fbp5500 import ScutFbp5500  # noqa: E402
from facet.evaluation.metrics import (  # noqa: E402
    calibration_metrics,
    full_report,
    group_report,
)
from facet.training.heads import BaggedRidgeHead, DistributionHead, RidgeRegressionHead  # noqa: E402
from facet.utils.run import RunManifest  # noqa: E402
from facet.utils.seed import seed_everything  # noqa: E402

FEATURE_DIR = ROOT / "artifacts/features"

# Representations to compare. Tuples are concatenation fusions (RESEARCH.md section 6.7).
REPRESENTATIONS: dict[str, tuple[str, ...]] = {
    "geometry": ("geometry",),
    "clip": ("clip",),
    "arcface_buffalo_l": ("arcface_buffalo_l",),
    "arcface_antelopev2": ("arcface_antelopev2",),
    "arcface_r50+geometry": ("arcface_buffalo_l", "geometry"),
    "arcface_r50+clip": ("arcface_buffalo_l", "clip"),
    "arcface_r50+r100": ("arcface_buffalo_l", "arcface_antelopev2"),
    "all": ("arcface_buffalo_l", "arcface_antelopev2", "clip", "geometry"),
}

PUBLISHED = {  # docs/RESEARCH.md appendix A, all verified from arXiv:1801.06345
    "5fold": {"AlexNet": 0.8634, "ResNet-18": 0.8900, "ResNeXt-50": 0.8997},
    "split6040": {"AlexNet": 0.8298, "ResNet-18": 0.8513, "ResNeXt-50": 0.8777},
    "human_inter_rater": 0.770,
}


def load_features(key: str, margin: float, size: int) -> tuple[np.ndarray, list[str]]:
    stem = f"{key}__m{margin:g}__s{size}"
    arr = np.load(FEATURE_DIR / f"{stem}.npy")
    meta = json.loads((FEATURE_DIR / f"{stem}.json").read_text())
    return arr, meta["filenames"]


def build_matrix(
    parts: tuple[str, ...], margin: float, size: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Concatenate feature blocks, verifying they share a row ordering."""
    mats, order, dims = [], None, {}
    for p in parts:
        arr, names = load_features(p, margin, size)
        if order is None:
            order = names
        elif names != order:
            raise RuntimeError(f"feature ordering mismatch between {parts[0]!r} and {p!r}")
        # L2-normalise each block so concatenation does not let one block dominate
        # purely through scale.
        arr = arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)
        mats.append(arr)
        dims[p] = arr.shape[1]
    return np.hstack(mats).astype(np.float64), dims


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(ROOT / "data/raw/SCUT-FBP5500_v2"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/exp001_frozen_linear_probe"))
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--reps", nargs="*", default=list(REPRESENTATIONS))
    ap.add_argument(
        "--splits", nargs="*", default=["cv1", "cv2", "cv3", "cv4", "cv5", "split6040"]
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--allow-partial-overwrite",
        action="store_true",
        help="permit a subset run to overwrite an existing full results.json",
    )
    args = ap.parse_args()

    # A subset run (e.g. --reps X --splits cv1, used for smoke tests and regression
    # checks) writes to the same results.json as a full run and would silently destroy
    # the experimental record. Refuse unless explicitly told otherwise, or redirected.
    partial = set(args.reps) != set(REPRESENTATIONS) or len(args.splits) < 6
    existing = Path(args.out_dir) / "results.json"
    if partial and existing.exists() and not args.allow_partial_overwrite:
        try:
            prior = json.loads(existing.read_text())
        except json.JSONDecodeError:
            prior = {}
        if len(prior) > len(args.reps):
            raise SystemExit(
                f"refusing to overwrite {existing} ({len(prior)} representations on disk) "
                f"with a partial run ({len(args.reps)} representation(s), "
                f"{len(args.splits)} split(s)).\n"
                "Use --out-dir <scratch> for a smoke test, or "
                "--allow-partial-overwrite if you really mean it."
            )

    seed_everything(args.seed)
    manifest = RunManifest(
        experiment="exp001_frozen_linear_probe",
        description=(
            "Frozen pretrained representations + linear heads on SCUT-FBP5500, evaluated "
            "on the dataset's own official splits. Answers E2 in docs/RESEARCH.md."
        ),
        config=vars(args),
        seed=args.seed,
        dataset="SCUT-FBP5500_v2",
        dataset_version="v2 (authors' Google Drive release, 2018-05-11)",
        split_methodology=(
            "Official 5-fold CV and official 60/40 split, as shipped with the dataset. "
            "NOTE: these are random IMAGE splits; subject-disjointness is UNVERIFIED "
            "(experiment E1). Treat all numbers as provisional."
        ),
    )

    ds = ScutFbp5500(args.data_root)
    labels = ds.labels
    results: dict[str, dict] = {}

    print(f"{'representation':<24} {'split':<10} {'dim':>6} {'PC':>7} {'MAE':>7} "
          f"{'RMSE':>7} {'rho':>7} {'pair':>7} {'ndcg':>7}")
    print("-" * 96)

    for rep_name in args.reps:
        parts = REPRESENTATIONS[rep_name]
        X_all, dims = build_matrix(parts, args.margin, args.size)
        _, order = load_features(parts[0], args.margin, args.size)
        index = {n: i for i, n in enumerate(order)}

        y_all = labels.loc[order, "mean"].to_numpy(np.float64)
        P_all = ds.histogram_matrix(order).astype(np.float64)
        sub_all = labels.loc[order, "subgroup"].to_numpy()

        rep_res: dict[str, dict] = {}
        for split_name in args.splits:
            split = ds.splits[split_name]
            tr = np.array([index[n] for n in split.train])
            te = np.array([index[n] for n in split.test])
            Xtr, Xte = X_all[tr], X_all[te]
            ytr, yte = y_all[tr], y_all[te]

            t0 = time.time()
            # --- head 1: plain ridge on the mean score ---------------------------------
            ridge = RidgeRegressionHead().fit(Xtr, ytr)
            pred_ridge = ridge.predict(Xte)

            # --- head 2: label-distribution head ---------------------------------------
            ldl = DistributionHead().fit(Xtr, P_all[tr])
            pred_ldl = ldl.predict(Xte)
            std_ldl = ldl.predict_std(Xte)
            p_ge4 = ldl.predict_p_ge(Xte, 4.0)

            # --- head 3: bagged ridge -> epistemic uncertainty --------------------------
            bagged = BaggedRidgeHead(n_members=10, alpha=ridge.chosen_alpha).fit(Xtr, ytr)
            pred_bag, std_bag = bagged.predict_with_std(Xte)
            fit_sec = time.time() - t0

            entry = {
                "n_train": len(tr),
                "n_test": len(te),
                "dim": int(X_all.shape[1]),
                "alpha": ridge.chosen_alpha,
                "fit_sec": round(fit_sec, 2),
                "ridge": full_report(yte, pred_ridge, k=100),
                "ldl": full_report(yte, pred_ldl, k=100),
                "bagged": full_report(yte, pred_bag, k=100),
                # Does the LDL head's predicted rater-disagreement track the TRUE
                # rater-disagreement? This is the claim in RESEARCH.md 6.3 / 11.1.
                "aleatoric_vs_true_std": float(
                    np.corrcoef(std_ldl, labels.loc[split.test, "std"].to_numpy())[0, 1]
                ),
                # Does the epistemic spread predict where the model is actually wrong?
                "epistemic_calibration": calibration_metrics(yte, pred_bag, std_bag),
                # Ranking by P(rating>=4) instead of by the mean.
                "rank_by_p_ge4": full_report(
                    labels.loc[split.test, "p_ge4"].to_numpy(), p_ge4, k=100
                ),
                "by_subgroup": group_report(yte, pred_ridge, sub_all[te], k=50),
            }
            rep_res[split_name] = entry

            r = entry["ridge"]
            print(
                f"{rep_name:<24} {split_name:<10} {entry['dim']:>6} {r['pc']:>7.4f} "
                f"{r['mae']:>7.4f} {r['rmse']:>7.4f} {r['spearman']:>7.4f} "
                f"{r['pairwise_acc']:>7.4f} {r['ndcg@100']:>7.4f}"
            )

        cv = [k for k in rep_res if k.startswith("cv")]
        if cv:
            rep_res["cv_mean"] = {
                head: {
                    m: float(np.mean([rep_res[k][head][m] for k in cv]))
                    for m in rep_res[cv[0]][head]
                }
                for head in ("ridge", "ldl", "bagged")
            }
            rep_res["cv_mean"]["aleatoric_vs_true_std"] = float(
                np.mean([rep_res[k]["aleatoric_vs_true_std"] for k in cv])
            )
        results[rep_name] = rep_res
        print("-" * 96)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))

    manifest.metrics = {
        rep: {
            "cv_mean_pc_ridge": results[rep].get("cv_mean", {}).get("ridge", {}).get("pc"),
            "cv_mean_pc_ldl": results[rep].get("cv_mean", {}).get("ldl", {}).get("pc"),
            "split6040_pc_ridge": results[rep].get("split6040", {}).get("ridge", {}).get("pc"),
        }
        for rep in results
    }
    manifest.artifacts = {"results": str(out_dir / "results.json")}
    manifest.notes = [
        "Reference (verified, arXiv:1801.06345): 5-fold PC AlexNet 0.8634 / ResNet-18 "
        "0.8900 / ResNeXt-50 0.8997; 60-40 PC 0.8298 / 0.8513 / 0.8777.",
        "Human inter-rater correlation on this dataset is ~0.770-0.785. Model correlations "
        "above that are NOT superhuman perception - the crowd mean is a low-variance target.",
        "Official splits are random image splits; subject-disjointness unverified (E1).",
    ]
    manifest.finish(out_dir)
    print(f"\n[ok] results -> {out_dir/'results.json'}")
    print(f"[ok] manifest -> {out_dir/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
