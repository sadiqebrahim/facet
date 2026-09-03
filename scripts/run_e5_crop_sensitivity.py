#!/usr/bin/env python
"""Experiment E5 - how much does the crop protocol matter?

Preprocessing is usually treated as a detail and then frozen by accident. This measures
it: crop margin (0 / 10 / 25 / 40 %) and alignment (ArcFace 5-point similarity transform
vs. a plain bbox crop), for each representation.

The prior recorded in docs/RESEARCH.md section 2.2 was that this "may matter more than the
choice of backbone". That is a testable claim and this tests it.

Crucially it reports TWO numbers per configuration, because E7 established that they can
disagree and that only the second one predicts whether the product works:

    in-benchmark   SCUT-FBP5500 official 5-fold Pearson correlation
    transfer       train on ALL of SCUT, test on ALL of held-out MEBeauty (Spearman)

A crop protocol that maximises in-benchmark accuracy while degrading transfer would be
exactly the wrong thing to freeze into the feature store.

Usage:
    python scripts/run_e5_crop_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.data.mebeauty import MEBeauty  # noqa: E402
from facet.data.scut_fbp5500 import ScutFbp5500  # noqa: E402
from facet.evaluation.metrics import pairwise_accuracy, regression_metrics  # noqa: E402
from facet.training.heads import RidgeRegressionHead  # noqa: E402
from facet.utils.run import RunManifest  # noqa: E402
from facet.utils.seed import seed_everything  # noqa: E402

FEATURE_DIR = ROOT / "artifacts/features"

#: (label, margin, align)
CROPS = [
    ("m0.00 template", 0.0, "template"),
    ("m0.10 template", 0.1, "template"),
    ("m0.25 template", 0.25, "template"),
    ("m0.40 template", 0.4, "template"),
    ("m0.25 bbox(no align)", 0.25, "bbox"),
]
ENCODERS = {
    "arcface_r50": ("arcface_buffalo_l",),
    "clip": ("clip",),
    "arcface_r50+clip": ("arcface_buffalo_l", "clip"),
}


def stem(dataset, key, margin, align, size=112):
    prefix = "" if dataset == "scut" else f"{dataset}__"
    suffix = "" if align == "template" else f"__{align}"
    return f"{prefix}{key}__m{margin:g}__s{size}{suffix}"


def load(dataset, parts, margin, align):
    mats, order = [], None
    for p in parts:
        st = stem(dataset, p, margin, align)
        arr = np.load(FEATURE_DIR / f"{st}.npy").astype(np.float64)
        names = json.loads((FEATURE_DIR / f"{st}.json").read_text())["filenames"]
        if order is None:
            order = names
        elif names != order:
            raise RuntimeError(f"ordering mismatch {st}")
        arr /= np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)
        mats.append(arr)
    return np.hstack(mats), order


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scut-root", default=str(ROOT / "data/raw/SCUT-FBP5500_v2"))
    ap.add_argument("--me-root", default=str(ROOT / "data/raw/MEBeauty"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/e5_crop_sensitivity"))
    args = ap.parse_args()

    seed_everything(1337)
    manifest = RunManifest(
        experiment="e5_crop_sensitivity",
        description="Does the crop protocol matter more than the backbone?",
        config=vars(args),
        dataset="SCUT-FBP5500_v2 (+ MEBeauty for transfer)",
        split_methodology=(
            "In-benchmark: SCUT official 5-fold CV. Transfer: train on all 5500 SCUT, "
            "test on all 2520 held-out MEBeauty (rank metrics only - scales differ)."
        ),
    )

    scut = ScutFbp5500(args.scut_root)
    me = MEBeauty(args.me_root)
    results: dict = {}

    print(f"{'encoder':<18} {'crop':<22} {'5fold PC':>9} {'5fold MAE':>10} "
          f"{'transfer rho':>13} {'transfer pair':>14}")
    print("-" * 92)

    for enc_name, parts in ENCODERS.items():
        results[enc_name] = {}
        for label, margin, align in CROPS:
            Xs, s_order = load("scut", parts, margin, align)
            Xm, m_order = load("mebeauty", parts, margin, align)
            ys = scut.labels.loc[s_order, "mean"].to_numpy(np.float64)
            ym = me.labels.loc[m_order, "mean"].to_numpy(np.float64)
            idx = {n: i for i, n in enumerate(s_order)}

            # ---- in-benchmark: official 5-fold -------------------------------------
            pcs, maes = [], []
            for k in range(1, 6):
                sp = scut.splits[f"cv{k}"]
                tr = np.array([idx[n] for n in sp.train])
                te = np.array([idx[n] for n in sp.test])
                m = RidgeRegressionHead().fit(Xs[tr], ys[tr])
                r = regression_metrics(ys[te], m.predict(Xs[te]))
                pcs.append(r["pc"])
                maes.append(r["mae"])

            # ---- transfer: all SCUT -> all MEBeauty --------------------------------
            mt = RidgeRegressionHead().fit(Xs, ys)
            pm = mt.predict(Xm)
            rho = float(stats.spearmanr(ym, pm).statistic)
            pair = pairwise_accuracy(ym, pm)

            results[enc_name][label] = {
                "margin": margin,
                "align": align,
                "cv_pc": float(np.mean(pcs)),
                "cv_pc_std": float(np.std(pcs)),
                "cv_mae": float(np.mean(maes)),
                "transfer_spearman": rho,
                "transfer_pairwise": pair,
            }
            print(f"{enc_name:<18} {label:<22} {np.mean(pcs):>9.4f} {np.mean(maes):>10.4f} "
                  f"{rho:>13.4f} {pair:>14.4f}")
        print("-" * 92)

    # ------------------------------------------------------------------ analysis
    print("\n=== How much does the crop protocol matter, vs. the backbone? ===")
    for enc_name in ENCODERS:
        tmpl = {k: v for k, v in results[enc_name].items() if v["align"] == "template"}
        pcs = [v["cv_pc"] for v in tmpl.values()]
        rhos = [v["transfer_spearman"] for v in tmpl.values()]
        best_pc = max(tmpl.items(), key=lambda kv: kv[1]["cv_pc"])
        best_rho = max(tmpl.items(), key=lambda kv: kv[1]["transfer_spearman"])
        print(f"  {enc_name:<18} margin sweep: PC range {max(pcs)-min(pcs):.4f}  "
              f"rho range {max(rhos)-min(rhos):.4f}   "
              f"best-PC={best_pc[0].split()[0]}  best-transfer={best_rho[0].split()[0]}")
    enc_pc = [max(v["cv_pc"] for v in results[e].values()) for e in ENCODERS]
    enc_rho = [max(v["transfer_spearman"] for v in results[e].values()) for e in ENCODERS]
    print(f"  {'backbone choice':<18} PC range {max(enc_pc)-min(enc_pc):.4f}  "
          f"rho range {max(enc_rho)-min(enc_rho):.4f}")

    print("\n=== Value of alignment (template vs plain bbox crop, both at margin 0.25) ===")
    for enc_name in ENCODERS:
        a = results[enc_name]["m0.25 template"]
        b = results[enc_name]["m0.25 bbox(no align)"]
        print(f"  {enc_name:<18} PC {b['cv_pc']:.4f} -> {a['cv_pc']:.4f} "
              f"({a['cv_pc']-b['cv_pc']:+.4f})   "
              f"transfer {b['transfer_spearman']:.4f} -> {a['transfer_spearman']:.4f} "
              f"({a['transfer_spearman']-b['transfer_spearman']:+.4f})")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    manifest.metrics = results
    manifest.finish(out)
    print(f"\n[ok] -> {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
