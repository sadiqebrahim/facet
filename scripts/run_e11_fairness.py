#!/usr/bin/env python
"""Experiment E11 - end-to-end fairness audit, and the attractiveness/quality confound.

docs/RESEARCH.md section 13.3 argues that per-component bias COMPOUNDS through the pipeline:
a face more likely to be missed by the detector is also more likely to be mis-gendered, and
if it belongs to a group absent from the beauty training data its attractiveness score is
pure extrapolation. So this audits the whole chain on one balanced set rather than each
component separately.

Run on FairFace validation (10,954 images, balanced across seven perceived-race groups,
CC BY 4.0). Five parts, matching section 14:

    (a) detection recall by group
    (b) gender accuracy by group
    (c) age error by group
    (d) attractiveness score DISTRIBUTIONS by group
    (e) the confound: does predicted attractiveness track predicted image QUALITY?

Part (e) is the one exp001 and E5 both pointed at. If attractiveness ratings substantially
encode photography rather than faces, the product must say so.

Usage:
    python scripts/run_e11_fairness.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.data.fairface import FairFace  # noqa: E402
from facet.data.scut_fbp5500 import ScutFbp5500  # noqa: E402
from facet.models.insightface_backend import GenderAgePredictor, align_to_template  # noqa: E402
from facet.models.quality import composite, per_crop_signals  # noqa: E402
from facet.training.objectives import (  # noqa: E402
    ScoreHead, build_targets, loss_distribution, to_score,
)
from facet.utils.run import RunManifest  # noqa: E402
from facet.utils.seed import seed_everything  # noqa: E402

FEATURE_DIR = ROOT / "artifacts/features"
DET_DIR = ROOT / "artifacts/detections"
PARTS = ("arcface_buffalo_l", "clip")
MARGIN = 0.25


def load(dataset, parts, margin=MARGIN, size=112):
    mats, order = [], None
    for p in parts:
        prefix = "" if dataset == "scut" else f"{dataset}__"
        st = f"{prefix}{p}__m{margin:g}__s{size}"
        arr = np.load(FEATURE_DIR / f"{st}.npy").astype(np.float32)
        names = json.loads((FEATURE_DIR / f"{st}.json").read_text())["filenames"]
        if order is None:
            order = names
        elif names != order:
            raise RuntimeError("ordering mismatch")
        arr /= np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)
        mats.append(arr)
    return np.hstack(mats), order


def by_group(values, groups, fn, min_n=30):
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if m.sum() < min_n:
            continue
        out[str(g)] = {"n": int(m.sum()), **fn(m)}
    return out


def spread(d, key):
    vals = [v[key] for v in d.values()]
    return float(max(vals) - min(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ff-root", default=str(ROOT / "data/raw/FairFace"))
    ap.add_argument("--scut-root", default=str(ROOT / "data/raw/SCUT-FBP5500_v2"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/e11_fairness"))
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = RunManifest(
        experiment="e11_fairness",
        description="End-to-end demographic audit + attractiveness/quality confound test",
        config=vars(args), seed=args.seed,
        dataset="FairFace validation (balanced) + SCUT-FBP5500 for the beauty head",
        split_methodology=(
            "Beauty head trained on ALL of SCUT-FBP5500 and applied to FairFace, which is "
            "entirely held out. FairFace labels are annotator-perceived and are used only "
            "to slice our own error rates."
        ),
    )

    ff = FairFace(args.ff_root)
    lab = ff.labels
    names = ff.filenames
    race = lab["race_label"].to_numpy()
    gender = lab["gender_label"].to_numpy()
    group = lab["group"].to_numpy()
    results: dict = {"n_images": len(names)}

    # ---------------------------------------------- (a) detection recall by group
    det = np.load(DET_DIR / "fairface__buffalo_l.npz", allow_pickle=True)
    assert list(det["names"]) == names
    det_score = det["scores"]
    bbox = det["bboxes"]
    found = det_score > 0
    face_px = np.where(found, np.maximum(bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]), 0.0)

    results["a_detection"] = {
        "overall_recall": float(found.mean()),
        "by_race": by_group(None, race, lambda m: {
            "recall": float(found[m].mean()),
            "mean_det_score": float(det_score[m][found[m]].mean()) if found[m].any() else 0.0,
        }),
        "by_group": by_group(None, group, lambda m: {"recall": float(found[m].mean())}),
    }
    print(f"=== (a) DETECTION recall (overall {found.mean():.4f}) ===")
    for g, v in results["a_detection"]["by_race"].items():
        print(f"  {g:<18} n={v['n']:<5} recall={v['recall']:.4f} det_score={v['mean_det_score']:.3f}")
    print(f"  -> recall spread across races: {spread(results['a_detection']['by_race'],'recall'):.4f}")

    # ------------------------------------ (b,c) gender / age, and per-crop quality
    print("\n=== building aligned crops + quality signals ===")
    ga = GenderAgePredictor(use_gpu=(device == "cuda"))
    crops = np.zeros((len(names), 112, 112, 3), dtype=np.uint8)
    sig = {k: np.zeros(len(names)) for k in
           ("blur", "luminance", "contrast", "clipped_dark", "clipped_bright")}
    for i, n in enumerate(names):
        img = cv2.imread(str(ff.image_path(n)))
        kps = det["kps"][i]
        crops[i] = (align_to_template(img, kps, size=112, margin=MARGIN)
                    if found[i] and np.isfinite(kps).all() else cv2.resize(img, (112, 112)))
        for k, v in per_crop_signals(crops[i]).items():
            sig[k][i] = v
        if (i + 1) % 4000 == 0:
            print(f"  {i+1}/{len(names)}")

    sig["det_score"] = det_score
    sig["face_pixels"] = face_px
    qual = composite(sig)
    g_male_prob, age_pred = ga.predict(crops)

    pred_female = g_male_prob < 0.5
    true_female = gender == "Female"
    correct = pred_female == true_female
    results["b_gender"] = {
        "overall_accuracy": float(correct.mean()),
        "by_race": by_group(None, race, lambda m: {"accuracy": float(correct[m].mean())}),
        "by_group": by_group(None, group, lambda m: {"accuracy": float(correct[m].mean())}),
    }
    print(f"\n=== (b) GENDER accuracy (overall {correct.mean():.4f}) ===")
    for g, v in results["b_gender"]["by_race"].items():
        print(f"  {g:<18} n={v['n']:<5} acc={v['accuracy']:.4f}")
    print(f"  -> accuracy spread across races: {spread(results['b_gender']['by_race'],'accuracy'):.4f}")
    worst = min(results["b_gender"]["by_group"].items(), key=lambda kv: kv[1]["accuracy"])
    best = max(results["b_gender"]["by_group"].items(), key=lambda kv: kv[1]["accuracy"])
    print(f"  -> best group {best[0]} {best[1]['accuracy']:.4f} | "
          f"worst group {worst[0]} {worst[1]['accuracy']:.4f}")

    age_true = lab["age_mid"].to_numpy(np.float64)
    age_err = np.abs(age_pred - age_true)
    results["c_age"] = {
        "overall_mae": float(age_err.mean()),
        "by_race": by_group(None, race, lambda m: {"mae": float(age_err[m].mean())}),
        "by_age_bucket": by_group(None, lab["age_bucket"].to_numpy(),
                                  lambda m: {"mae": float(age_err[m].mean())}),
    }
    print(f"\n=== (c) AGE MAE vs bucket midpoint (overall {age_err.mean():.2f} years) ===")
    for g, v in results["c_age"]["by_race"].items():
        print(f"  {g:<18} n={v['n']:<5} MAE={v['mae']:.2f}")
    print(f"  -> MAE spread across races: {spread(results['c_age']['by_race'],'mae'):.2f} years")
    print("  by age bucket:")
    for g, v in results["c_age"]["by_age_bucket"].items():
        print(f"    {g:<8} n={v['n']:<5} MAE={v['mae']:.2f}")

    # ------------------------------- (d) attractiveness distributions by group
    print("\n=== (d) ATTRACTIVENESS score distribution by group ===")
    scut = ScutFbp5500(args.scut_root)
    Xs, s_order = load("scut", PARTS)
    Xf, f_order = load("fairface", PARTS)
    assert f_order == names
    tgt = build_targets(scut.histogram_matrix(s_order).astype(np.float64))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Xs)); nv = len(Xs) // 10
    va, tr = perm[:nv], perm[nv:]
    mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-6
    head = ScoreHead(Xs.shape[1], 5).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(((Xs[tr] - mu) / sd).astype(np.float32)).to(device)
    xv = torch.from_numpy(((Xs[va] - mu) / sd).astype(np.float32)).to(device)
    batch = {"hist": torch.from_numpy(tgt["hist"][tr]).float().to(device)}
    best, state, bad = -np.inf, None, 0
    for ep in range(400):
        head.train(); loss = loss_distribution(head(xt), batch)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0:
            head.eval()
            with torch.no_grad():
                s = to_score("distribution", head(xv)).cpu().numpy()
            rho = stats.spearmanr(tgt["mean"][va], s).statistic
            if rho > best:
                best, bad, state = rho, 0, {k: v.clone() for k, v in head.state_dict().items()}
            else:
                bad += 1
                if bad > 10: break
    head.load_state_dict(state); head.eval()
    with torch.no_grad():
        beauty = to_score("distribution",
                          head(torch.from_numpy(((Xf - mu) / sd).astype(np.float32)).to(device))
                          ).cpu().numpy().astype(np.float64)

    pct = stats.rankdata(beauty) / len(beauty)
    results["d_attractiveness"] = {
        "by_race": by_group(None, race, lambda m: {
            "mean_score": float(beauty[m].mean()),
            "mean_percentile": float(pct[m].mean()),
        }),
        "by_group": by_group(None, group, lambda m: {
            "mean_score": float(beauty[m].mean()), "mean_percentile": float(pct[m].mean()),
        }),
    }
    for g, v in sorted(results["d_attractiveness"]["by_race"].items(),
                       key=lambda kv: kv[1]["mean_percentile"]):
        print(f"  {g:<18} n={v['n']:<5} mean={v['mean_score']:.3f} pct={v['mean_percentile']:.3f}")
    rspread = spread(results["d_attractiveness"]["by_race"], "mean_percentile")
    gspread = spread(results["d_attractiveness"]["by_group"], "mean_percentile")
    results["d_attractiveness"]["percentile_spread_race"] = rspread
    results["d_attractiveness"]["percentile_spread_group"] = gspread
    print(f"  -> percentile spread: {rspread:.3f} across races, {gspread:.3f} across race x gender")
    top = np.argsort(-beauty)[:100]
    comp = {r: int((race[top] == r).sum()) for r in sorted(set(race.tolist()))}
    expected = {r: round(100 * (race == r).mean(), 1) for r in comp}
    results["d_attractiveness"]["top100_composition"] = comp
    results["d_attractiveness"]["top100_expected_if_uniform"] = expected
    print("  top-100 composition (vs share of the balanced set):")
    for r in comp:
        print(f"    {r:<18} {comp[r]:>3}  (expected ~{expected[r]:.0f})")

    # -------------------------------------------- (e) the quality confound
    print("\n=== (e) CONFOUND: attractiveness vs image quality ===")
    conf = {
        "quality_composite": float(stats.spearmanr(beauty, qual).statistic),
        "blur": float(stats.spearmanr(beauty, sig["blur"]).statistic),
        "contrast": float(stats.spearmanr(beauty, sig["contrast"]).statistic),
        "luminance": float(stats.spearmanr(beauty, sig["luminance"]).statistic),
        "face_pixels": float(stats.spearmanr(beauty, face_px).statistic),
        "det_score": float(stats.spearmanr(beauty, det_score).statistic),
    }
    results["e_confound"] = conf
    for k, v in sorted(conf.items(), key=lambda kv: -abs(kv[1])):
        print(f"  corr(attractiveness, {k:<18}) = {v:+.3f}")
    # How much of the attractiveness variance do pure photography signals explain?
    Q = np.stack([sig["blur"], sig["contrast"], sig["luminance"], face_px, det_score,
                  sig["clipped_dark"], sig["clipped_bright"]], axis=1)
    Q = (Q - Q.mean(0)) / (Q.std(0) + 1e-9)
    Q = np.hstack([Q, np.ones((len(Q), 1))])
    coef, *_ = np.linalg.lstsq(Q, beauty, rcond=None)
    r2 = 1.0 - ((beauty - Q @ coef) ** 2).sum() / ((beauty - beauty.mean()) ** 2).sum()
    results["e_confound"]["photography_only_r2"] = float(r2)
    print(f"  -> R^2 of attractiveness predicted from photography signals ALONE: {r2:.4f}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    manifest.metrics = results
    manifest.finish(out)
    print(f"\n[ok] -> {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
