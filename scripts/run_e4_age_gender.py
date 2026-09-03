#!/usr/bin/env python
"""Experiment E4 - benchmark off-the-shelf age/gender models.

docs/RESEARCH.md 2.5 recommends adopting MiVOLO rather than training an age model, on the
strength of published numbers and an Apache-2.0 licence. Published numbers are not a valid
basis for selection (1.2), so this measures both candidates ourselves, on one balanced set,
with per-group reporting.

    mivolo_v2      MiVOLO d1 384, face-only mode, Apache-2.0
    insightface    the genderage model already on disk, research-only licence

Crop protocol is swept per model rather than fixed, because the two were trained on
different framings and a benchmark that feeds one model the other's preferred crop measures
the crop, not the model.

CONTAMINATION WARNING, stated up front: MiVOLO v2's card says it was trained on
"proprietary and open-source datasets" without enumerating them, so FairFace may be in its
training data. The InsightFace baseline is certainly not trained on FairFace. That asymmetry
favours MiVOLO and cannot be resolved from the outside - the numbers below are an upper
bound on MiVOLO's advantage, not a clean comparison.

Usage:
    python scripts/run_e4_age_gender.py
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

from facet.data.fairface import FairFace  # noqa: E402
from facet.models.insightface_backend import (  # noqa: E402
    GenderAgePredictor, align_to_template, crop_bbox,
)
from facet.models.mivolo_backend import MiVOLOPredictor  # noqa: E402
from facet.utils.run import RunManifest  # noqa: E402
from facet.utils.seed import seed_everything  # noqa: E402

DET_DIR = ROOT / "artifacts/detections"


def build(ff, names, det, kind, size):
    """kind: 'full' = whole image, 'bbox' = detector box crop, 'align' = 5-point template."""
    out = np.zeros((len(names), size, size, 3), dtype=np.uint8)
    for i, n in enumerate(names):
        img = cv2.imread(str(ff.image_path(n)))
        ok = det["scores"][i] > 0 and np.isfinite(det["kps"][i]).all()
        if kind == "full" or not ok:
            out[i] = cv2.resize(img, (size, size))
        elif kind == "bbox":
            out[i] = crop_bbox(img, tuple(det["bboxes"][i]), size=size, margin=0.4)
        else:
            out[i] = align_to_template(img, det["kps"][i], size=size, margin=0.25)
    return out


def report(pred_f, age, lab, tag):
    true_f = (lab["gender_label"] == "Female").to_numpy()
    correct = (pred_f >= 0.5) == true_f
    age_t = lab["age_mid"].to_numpy(np.float64)
    err = np.abs(age - age_t)
    race = lab["race_label"].to_numpy()
    bucket = lab["age_bucket"].to_numpy()

    def grp(keys, vals, fn):
        o = {}
        for g in sorted(set(keys.tolist())):
            m = keys == g
            if m.sum() >= 30:
                o[str(g)] = {"n": int(m.sum()), **fn(m)}
        return o

    return {
        "tag": tag,
        "gender_accuracy": float(correct.mean()),
        "age_mae": float(err.mean()),
        "gender_by_race": grp(race, None, lambda m: {"accuracy": float(correct[m].mean())}),
        "age_by_race": grp(race, None, lambda m: {"mae": float(err[m].mean())}),
        "age_by_bucket": grp(bucket, None, lambda m: {"mae": float(err[m].mean())}),
    }


def spread(d, k):
    v = [x[k] for x in d.values()]
    return float(max(v) - min(v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ff-root", default=str(ROOT / "data/raw/FairFace"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/e4_age_gender"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    seed_everything(1337)
    manifest = RunManifest(
        experiment="e4_age_gender",
        description="MiVOLO vs InsightFace genderage on FairFace, per-group",
        config=vars(args),
        dataset="FairFace validation (balanced)",
        split_methodology=(
            "Whole balanced validation split; both models evaluated on identical images. "
            "Crop protocol swept per model. Age compared against FairFace bucket midpoints. "
            "MiVOLO's training data is not fully disclosed - FairFace contamination possible."
        ),
    )

    ff = FairFace(args.ff_root)
    names = ff.filenames[: args.limit] if args.limit else ff.filenames
    lab = ff.labels.loc[names]
    det = np.load(DET_DIR / "fairface__buffalo_l.npz", allow_pickle=True)
    assert list(det["names"])[: len(names)] == names
    det = {k: det[k][: len(names)] for k in ("scores", "kps", "bboxes")}
    print(f"FairFace: {len(names)} images\n")

    results: dict = {}
    print(f"{'model':<14} {'crop':<7} {'gender acc':>11} {'age MAE':>9} "
          f"{'g-spread':>9} {'a-spread':>9} {'img/s':>7}")
    print("-" * 74)

    for kind in ("full", "bbox", "align"):
        crops = build(ff, names, det, kind, 384)
        m = MiVOLOPredictor()
        t0 = time.time()
        pf, ag = m.predict(crops, batch_size=64)
        sp = len(names) / (time.time() - t0)
        r = report(pf, ag, lab, f"mivolo_{kind}")
        r["images_per_sec"] = sp
        results[f"mivolo_{kind}"] = r
        print(f"{'mivolo_v2':<14} {kind:<7} {r['gender_accuracy']:>11.4f} {r['age_mae']:>9.2f} "
              f"{spread(r['gender_by_race'],'accuracy'):>9.4f} "
              f"{spread(r['age_by_race'],'mae'):>9.2f} {sp:>7.1f}")
        # `del` drops the Python reference but not the CUDA allocator's cached blocks, so
        # three sequential model loads otherwise pile up ~46 GB of VRAM.
        del m
        import torch
        torch.cuda.empty_cache()

    ga = GenderAgePredictor()
    for kind in ("bbox", "align"):
        crops = build(ff, names, det, kind, 112)
        t0 = time.time()
        pm, ag = ga.predict(crops)
        sp = len(names) / (time.time() - t0)
        r = report(1.0 - pm, ag, lab, f"insightface_{kind}")   # pm = P(male)
        r["images_per_sec"] = sp
        results[f"insightface_{kind}"] = r
        print(f"{'insightface':<14} {kind:<7} {r['gender_accuracy']:>11.4f} {r['age_mae']:>9.2f} "
              f"{spread(r['gender_by_race'],'accuracy'):>9.4f} "
              f"{spread(r['age_by_race'],'mae'):>9.2f} {sp:>7.1f}")

    bm = max((k for k in results if k.startswith("mivolo")),
             key=lambda k: results[k]["gender_accuracy"] - results[k]["age_mae"] / 100)
    bi = max((k for k in results if k.startswith("insightface")),
             key=lambda k: results[k]["gender_accuracy"] - results[k]["age_mae"] / 100)
    results["best_mivolo"], results["best_insightface"] = bm, bi
    print(f"\nbest MiVOLO config: {bm} | best InsightFace config: {bi}")

    print(f"\n=== gender accuracy by race ({bm} vs {bi}) ===")
    print(f"{'race':<18} {'MiVOLO':>9} {'InsightFace':>13} {'delta':>8}")
    for g in results[bm]["gender_by_race"]:
        a = results[bm]["gender_by_race"][g]["accuracy"]
        b = results[bi]["gender_by_race"][g]["accuracy"]
        print(f"  {g:<16} {a:>9.4f} {b:>13.4f} {a-b:>+8.4f}")

    print(f"\n=== age MAE by bucket ({bm} vs {bi}) ===")
    print(f"{'bucket':<10} {'MiVOLO':>9} {'InsightFace':>13} {'delta':>8}")
    for g in sorted(results[bm]["age_by_bucket"]):
        a = results[bm]["age_by_bucket"][g]["mae"]
        b = results[bi]["age_by_bucket"][g]["mae"]
        print(f"  {g:<8} {a:>9.2f} {b:>13.2f} {a-b:>+8.2f}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    manifest.metrics = results
    manifest.finish(out)
    print(f"\n[ok] -> {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
