#!/usr/bin/env python
"""Experiment E14 - can we learn an individual's taste, and how many labels does it take?

E7 promoted this from a Phase-10 extra to a core question: two rater pools disagreed on
80% of a top-100, so a single population ranking cannot be the honest answer. The obvious
fix is a per-user model. This asks whether that actually works, and at what label cost.

Simulating users is possible because both datasets ship per-rater scores:
    SCUT-FBP5500  60 raters x 5500 images, COMPLETE matrix (330,000 ratings)
    MEBeauty      360 raters x 2520 images, sparse (61,404 ratings)

Three models, all over the same cached frozen features:
    population  ridge fit on the mean of the OTHER raters (leave-one-rater-out, so the
                target rater's own ratings never enter the population target)
    personal    ridge fit only on this rater's n labels
    residual    population prediction + ridge on (this rater's rating - population
                prediction), i.e. the decomposition in docs/RESEARCH.md section 15.5

Two ceilings, because "how good is good" is not obvious for a noisy human target:
    self-consistency  test-retest reliability from SCUT's re-rated images. Seven raters
                      rated all 5500 images twice.
    attenuation limit sqrt(reliability) - the maximum correlation ANY predictor can have
                      with a single noisy rating. Comparing a model to 1.0 is meaningless;
                      this is the real bar.

Usage:
    python scripts/run_e14_personalisation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.data.mebeauty import MEBeauty  # noqa: E402
from facet.data.scut_fbp5500 import ScutFbp5500  # noqa: E402
from facet.evaluation.metrics import pairwise_accuracy  # noqa: E402
from facet.utils.run import RunManifest  # noqa: E402
from facet.utils.seed import seed_everything  # noqa: E402

FEATURE_DIR = ROOT / "artifacts/features"
SUPPORT_SIZES = (0, 10, 25, 50, 100, 250, 500, 1000, 2000)


def load_features(dataset: str, parts: tuple[str, ...], margin=0.0, size=112):
    mats, order = [], None
    for p in parts:
        prefix = "" if dataset == "scut" else f"{dataset}__"
        stem = f"{prefix}{p}__m{margin:g}__s{size}"
        arr = np.load(FEATURE_DIR / f"{stem}.npy").astype(np.float64)
        names = json.loads((FEATURE_DIR / f"{stem}.json").read_text())["filenames"]
        if order is None:
            order = names
        elif names != order:
            raise RuntimeError("feature ordering mismatch")
        arr /= np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)
        mats.append(arr)
    return np.hstack(mats), order


def self_consistency(ds: ScutFbp5500) -> dict:
    """Test-retest reliability from SCUT's repeated ratings.

    ~10% of faces were re-shown to raters during annotation; seven raters ended up rating
    the entire set twice. Correlating the two passes gives each rater's reliability, and
    sqrt(reliability) is the attenuation limit - the best correlation any model could
    achieve against a single one of their noisy ratings.
    """
    sub = ds.ratings[ds.ratings["original Rating"].notna()]
    per = {}
    for rid, g in sub.groupby("Rater"):
        if len(g) >= 30:
            rho = float(stats.spearmanr(g["Rating"], g["original Rating"]).statistic)
            per[int(rid)] = {"n": int(len(g)), "test_retest_spearman": rho}
    vals = [v["test_retest_spearman"] for v in per.values()]
    mean_rel = float(np.mean(vals))
    return {
        "n_raters_with_repeats": len(per),
        "mean_test_retest_spearman": mean_rel,
        "median_test_retest_spearman": float(np.median(vals)),
        "attenuation_limit": float(np.sqrt(max(mean_rel, 0.0))),
        "per_rater": per,
    }


def fit_ridge(X, y, alpha):
    sc = StandardScaler().fit(X)
    return sc, Ridge(alpha=alpha).fit(sc.transform(X), y)


def run_dataset(name, X, R, train_idx, test_idx, alpha, seeds=3, min_support=10):
    """R is (n_images, n_raters) with NaN for unrated. Returns per-rater learning curves."""
    n_raters = R.shape[1]
    curves: dict[str, dict] = {}
    consensus_fit = []

    for j in range(n_raters):
        rated = np.isfinite(R[:, j])
        tr = np.array([i for i in train_idx if rated[i]])
        te = np.array([i for i in test_idx if rated[i]])
        if len(tr) < min_support or len(te) < 30:
            continue

        y_self_te = R[te, j]
        if np.ptp(y_self_te) == 0:
            continue

        # ---- population model: target is the mean of all OTHER raters --------------
        others = np.delete(R, j, axis=1)
        with np.errstate(invalid="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pop_target = np.nanmean(others, axis=1)
        ok = np.isfinite(pop_target)
        tr_pop = np.array([i for i in train_idx if ok[i]])
        sc, pop = fit_ridge(X[tr_pop], pop_target[tr_pop], alpha)
        pop_te = pop.predict(sc.transform(X[te]))

        rho_pop = float(stats.spearmanr(y_self_te, pop_te).statistic)
        consensus_fit.append(rho_pop)

        entry = {
            "n_train_available": int(len(tr)),
            "n_test": int(len(te)),
            "population_spearman": rho_pop,
            "population_pairwise": pairwise_accuracy(y_self_te, pop_te, n_pairs=20000),
            "curve": {},
        }

        rng = np.random.default_rng(1000 + j)
        for n in SUPPORT_SIZES:
            if n == 0:
                entry["curve"]["0"] = {
                    "personal_spearman": float("nan"),
                    "residual_spearman": rho_pop,
                    "residual_pairwise": entry["population_pairwise"],
                }
                continue
            if n > len(tr):
                continue
            pers_r, resid_r, resid_p = [], [], []
            for _ in range(seeds):
                sup = rng.choice(tr, size=n, replace=False)
                y_sup = R[sup, j]
                if np.ptp(y_sup) == 0:
                    continue
                # personal-only
                sc_p, mp = fit_ridge(X[sup], y_sup, alpha)
                pred_p = mp.predict(sc_p.transform(X[te]))
                pers_r.append(stats.spearmanr(y_self_te, pred_p).statistic)
                # residual on top of the population model
                # Residual formulation: learn only this user's DEVIATION from the
                # population prediction, so the model degrades to the population model
                # when the user has given few labels (docs/RESEARCH.md 15.5).
                resid_target = y_sup - pop.predict(sc.transform(X[sup]))
                sc_r, mr = fit_ridge(X[sup], resid_target, alpha)
                pred_r = pop_te + mr.predict(sc_r.transform(X[te]))
                resid_r.append(stats.spearmanr(y_self_te, pred_r).statistic)
                resid_p.append(pairwise_accuracy(y_self_te, pred_r, n_pairs=20000))
            if pers_r:
                entry["curve"][str(n)] = {
                    "personal_spearman": float(np.nanmean(pers_r)),
                    "residual_spearman": float(np.nanmean(resid_r)),
                    "residual_pairwise": float(np.nanmean(resid_p)),
                }
        curves[str(j)] = entry

    return curves, float(np.mean(consensus_fit)) if consensus_fit else float("nan")


def aggregate(curves: dict) -> dict:
    """Mean learning curve across simulated users.

    The population baseline is recomputed on the SAME user cohort available at each
    support size. Users differ in how many images they rated, so the cohort shrinks as n
    grows; comparing a large-n mean against the all-users baseline would be comparing
    different populations and can invent or hide an effect entirely.
    """
    agg: dict[str, dict] = {}
    for n in SUPPORT_SIZES:
        k = str(n)
        sub = [
            c for c in curves.values()
            if k in c["curve"] and np.isfinite(c["curve"][k]["residual_spearman"])
        ]
        if len(sub) < 15:
            continue
        pop_matched = float(np.mean([c["population_spearman"] for c in sub]))
        res = float(np.mean([c["curve"][k]["residual_spearman"] for c in sub]))
        agg[k] = {
            "n_users": len(sub),
            "personal_spearman": float(np.nanmean([c["curve"][k]["personal_spearman"] for c in sub])),
            "residual_spearman": res,
            "population_spearman_matched_cohort": pop_matched,
            "gain_vs_population": res - pop_matched,
            "fraction_of_users_helped": float(
                np.mean([c["curve"][k]["residual_spearman"] > c["population_spearman"] for c in sub])
            ),
        }
    pop = float(np.mean([c["population_spearman"] for c in curves.values()]))
    beats = None
    for n in SUPPORT_SIZES:
        k = str(n)
        if n > 0 and k in agg and agg[k]["gain_vs_population"] > 0:
            beats = n
            break
    return {"population_spearman": pop, "curve": agg, "labels_to_beat_population": beats}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scut-root", default=str(ROOT / "data/raw/SCUT-FBP5500_v2"))
    ap.add_argument("--me-root", default=str(ROOT / "data/raw/MEBeauty"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/e14_personalisation"))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    seed_everything(1337)
    manifest = RunManifest(
        experiment="e14_personalisation",
        description="Is there learnable individual taste beyond the population consensus?",
        config=vars(args),
        dataset="SCUT-FBP5500 (60 raters) + MEBeauty (360 raters)",
        split_methodology=(
            "Simulated users = individual raters. Population target is the mean of all "
            "OTHER raters (leave-one-rater-out), so a rater's own labels never leak into "
            "the population model. Image split is SCUT's official cv1; MEBeauty uses its "
            "official split."
        ),
    )
    results: dict = {}

    # ------------------------------------------------------------------ ceilings
    scut = ScutFbp5500(args.scut_root)
    sc_ceiling = self_consistency(scut)
    results["ceilings"] = sc_ceiling
    print("=== Ceilings (SCUT-FBP5500) ===")
    print(f"  raters who rated everything twice : {sc_ceiling['n_raters_with_repeats']}")
    print(f"  test-retest self-consistency      : {sc_ceiling['mean_test_retest_spearman']:.4f}")
    print(f"  attenuation limit sqrt(reliability): {sc_ceiling['attenuation_limit']:.4f}")
    print("  -> no model can correlate above the attenuation limit with a single rating\n")

    X, order = load_features("scut", ("arcface_buffalo_l", "clip"))
    idx = {n: i for i, n in enumerate(order)}
    piv = scut.ratings.pivot_table(
        index="Filename", columns="Rater", values="Rating", aggfunc="mean"
    ).reindex(order)
    R = piv.to_numpy(dtype=float)

    split = scut.splits["cv1"]
    tr_idx = np.array([idx[n] for n in split.train])
    te_idx = np.array([idx[n] for n in split.test])

    # one alpha, chosen once on the population target, reused everywhere for comparability
    with np.errstate(invalid="ignore"):
        pop_all = np.nanmean(R, axis=1)
    sc0 = StandardScaler().fit(X[tr_idx])
    alpha = float(RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc0.transform(X[tr_idx]),
                                                            pop_all[tr_idx]).alpha_)
    print(f"ridge alpha (chosen once on the population target): {alpha:g}\n")

    print("=== SCUT-FBP5500: 60 simulated users ===")
    curves, cons = run_dataset("scut", X, R, tr_idx, te_idx, alpha, seeds=args.seeds)
    scut_agg = aggregate(curves)
    results["scut"] = {"aggregate": scut_agg, "per_user": curves}

    print(f"{'labels':>7} {'users':>6} {'pop(matched)':>13} {'personal':>10} "
          f"{'residual':>10} {'gain':>9} {'helped':>8}")
    print("-" * 68)
    for n in SUPPORT_SIZES:
        k = str(n)
        if k not in scut_agg["curve"]:
            continue
        c = scut_agg["curve"][k]
        p = "-" if n == 0 else f"{c['personal_spearman']:.4f}"
        print(f"{n:>7} {c['n_users']:>6} {c['population_spearman_matched_cohort']:>13.4f} "
              f"{p:>10} {c['residual_spearman']:>10.4f} {c['gain_vs_population']:>+9.4f} "
              f"{100*c['fraction_of_users_helped']:>7.0f}%")
    print("-" * 68)
    print(f"population-only (0 labels)          : {scut_agg['population_spearman']:.4f}")
    print(f"attenuation limit                   : {sc_ceiling['attenuation_limit']:.4f}")
    print(f"labels needed to beat population    : {scut_agg['labels_to_beat_population']}")

    # ------------------------------------------------------------------ MEBeauty
    print("\n=== MEBeauty: simulated users (sparse ratings, diverse pool) ===")
    me = MEBeauty(args.me_root)
    Xm, me_order = load_features("mebeauty", ("arcface_buffalo_l", "clip"))
    Rm, _ = me.rater_matrix()
    midx = {k: i for i, k in enumerate(me.keys)}
    msp = me.splits["official"]
    mtr = np.array([midx[k] for k in msp.train])
    mte = np.array([midx[k] for k in msp.test])
    me_curves, me_cons = run_dataset("mebeauty", Xm, Rm, mtr, mte, alpha,
                                     seeds=args.seeds, min_support=10)
    me_agg = aggregate(me_curves)
    results["mebeauty"] = {"aggregate": me_agg, "per_user": me_curves}
    print(f"  simulated users with enough data: {len(me_curves)}")
    print(f"{'labels':>7} {'users':>6} {'pop(matched)':>13} {'personal':>10} "
          f"{'residual':>10} {'gain':>9} {'helped':>8}")
    print("-" * 68)
    for n in SUPPORT_SIZES:
        k = str(n)
        if k not in me_agg["curve"]:
            continue
        c = me_agg["curve"][k]
        p = "-" if n == 0 else f"{c['personal_spearman']:.4f}"
        print(f"{n:>7} {c['n_users']:>6} {c['population_spearman_matched_cohort']:>13.4f} "
              f"{p:>10} {c['residual_spearman']:>10.4f} {c['gain_vs_population']:>+9.4f} "
              f"{100*c['fraction_of_users_helped']:>7.0f}%")
    print("-" * 68)
    print(f"population-only : {me_agg['population_spearman']:.4f}")
    print(f"labels needed to beat population: {me_agg['labels_to_beat_population']}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    manifest.metrics = {
        "ceilings": sc_ceiling,
        "scut_aggregate": scut_agg,
        "mebeauty_aggregate": me_agg,
    }
    manifest.finish(out)
    print(f"\n[ok] -> {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
