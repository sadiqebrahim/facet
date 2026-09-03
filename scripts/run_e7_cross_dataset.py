#!/usr/bin/env python
"""Experiment E7 - cross-dataset generalisation, SCUT-FBP5500 -> MEBeauty.

THE GATE. Everything before this is research on one saturated benchmark whose human
inter-rater ceiling is r~0.77; this asks whether any of it transfers to images that do
not look like that benchmark. If nothing transfers, the product does not work, and no
amount of in-benchmark PC fixes it (docs/RESEARCH.md section 14).

Because the rating scales differ (SCUT 1-5, MEBeauty 1-10) everything is reported by RANK
correlation. Comparing MAE across the two would be meaningless.

Four arms, because a raw cross-dataset number is uninterpretable on its own:

  A  train SCUT      -> test MEBeauty   the actual question
  B  train MEBeauty  -> test MEBeauty   within-dataset upper bound on the SAME test set,
                                        so we can separate "domain shift" from "this data
                                        is just harder"
  C  train MEBeauty  -> test SCUT       the reverse direction, for symmetry
  D  human ceiling                      split the rater pool in half and correlate; the
                                        only fair yardstick for either dataset

It also breaks arm A down by ethnicity, splitting MEBeauty into the groups SCUT-FBP5500
contains (Asian, Caucasian) and the four it contains NONE of (Black, Indian, Hispanic,
Middle Eastern). That is a direct test of the out-of-distribution claim in section 13.1.3.

Usage:
    python scripts/run_e7_cross_dataset.py
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
from facet.evaluation.metrics import kendall_tau, ndcg_at_k, pairwise_accuracy  # noqa: E402
from facet.training.heads import DistributionHead, RidgeRegressionHead  # noqa: E402
from facet.utils.run import RunManifest  # noqa: E402
from facet.utils.seed import seed_everything  # noqa: E402

FEATURE_DIR = ROOT / "artifacts/features"

REPRESENTATIONS: dict[str, tuple[str, ...]] = {
    "arcface_buffalo_l": ("arcface_buffalo_l",),
    "arcface_antelopev2": ("arcface_antelopev2",),
    "clip": ("clip",),
    "arcface_r50+clip": ("arcface_buffalo_l", "clip"),
}


def load_block(dataset: str, key: str, margin: float, size: int):
    prefix = "" if dataset == "scut" else f"{dataset}__"
    stem = f"{prefix}{key}__m{margin:g}__s{size}"
    arr = np.load(FEATURE_DIR / f"{stem}.npy").astype(np.float64)
    meta = json.loads((FEATURE_DIR / f"{stem}.json").read_text())
    arr /= np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)
    return arr, meta["filenames"]


def build(dataset: str, parts: tuple[str, ...], margin: float, size: int):
    mats, order = [], None
    for p in parts:
        arr, names = load_block(dataset, p, margin, size)
        if order is None:
            order = names
        elif names != order:
            raise RuntimeError(f"feature ordering mismatch in {dataset}/{p}")
        mats.append(arr)
    return np.hstack(mats), order


def rank_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> dict[str, float]:
    """Scale-free metrics only - the two datasets use different rating scales."""
    return {
        "spearman": float(stats.spearmanr(y_true, y_pred).statistic),
        "kendall_tau": kendall_tau(y_true, y_pred),
        "pairwise_acc": pairwise_accuracy(y_true, y_pred),
        f"ndcg@{k}": ndcg_at_k(y_true, y_pred, k=k),
        "n": int(len(y_true)),
    }


def cross_group_calibration(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, min_n: int = 30
) -> dict:
    """Do predictions preserve BETWEEN-group score levels, not just within-group order?

    Spearman computed inside each group is blind to this: a model can rank perfectly
    within every group while systematically placing one group too high or too low
    overall. For a product that ranks all faces together and shows the top N, that
    systematic offset is what decides who gets surfaced - so it must be measured
    separately.

    Both series are converted to percentile ranks over the whole set, so the 1-5 vs
    1-10 scale difference is removed and `bias` reads directly as "this group is placed
    N percentile points higher/lower than the raters placed it".
    """
    tr = stats.rankdata(y_true) / len(y_true)
    pr = stats.rankdata(y_pred) / len(y_pred)
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if m.sum() < min_n:
            continue
        out[str(g)] = {
            "n": int(m.sum()),
            "true_mean_percentile": float(tr[m].mean()),
            "pred_mean_percentile": float(pr[m].mean()),
            "bias": float(pr[m].mean() - tr[m].mean()),
        }
    biases = [v["bias"] for v in out.values()]
    return {
        "per_group": out,
        "max_abs_bias": float(max(abs(b) for b in biases)) if biases else float("nan"),
        "bias_spread": float(max(biases) - min(biases)) if biases else float("nan"),
    }


def topk_composition(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k: int = 100
) -> dict:
    """Who actually gets surfaced in a top-k selection, predicted vs. ground truth.

    This is the product metric. A user of a "find the best faces" tool sees the top k and
    nothing else, so agreement on that set - and its demographic composition - matters
    more than global correlation.
    """
    top_pred = set(np.argsort(-y_pred)[:k].tolist())
    top_true = set(np.argsort(-y_true)[:k].tolist())
    comp = {}
    for g in sorted(set(groups.tolist())):
        idx = set(np.where(groups == g)[0].tolist())
        comp[str(g)] = {
            "in_true_topk": len(idx & top_true),
            "in_pred_topk": len(idx & top_pred),
        }
    return {"k": k, "overlap": len(top_pred & top_true), "composition": comp}


def human_ceiling(R: np.ndarray, n_trials: int = 200, seed: int = 0) -> dict[str, float]:
    """Split-half rater reliability: correlate two disjoint halves of the rater pool.

    This is the fair yardstick. A model cannot meaningfully be said to "beat humans" on a
    target that is itself a noisy human aggregate - and a model's rank correlation should
    be read relative to this number, not to 1.0.
    """
    rng = np.random.default_rng(seed)
    n_raters = R.shape[1]
    out = []
    for _ in range(n_trials):
        perm = rng.permutation(n_raters)
        a, b = perm[: n_raters // 2], perm[n_raters // 2 :]
        with np.errstate(invalid="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ma = np.nanmean(R[:, a], axis=1)
                mb = np.nanmean(R[:, b], axis=1)
        ok = np.isfinite(ma) & np.isfinite(mb)
        if ok.sum() > 30:
            out.append(stats.spearmanr(ma[ok], mb[ok]).statistic)
    arr = np.array(out)
    half = float(arr.mean())
    # Spearman-Brown: correct a half-pool correlation up to the full pool.
    return {
        "split_half_spearman": half,
        "spearman_brown_full_pool": float(2 * half / (1 + half)),
        "n_trials": len(arr),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scut-root", default=str(ROOT / "data/raw/SCUT-FBP5500_v2"))
    ap.add_argument("--me-root", default=str(ROOT / "data/raw/MEBeauty"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/e7_cross_dataset"))
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    seed_everything(args.seed)
    manifest = RunManifest(
        experiment="e7_cross_dataset",
        description="Does anything learned on SCUT-FBP5500 transfer to MEBeauty?",
        config=vars(args),
        seed=args.seed,
        dataset="SCUT-FBP5500_v2 -> MEBeauty",
        dataset_version="SCUT v2 (2018) / MEBeauty 2022 scores",
        split_methodology=(
            "Train on ALL of SCUT-FBP5500 (5500). Test on ALL of MEBeauty (2520), which "
            "is fully held out - no MEBeauty image influences the SCUT-trained model. "
            "Within-MEBeauty arm uses MEBeauty's own official train/test split. Rank "
            "metrics only: the rating scales differ (1-5 vs 1-10)."
        ),
    )

    scut = ScutFbp5500(args.scut_root)
    me = MEBeauty(args.me_root)
    results: dict = {}

    # ---------------------------------------------------------------- arm D: humans
    R, _ = me.rater_matrix()
    results["human_ceiling_mebeauty"] = human_ceiling(R)
    print("=== D. Human ceiling (split-half rater reliability) ===")
    print(f"  MEBeauty  split-half rho = {results['human_ceiling_mebeauty']['split_half_spearman']:.4f}"
          f"   full-pool (Spearman-Brown) = "
          f"{results['human_ceiling_mebeauty']['spearman_brown_full_pool']:.4f}")
    print("  SCUT-FBP5500 published inter-group rater correlation = 0.770 (paper Table III)")
    print()

    me_keys = me.keys
    y_me = me.labels.loc[me_keys, "mean"].to_numpy(np.float64)
    eth = me.labels.loc[me_keys, "ethnicity"].to_numpy()
    in_scut = me.labels.loc[me_keys, "in_scut_distribution"].to_numpy()

    print(f"{'representation':<22} {'arm':<28} {'rho':>8} {'tau':>8} {'pair':>8} {'ndcg':>8} {'n':>6}")
    print("-" * 92)

    for rep, parts in REPRESENTATIONS.items():
        Xs, scut_names = build("scut", parts, args.margin, args.size)
        Xm, me_names = build("mebeauty", parts, args.margin, args.size)
        assert me_names == me_keys, "MEBeauty feature order does not match loader order"

        y_scut = scut.labels.loc[scut_names, "mean"].to_numpy(np.float64)
        P_scut = scut.histogram_matrix(scut_names).astype(np.float64)

        entry: dict = {}

        # ---- arm A: train SCUT (all), test MEBeauty (all) -------------------------
        ridge = RidgeRegressionHead().fit(Xs, y_scut)
        pred_a = ridge.predict(Xm)
        ldl = DistributionHead().fit(Xs, P_scut)
        pred_a_ldl = ldl.predict(Xm)
        entry["A_scut_to_mebeauty"] = rank_metrics(y_me, pred_a)
        entry["A_scut_to_mebeauty_ldl"] = rank_metrics(y_me, pred_a_ldl)

        # by ethnicity, and by in/out of SCUT's training distribution
        entry["A_by_ethnicity"] = {
            e: rank_metrics(y_me[eth == e], pred_a[eth == e], k=50)
            for e in sorted(set(eth.tolist()))
        }
        entry["A_in_scut_distribution"] = rank_metrics(y_me[in_scut], pred_a[in_scut])
        entry["A_out_of_scut_distribution"] = rank_metrics(y_me[~in_scut], pred_a[~in_scut])

        # Within-group ranking can look fine while between-group placement is badly off.
        group = me.labels.loc[me_keys, "group"].to_numpy()
        entry["A_cross_group_calibration"] = cross_group_calibration(y_me, pred_a, group)
        entry["A_cross_ethnicity_calibration"] = cross_group_calibration(y_me, pred_a, eth)
        entry["A_top100"] = topk_composition(y_me, pred_a, eth, k=100)

        # ---- arm B: train MEBeauty, test MEBeauty (upper bound on same test set) ---
        sp = me.splits["official"]
        idx = {k: i for i, k in enumerate(me_keys)}
        tr = np.array([idx[k] for k in sp.train])
        te = np.array([idx[k] for k in sp.test])
        ridge_b = RidgeRegressionHead().fit(Xm[tr], y_me[tr])
        entry["B_mebeauty_to_mebeauty"] = rank_metrics(y_me[te], ridge_b.predict(Xm[te]))
        # arm A restricted to the same test rows, so B and A are directly comparable
        entry["A_on_B_testset"] = rank_metrics(y_me[te], pred_a[te])

        # ---- arm C: train MEBeauty (all), test SCUT (all) -------------------------
        ridge_c = RidgeRegressionHead().fit(Xm, y_me)
        entry["C_mebeauty_to_scut"] = rank_metrics(y_scut, ridge_c.predict(Xs))

        results[rep] = entry

        for arm in ("A_scut_to_mebeauty", "A_on_B_testset", "B_mebeauty_to_mebeauty",
                    "C_mebeauty_to_scut"):
            m = entry[arm]
            print(f"{rep:<22} {arm:<28} {m['spearman']:>8.4f} {m['kendall_tau']:>8.4f} "
                  f"{m['pairwise_acc']:>8.4f} {m['ndcg@100']:>8.4f} {m['n']:>6}")
        print("-" * 92)

    # ---------------------------------------------------------------- OOD summary
    print("\n=== Arm A by ethnicity (SCUT-trained model on MEBeauty), Spearman ===")
    best = "arcface_r50+clip"
    hdr = sorted(results[best]["A_by_ethnicity"])
    print(f"{'representation':<22} " + " ".join(f"{e:>11}" for e in hdr))
    for rep in REPRESENTATIONS:
        row = results[rep]["A_by_ethnicity"]
        print(f"{rep:<22} " + " ".join(f"{row[e]['spearman']:>11.4f}" for e in hdr))
    print()
    print(f"{'representation':<22} {'in-SCUT-dist':>14} {'out-of-dist':>14} {'gap':>8}")
    for rep in REPRESENTATIONS:
        a = results[rep]["A_in_scut_distribution"]["spearman"]
        b = results[rep]["A_out_of_scut_distribution"]["spearman"]
        print(f"{rep:<22} {a:>14.4f} {b:>14.4f} {a-b:>8.4f}")

    print("\n=== Cross-group CALIBRATION (what within-group Spearman hides) ===")
    print(f"{'representation':<22} {'max|bias|':>11} {'bias spread':>13} {'top100 overlap':>16}")
    for rep in REPRESENTATIONS:
        c = results[rep]["A_cross_group_calibration"]
        t = results[rep]["A_top100"]
        print(f"{rep:<22} {c['max_abs_bias']:>11.3f} {c['bias_spread']:>13.3f} "
              f"{t['overlap']:>13}/100")

    print(f"\n=== {best}: per-group placement bias (percentile points) ===")
    pg = results[best]["A_cross_group_calibration"]["per_group"]
    print(f"{'group':<22} {'n':>5} {'true %ile':>10} {'pred %ile':>10} {'bias':>8}")
    for g, v in sorted(pg.items(), key=lambda kv: kv[1]["bias"]):
        print(f"{g:<22} {v['n']:>5} {v['true_mean_percentile']:>10.3f} "
              f"{v['pred_mean_percentile']:>10.3f} {v['bias']:>+8.3f}")

    print(f"\n=== {best}: top-100 composition (who gets surfaced) ===")
    comp = results[best]["A_top100"]["composition"]
    print(f"{'ethnicity':<14} {'in true top100':>15} {'in pred top100':>15}")
    for g, v in comp.items():
        print(f"{g:<14} {v['in_true_topk']:>15} {v['in_pred_topk']:>15}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    manifest.metrics = {
        rep: {
            "A_scut_to_mebeauty_spearman": results[rep]["A_scut_to_mebeauty"]["spearman"],
            "B_within_mebeauty_spearman": results[rep]["B_mebeauty_to_mebeauty"]["spearman"],
            "C_mebeauty_to_scut_spearman": results[rep]["C_mebeauty_to_scut"]["spearman"],
        }
        for rep in REPRESENTATIONS
    }
    manifest.metrics["human_ceiling_mebeauty"] = results["human_ceiling_mebeauty"]
    manifest.notes = [
        "Rank metrics only: SCUT is 1-5, MEBeauty is 1-10.",
        "MEBeauty is fully held out from the SCUT-trained model.",
        "Read model rho relative to the human ceiling, not to 1.0.",
    ]
    manifest.finish(out)
    print(f"\n[ok] -> {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
