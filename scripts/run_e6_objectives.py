#!/usr/bin/env python
"""Experiment E6 - which training objective should the beauty head use?

The core beauty experiment (docs/RESEARCH.md section 14). Five objectives over identical
frozen features, identical head architecture, optimiser, schedule, seed and early-stopping
criterion. Only the loss differs, so any difference is attributable to the objective.

    regression    MSE against the mean rating (the conventional baseline)
    ordinal       CORAL-style cumulative logits vs empirical P(rating > k)
    distribution  softmax + KL against the real 60-rater histogram (LDL)
    pairwise      Bradley-Terry on empirical preference probabilities
    hybrid        KL + lambda * Bradley-Terry

Judged on four things, not one:
    in-benchmark  SCUT official 5-fold: PC, MAE, Spearman, pairwise accuracy, NDCG@100
    transfer      train on all SCUT -> all held-out MEBeauty (the E7 lesson)
    distribution  KL to the true rating histogram, where the objective produces one
    uncertainty   does predicted spread track real rater disagreement?

Scores are affine-calibrated on the training split before MAE is computed, so that the
pairwise head - which has no inherent scale - is still comparable. Rank metrics are
unaffected by that calibration.

Uses the E5 production crop (margin 0.25).

Usage:
    python scripts/run_e6_objectives.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.data.mebeauty import MEBeauty  # noqa: E402
from facet.data.scut_fbp5500 import ScutFbp5500  # noqa: E402
from facet.evaluation.metrics import (  # noqa: E402
    kl_divergence, ndcg_at_k, pairwise_accuracy, regression_metrics,
)
from facet.training.objectives import (  # noqa: E402
    OBJECTIVES, ScoreHead, aleatoric_std, build_targets, empirical_preference,
    loss_hybrid, to_distribution, to_score,
)
from facet.utils.run import RunManifest  # noqa: E402
from facet.utils.seed import seed_everything  # noqa: E402

FEATURE_DIR = ROOT / "artifacts/features"
PARTS = ("arcface_buffalo_l", "clip")
MARGIN = 0.25  # E5 production protocol


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


def train_head(name, Xtr, Xva, tgt_tr, tgt_va, R_tr, device, epochs=400, patience=50,
               lr=1e-3, wd=1e-4, n_pairs=8192, seed=0):
    """Train one objective. Early-stops on validation Spearman - the same criterion for
    every arm, so no objective is given a home-field advantage in model selection."""
    torch.manual_seed(seed)
    width, loss_fn, needs_pairs = OBJECTIVES[name]
    head = ScoreHead(Xtr.shape[1], width).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)

    xt = torch.from_numpy(Xtr).to(device)
    xv = torch.from_numpy(Xva).to(device)
    batch = {
        "mean": torch.from_numpy(tgt_tr["mean"]).float().to(device),
        "hist": torch.from_numpy(tgt_tr["hist"]).float().to(device),
        "cum": torch.from_numpy(tgt_tr["cum"]).float().to(device),
    }
    y_va = tgt_va["mean"]
    rng = np.random.default_rng(seed)

    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        head.train()
        if needs_pairs:
            i = rng.integers(0, len(Xtr), n_pairs)
            j = rng.integers(0, len(Xtr), n_pairs)
            keep = i != j
            i, j = i[keep], j[keep]
            p = empirical_preference(R_tr, i, j)
            batch["pair_i"] = torch.from_numpy(i).long().to(device)
            batch["pair_j"] = torch.from_numpy(j).long().to(device)
            batch["pair_p"] = torch.from_numpy(p).float().to(device)
        out = head(xt)
        loss = loss_hybrid(out, batch) if name == "hybrid" else loss_fn(out, batch)
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % 5 == 0 or ep == epochs - 1:
            head.eval()
            with torch.no_grad():
                s = to_score(name, head(xv)).cpu().numpy()
            rho = stats.spearmanr(y_va, s).statistic
            if np.isfinite(rho) and rho > best:
                best, bad = rho, 0
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            else:
                bad += 1
                if bad > patience // 5:
                    break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head


def evaluate(name, head, X, y_true, hist_true, std_true, device, calib=None, k=100):
    with torch.no_grad():
        out = head(torch.from_numpy(X).to(device))
        s = to_score(name, out).cpu().numpy().astype(np.float64)
        dist = to_distribution(name, out)
        sd = aleatoric_std(name, out)
    s_cal = calib[0] * s + calib[1] if calib is not None else s
    m = regression_metrics(y_true, s_cal)
    res = {
        "pc": m["pc"], "mae": m["mae"], "rmse": m["rmse"], "spearman": m["spearman"],
        "pairwise_acc": pairwise_accuracy(y_true, s),
        f"ndcg@{k}": ndcg_at_k(y_true, s, k=k),
    }
    if dist is not None and hist_true is not None:
        res["kl_to_true_hist"] = kl_divergence(hist_true, dist.cpu().numpy())
    if sd is not None and std_true is not None:
        res["aleatoric_vs_true_std"] = float(
            stats.spearmanr(sd.cpu().numpy(), std_true).statistic
        )
    return res


def affine_calibration(s, y):
    """Least-squares scale+shift so MAE is comparable across objectives."""
    A = np.stack([s, np.ones_like(s)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scut-root", default=str(ROOT / "data/raw/SCUT-FBP5500_v2"))
    ap.add_argument("--me-root", default=str(ROOT / "data/raw/MEBeauty"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/e6_objectives"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--transfer-seeds", type=int, default=5,
                    help="repeat the transfer arm N times; the objectives differ by less "
                         "than seed noise, so a single run would be over-read")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = RunManifest(
        experiment="e6_objectives",
        description="Which training objective should the beauty head use?",
        config=vars(args), seed=args.seed,
        dataset="SCUT-FBP5500_v2 (+ MEBeauty transfer)",
        split_methodology=(
            "SCUT official 5-fold; 10% of each training split held out for early stopping "
            "(uniform criterion: validation Spearman). Transfer: train on all 5500 SCUT, "
            "test on all 2520 held-out MEBeauty. Crop margin 0.25 (E5 protocol)."
        ),
    )

    scut = ScutFbp5500(args.scut_root)
    me = MEBeauty(args.me_root)
    X, order = load("scut", PARTS)
    Xm, me_order = load("mebeauty", PARTS)
    idx = {n: i for i, n in enumerate(order)}

    hist = scut.histogram_matrix(order).astype(np.float64)
    tgt_all = build_targets(hist)
    std_all = scut.labels.loc[order, "std"].to_numpy(np.float64)
    y_all = tgt_all["mean"]
    ym = me.labels.loc[me_order, "mean"].to_numpy(np.float64)

    piv = scut.ratings.pivot_table(
        index="Filename", columns="Rater", values="Rating", aggfunc="mean"
    ).reindex(order)
    R = piv.to_numpy(dtype=np.float64)

    print(f"device={device}  features={X.shape}  crop margin={MARGIN}\n")
    results: dict = {}

    for name in OBJECTIVES:
        folds = []
        for k in range(1, 6):
            sp = scut.splits[f"cv{k}"]
            tr_all = np.array([idx[n] for n in sp.train])
            te = np.array([idx[n] for n in sp.test])
            rng = np.random.default_rng(args.seed + k)
            perm = rng.permutation(len(tr_all))
            n_va = len(tr_all) // 10
            va, tr = tr_all[perm[:n_va]], tr_all[perm[n_va:]]

            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xtr, Xva, Xte = (X[tr] - mu) / sd, (X[va] - mu) / sd, (X[te] - mu) / sd

            head = train_head(
                name, Xtr, Xva,
                {kk: vv[tr] for kk, vv in tgt_all.items()},
                {kk: vv[va] for kk, vv in tgt_all.items()},
                R[tr], device, seed=args.seed + k,
            )
            with torch.no_grad():
                s_tr = to_score(name, head(torch.from_numpy(Xtr).to(device))).cpu().numpy()
            calib = affine_calibration(s_tr.astype(np.float64), y_all[tr])
            folds.append(evaluate(name, head, Xte, y_all[te], hist[te], std_all[te],
                                  device, calib=calib))

        agg = {m: float(np.mean([f[m] for f in folds])) for m in folds[0]}

        # ---- transfer: train on ALL of SCUT, test on held-out MEBeauty --------------
        # Repeated over seeds: the objectives turn out to differ by roughly one seed
        # standard deviation, so a single run would invite reading noise as a result.
        t_rho, t_pair = [], []
        for si in range(args.transfer_seeds):
            rng = np.random.default_rng(1000 + si)
            perm = rng.permutation(len(X))
            n_va = len(X) // 10
            va, tr = perm[:n_va], perm[n_va:]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            head = train_head(
                name, (X[tr] - mu) / sd, (X[va] - mu) / sd,
                {kk: vv[tr] for kk, vv in tgt_all.items()},
                {kk: vv[va] for kk, vv in tgt_all.items()},
                R[tr], device, seed=1000 + si,
            )
            with torch.no_grad():
                sm = to_score(
                    name, head(torch.from_numpy(((Xm - mu) / sd).astype(np.float32)).to(device))
                ).cpu().numpy().astype(np.float64)
            t_rho.append(float(stats.spearmanr(ym, sm).statistic))
            t_pair.append(pairwise_accuracy(ym, sm))
        agg["transfer_spearman"] = float(np.mean(t_rho))
        agg["transfer_spearman_std"] = float(np.std(t_rho))
        agg["transfer_spearman_seeds"] = t_rho
        agg["transfer_pairwise"] = float(np.mean(t_pair))
        results[name] = {"cv_mean": agg, "folds": folds}

        print(f"{name:<13} PC={agg['pc']:.4f} MAE={agg['mae']:.4f} rho={agg['spearman']:.4f} "
              f"pair={agg['pairwise_acc']:.4f} ndcg={agg['ndcg@100']:.4f} | "
              f"transfer rho={agg['transfer_spearman']:.4f}"
              f"+-{agg['transfer_spearman_std']:.4f} pair={agg['transfer_pairwise']:.4f}"
              + (f" | KL={agg['kl_to_true_hist']:.4f}" if "kl_to_true_hist" in agg else "")
              + (f" alea_r={agg['aleatoric_vs_true_std']:.3f}"
                 if "aleatoric_vs_true_std" in agg else ""))

    # Is any objective actually distinguishable from any other on transfer?
    names = list(results)
    rhos = {n: np.array(results[n]["cv_mean"]["transfer_spearman_seeds"]) for n in names}
    best = max(names, key=lambda n: rhos[n].mean())
    worst = min(names, key=lambda n: rhos[n].mean())
    pooled = float(np.sqrt((rhos[best].var(ddof=1) + rhos[worst].var(ddof=1)) / 2))
    gap = float(rhos[best].mean() - rhos[worst].mean())
    tt = stats.ttest_ind(rhos[best], rhos[worst])
    results["_significance"] = {
        "best": best, "worst": worst, "gap": gap, "pooled_seed_std": pooled,
        "gap_in_seed_stds": gap / max(pooled, 1e-9), "welch_p": float(tt.pvalue),
    }
    print(f"\nbest={best} ({rhos[best].mean():.4f})  worst={worst} ({rhos[worst].mean():.4f})"
          f"  gap={gap:.4f} = {gap/max(pooled,1e-9):.1f} seed-std  (Welch p={tt.pvalue:.3f})")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    manifest.metrics = {k: v["cv_mean"] for k, v in results.items() if k != "_significance"}
    manifest.metrics["_significance"] = results["_significance"]
    manifest.finish(out)
    print(f"\n[ok] -> {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
