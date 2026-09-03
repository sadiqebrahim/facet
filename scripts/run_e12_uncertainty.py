#!/usr/bin/env python
"""Experiment E12 - which uncertainty method gives a confidence number the UI can show?

exp001 found a concrete defect: bootstrap-ensemble spread achieved 22% coverage at a 68%
nominal level and correlated with actual error at only 0.10. Exposing that as "confidence"
would put an authoritative-looking meaningless number in front of the user, which is the
specific failure docs/RESEARCH.md section 11.4 warns against. This finds a method that works.

Methods compared, all on the E6 production head (LDL) over the E5 production crop (m0.25):

    aleatoric    spread of the predicted rating distribution - irreducible rater disagreement
    ensemble     5 independently seeded heads; spread across members (epistemic)
    mc_dropout   dropout at inference, 30 passes
    tta          horizontal-flip test-time augmentation, 2 views
    combined     sqrt(aleatoric^2 + epistemic^2)
    conformal    split-conformal intervals calibrated on held-out data

Judged on what a confidence number actually has to do:

    coverage      does a nominal 68/90/95% interval contain the truth that often?
    sharpness     how wide are the intervals - a useless-but-honest wide interval is not free
    discrimination does predicted uncertainty rank where the model is actually wrong?

And then the honest test: **does calibration survive the domain shift to MEBeauty?**
Conformal prediction guarantees coverage only under exchangeability, which a domain shift
breaks by construction. Section 11.2 recommends conformal for the UI, so that recommendation
needs checking, not assuming.

Usage:
    python scripts/run_e12_uncertainty.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.data.mebeauty import MEBeauty  # noqa: E402
from facet.data.scut_fbp5500 import ScutFbp5500  # noqa: E402
from facet.training.objectives import (  # noqa: E402
    ScoreHead, aleatoric_std, build_targets, loss_distribution, to_score,
)
from facet.utils.run import RunManifest  # noqa: E402
from facet.utils.seed import seed_everything  # noqa: E402

FEATURE_DIR = ROOT / "artifacts/features"
PARTS = ("arcface_buffalo_l", "clip")
MARGIN = 0.25
LEVELS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def load(dataset, parts, flip=False, margin=MARGIN, size=112):
    mats, order = [], None
    for p in parts:
        prefix = "" if dataset == "scut" else f"{dataset}__"
        st = f"{prefix}{p}__m{margin:g}__s{size}" + ("__flip" if flip else "")
        arr = np.load(FEATURE_DIR / f"{st}.npy").astype(np.float32)
        names = json.loads((FEATURE_DIR / f"{st}.json").read_text())["filenames"]
        if order is None:
            order = names
        elif names != order:
            raise RuntimeError("ordering mismatch")
        arr /= np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)
        mats.append(arr)
    return np.hstack(mats), order


class DropoutHead(nn.Module):
    """LDL head with dropout, so the same weights can be sampled at inference."""

    def __init__(self, in_dim, out_dim=5, p=0.2):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, x):
        return self.fc(self.drop(x))


def train(head, Xtr, tgt_tr, Xva, y_va, device, epochs=400, patience=10, lr=1e-3, wd=1e-4,
          seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    xt, xv = torch.from_numpy(Xtr).to(device), torch.from_numpy(Xva).to(device)
    batch = {"hist": torch.from_numpy(tgt_tr["hist"]).float().to(device)}
    best, state, bad = -np.inf, None, 0
    for ep in range(epochs):
        head.train()
        loss = loss_distribution(head(xt), batch)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0:
            head.eval()
            with torch.no_grad():
                s = to_score("distribution", head(xv)).cpu().numpy()
            rho = stats.spearmanr(y_va, s).statistic
            if np.isfinite(rho) and rho > best:
                best, bad, state = rho, 0, {k: v.detach().clone()
                                            for k, v in head.state_dict().items()}
            else:
                bad += 1
                if bad > patience:
                    break
    if state:
        head.load_state_dict(state)
    head.eval()
    return head


def coverage_report(y, mu, sigma, z_levels=((0.68, 1.0), (0.90, 1.645), (0.95, 1.96))):
    """Gaussian-interval coverage plus sharpness and discrimination."""
    err = np.abs(mu - y)
    sigma = np.clip(sigma, 1e-9, None)
    out = {f"coverage_{int(100*p)}": float((err <= z * sigma).mean()) for p, z in z_levels}
    out["mean_interval_width_90"] = float(2 * 1.645 * sigma.mean())
    out["err_sigma_spearman"] = float(stats.spearmanr(sigma, err).statistic)
    out["mean_sigma"] = float(sigma.mean())
    out["mean_abs_err"] = float(err.mean())
    return out


def conformal_quantile(err_cal, sigma_cal, alpha):
    """Split-conformal: quantile of normalised residuals on a calibration set."""
    n = len(err_cal)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(err_cal / np.clip(sigma_cal, 1e-9, None), q))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scut-root", default=str(ROOT / "data/raw/SCUT-FBP5500_v2"))
    ap.add_argument("--me-root", default=str(ROOT / "data/raw/MEBeauty"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/e12_uncertainty"))
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = RunManifest(
        experiment="e12_uncertainty",
        description="Which uncertainty method produces a confidence the UI can honestly show?",
        config=vars(args), seed=args.seed,
        dataset="SCUT-FBP5500_v2 (+ MEBeauty for shifted-domain calibration)",
        split_methodology=(
            "SCUT official cv1. Training split further divided into fit / conformal "
            "calibration. Domain-shift test: intervals calibrated on SCUT, applied to "
            "MEBeauty after rescaling MEBeauty's 1-10 scale onto SCUT's 1-5."
        ),
    )

    scut = ScutFbp5500(args.scut_root)
    me = MEBeauty(args.me_root)
    X, order = load("scut", PARTS)
    Xf, _ = load("scut", PARTS, flip=True)
    Xm, me_order = load("mebeauty", PARTS)
    Xmf, _ = load("mebeauty", PARTS, flip=True)
    idx = {n: i for i, n in enumerate(order)}

    hist = scut.histogram_matrix(order).astype(np.float64)
    tgt = build_targets(hist)
    y_all = tgt["mean"]
    true_std = scut.labels.loc[order, "std"].to_numpy(np.float64)

    sp = scut.splits["cv1"]
    tr_all = np.array([idx[n] for n in sp.train])
    te = np.array([idx[n] for n in sp.test])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tr_all))
    n_va, n_cal = len(tr_all) // 10, len(tr_all) // 5
    va = tr_all[perm[:n_va]]
    cal = tr_all[perm[n_va : n_va + n_cal]]
    fit = tr_all[perm[n_va + n_cal :]]
    print(f"device={device}  fit={len(fit)} calib={len(cal)} val={len(va)} test={len(te)}\n")

    mu_s, sd_s = X[fit].mean(0), X[fit].std(0) + 1e-6
    def norm(A): return ((A - mu_s) / sd_s).astype(np.float32)

    # ---- deep ensemble of LDL heads -------------------------------------------------
    heads = []
    for m in range(args.members):
        h = ScoreHead(X.shape[1], 5).to(device)
        heads.append(train(h, norm(X[fit]), {k: v[fit] for k, v in tgt.items()},
                           norm(X[va]), y_all[va], device, seed=args.seed + m))
    drop = train(DropoutHead(X.shape[1]).to(device), norm(X[fit]),
                 {k: v[fit] for k, v in tgt.items()}, norm(X[va]), y_all[va], device,
                 seed=args.seed)

    def predict(A, Aflip):
        """Return every uncertainty channel for a feature matrix."""
        t = torch.from_numpy(norm(A)).to(device)
        tf = torch.from_numpy(norm(Aflip)).to(device)
        with torch.no_grad():
            outs = [h(t) for h in heads]
            scores = np.stack([to_score("distribution", o).cpu().numpy() for o in outs])
            alea = np.stack([aleatoric_std("distribution", o).cpu().numpy() for o in outs])
            mu = scores.mean(0)
            epi = scores.std(0)
            alea_m = alea.mean(0)
            flip_s = np.stack([to_score("distribution", h(tf)).cpu().numpy() for h in heads])
            tta = np.stack([scores.mean(0), flip_s.mean(0)]).std(0)
            drop.train()  # dropout active at inference
            mc = np.stack([to_score("distribution", drop(t)).cpu().numpy() for _ in range(30)])
            drop.eval()
            mcd = mc.std(0)
        return {
            "mu": mu.astype(np.float64),
            "aleatoric": alea_m.astype(np.float64),
            "ensemble": epi.astype(np.float64),
            "mc_dropout": mcd.astype(np.float64),
            "tta": tta.astype(np.float64),
            "combined": np.sqrt(alea_m**2 + epi**2).astype(np.float64),
        }

    P_cal = predict(X[cal], Xf[cal])
    P_te = predict(X[te], Xf[te])
    channels = ["aleatoric", "ensemble", "mc_dropout", "tta", "combined"]
    results: dict = {"in_domain": {}, "shifted": {}, "conformal": {}}

    print("=== In-domain (SCUT cv1 test): raw Gaussian intervals ===")
    print(f"{'method':<12} {'cov68':>7} {'cov90':>7} {'cov95':>7} {'mean sig':>9} "
          f"{'width90':>8} {'err corr':>9}")
    for ch in channels:
        r = coverage_report(y_all[te], P_te["mu"], P_te[ch])
        results["in_domain"][ch] = r
        print(f"{ch:<12} {r['coverage_68']:>7.3f} {r['coverage_90']:>7.3f} "
              f"{r['coverage_95']:>7.3f} {r['mean_sigma']:>9.4f} "
              f"{r['mean_interval_width_90']:>8.3f} {r['err_sigma_spearman']:>9.3f}")
    print(f"{'(true MAE)':<12} {'':>7} {'':>7} {'':>7} {'':>9} {'':>8} "
          f"  MAE={np.abs(P_te['mu']-y_all[te]).mean():.4f}")

    print("\n=== Conformal recalibration (scale factor fitted on held-out calibration set) ===")
    print(f"{'method':<12} {'q68':>7} {'q90':>7} {'q95':>7} | {'cov68':>7} {'cov90':>7} {'cov95':>7} {'width90':>8}")
    err_cal = np.abs(P_cal["mu"] - y_all[cal])
    for ch in channels:
        qs = {a: conformal_quantile(err_cal, P_cal[ch], a) for a in (0.32, 0.10, 0.05)}
        err_te = np.abs(P_te["mu"] - y_all[te])
        cov = {a: float((err_te <= qs[a] * P_te[ch]).mean()) for a in qs}
        results["conformal"][ch] = {
            "q": {str(k): v for k, v in qs.items()},
            "coverage": {str(k): v for k, v in cov.items()},
            "width90": float(2 * qs[0.10] * P_te[ch].mean()),
        }
        print(f"{ch:<12} {qs[0.32]:>7.2f} {qs[0.10]:>7.2f} {qs[0.05]:>7.2f} | "
              f"{cov[0.32]:>7.3f} {cov[0.10]:>7.3f} {cov[0.05]:>7.3f} "
              f"{2*qs[0.10]*P_te[ch].mean():>8.3f}")

    # ---- does calibration survive a domain shift? ------------------------------------
    print("\n=== Domain shift: intervals calibrated on SCUT, applied to MEBeauty ===")
    P_me = predict(Xm, Xmf)
    ym = me.labels.loc[me_order, "mean"].to_numpy(np.float64)
    # MEBeauty is 1-10, SCUT is 1-5. Map by matching rank position, which is the only
    # scale-free way to compare an absolute interval across incommensurable scales.
    ym_r = stats.rankdata(ym) / len(ym)
    y_scut_sorted = np.sort(y_all)
    ym_on_scut = y_scut_sorted[np.clip((ym_r * len(y_scut_sorted)).astype(int), 0,
                                       len(y_scut_sorted) - 1)]
    print(f"{'method':<12} {'cov68':>7} {'cov90':>7} {'cov95':>7}   (nominal 0.68 / 0.90 / 0.95)")
    err_me = np.abs(P_me["mu"] - ym_on_scut)
    for ch in channels:
        qs = results["conformal"][ch]["q"]
        cov = {a: float((err_me <= float(qs[a]) * P_me[ch]).mean()) for a in qs}
        results["shifted"][ch] = cov
        print(f"{ch:<12} {cov['0.32']:>7.3f} {cov['0.1']:>7.3f} {cov['0.05']:>7.3f}")

    results["aleatoric_vs_true_rater_std"] = float(
        stats.spearmanr(P_te["aleatoric"], true_std[te]).statistic
    )
    print(f"\ncorr(predicted aleatoric, TRUE rater std) = "
          f"{results['aleatoric_vs_true_rater_std']:.3f}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    manifest.metrics = results
    manifest.finish(out)
    print(f"\n[ok] -> {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
